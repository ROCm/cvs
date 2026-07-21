'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Training-specific config schema for the jaxmaxtext suite.

The framework-agnostic machinery (paths/model/container schema, the 3-pass
placeholder substitution, the `enforce_thresholds` gate, and the
`substitute_config` file-read helper) lives in `cvs.lib.utils.config_loader`.
This module holds the training half: the MaxText config, tokenizer, NCCL,
JAX distributed settings, RDMA lib, and `TrainingVariantConfig(BaseVariantConfig)`
with the STEPS/BATCH/SEQLEN/NODES `cell_key`.

A training suite does not sweep cells the way inference does (no NxM matrix of
ISL/OSL/concurrency). Instead, a single training run produces one cell key
derived from steps, batch size, sequence length, and node count. The node count
is injected at runtime from `len(orch.hosts)` via `cell_key(num_nodes=...)`.
'''

from __future__ import annotations

import warnings
from typing import Any, Dict, Literal

from cvs.lib.utils.config_loader import BaseVariantConfig, _Allow, _Forbid, substitute_config
from cvs.lib.training.jax.utils.maxtext_parsing import GATED_METRICS


class Tokenizer(_Forbid):
    hf_model_id: str
    tokenizer_path: str


class NcclConfig(_Allow):
    ib_hca_list: str = ""
    ib_hca: str = ""
    socket_ifname: str = ""
    gloo_socket_ifname: str = ""
    ib_tc: str = "41"
    ib_sl: str = "0"
    ib_gid_index: str = "3"


class JaxDistributed(_Forbid):
    coordinator_ip: str = ""
    coordinator_port: str = "12346"
    initialization_timeout_seconds: str = "1800"
    heartbeat_timeout_seconds: str = "900"


class RdmaLib(_Allow):
    host_source_file: str = ""
    container_mount_file: str = ""
    container_dest_file: str = ""


class TrainingConfig(_Allow):
    distributed: bool = True
    steps: int = 30
    enable_checkpointing: bool = False
    train_script: str = "/workspace/maxtext/src/MaxText/train.py"
    maxtext_config: Dict[str, Any] = {}
    tokenizer: Tokenizer
    nic_type: str = "thor2"
    rdma_lib: RdmaLib = RdmaLib()
    env_vars: Dict[str, str] = {}
    xla_flags: Dict[str, str] = {}
    nccl: NcclConfig = NcclConfig()
    jax_distributed: JaxDistributed = JaxDistributed()


def validate_thresholds_cover_training(
    *,
    expected_cells,
    thresholds,
    enforce_thresholds: bool,
    gated_metrics=None,
) -> None:
    """Shared training threshold/cell coverage check."""
    expected = set(expected_cells)
    present = set(thresholds.keys())
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    problems = []
    if missing:
        problems.append(f"training cells with no threshold entry: {missing}")
    if extra:
        problems.append(f"threshold keys matching no training cell (typo?): {extra}")
    gated = gated_metrics if gated_metrics is not None else GATED_METRICS
    gated_keys = [f"training.{m}" for m in sorted(gated)]
    gated_gaps = {}
    for cell in sorted(expected & present):
        specs = thresholds.get(cell) or {}
        absent = [k for k in gated_keys if k not in specs]
        if absent:
            gated_gaps[cell] = absent
    if gated_gaps:
        problems.append(f"cells missing gated-metric specs: {gated_gaps}")
    if problems:
        msg = "threshold.json does not match the training config; " + "; ".join(problems)
        if enforce_thresholds:
            raise ValueError(msg)
        warnings.warn(f"{msg} (enforce_thresholds=false -> record-only)", stacklevel=3)


class TrainingVariantConfig(BaseVariantConfig):
    framework: Literal["jaxmaxtext"]
    gpu_arch: str
    training: TrainingConfig

    def cell_key(self, num_nodes=1):
        """Canonical threshold key for this training run.

        `num_nodes` is injected at runtime from `len(orch.hosts)` so the same
        config works on different cluster sizes without editing the JSON.
        """
        t = self.training
        mc = t.maxtext_config
        batch = mc.get("per_device_batch_size", "-")
        seqlen = mc.get("max_target_length", "-")
        return f"STEPS={t.steps},BATCH={batch},SEQLEN={seqlen},NODES={num_nodes}"

    def expected_cells(self, num_nodes=1):
        """The single cell this training config produces."""
        return [self.cell_key(num_nodes=num_nodes)]


# ---------- public API (training) ----------


def load_training_variant(config_path, cluster_dict):
    """Load and validate a jaxmaxtext variant config + its sibling threshold file.

    Delegates the file read + placeholder substitution + threshold discovery to
    the generic `substitute_config`, then attaches the thresholds and builds the
    typed `TrainingVariantConfig`.
    """
    raw, thresholds = substitute_config(config_path, cluster_dict)
    raw["thresholds"] = thresholds
    return TrainingVariantConfig(**raw)
