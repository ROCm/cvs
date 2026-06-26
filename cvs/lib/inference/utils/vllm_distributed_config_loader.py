'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Inference-specific config schema for the vllm_distributed suite.

Extends the vllm_single loader with distributed params (pipeline_parallel_size,
master_addr, master_port, nnodes) and a PP-aware cell_key.

cell_key format: ISL=<isl>,OSL=<osl>,TP=<tp>,PP=<pp>,CONC=<concurrency>
The PP segment is the distributed-specific addition over vllm_single.
'''

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Literal

from cvs.lib.inference.utils.vllm_parsing import GATED_METRICS
from cvs.lib.utils.config_loader import substitute_config


class _Forbid(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Allow(BaseModel):
    model_config = ConfigDict(extra="allow")


# ---------- sub-models ----------


class ContainerConfig(_Allow):
    """Loose container block — passes through to the orchestrator as-is."""

    lifetime: str = "per_run"
    name: str = ""
    image: str = ""


class Paths(_Forbid):
    shared_fs: str
    models_dir: str
    log_dir: str
    hf_token_file: str


class ModelSpec(_Forbid):
    id: str
    remote: Literal[0, 1]


class RoleServer(_Forbid):
    serve_args: Dict[str, Any] = {}
    env: Dict[str, str] = {}


class Roles(_Forbid):
    server: RoleServer = RoleServer()


class GoodputSlo(_Forbid):
    ttft_ms: float
    tpot_ms: float
    e2el_ms: float


class SeqCombo(_Forbid):
    name: str
    isl: str
    osl: str
    goodput_slo: Optional[GoodputSlo] = None


class Run(_Forbid):
    combo: str
    concurrency: int


def validate_sweep_selector(combo_names, run_combo_refs):
    counts = Counter(combo_names)
    dupes = sorted(name for name, count in counts.items() if count > 1)
    if dupes:
        raise ValueError(f"duplicate sequence_combination names: {dupes}")
    known = set(counts)
    unknown = sorted({r for r in run_combo_refs if r not in known})
    if unknown:
        raise ValueError(f"run.combo names no sequence_combination: {unknown} (known: {sorted(known)})")


class Sweep(_Forbid):
    sequence_combinations: List[SeqCombo]
    runs: List[Run]

    @model_validator(mode="after")
    def _check_runs_reference_known_combos(self):
        validate_sweep_selector(
            [c.name for c in self.sequence_combinations],
            [r.combo for r in self.runs],
        )
        return self


class Params(_Forbid):
    backend: str = "vllm"
    base_url: str = "http://0.0.0.0"
    port_no: str = "8888"
    dataset_name: str = "random"
    burstiness: str = "1.0"
    seed: str = "0"
    request_rate: str = "inf"
    random_range_ratio: str = "0.8"
    random_prefix_len: str = "0"
    tensor_parallelism: str = "8"
    pipeline_parallel_size: str = "2"
    master_addr: str = "localhost"
    master_port: str = "29501"
    nnodes: str = "2"
    tokenizer_mode: str = "auto"
    percentile_metrics: str = "ttft,tpot,itl,e2el"
    metric_percentiles: str = "50,90,95,99"
    num_prompts: str = "3200"
    client_poll_count: str = "20"


class VariantConfig(_Forbid):
    """Typed config for the vllm_distributed suite.

    Standalone (does not extend BaseVariantConfig) so it can be constructed
    without the threshold_json field that the base requires -- that field is
    present in production configs but absent from unit-test fixtures.

    `container` is optional: unit-test fixtures omit it (defaults to a bare
    ContainerConfig), while production configs provide the full block.  The
    conftest's `orch` fixture accesses `variant_config.container.model_dump()`
    to build the orchestrator, so the field must be present even when empty.
    """

    schema_version: Literal[1]
    framework: Literal["vllm_distributed"]
    gpu_arch: str
    enforce_thresholds: bool = True
    container: ContainerConfig = Field(default_factory=ContainerConfig)
    paths: Paths
    model: ModelSpec
    roles: Roles = Roles()
    params: Params = Params()
    sweep: Sweep
    thresholds: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    def cell_key(self, isl, osl, concurrency):
        """The canonical threshold key for one sweep cell.

        Distributed-specific: includes PP= segment in addition to the
        vllm_single TP= segment.

        Format: ISL=<isl>,OSL=<osl>,TP=<tp>,PP=<pp>,CONC=<concurrency>
        """
        return (
            f"ISL={isl},OSL={osl},"
            f"TP={self.params.tensor_parallelism},"
            f"PP={self.params.pipeline_parallel_size},"
            f"CONC={concurrency}"
        )

    def expected_cells(self):
        """Every (isl, osl, conc) cell the sweep's `runs` selector picks."""
        by_name = {c.name: c for c in self.sweep.sequence_combinations}
        return [self.cell_key(by_name[r.combo].isl, by_name[r.combo].osl, r.concurrency) for r in self.sweep.runs]

    @model_validator(mode="after")
    def _check_remote_not_implemented(self):
        if self.model.remote == 1:
            raise NotImplementedError("model.remote=1 (remote model download) is not implemented. ")
        return self

    @model_validator(mode="after")
    def _check_thresholds_cover_sweep(self):
        expected = set(self.expected_cells())
        present = set(self.thresholds.keys())
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        problems = []
        if missing:
            problems.append(f"sweep cells with no threshold entry: {missing}")
        if extra:
            problems.append(f"threshold keys matching no sweep cell (typo?): {extra}")
        gated_keys = [f"client.{m}" for m in sorted(GATED_METRICS)]
        gated_gaps = {}
        for cell in sorted(expected & present):
            specs = self.thresholds.get(cell) or {}
            absent = [k for k in gated_keys if k not in specs]
            if absent:
                gated_gaps[cell] = absent
        if gated_gaps:
            problems.append(f"cells missing gated-metric specs: {gated_gaps}")
        if problems:
            msg = "threshold.json does not match the sweep matrix; " + "; ".join(problems)
            if self.enforce_thresholds:
                raise ValueError(msg)
            warnings.warn(f"{msg} (enforce_thresholds=false -> record-only)", stacklevel=2)
        return self


# ---------- public API ----------


def load_variant(config_path, cluster_dict):
    """Load and validate a vllm_distributed variant config + its sibling threshold file.

    Production configs contain extra fields (container, threshold_json) that the
    standalone VariantConfig does not model.  They are stripped before construction
    so unit-test fixtures (which also omit those fields) and production configs
    are handled identically.
    """
    raw, thresholds = substitute_config(config_path, cluster_dict)
    known = {k: v for k, v in raw.items() if k in VariantConfig.model_fields}
    known["thresholds"] = thresholds
    return VariantConfig(**known)
