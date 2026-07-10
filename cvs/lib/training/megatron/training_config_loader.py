'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Training-specific config schema for Megatron suites (single-node and distributed).

The framework-agnostic machinery (ContainerSpec, RuntimeSpec, placeholder
substitution helpers) lives in `cvs.lib.utils.config_loader`. This module holds
the training half: MegatronSweepCombo, MegatronSweep, MegatronVariantConfig,
and load_training_variant.

Unlike inference configs, Megatron configs carry thresholds inline inside each
sweep combo's result_dict — no sibling *threshold.json file is needed.
enforce_thresholds gates whether those inline thresholds are asserted in
test_throughput.

Both megatron_single and megatron_distributed are covered by MegatronVariantConfig
via the framework field. The test suite reads variant_config.framework to decide
whether to pass distributed_training=True to MegatronTrainingJob.
'''

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from pydantic import Field, model_validator
from typing_extensions import Literal

from cvs.lib.utils.config_loader import (
    ContainerSpec,
    _Forbid,
    _resolve_cluster_mapping,
    _walk_substitute,
)


# ---------- pydantic models (training) ----------


class MegatronSweepCombo(_Forbid):
    name: str
    micro_batch_size: str
    global_batch_size: str
    precision: str = ""
    result_dict: Dict[str, Any] = Field(default_factory=dict)


def validate_sweep_selector(combo_keys, run_refs):
    """The sweep-selector rule: combination keys unique, every run references one.

    Single home for this check, shared by the typed MegatronSweep validator
    (load time) and pytest_generate_tests (collection time, which reads raw
    JSON before the loader runs) so the two can never drift.

    Without it a typo'd run key is a silently-dropped cell — the sweep runs
    a different matrix than the config reads.
    """
    counts = Counter(combo_keys)
    dupes = sorted(k for k, count in counts.items() if count > 1)
    if dupes:
        raise ValueError(f"duplicate sweep.combinations keys: {dupes}")
    known = set(counts)
    unknown = sorted(r for r in run_refs if r not in known)
    if unknown:
        raise ValueError(
            f"sweep.runs references unknown combinations: {unknown} (known: {sorted(known)})"
        )


class MegatronSweep(_Forbid):
    combinations: Dict[str, MegatronSweepCombo]
    runs: List[str]

    @model_validator(mode="after")
    def _check_runs_reference_known_combos(self):
        validate_sweep_selector(
            list(self.combinations.keys()),
            self.runs,
        )
        return self


class MegatronVariantConfig(_Forbid):
    schema_version: Literal[1]
    framework: Literal["megatron_single", "megatron_distributed"]
    gpu_arch: str
    enforce_thresholds: bool = True
    config: Dict[str, Any]        # training knobs: megatron_root, nccl_*, nic_type, ...
    model_params: Dict[str, Any]  # model knobs: model_name, precision, tp, pp, ...
    container: ContainerSpec
    sweep: MegatronSweep


# ---------- public API (training) ----------


def _check_no_changeme(node, path="", _offenders=None):
    """Recursively collect config fields whose value still contains '<changeme>'.

    Collects all offending dotted paths so the caller can report them all at once.
    """
    if _offenders is None:
        _offenders = []
    if isinstance(node, dict):
        for k, v in node.items():
            _check_no_changeme(v, f"{path}.{k}" if path else k, _offenders)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _check_no_changeme(v, f"{path}[{i}]", _offenders)
    elif isinstance(node, str) and "<changeme>" in node:
        _offenders.append(path)
    if not path:
        if _offenders:
            raise ValueError(
                f"config has unfilled placeholder '<changeme>' in: {', '.join(_offenders)}"
            )


def load_training_variant(config_path, cluster_dict) -> MegatronVariantConfig:
    """Load and validate a Megatron training variant config.

    Delegates placeholder substitution to _resolve_cluster_mapping +
    _walk_substitute (pass 1 only — no paths block, no cross-block refs).
    No sibling threshold file is required — thresholds live inline in
    sweep.combinations.<id>.result_dict.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"training config not found: {config_path}")

    raw = json.loads(config_path.read_text())
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}

    cluster_map = _resolve_cluster_mapping(cluster_dict)
    raw = _walk_substitute(raw, cluster_map)

    _check_no_changeme(raw)

    return MegatronVariantConfig(**raw)
