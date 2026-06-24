'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

InferenceX SGLang parity suite config schema (``framework: inferencex_atom_sglang_single``).

Single-node ROCm SGLang: ``sglang.launch_server`` + ``sglang.bench_serving``.
'''

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from pydantic import model_validator
from typing_extensions import Literal

from cvs.lib.inference.utils.inferencex_atom_config_loader import (
    InferenceXAtomRunCard,
    placeholder_gated_threshold_cell,
)
from cvs.lib.inference.utils.inferencing_config_loader import RoleServer, Sweep
from cvs.lib.inference.utils.vllm_parsing import GATED_METRICS
from cvs.lib.utils.config_loader import BaseVariantConfig, _Forbid, substitute_config


class InferenceXAtomSglangRoleServer(RoleServer):
    # Extra CLI tokens for ``python -m sglang.launch_server`` after base flags.
    sglang_args: List[str] = []


class InferenceXAtomSglangRoles(_Forbid):
    server: InferenceXAtomSglangRoleServer = InferenceXAtomSglangRoleServer()


class InferenceXAtomSglangParams(_Forbid):
    backend: str = "vllm"
    base_url: str = "http://0.0.0.0"
    port_no: str = "8000"
    dataset_name: str = "random"
    burstiness: str = "1.0"
    seed: str = "0"
    request_rate: str = "inf"
    random_range_ratio: str = "0.8"
    random_prefix_len: str = "0"
    tensor_parallelism: str = "8"
    tokenizer_mode: str = "auto"
    percentile_metrics: str = "ttft,tpot,itl,e2el"
    metric_percentiles: str = "99"
    num_prompts: str = "1000"
    max_model_length: str = "8192"
    client_poll_count: str = "50"
    client_poll_wait_time: str = "60"
    bench_max_failed_requests: str = "0"
    bench_extra_args: str = ""
    result_filename: str = "results"


class InferenceXAtomSglangVariantConfig(BaseVariantConfig):
    framework: Literal["inferencex_atom_sglang_single"]
    gpu_arch: str
    ix_recipe_id: Optional[str] = None
    run_card: InferenceXAtomRunCard = InferenceXAtomRunCard()
    roles: InferenceXAtomSglangRoles = InferenceXAtomSglangRoles()
    params: InferenceXAtomSglangParams
    sweep: Sweep

    def cell_key(self, isl, osl, concurrency):
        return f"ISL={isl},OSL={osl},TP={self.params.tensor_parallelism},CONC={concurrency}"

    def expected_cells(self) -> List[str]:
        by_name = {c.name: c for c in self.sweep.sequence_combinations}
        return [
            self.cell_key(by_name[r.combo].isl, by_name[r.combo].osl, r.concurrency)
            for r in self.sweep.runs
        ]

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


def load_variant(config_path, cluster_dict) -> InferenceXAtomSglangVariantConfig:
    """Load SGLang parity variant config + sibling ``*threshold.json``."""
    raw, thresholds = substitute_config(config_path, cluster_dict)
    raw["thresholds"] = thresholds
    return InferenceXAtomSglangVariantConfig(**raw)


def orchestrator_container_from_variant(variant: InferenceXAtomSglangVariantConfig) -> Dict[str, Any]:
    block = variant.container.model_dump()
    server_env = variant.roles.server.env
    if server_env:
        block = {**block, "env": dict(server_env)}
    return block


__all__ = [
    "InferenceXAtomSglangVariantConfig",
    "load_variant",
    "orchestrator_container_from_variant",
    "placeholder_gated_threshold_cell",
]
