'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

InferenceX ATOM suite config schema (``framework: inferencex_atom_single``).

Generic paths/model/container/threshold plumbing lives in
:mod:`cvs.lib.utils.config_loader`. Sweep selector types are shared with
:mod:`cvs.lib.inference.utils.inferencing_config_loader`.
'''

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from pydantic import model_validator
from typing_extensions import Literal

from cvs.lib.inference.utils.inferencex_atom_recipes import apply_ix_recipe

from cvs.lib.inference.utils.inferencing_config_loader import (
    RoleServer,
    Sweep,
    validate_sweep_selector,
    validate_thresholds_cover_sweep,
)
from cvs.lib.inference.utils.inferencex_atom_parsing import GATED_METRICS
from cvs.lib.utils.config_loader import BaseVariantConfig, _Forbid, substitute_config


class InferenceXAtomRoleServer(RoleServer):
    # Extra CLI tokens for ``python -m atom.entrypoints.openai_server`` after
    # ``--model`` / ``--server-port`` (e.g. ``-tp``, ``--kv_cache_dtype``).
    atom_args: List[str] = []


class InferenceXAtomRoles(_Forbid):
    server: InferenceXAtomRoleServer = InferenceXAtomRoleServer()


class InferenceXAtomParams(_Forbid):
    # ``atom`` uses ATOM openai_server + benchmark_serving; ``vllm`` keeps the
    # interim uplift path (vllm serve + vllm bench serve).
    driver: Literal["atom", "vllm"] = "vllm"
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
    metric_percentiles: str = "95,99"
    num_prompts: str = "1000"
    max_model_length: str = "8192"
    client_poll_count: str = "50"
    client_poll_wait_time: str = "60"
    client_initial_wait_s: str = "120"
    server_precheck_wait_s: str = "30"
    server_warmup_wait_s: str = "330"
    server_poll_count: str = "60"
    server_poll_wait_time: str = "60"
    reuse_server_across_sweep: str = "true"
    bench_max_failed_requests: str = "0"
    bench_extra_args: str = ""
    result_filename: str = "results"


class InferenceXAtomRunCard(_Forbid):
    upstream_run_url: str = ""
    atom_image_pin: str = ""
    notes: str = ""


class InferenceXAtomVariantConfig(BaseVariantConfig):
    framework: Literal["inferencex_atom_single"]
    gpu_arch: str
    ix_recipe_id: Optional[str] = None
    run_card: InferenceXAtomRunCard = InferenceXAtomRunCard()
    roles: InferenceXAtomRoles = InferenceXAtomRoles()
    params: InferenceXAtomParams
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
        validate_thresholds_cover_sweep(
            expected_cells=self.expected_cells(),
            thresholds=self.thresholds,
            enforce_thresholds=self.enforce_thresholds,
            gated_metrics=GATED_METRICS,
        )
        return self


def expand_sweep(sweep):
    """Expand a sweep into ``(cases, ids)`` for pytest parametrization."""
    if hasattr(sweep, "sequence_combinations"):
        combos = [c.model_dump() for c in sweep.sequence_combinations]
        runs = [r.model_dump() for r in sweep.runs]
    else:
        combos = sweep.get("sequence_combinations", [])
        runs = sweep.get("runs", [])
    validate_sweep_selector([c["name"] for c in combos], [r["combo"] for r in runs])
    by_name = {c["name"]: c for c in combos}
    cases = []
    ids = []
    for run in runs:
        combo = by_name[run["combo"]]
        conc = run["concurrency"]
        cases.append((combo, conc))
        ids.append(f"{run['combo']}-conc{conc}")
    return cases, ids


def load_variant(config_path, cluster_dict) -> InferenceXAtomVariantConfig:
    raw, thresholds = substitute_config(config_path, cluster_dict)
    raw = apply_ix_recipe(raw)
    raw["thresholds"] = thresholds
    return InferenceXAtomVariantConfig(**raw)


def placeholder_gated_threshold_cell(
    *,
    output_throughput_min: float = 0,
    total_token_throughput_min: float = 0,
    per_gpu_throughput_min: float = 0,
    output_tput_per_gpu_min: float = 0,
    mean_ttft_max_ms: float = 1_000_000,
    p99_ttft_max_ms: float = 1_000_000,
    mean_tpot_max_ms: float = 1_000_000,
    p95_tpot_max_ms: float = 1_000_000,
    failed_max: int = 1_000_000_000,
    success_rate_min: float = 0,
) -> Dict[str, Any]:
    """Return one sweep cell's ``client.*`` specs covering every ``GATED_METRICS`` member."""
    loose_ms = {"kind": "max_ms", "value": 1_000_000}
    loose_tok = {"kind": "min_tok_s", "value": 0}
    return {
        "client.total_token_throughput": {"kind": "min_tok_s", "value": total_token_throughput_min},
        "client.output_throughput": {"kind": "min_tok_s", "value": output_throughput_min},
        "client.per_gpu_throughput": {"kind": "min_tok_s", "value": per_gpu_throughput_min},
        "client.output_tput_per_gpu": {"kind": "min_tok_s", "value": output_tput_per_gpu_min},
        "client.mean_ttft_ms": {"kind": "max_ms", "value": mean_ttft_max_ms},
        "client.median_ttft_ms": loose_ms,
        "client.p90_ttft_ms": loose_ms,
        "client.p95_ttft_ms": loose_ms,
        "client.p99_ttft_ms": {"kind": "max_ms", "value": p99_ttft_max_ms},
        "client.mean_tpot_ms": {"kind": "max_ms", "value": mean_tpot_max_ms},
        "client.median_tpot_ms": loose_ms,
        "client.p90_tpot_ms": loose_ms,
        "client.p95_tpot_ms": {"kind": "max_ms", "value": p95_tpot_max_ms},
        "client.p99_tpot_ms": loose_ms,
        "client.mean_itl_ms": loose_ms,
        "client.median_itl_ms": loose_ms,
        "client.p95_itl_ms": loose_ms,
        "client.p99_itl_ms": loose_ms,
        "client.mean_e2el_ms": loose_ms,
        "client.median_e2el_ms": loose_ms,
        "client.p90_e2el_ms": loose_ms,
        "client.p95_e2el_ms": loose_ms,
        "client.p99_e2el_ms": loose_ms,
        "client.success_rate": {"kind": "min", "value": success_rate_min},
        "client.failed": {"kind": "max", "value": failed_max},
    }


def orchestrator_container_from_variant(variant: InferenceXAtomVariantConfig) -> Dict[str, Any]:
    """``container`` block for :class:`OrchestratorConfig` (includes server env)."""
    block = variant.container.model_dump()
    server_env = variant.roles.server.env
    if server_env:
        block = {**block, "env": dict(server_env)}
    return block
