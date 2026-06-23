'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

InferenceMax suite config schema (``framework: inferencemax_single``).

Generic paths/model/container/threshold plumbing lives in
:mod:`cvs.lib.utils.config_loader`. Sweep selector types are shared with
:mod:`cvs.lib.inference.utils.inferencing_config_loader`. This module adds
InferenceMax-specific ``params``, ``inferencemax`` options, and adapters that
produce the legacy ``inference_dict`` / ``benchmark_params`` shapes still
consumed by :class:`cvs.lib.inference.inferencemax_orch.InferenceMaxJob` until
the driver is ported (Phase 3).
'''

from __future__ import annotations

import warnings
from typing import Any, Dict, List

from pydantic import model_validator
from typing_extensions import Literal

from cvs.lib.inference.utils.inferencing_config_loader import (
    Roles,
    Sweep,
)
from cvs.lib.inference.utils.vllm_parsing import GATED_METRICS
from cvs.lib.utils.config_loader import BaseVariantConfig, _Forbid, substitute_config

# Legacy verify_inference_results metric names keyed from client.* threshold specs.
_CLIENT_TO_LEGACY_RESULT = {
    "client.output_throughput": "output_throughput_per_sec",
    "client.mean_ttft_ms": "mean_ttft_ms",
    "client.mean_tpot_ms": "mean_tpot_ms",
}


class InferenceMaxOptions(_Forbid):
    use_host_mounted_server_script: bool = True
    benchmark_server_script_path: str = "auto"
    host_benchmark_scripts_relpath: str = "lib/dtni/vllm_benchmark_scripts"
    vllm_enforce_eager: bool = True
    inferencemax_repo: str = "https://github.com/SemiAnalysisAI/InferenceX.git"
    benchmark_script_repo: str = "https://github.com/kimbochen/bench_serving.git"


class InferenceMaxParams(_Forbid):
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
    server_script: str = "fixed_seq_len/vllm_serve_mi300x.sh"
    bench_serv_script: str = "benchmark_serving.py"


class InferenceMaxVariantConfig(BaseVariantConfig):
    framework: Literal["inferencemax_single"]
    gpu_arch: str
    roles: Roles = Roles()
    params: InferenceMaxParams
    sweep: Sweep
    inferencemax: InferenceMaxOptions = InferenceMaxOptions()

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


def load_variant(config_path, cluster_dict) -> InferenceMaxVariantConfig:
    """Load and validate an InferenceMax variant config + sibling ``*threshold.json``."""
    raw, thresholds = substitute_config(config_path, cluster_dict)
    raw["thresholds"] = thresholds
    return InferenceMaxVariantConfig(**raw)


def benchmark_model_key(variant: InferenceMaxVariantConfig) -> str:
    """``benchmark_params`` dict key (basename of ``model.id``)."""
    return variant.model.id.rsplit("/", 1)[-1]


def legacy_inference_dict_from_variant(variant: InferenceMaxVariantConfig) -> Dict[str, Any]:
    """Build the legacy ``config`` object for :class:`InferenceMaxJob`."""
    runtime_args = variant.container.runtime.args
    env = dict(variant.roles.server.env)
    return {
        "container_image": variant.container.image,
        "container_name": variant.container.name,
        "use_host_mounted_server_script": variant.inferencemax.use_host_mounted_server_script,
        "benchmark_server_script_path": variant.inferencemax.benchmark_server_script_path,
        "host_benchmark_scripts_relpath": variant.inferencemax.host_benchmark_scripts_relpath,
        "vllm_enforce_eager": variant.inferencemax.vllm_enforce_eager,
        "inferencemax_repo": variant.inferencemax.inferencemax_repo,
        "benchmark_script_repo": variant.inferencemax.benchmark_script_repo,
        "hf_token_file": variant.paths.hf_token_file,
        "shm_size": str(runtime_args.get("shm_size", "128G")),
        "log_dir": variant.paths.log_dir,
        "nnodes": "1",
        "distributed_inference": False,
        "container_config": {
            "device_list": list(runtime_args.get("devices") or ["/dev/dri", "/dev/kfd"]),
            "volume_dict": _volume_dict_from_container(variant),
            "env_dict": env,
        },
    }


def _legacy_result_dict_from_thresholds(variant: InferenceMaxVariantConfig) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for cell_key, specs in variant.thresholds.items():
        if not isinstance(specs, dict):
            continue
        legacy: Dict[str, str] = {}
        for client_key, legacy_key in _CLIENT_TO_LEGACY_RESULT.items():
            spec = specs.get(client_key)
            if isinstance(spec, dict) and "value" in spec:
                legacy[legacy_key] = str(spec["value"])
        if legacy:
            out[cell_key] = legacy
    return out


def legacy_benchmark_params_from_variant(variant: InferenceMaxVariantConfig) -> Dict[str, Dict[str, Any]]:
    """Build the legacy ``benchmark_params`` object for :class:`InferenceMaxJob`."""
    key = benchmark_model_key(variant)
    by_name = {c.name: c for c in variant.sweep.sequence_combinations}
    first = variant.sweep.runs[0]
    combo = by_name[first.combo]
    cell = {
        **variant.params.model_dump(),
        "model": variant.model.id,
        "input_sequence_length": str(combo.isl),
        "output_sequence_length": str(combo.osl),
        "max_concurrency": str(first.concurrency),
        "result_dict": _legacy_result_dict_from_thresholds(variant),
    }
    return {key: cell}


def placeholder_gated_threshold_cell(
    *,
    output_throughput_min: float = 0,
    mean_ttft_max_ms: float = 1_000_000,
    mean_tpot_max_ms: float = 1_000_000,
    failed_max: int = 1_000_000_000,
    success_rate_min: float = 0,
) -> Dict[str, Any]:
    """Return one sweep cell's ``client.*`` specs covering every ``GATED_METRICS`` member."""
    loose_ms = {"kind": "max_ms", "value": 1_000_000}
    loose_tok = {"kind": "min_tok_s", "value": 0}
    cell: Dict[str, Any] = {
        "client.total_token_throughput": loose_tok,
        "client.output_throughput": {"kind": "min_tok_s", "value": output_throughput_min},
        "client.mean_ttft_ms": {"kind": "max_ms", "value": mean_ttft_max_ms},
        "client.median_ttft_ms": loose_ms,
        "client.p90_ttft_ms": loose_ms,
        "client.p95_ttft_ms": loose_ms,
        "client.p99_ttft_ms": loose_ms,
        "client.mean_tpot_ms": {"kind": "max_ms", "value": mean_tpot_max_ms},
        "client.median_tpot_ms": loose_ms,
        "client.p90_tpot_ms": loose_ms,
        "client.p95_tpot_ms": loose_ms,
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
    return cell


def _volume_dict_from_container(variant: InferenceMaxVariantConfig) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for vol in variant.container.runtime.args.get("volumes") or []:
        if isinstance(vol, str) and ":" in vol:
            host, container = vol.split(":", 1)
            out[host] = container
    return out


def orchestrator_container_from_variant(variant: InferenceMaxVariantConfig) -> Dict[str, Any]:
    """``container`` block for :class:`OrchestratorConfig` (includes server env)."""
    block = variant.container.model_dump()
    server_env = variant.roles.server.env
    if server_env:
        block = {**block, "env": dict(server_env)}
    return block
