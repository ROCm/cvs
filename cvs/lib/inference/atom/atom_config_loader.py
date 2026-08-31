'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

ATOM suite config loader (``atom``).

Pydantic models live in ``cvs.schema.config_file.inference.atom.variant``.
'''

from __future__ import annotations

from typing import Any

from cvs.lib.utils.config_loader import substitute_config
from cvs.schema.config_file.inference.atom.variant import (
    ATOM_DRIVERS,
    ATOM_PP_DRIVERS,
    AtomParams,
    AtomRoleServer,
    AtomRoles,
    AtomRunCard,
    AtomVariantConfig,
    MtpQualityConfig,
    QuantParityConfig,
    merge_mxfp4_triton_env,
)
from cvs.schema.config_file.inference.common.sweep import validate_sweep_selector

__all__ = [
    "ATOM_DRIVERS",
    "ATOM_PP_DRIVERS",
    "AtomParams",
    "AtomRoleServer",
    "AtomRoles",
    "AtomRunCard",
    "AtomVariantConfig",
    "MtpQualityConfig",
    "QuantParityConfig",
    "expand_sweep",
    "expand_sweep_parametrize",
    "load_variant",
    "merge_mxfp4_triton_env",
    "orchestrator_container_from_variant",
    "placeholder_gated_threshold_cell",
    "reuse_server_flag",
    "server_session_key",
    "validate_sweep_selector",
]


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


def reuse_server_flag(params) -> bool:
    raw = str(getattr(params, "reuse_server_across_sweep", "false")).strip().lower()
    return raw in ("true", "1", "yes")


def server_session_key(variant_config, isl, osl):
    p = variant_config.params
    roles = variant_config.roles.server
    if p.driver == "atom":
        server_tokens = tuple(roles.atom_args)
    elif p.driver == "sglang":
        server_tokens = tuple(roles.sglang_args)
    else:
        server_tokens = tuple(sorted(roles.serve_args.items()))
    return (
        variant_config.model.id,
        p.driver,
        str(isl),
        str(osl),
        server_tokens,
        p.tensor_parallelism,
        p.nnodes,
        p.pipeline_parallel_size,
        p.master_addr,
        p.master_port,
    )


def expand_sweep_parametrize(sweep, fixturenames):
    from cvs.lib.inference.atom.atom_parsing import METRIC_TIER_ORDER

    cases, ids = expand_sweep(sweep)
    if "metric_tier" in fixturenames:
        if not cases:
            return None
        tier_cases = []
        tier_ids = []
        for (combo, c), cid in zip(cases, ids):
            for tier in METRIC_TIER_ORDER:
                tier_cases.append((combo, c, tier))
                tier_ids.append(f"{cid}-{tier}")
        return ("seq_combo,concurrency,metric_tier", tier_cases, tier_ids)
    if "seq_combo" in fixturenames and "concurrency" in fixturenames and cases:
        return ("seq_combo,concurrency", cases, ids)
    return None


def load_variant(config_path, cluster_dict) -> AtomVariantConfig:
    raw, thresholds = substitute_config(config_path, cluster_dict)
    raw["thresholds"] = thresholds
    return AtomVariantConfig(**raw)


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
) -> dict[str, Any]:
    """Return one sweep cell's ``client.*`` specs covering every gated metric."""
    from cvs.lib.inference.atom.atom_parsing import GATED_METRICS

    loose_ms = {"kind": "max_ms", "value": 1_000_000}
    out = {
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
    for m in GATED_METRICS:
        key = f"client.{m}"
        if key not in out:
            kind = "max_ms" if m.endswith("_ms") else "max" if m == "failed" else "min"
            out[key] = {"kind": kind, "value": 0 if kind == "min" else 1_000_000}
    return out


def orchestrator_container_from_variant(variant: AtomVariantConfig) -> dict[str, Any]:
    block = variant.container.model_dump()
    server_env = variant.roles.server.env
    if server_env:
        block = {**block, "env": dict(server_env)}
    return block
