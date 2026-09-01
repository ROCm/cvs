'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

ATOM suite config schema (``atom``).

Generic paths/model/container/threshold plumbing lives in
:mod:`cvs.lib.utils.config_loader`. Sweep selector types are shared with
:mod:`cvs.lib.inference.utils.inferencing_config_loader`.
'''

from __future__ import annotations

import re
from typing import Any

from pydantic import Field, field_validator, model_validator
from typing_extensions import Literal

from cvs.lib import globals
from cvs.lib.inference.atom.atom_parsing import GATED_METRICS
from cvs.lib.inference.utils.accuracy_config import AccuracyConfig
from cvs.lib.inference.utils.functional_config import FunctionalConfig
from cvs.lib.inference.utils.inferencing_config_loader import (
    RoleServer,
    Sweep,
    validate_sweep_selector,
    validate_thresholds_cover_sweep,
)
from cvs.lib.inference.utils.long_context_accuracy_config import LongContextAccuracyConfig
from cvs.lib.inference.utils.platform_config import PlatformConfig
from cvs.lib.utils.config_loader import BaseVariantConfig, _Forbid, substitute_config

ATOM_DRIVERS = ("atom", "vllm", "vllm_atom", "sglang")
ATOM_PP_DRIVERS = ("vllm", "vllm_atom", "sglang")
# MI300X MXFP4 MoE + A4W4 GEMM require Triton; aiter A4W4 is unsupported on gfx942.
# atom.utils.envs treats only "1" as true — "true" is ignored.
_MXFP4_TRITON_ENV = {
    "ATOM_USE_TRITON_MOE": "1",
    "ATOM_USE_TRITON_GEMM": "1",
}


def merge_mxfp4_triton_env(precision: str, env: dict[str, str]) -> dict[str, str]:
    """Return server env with MXFP4 Triton defaults applied when unset."""
    merged = dict(env or {})
    if (precision or "").lower() == "mxfp4":
        for key, value in _MXFP4_TRITON_ENV.items():
            merged.setdefault(key, value)
        for key in _MXFP4_TRITON_ENV:
            if str(merged.get(key, "")).lower() == "true":
                merged[key] = "1"
    return merged


log = globals.log

# Written by test_discover_topology / resolve_multinode_fabric — not user env.
_ORCH_MANAGED_NETWORK_ENV = frozenset({"NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "NCCL_IB_HCA"})
_IB_HCA_NETDEV_RE = re.compile(r"^mlx5_\d+$", re.IGNORECASE)


class AtomRoleServer(RoleServer):
    # Extra CLI tokens for ``python -m atom.entrypoints.openai_server`` after
    # ``--model`` / ``--server-port`` (e.g. ``-tp``, ``--kv_cache_dtype``).
    atom_args: list[str] = []
    # Extra CLI tokens appended to ``python3 -m sglang.launch_server`` (driver=sglang).
    sglang_args: list[str] = []
    # IB HCA devices for NCCL_IB_HCA (multinode only).
    # absent or "auto" -> use whatever ibv_devinfo -l reports (test_discover_topology).
    # explicit list -> validated at preflight against ibv_devinfo output.
    ib_hca_devices: Literal["auto"] | list[str] | None = None
    # Linux netdev for NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME on multinode PP runs.
    # absent or "auto" -> resolved at runtime by test_discover_topology from cluster IPs.
    ib_netdev: Literal["auto"] | str | None = None

    @field_validator("ib_netdev", mode="after")
    @classmethod
    def _normalize_ib_netdev(cls, v):
        raw = (v or "").strip()
        if raw and raw.lower() != "auto" and _IB_HCA_NETDEV_RE.match(raw):
            log.warning(
                "roles.server.ib_netdev=%r looks like an IB HCA name; coercing to 'auto' "
                "(socket netdev is discovered from cluster IPs at runtime)",
                raw,
            )
            return "auto"
        return v

    @model_validator(mode="after")
    def _strip_orchestrator_managed_network_env(self):
        if not self.env:
            return self
        dropped = sorted(k for k in self.env if k in _ORCH_MANAGED_NETWORK_ENV)
        if not dropped:
            return self
        log.warning(
            "roles.server.env drops orchestrator-managed keys %s "
            "(set by test_discover_topology / build_server_cmd instead)",
            dropped,
        )
        self.env = {k: v for k, v in self.env.items() if k not in _ORCH_MANAGED_NETWORK_ENV}
        return self


class AtomRoles(_Forbid):
    server: AtomRoleServer = AtomRoleServer()


class AtomParams(_Forbid):
    # ``atom`` = standalone ATOM openai_server + benchmark_serving.
    # ``vllm_atom`` = vLLM coordinator + ATOM local kernels (true multinode PP).
    # ``vllm`` = interim ROCm vLLM uplift (vllm serve + vllm bench serve).
    # ``sglang`` = SGLang coordinator (launch_server + bench_serving) for PP runs.
    driver: Literal["atom", "vllm", "vllm_atom", "sglang"] = "vllm"
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
    reuse_server_across_sweep: str = "false"
    bench_max_failed_requests: str = "0"
    bench_extra_args: str = ""
    result_filename: str = "results"
    # Multinode (M5): omit or set nnodes=1 for single-node runs. When nnodes>1,
    # cluster node_dict must list the same number of hosts and test_setup_sshd runs.
    nnodes: str = "1"
    pipeline_parallel_size: str = "1"
    master_addr: str = ""
    master_port: str = "29501"
    # Optional single-node reference output_throughput for scaling.efficiency_pct.
    scaling_baseline_output_throughput: str = ""


class AtomRunCard(_Forbid):
    upstream_run_url: str = ""
    atom_image_pin: str = ""
    notes: str = ""


class MtpQualityConfig(_Forbid):
    enabled: bool = False
    chat_template_prompt: str = "Say hello in one short sentence."
    chat_template_expected_sha256: str = ""


class QuantParityConfig(_Forbid):
    enabled: bool = False
    probe_prompt: str = "The capital of France is"
    reference_config_stem: str = ""


ATOM_FRAMEWORKS = ("atom",)


class AtomVariantConfig(BaseVariantConfig):
    framework: Literal["atom"]

    gpu_arch: str
    run_card: AtomRunCard = AtomRunCard()
    roles: AtomRoles = AtomRoles()
    params: AtomParams
    sweep: Sweep
    accuracy: AccuracyConfig = Field(default_factory=AccuracyConfig)
    mtp_quality: MtpQualityConfig = Field(default_factory=MtpQualityConfig)
    quant_parity: QuantParityConfig = Field(default_factory=QuantParityConfig)
    functional: FunctionalConfig = Field(default_factory=FunctionalConfig)
    long_context_accuracy: LongContextAccuracyConfig = Field(default_factory=LongContextAccuracyConfig)
    platform: PlatformConfig = Field(default_factory=PlatformConfig)

    def cell_key(self, isl, osl, concurrency):
        p = self.params
        return f"ISL={isl},OSL={osl},TP={p.tensor_parallelism},PP={p.pipeline_parallel_size},CONC={concurrency}"

    def expected_cells(self) -> list[str]:
        by_name = {c.name: c for c in self.sweep.sequence_combinations}
        return [self.cell_key(by_name[r.combo].isl, by_name[r.combo].osl, r.concurrency) for r in self.sweep.runs]

    @model_validator(mode="after")
    def _apply_mxfp4_triton_env_defaults(self):
        self.roles.server.env = merge_mxfp4_triton_env(self.model.precision, self.roles.server.env)
        return self

    @model_validator(mode="after")
    def _check_thresholds_cover_sweep(self):
        validate_thresholds_cover_sweep(
            expected_cells=self.expected_cells(),
            thresholds=self.thresholds,
            enforce_thresholds=self.enforce_thresholds,
            gated_metrics=GATED_METRICS,
            gated_metric_prefix="",
        )
        if int(self.params.nnodes) > 1 and (self.params.scaling_baseline_output_throughput or "").strip():
            missing = []
            for cell in self.expected_cells():
                specs = self.thresholds.get(cell) or {}
                if "scaling.efficiency_pct" not in specs:
                    missing.append(cell)
            if missing:
                msg = (
                    "multinode variant with scaling_baseline_output_throughput requires "
                    f"scaling.efficiency_pct in every cell; missing: {missing}"
                )
                if self.enforce_thresholds:
                    raise ValueError(msg)
                import warnings

                warnings.warn(f"{msg} (enforce_thresholds=false -> record-only)", stacklevel=2)
        return self

    @model_validator(mode="after")
    def _atom_multinode_uses_dp_not_pp(self):
        if self.params.driver == "atom" and int(self.params.nnodes) > 1:
            if int(self.params.pipeline_parallel_size) > 1:
                raise ValueError(
                    "params.driver='atom' with nnodes>1 uses ATOM SPMD data parallel (-dp); "
                    "standalone ATOM cannot execute pipeline parallel. For true PP>1 use "
                    "params.driver='vllm_atom' or 'sglang'."
                )
        return self

    @model_validator(mode="after")
    def _pp_driver_distributed_consistency(self):
        driver = self.params.driver
        if driver not in ATOM_PP_DRIVERS:
            return self
        nn = int(self.params.nnodes)
        pp = int(self.params.pipeline_parallel_size)
        is_ray = self.roles.server.serve_args.get("distributed-executor-backend") == "ray"
        if nn > 1 and pp == 1 and not is_ray:
            raise ValueError(
                f"params.driver={driver!r} with nnodes={nn} requires pipeline_parallel_size>1 "
                f"(got pp={pp}) for multinode pipeline parallel"
            )
        if pp > 1 and nn == 1:
            raise ValueError(
                f"pipeline_parallel_size={pp} > 1 requires nnodes > 1 (got nnodes={nn}) for params.driver={driver!r}"
            )
        return self

    @model_validator(mode="after")
    def _atom_driver_requires_inline_server_args(self):
        if self.params.driver == "atom" and not self.roles.server.atom_args:
            raise ValueError(
                "params.driver='atom' requires roles.server.atom_args "
                "(inline ATOM openai_server CLI tokens, vLLM-style)"
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


def reuse_server_flag(params) -> bool:
    """Return True when ``params.reuse_server_across_sweep`` is a truthy string."""
    raw = str(getattr(params, "reuse_server_across_sweep", "false")).strip().lower()
    return raw in ("true", "1", "yes")


def server_session_key(variant_config, isl, osl):
    """Stable key for server reuse across sweep cells with identical model/shape."""
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
    """Build pytest parametrize args for inference or metric-tier collection."""
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


_PROFILE_SHARED_KEYS = frozenset(
    {
        "schema_version",
        "framework",
        "gpu_arch",
        "paths",
        "model",
        "threshold_json",
    }
)


def _strip_profile_meta(raw: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in raw.items() if k not in _PROFILE_SHARED_KEYS and k != "profiles"}


def _is_w2_gpt_oss_mxfp4(raw: dict[str, Any]) -> bool:
    model = raw.get("model") or {}
    if (model.get("precision") or "").lower() != "mxfp4":
        return False
    mid = (model.get("id") or "").lower()
    return mid.endswith("gpt-oss-120b") or "gpt-oss-120b" in mid


def _w2_perf_native_atom_fallback(
    raw: dict[str, Any],
    merged: dict[str, Any],
    selected: str,
) -> dict[str, Any]:
    """Route W2 perf off native atom until MXFP4 openai_server is customer-qualified."""
    if selected != "perf":
        return merged
    if (merged.get("params") or {}).get("driver") != "atom":
        return merged
    if not _is_w2_gpt_oss_mxfp4(raw):
        return merged
    profiles = raw.get("profiles") or {}
    vllm_body = profiles.get("vllm")
    if not isinstance(vllm_body, dict):
        return merged
    log.info(
        "W2 GPT-OSS MXFP4 perf requested driver=atom; using vllm_atom stack "
        "(opt into native openai_server with --config_profile native when qualified)"
    )
    overlay = dict(vllm_body)
    for key in ("sweep", "enforce_thresholds", "run_card", "platform", "functional"):
        if key in merged:
            overlay[key] = merged[key]
    rc = dict(overlay.get("run_card") or {})
    prefix = "W2 perf via vllm_atom (native atom: --config_profile native)"
    existing = (rc.get("notes") or "").strip()
    rc["notes"] = f"{existing}; {prefix}" if existing else prefix
    overlay["run_card"] = rc
    shared = {k: raw[k] for k in _PROFILE_SHARED_KEYS if k in raw}
    return {**shared, **overlay, "schema_version": 1}


def resolve_atom_profile(
    raw: dict[str, Any],
    thresholds: dict[str, Any],
    profile: str | None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Flatten a schema_version 2 multi-profile config into one variant dict."""
    profiles = raw.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        threshold_profiles = thresholds.get("profiles") if isinstance(thresholds.get("profiles"), dict) else None
        if threshold_profiles is not None:
            selected = (profile or "perf").strip()
            if selected not in threshold_profiles:
                known = ", ".join(sorted(threshold_profiles))
                raise ValueError(
                    f"profiled threshold file requires --config_profile or CVS_CONFIG_PROFILE "
                    f"(requested {selected!r}; known: {known})"
                )
            return raw, dict(threshold_profiles[selected]), selected
        return raw, thresholds, profile or "perf"

    selected = (profile or raw.get("default_profile") or "").strip()
    if not selected:
        known = ", ".join(sorted(profiles))
        raise ValueError(f"multi-profile config requires --config_profile or default_profile (known: {known})")
    if selected not in profiles:
        known = ", ".join(sorted(profiles))
        raise ValueError(f"unknown config profile {selected!r} (known: {known})")

    profile_body_raw = dict(profiles[selected])
    threshold_profile = (profile_body_raw.pop("threshold_profile", None) or selected).strip()

    shared = {k: raw[k] for k in _PROFILE_SHARED_KEYS if k in raw}
    merged = {**shared, **profile_body_raw}
    merged["schema_version"] = 1

    threshold_profiles = thresholds.get("profiles") if isinstance(thresholds.get("profiles"), dict) else None
    if threshold_profiles is not None:
        if threshold_profile not in threshold_profiles:
            known = ", ".join(sorted(threshold_profiles))
            raise ValueError(
                f"threshold file missing profile {threshold_profile!r} referenced by config (known: {known})"
            )
        merged_thresholds = dict(threshold_profiles[threshold_profile])
    else:
        merged_thresholds = dict(thresholds)

    merged = _w2_perf_native_atom_fallback(raw, merged, selected)
    return merged, merged_thresholds, selected


def _expected_cells_from_raw(raw: dict[str, Any]) -> list[str]:
    """Compute sweep cell keys from a resolved flat config dict (pre-pydantic)."""
    params = raw["params"]
    sweep = raw["sweep"]
    by_name = {c["name"]: c for c in sweep["sequence_combinations"]}
    tp = params["tensor_parallelism"]
    pp = params.get("pipeline_parallel_size", "1")
    cells = []
    for run in sweep["runs"]:
        combo = by_name[run["combo"]]
        isl, osl, conc = combo["isl"], combo["osl"], run["concurrency"]
        cells.append(f"ISL={isl},OSL={osl},TP={tp},PP={pp},CONC={conc}")
    return cells


def _prune_orphan_sweep_thresholds(thresholds: dict[str, Any], expected_cells: list[str]) -> dict[str, Any]:
    """Drop sweep-cell threshold keys that belong to another topology/profile."""
    expected = set(expected_cells)
    pruned: dict[str, Any] = {}
    for key, value in thresholds.items():
        if key.startswith("ISL=") and key not in expected:
            continue
        pruned[key] = value
    return pruned


def load_variant(config_path, cluster_dict, profile: str | None = None) -> AtomVariantConfig:
    raw, thresholds = substitute_config(config_path, cluster_dict)
    raw, thresholds, _ = resolve_atom_profile(raw, thresholds, profile)
    thresholds = _prune_orphan_sweep_thresholds(thresholds, _expected_cells_from_raw(raw))
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
    """Return one sweep cell's bare-metric specs covering every ``GATED_METRICS`` member."""
    loose_ms = {"kind": "max_ms", "value": 1_000_000}
    return {
        "total_token_throughput": {"kind": "min_tok_s", "value": total_token_throughput_min},
        "output_throughput": {"kind": "min_tok_s", "value": output_throughput_min},
        "per_gpu_throughput": {"kind": "min_tok_s", "value": per_gpu_throughput_min},
        "output_tput_per_gpu": {"kind": "min_tok_s", "value": output_tput_per_gpu_min},
        "mean_ttft_ms": {"kind": "max_ms", "value": mean_ttft_max_ms},
        "median_ttft_ms": loose_ms,
        "p90_ttft_ms": loose_ms,
        "p95_ttft_ms": loose_ms,
        "p99_ttft_ms": {"kind": "max_ms", "value": p99_ttft_max_ms},
        "mean_tpot_ms": {"kind": "max_ms", "value": mean_tpot_max_ms},
        "median_tpot_ms": loose_ms,
        "p90_tpot_ms": loose_ms,
        "p95_tpot_ms": {"kind": "max_ms", "value": p95_tpot_max_ms},
        "p99_tpot_ms": loose_ms,
        "mean_itl_ms": loose_ms,
        "median_itl_ms": loose_ms,
        "p95_itl_ms": loose_ms,
        "p99_itl_ms": loose_ms,
        "mean_e2el_ms": loose_ms,
        "median_e2el_ms": loose_ms,
        "p90_e2el_ms": loose_ms,
        "p95_e2el_ms": loose_ms,
        "p99_e2el_ms": loose_ms,
        "success_rate": {"kind": "min", "value": success_rate_min},
        "failed": {"kind": "max", "value": failed_max},
    }


def orchestrator_container_from_variant(variant: AtomVariantConfig) -> dict[str, Any]:
    """``container`` block for :class:`OrchestratorConfig` (includes server env)."""
    block = variant.container.model_dump()
    server_env = variant.roles.server.env
    if server_env:
        block = {**block, "env": dict(server_env)}
    return block
