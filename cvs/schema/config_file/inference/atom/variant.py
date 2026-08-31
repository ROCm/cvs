"""
ATOM inference variant config schema.

Mirrors ``cvs/input/config_file/inference/atom/``.
"""

from __future__ import annotations

import re
import warnings

from pydantic import Field, field_validator, model_validator
from typing_extensions import Literal

from cvs.lib import globals
from cvs.lib.inference.atom.atom_parsing import GATED_METRICS
from cvs.schema.common.base import BaseVariantConfig
from cvs.schema.base import _Forbid
from cvs.schema.config_file.inference.common.accuracy import AccuracyConfig
from cvs.schema.config_file.inference.common.functional import FunctionalConfig
from cvs.schema.config_file.inference.common.long_context_accuracy import LongContextAccuracyConfig
from cvs.schema.config_file.inference.common.platform import PlatformConfig
from cvs.schema.config_file.inference.common.sweep import (
    RoleServer,
    Sweep,
    validate_thresholds_cover_sweep,
)

ATOM_DRIVERS = ("atom", "vllm", "vllm_atom", "sglang")
ATOM_PP_DRIVERS = ("vllm", "vllm_atom", "sglang")
_MXFP4_TRITON_ENV = {
    "ATOM_USE_TRITON_MOE": "1",
    "ATOM_USE_TRITON_GEMM": "1",
}

log = globals.log
_ORCH_MANAGED_NETWORK_ENV = frozenset({"NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "NCCL_IB_HCA"})
_IB_HCA_NETDEV_RE = re.compile(r"^mlx5_\d+$", re.IGNORECASE)


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


class AtomRoleServer(RoleServer):
    atom_args: list[str] = []
    sglang_args: list[str] = []
    ib_hca_devices: Literal["auto"] | list[str] | None = None
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
    server: AtomRoleServer = Field(default_factory=AtomRoleServer)


class AtomParams(_Forbid):
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
    nnodes: str = "1"
    pipeline_parallel_size: str = "1"
    master_addr: str = ""
    master_port: str = "29501"
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


class AtomVariantConfig(BaseVariantConfig):
    framework: Literal["atom"]
    gpu_arch: str
    run_card: AtomRunCard = Field(default_factory=AtomRunCard)
    roles: AtomRoles = Field(default_factory=AtomRoles)
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
        key = f"ISL={isl},OSL={osl},TP={p.tensor_parallelism}"
        nnodes = int(p.nnodes)
        pp = int(p.pipeline_parallel_size)
        if p.driver == "atom":
            if nnodes > 1:
                key += f",DP={nnodes},NNODES={nnodes}"
        elif p.driver in ATOM_PP_DRIVERS:
            if pp > 1 or nnodes > 1:
                key += f",PP={p.pipeline_parallel_size}"
            if nnodes > 1:
                key += f",NNODES={p.nnodes}"
        return f"{key},CONC={concurrency}"

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
