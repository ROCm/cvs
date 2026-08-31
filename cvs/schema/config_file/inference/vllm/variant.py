"""
Unified vLLM inference variant config schema.

Mirrors ``cvs/input/config_file/inference/vllm/``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import Field, field_validator, model_validator
from typing_extensions import Literal

from cvs.schema.base import _Allow, _Forbid
from cvs.schema.config_file.inference.common.accuracy import AccuracyConfig
from cvs.schema.config_file.inference.common.sweep import (
    RoleServer,
    Sweep,
    validate_thresholds_cover_sweep,
)

_VLLM_LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}


class ContainerConfig(_Allow):
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


class VllmRoleServer(RoleServer):
    ib_hca_devices: Union[Literal["auto"], List[str], None] = None
    ib_netdev: Optional[str] = None

    @field_validator("serve_args", mode="after")
    @classmethod
    def _check_log_level(cls, v):
        level = v.get("log-level")
        if level is not None and level not in _VLLM_LOG_LEVELS:
            raise ValueError(f"serve_args.log-level must be one of {sorted(_VLLM_LOG_LEVELS)}, got: {level!r}")
        return v


class VllmRoles(_Forbid):
    server: VllmRoleServer = Field(default_factory=VllmRoleServer)


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
    pipeline_parallel_size: str = "1"
    master_addr: str = "localhost"
    master_port: str = "29501"
    nnodes: str = "1"
    tokenizer_mode: str = "auto"
    percentile_metrics: str = "ttft,tpot,itl,e2el"
    metric_percentiles: str = "50,90,95,99"
    num_prompts: str = "3200"
    client_poll_count: str = "20"


class VariantConfig(_Forbid):
    """Unified typed config for both single-node and distributed vllm runs."""

    schema_version: Literal[1]
    framework: Literal["vllm"]
    gpu_arch: str
    enforce_thresholds: bool = True
    container: ContainerConfig = Field(default_factory=ContainerConfig)
    paths: Paths
    model: ModelSpec
    roles: VllmRoles = Field(default_factory=VllmRoles)
    params: Params = Field(default_factory=Params)
    sweep: Sweep
    thresholds: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    accuracy: AccuracyConfig = Field(default_factory=AccuracyConfig)

    @model_validator(mode="after")
    def _check_distributed_consistency(self):
        nn = int(self.params.nnodes)
        pp = int(self.params.pipeline_parallel_size)
        is_ray = self.roles.server.serve_args.get("distributed-executor-backend") == "ray"
        if nn > 1 and pp == 1 and not is_ray:
            raise ValueError(f"nnodes={nn} > 1 requires pipeline_parallel_size > 1 (got pp={pp})")
        if pp > 1 and nn == 1:
            raise ValueError(f"pipeline_parallel_size={pp} > 1 requires nnodes > 1 (got nnodes={nn})")
        if nn > 1 and not self.roles.server.ib_netdev:
            raise ValueError(
                "ib_netdev is required in roles.server when nnodes > 1. "
                "Set it to the Linux network interface name for NCCL_SOCKET_IFNAME "
                '(e.g. "ens51f1np1"). Cannot be auto-derived from HCA names.'
            )
        return self

    @model_validator(mode="after")
    def _check_remote_not_implemented(self):
        if self.model.remote == 1:
            raise NotImplementedError("model.remote=1 (remote model download) is not implemented.")
        return self

    def cell_key(self, isl, osl, concurrency):
        base = f"ISL={isl},OSL={osl},TP={self.params.tensor_parallelism},"
        if int(self.params.pipeline_parallel_size) > 1:
            base += f"PP={self.params.pipeline_parallel_size},"
        return base + f"CONC={concurrency}"

    def expected_cells(self):
        by_name = {c.name: c for c in self.sweep.sequence_combinations}
        return [self.cell_key(by_name[r.combo].isl, by_name[r.combo].osl, r.concurrency) for r in self.sweep.runs]

    @model_validator(mode="after")
    def _check_thresholds_cover_sweep(self):
        validate_thresholds_cover_sweep(
            expected_cells=self.expected_cells(),
            thresholds=self.thresholds,
            enforce_thresholds=self.enforce_thresholds,
            gated_metrics=set(),
        )
        return self
