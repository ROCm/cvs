'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Engine configs on ``LlmServingConfig``: ``VllmConfig`` and its blocks.

``vllm serve`` flags go in ``server_params.engine_args`` and ``vllm bench
serve`` flags in ``benchmark_params.client_args``, in snake_case, the spelling
the commands are built from. Neither may name a field the harness sets.
'''

import json
import re
from typing import Annotated, Any, Dict, List, Literal, Union

from pydantic import AfterValidator, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from cvs.config_validation.schemas.base import ConfigStr, SecretInputError, drop_comment_keys
from cvs.config_validation.schemas.common_training_inference import ContainerSpec, RuntimeArgs, RuntimeSpec
from cvs.config_validation.schemas.inference.llm_serving import (
    LlmServingConfig,
    ServingBenchmarkParams,
    ServingServerParams,
)
from cvs.config_validation.schemas.sweeps import Axis, parse_cell_key
from cvs.config_validation.schemas.thresholds import Threshold, ThresholdFile
from cvs.lib.conf_lib.config_blocks import PLACEHOLDER_MESSAGE, has_placeholder, placeholders_rejected
from cvs.lib.inference.utils.vllm_metrics import METRICS_BY_NAME

ISL = Axis("ISL", "isl")
OSL = Axis("OSL", "osl")
TP = Axis("TP", "tp")
PP = Axis("PP", "pp")
CONC = Axis("CONC", "concurrency")
_AXES = (ISL, OSL, TP, PP, CONC)

_SOCKET_ENV = {"NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME"}
_HF_TOKEN_ENV = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}
_METADATA_PREFIXES = ("_comment", "_example")
_FLAG_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# A server_params field, a former field name, or a flag the launcher sets; as a vllm
# serve flag, each would repeat or contradict the harness.
SERVER_RESERVED = frozenset(
    {
        "backend",
        "engine",
        "model",
        "tensor_parallel_size",
        "tp",
        "pipeline_parallel_size",
        "pp",
        "dp",
        "ep",
        "port",
        "dist_init_port",
        "master_port",
        "master_addr",
        "nnodes",
        "node_rank",
        "headless",
        "server_poll_iterations",
        "server_poll_wait_s",
        "server_warmup_wait_s",
        "distributed_executor_backend",
    }
)

# Each name is a benchmark_params field or a flag the harness sets on vllm bench serve.
CLIENT_RESERVED = frozenset(
    {
        "backend",
        "base_url",
        "model",
        "dataset_name",
        "num_prompts",
        "request_rate",
        "burstiness",
        "tokenizer_mode",
        "seed",
        "random_range_ratio",
        "random_prefix_len",
        "random_input_len",
        "random_output_len",
        "max_concurrency",
        "save_result",
        "result_dir",
        "result_filename",
        "percentile_metrics",
        "metric_percentiles",
        "client_poll_iterations",
        "client_poll_wait_s",
        "client_initial_wait_s",
        "trust_remote_code",
        "ignore_eos",
    }
)


def _reject_flag_placeholder(value, info):
    if not placeholders_rejected(info):
        return value
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if has_placeholder(text):
        raise ValueError(PLACEHOLDER_MESSAGE)
    return value


FlagValue = Annotated[Any, AfterValidator(_reject_flag_placeholder)]


def _check_flags(flags, reserved):
    for name, value in flags.items():
        if not _FLAG_NAME_RE.fullmatch(name):
            raise ValueError(f"flag names must be snake_case: {name!r}")
        if value is False:
            raise ValueError(f"{name}=false is ambiguous; omit it or use the flag's negative form")
        if isinstance(value, list) and any(isinstance(item, (list, dict)) for item in value):
            raise ValueError(f"{name}: lists may contain scalar values only")
        if isinstance(value, dict):
            try:
                json.dumps(value, separators=(",", ":"))
            except (TypeError, ValueError):
                raise ValueError(f"{name} must be JSON serializable") from None
    collisions = sorted(reserved & set(flags))
    if collisions:
        raise ValueError(f"cannot set harness fields: {collisions}")
    return flags


class VllmRuntimeArgs(RuntimeArgs):
    model_config = ConfigDict(extra="forbid")

    network: str = "host"
    ipc: str = "host"
    privileged: bool = True
    volumes: List[ConfigStr] = Field(default_factory=list)
    devices: List[str] = Field(default_factory=list)

    # container.env is the one env block, so the Hugging Face token check
    # cannot be bypassed through docker run -e arguments built from here.
    @model_validator(mode="before")
    @classmethod
    def _reject_env(cls, data):
        if isinstance(data, dict) and "env" in data:
            raise ValueError("vLLM does not read runtime.args.env; set these variables in container.env")
        return data


class VllmRuntime(RuntimeSpec):
    model_config = ConfigDict(extra="forbid")

    name: Literal["docker"] = "docker"
    args: VllmRuntimeArgs = Field(default_factory=VllmRuntimeArgs)


class ContainerEnvTokenError(SecretInputError):
    """A Hugging Face token name in container.env."""


class VllmContainerSpec(ContainerSpec):
    runtime: VllmRuntime = Field(default_factory=VllmRuntime)

    @model_validator(mode="before")
    @classmethod
    def _reject_hf_token_env(cls, data):
        env = data.get("env") if isinstance(data, dict) else None
        tokens = sorted(name for name in env or {} if str(name).upper() in _HF_TOKEN_ENV)
        if tokens:
            raise ContainerEnvTokenError(
                f"container.env must not set {tokens}: container.env values become docker run -e arguments. "
                "Put the Hugging Face token in the file named by paths.hf_token_file instead."
            )
        return data

    @model_validator(mode="after")
    def _validate_network_env(self):
        configured_sockets = _SOCKET_ENV & set(self.env)
        if configured_sockets and configured_sockets != _SOCKET_ENV:
            missing = sorted(_SOCKET_ENV - configured_sockets)
            raise ValueError(f"container.env must set all socket interface variables together; missing: {missing}")
        empty_sockets = sorted(name for name in configured_sockets if not self.env[name].strip())
        if empty_sockets:
            raise ValueError(f"container.env socket interface variables cannot be empty: {empty_sockets}")
        if "NCCL_IB_HCA" in self.env and not self.nccl_ib_hcas:
            raise ValueError("container.env.NCCL_IB_HCA must name at least one device")
        return self

    @property
    def nccl_ib_hcas(self):
        value = self.env.get("NCCL_IB_HCA")
        if value is None:
            return None
        return [device.strip() for device in value.split(",") if device.strip()]

    @property
    def socket_env_configured(self):
        return bool(_SOCKET_ENV & set(self.env))


class VllmServerParams(ServingServerParams):
    """The ``vllm serve`` harness fields; ``engine_args`` holds every other serve flag.

    vLLM has no data or expert parallel fields, so ``dp`` and ``ep`` stay 1;
    their vLLM flags go in ``engine_args``.
    """

    engine: Literal["vllm"]
    dp: Literal[1] = 1
    ep: Literal[1] = 1
    port: int = 8888
    master_port: int = 29501
    server_poll_iterations: int = 60
    server_poll_wait_s: int = 60
    server_warmup_wait_s: int = 330
    distributed_executor_backend: Literal["mp", "ray"] = "mp"
    engine_args: Dict[str, FlagValue] = Field(default_factory=dict)

    @field_validator("engine_args")
    @classmethod
    def _check_engine_args(cls, engine_args):
        return _check_flags(engine_args, SERVER_RESERVED)


class VllmBenchmarkParams(ServingBenchmarkParams):
    """The ``vllm bench serve`` harness fields; ``client_args`` holds every other client flag."""

    dataset_name: str = "random"
    num_prompts: int = 3200
    request_rate: Union[str, float] = "inf"
    burstiness: float = 1.0
    tokenizer_mode: str = "auto"
    seed: int = 0
    random_range_ratio: float = 0.0
    random_prefix_len: int = 0
    client_poll_iterations: int = 20
    client_poll_wait_s: int = 60
    client_initial_wait_s: int = 120
    trust_remote_code: bool = False
    ignore_eos: bool = True
    client_args: Dict[str, FlagValue] = Field(default_factory=dict)

    @field_validator("client_args")
    @classmethod
    def _check_client_args(cls, client_args):
        return _check_flags(client_args, CLIENT_RESERVED)


class VllmThreshold(Threshold):
    @model_validator(mode="before")
    @classmethod
    def _reject_underscore_fields(cls, data):
        if not isinstance(data, dict):
            return data
        extra = sorted(key for key in data if isinstance(key, str) and key.startswith("_"))
        if extra:
            raise ValueError(f"unknown threshold fields {extra}")
        return data

    @model_validator(mode="after")
    def _reject_optional(self):
        if "optional" in self.model_fields_set:
            raise ValueError("vLLM thresholds do not support optional")
        return self


def _reject_underscore_metrics(cell):
    # A misspelled metric such as _output_throughput would otherwise be dropped
    # as a comment, and enforcement only checks that the cell exists.
    if isinstance(cell, dict):
        for metric in cell:
            if isinstance(metric, str) and metric.startswith("_") and not metric.startswith(_METADATA_PREFIXES):
                raise ValueError(f"unknown vLLM threshold metric {metric!r}")
    return drop_comment_keys(cell)


def _check_metric_kinds(cell):
    for metric, spec in cell.items():
        definition = METRICS_BY_NAME.get(metric)
        if definition is None:
            raise ValueError(f"unknown vLLM threshold metric {metric!r}")
        if spec.kind not in (definition.direction, "info"):
            raise ValueError(f"{metric} threshold kind must be {definition.direction!r} or 'info'")
    return cell


VllmThresholdCell = Annotated[
    Dict[str, VllmThreshold], BeforeValidator(_reject_underscore_metrics), AfterValidator(_check_metric_kinds)
]


class VllmThresholdFile(ThresholdFile):
    """A vLLM threshold file: cell keys on the vLLM axes, each naming registered vLLM metrics.

    Each kind is its metric's direction or ``info``. ``accuracy`` keeps the
    shared rules, since lm-eval metrics are not vLLM metrics.
    """

    __pydantic_extra__: Dict[str, VllmThresholdCell]

    # Pydantic before 2.11 ignores a key validator on __pydantic_extra__.
    @model_validator(mode="after")
    def _check_cell_keys(self):
        for key in self.model_extra:
            parse_cell_key(key, _AXES)
        return self


class VllmConfig(LlmServingConfig):
    """A vLLM suite config.

    The TP and PP of each cell, and of each threshold cell, equal
    ``server_params``. A threshold cell need not be one of ``cells``, so a
    config can run some of the cells its threshold file covers. With
    ``enforce_thresholds`` every cell has a threshold cell.
    """

    suite = "vllm"
    axes = _AXES

    container: VllmContainerSpec
    server_params: VllmServerParams
    benchmark_params: VllmBenchmarkParams = Field(default_factory=VllmBenchmarkParams)
    thresholds: VllmThresholdFile = Field(default_factory=VllmThresholdFile)

    @model_validator(mode="after")
    def _check_cells(self):
        server = (self.server_params.tp, self.server_params.pp)
        for cell in self.parsed_cells().values():
            if (cell.tp, cell.pp) != server:
                raise ValueError(f"cell {cell.key} conflicts with server_params.tp or server_params.pp")
        threshold_cells = self.thresholds.model_extra or {}
        for key in threshold_cells:
            cell = parse_cell_key(key, self.axes)
            if (cell.tp, cell.pp) != server:
                raise ValueError(f"threshold cell {key} conflicts with server_params.tp or server_params.pp")
        if self.enforce_thresholds:
            missing = [key for key in self.cells if key not in threshold_cells]
            if missing:
                raise ValueError(f"cells missing threshold coverage: {missing}")
        return self
