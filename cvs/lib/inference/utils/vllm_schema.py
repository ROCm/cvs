'''Typed config schema and run resolver for CVS vLLM workloads.'''

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import AfterValidator, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from cvs.lib.inference.utils.accuracy_config import AccuracyConfig
from cvs.lib.inference.utils.vllm_metrics import validate_threshold_spec
from cvs.lib.utils.config_blocks import (
    PLACEHOLDER_MESSAGE,
    ConfigStr,
    ContainerSpec,
    Paths,
    RuntimeArgs,
    RuntimeSpec,
    ThresholdFile,
    _Allow,
    _Forbid,
    has_placeholder,
    placeholders_rejected,
)

_CELL_RE = re.compile(
    r"^ISL=(?P<isl>[1-9]\d*),OSL=(?P<osl>[1-9]\d*),TP=(?P<tp>[1-9]\d*),PP=(?P<pp>[1-9]\d*),CONC=(?P<concurrency>[1-9]\d*)$"
)
_SOCKET_ENV = {"NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME"}
_HF_TOKEN_ENV = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}
_SERVER_RESERVED = {"master_addr", "master_port", "nnodes", "node_rank", "headless"}


def _reject_option_placeholder(value, info):
    if not placeholders_rejected(info):
        return value
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if has_placeholder(text):
        raise ValueError(PLACEHOLDER_MESSAGE)
    return value


OptionValue = Annotated[Any, AfterValidator(_reject_option_placeholder)]


class _Options(_Allow):
    """Strict at the section boundary, flexible only for upstream CLI options."""

    __pydantic_extra__: Dict[str, OptionValue]

    def extra_options(self):
        return dict(self.model_extra or {})


def _validate_vllm_thresholds(thresholds):
    """Check every cell spec and return the thresholds without comment keys.

    The report code reads ``variant.thresholds`` as plain dicts, so the specs
    are returned as written rather than as ``ThresholdSpec`` models.
    """
    cells = ThresholdFile.model_validate(thresholds).cells()
    for cell_key, cell in cells.items():
        for metric, spec in cell.items():
            if "optional" in spec.model_fields_set:
                raise ValueError(f"{cell_key}.{metric}: vLLM thresholds do not support optional")
            validate_threshold_spec(metric, spec.model_dump())
    cleaned = {cell_key: {metric: thresholds[cell_key][metric] for metric in cell} for cell_key, cell in cells.items()}
    if "accuracy" in thresholds:
        cleaned["accuracy"] = thresholds["accuracy"]
    return cleaned


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


class ContainerEnvTokenError(Exception):
    """A Hugging Face token name in container.env.

    Not a ValueError, so pydantic raises it as is instead of wrapping it in a
    ValidationError whose str(), errors() and json() would carry the token value.
    """


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


class ServerParams(_Options):
    """Harness fields plus arbitrary vLLM ``serve`` flags in snake_case."""

    backend: Literal["vllm"] = "vllm"
    model: ConfigStr
    tensor_parallel_size: int
    pipeline_parallel_size: int = 1
    port: int = 8888
    dist_init_port: int = 29501
    server_poll_iterations: int = 60
    server_poll_wait_s: int = 60
    server_warmup_wait_s: int = 330
    distributed_executor_backend: Literal["mp", "ray"] = "mp"

    @model_validator(mode="after")
    def _validate_upstream_options(self):
        options = self.extra_options()
        _validate_cli_option_map(options, section="server_params")
        collisions = sorted(_SERVER_RESERVED & set(options))
        if collisions:
            raise ValueError(f"server_params cannot override harness fields: {collisions}")
        return self


class BenchmarkParams(_Options):
    """Harness fields plus arbitrary vLLM ``bench serve`` flags in snake_case."""

    backend: Literal["vllm"] = "vllm"
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

    @model_validator(mode="after")
    def _validate_upstream_options(self):
        options = self.extra_options()
        _validate_cli_option_map(options, section="benchmark_params")
        collisions = sorted(_BENCHMARK_RESERVED & set(options))
        if collisions:
            raise ValueError(f"benchmark_params cannot override harness fields: {collisions}")
        return self


_BENCHMARK_RESERVED = {
    "backend",
    "base_url",
    "model",
    "max_concurrency",
    "random_input_len",
    "random_output_len",
    "result_dir",
    "result_filename",
    "percentile_metrics",
    "metric_percentiles",
}
_OPTION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_cli_option_map(options, *, section):
    for name, value in options.items():
        if not _OPTION_NAME_RE.fullmatch(name):
            raise ValueError(f"{section} option names must be snake_case: {name!r}")
        if value is False:
            raise ValueError(f"{section}.{name}=false is ambiguous; omit it or use the option's negative flag")
        if isinstance(value, list) and any(isinstance(item, (list, dict)) for item in value):
            raise ValueError(f"{section}.{name} lists may contain scalar values only")
        if isinstance(value, dict):
            try:
                json.dumps(value, separators=(",", ":"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{section}.{name} must be JSON serializable") from exc


def serialize_cli_options(options):
    """Serialize generic snake-case option maps for vLLM command lines."""
    _validate_cli_option_map(options, section="option")
    argv = []
    for name, value in options.items():
        if value is None:
            continue
        flag = f"--{name.replace('_', '-')}"
        if value is True:
            argv.append(flag)
        elif isinstance(value, list):
            argv.extend([flag, *(str(item) for item in value)])
        elif isinstance(value, dict):
            argv.extend([flag, json.dumps(value, separators=(",", ":"))])
        else:
            argv.extend([flag, str(value)])
    return argv


@dataclass(frozen=True)
class RunCell:
    key: str
    isl: int
    osl: int
    tp: int
    pp: int
    concurrency: int

    @classmethod
    def parse(cls, value):
        match = _CELL_RE.fullmatch(value)
        if not match:
            raise ValueError("run cell must be canonical ISL=<n>,OSL=<n>,TP=<n>,PP=<n>,CONC=<n>")
        values = {name: int(number) for name, number in match.groupdict().items()}
        return cls(value, **values)


@dataclass(frozen=True)
class ResolvedRun:
    cell: RunCell
    benchmark_params: Dict[str, Any]


class VariantConfig(_Forbid):
    enforce_thresholds: bool = True
    threshold_json: str
    ib_hca_devices: Union[Literal["auto"], List[str], None] = "auto"
    ib_netdev: Optional[str] = None
    paths: Paths
    container: VllmContainerSpec
    server_params: ServerParams
    benchmark_params: BenchmarkParams = Field(default_factory=BenchmarkParams)
    sweeps: Dict[str, Dict[str, OptionValue]]
    runs: List[str]
    thresholds: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    accuracy: AccuracyConfig = Field(default_factory=AccuracyConfig)
    _effective_topology: Any = PrivateAttr(default=None)
    _run_cells: Dict[str, RunCell] = PrivateAttr(default_factory=dict)

    @field_validator("sweeps", mode="before")
    @classmethod
    def _drop_sweep_comments(cls, value):
        if not isinstance(value, dict):
            return value
        return {
            key: {name: item for name, item in entry.items() if not name.startswith("_")}
            if isinstance(entry, dict)
            else entry
            for key, entry in value.items()
            if not key.startswith("_")
        }

    @field_validator("thresholds")
    @classmethod
    def _check_thresholds(cls, value):
        return _validate_vllm_thresholds(value)

    @model_validator(mode="after")
    def _validate_runs_and_thresholds(self):
        if not self.runs:
            raise ValueError("runs must be a nonempty explicit list")
        parsed = {key: RunCell.parse(key) for key in self.sweeps}
        if len(set(self.runs)) != len(self.runs):
            raise ValueError("runs contains duplicate cells")
        unknown_runs = sorted(set(self.runs) - set(parsed))
        if unknown_runs:
            raise ValueError(f"runs reference unknown sweeps: {unknown_runs}")
        for cell in parsed.values():
            if (
                cell.tp != self.server_params.tensor_parallel_size
                or cell.pp != self.server_params.pipeline_parallel_size
            ):
                raise ValueError(f"{cell.key} conflicts with server_params tensor/pipeline parallel size")
        for cell_key, overrides in self.sweeps.items():
            if not isinstance(overrides, dict):
                raise ValueError(f"sweeps.{cell_key} must be an object")
            reserved = sorted(_BENCHMARK_RESERVED & set(overrides))
            if reserved:
                raise ValueError(f"sweeps.{cell_key} cannot override harness fields: {reserved}")
            boolean_fields = {"ignore_eos", "trust_remote_code"} & set(overrides)
            for name in boolean_fields:
                if not isinstance(overrides[name], bool):
                    raise ValueError(f"sweeps.{cell_key}.{name} must be boolean")
            cli_overrides = {name: value for name, value in overrides.items() if name not in boolean_fields}
            _validate_cli_option_map(cli_overrides, section=f"sweeps.{cell_key}")
        threshold_cells = set(self.thresholds) - {"accuracy"}
        unknown_thresholds = sorted(threshold_cells - set(parsed))
        if unknown_thresholds:
            raise ValueError(f"threshold cells have no matching sweep: {unknown_thresholds}")
        if self.enforce_thresholds:
            missing = [key for key in self.runs if key not in threshold_cells]
            if missing:
                raise ValueError(f"selected runs missing threshold coverage: {missing}")
        self._run_cells = parsed
        return self

    @property
    def model_id(self):
        return self.server_params.model

    def bind_effective_topology(self, topology):
        self._effective_topology = topology

    def cell(self, key):
        return self._run_cells[key]

    def expected_cells(self):
        return list(self.runs)

    def cell_key(self, isl, osl, concurrency, **_unused):
        return (
            f"ISL={isl},OSL={osl},TP={self.server_params.tensor_parallel_size},"
            f"PP={self.server_params.pipeline_parallel_size},CONC={concurrency}"
        )

    def resolved_runs(self):
        base = self.benchmark_params.model_dump()
        extras = self.benchmark_params.extra_options()
        resolved = []
        for key in self.runs:
            values = {**base, **extras, **self.sweeps[key]}
            values.update(
                {
                    "random_input_len": self.cell(key).isl,
                    "random_output_len": self.cell(key).osl,
                    "max_concurrency": self.cell(key).concurrency,
                }
            )
            resolved.append(ResolvedRun(cell=self.cell(key), benchmark_params=values))
        return resolved
