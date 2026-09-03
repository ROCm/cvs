'''Typed config schema and run resolver for CVS vLLM workloads.'''

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from cvs.lib.inference.utils.accuracy_config import AccuracyConfig
from cvs.lib.inference.utils.vllm_server_metrics import PROM_METRICS
from cvs.lib.utils.config_loader import substitute_config
from cvs.lib.utils.gpu import GPU_METRICS

GATED_GPU_METRICS = {key for key, _unit in GPU_METRICS}
GATED_PROM_METRICS = {key for key, _unit in PROM_METRICS}
_CELL_RE = re.compile(
    r"^ISL=(?P<isl>[1-9]\d*),OSL=(?P<osl>[1-9]\d*),TP=(?P<tp>[1-9]\d*),PP=(?P<pp>[1-9]\d*),CONC=(?P<concurrency>[1-9]\d*)$"
)
_METADATA_PREFIXES = ("_comment", "_example")
_NETWORK_ENV = {"NCCL_IB_HCA", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME"}
_SERVER_RESERVED = {"master_addr", "master_port", "nnodes", "node_rank", "headless"}


class _Forbid(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Options(BaseModel):
    """Strict at the section boundary, flexible only for upstream CLI options."""

    model_config = ConfigDict(extra="allow")

    def extra_options(self) -> Dict[str, Any]:
        return dict(self.model_extra or {})


class RuntimeArgs(_Forbid):
    network: str = "host"
    ipc: str = "host"
    privileged: bool = True
    volumes: List[str] = Field(default_factory=list)
    devices: List[str] = Field(default_factory=list)


class Runtime(_Forbid):
    name: Literal["docker"] = "docker"
    args: RuntimeArgs = Field(default_factory=RuntimeArgs)


class ContainerConfig(_Forbid):
    lifetime: Literal["no_launch", "per_run", "persistent"] = "per_run"
    name: str
    image: str
    env: Dict[str, str] = Field(default_factory=dict)
    runtime: Runtime = Field(default_factory=Runtime)

    @model_validator(mode="after")
    def _reject_generated_network_env(self):
        collisions = sorted(_NETWORK_ENV & set(self.env))
        if collisions:
            raise ValueError(f"container.env cannot set generated network variables: {collisions}")
        return self


class Paths(_Forbid):
    shared_fs: str
    models_dir: str
    log_dir: str
    hf_token_file: str


class ServerParams(_Options):
    """Harness fields plus arbitrary vLLM ``serve`` flags in snake_case."""

    backend: Literal["vllm"] = "vllm"
    model: str
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
        _validate_cli_option_map(self.extra_options(), section="benchmark_params")
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


def _validate_cli_option_map(options: Dict[str, Any], *, section: str) -> None:
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


def serialize_cli_options(options: Dict[str, Any]) -> List[str]:
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
    def parse(cls, value: str) -> RunCell:
        match = _CELL_RE.fullmatch(value)
        if not match:
            raise ValueError("run cell must be canonical ISL=<n>,OSL=<n>,TP=<n>,PP=<n>,CONC=<n>")
        values = {name: int(number) for name, number in match.groupdict().items()}
        return cls(value, **values)


@dataclass(frozen=True)
class ResolvedRun:
    cell: RunCell
    benchmark_params: Dict[str, Any]


def _strip_metadata(value: Any) -> Any:
    """Remove local comments at schema boundaries without touching option payloads."""
    if not isinstance(value, dict):
        return value
    cleaned = {key: item for key, item in value.items() if not key.startswith(_METADATA_PREFIXES)}
    for section in ("paths", "container", "runtime", "args", "accuracy"):
        if isinstance(cleaned.get(section), dict):
            cleaned[section] = _strip_metadata(cleaned[section])
    if isinstance(cleaned.get("tasks"), list):
        cleaned["tasks"] = [
            {key: item for key, item in task.items() if not key.startswith(_METADATA_PREFIXES)}
            if isinstance(task, dict)
            else task
            for task in cleaned["tasks"]
        ]
    for section in ("server_params", "benchmark_params"):
        if isinstance(cleaned.get(section), dict):
            cleaned[section] = {
                key: item for key, item in cleaned[section].items() if not key.startswith(_METADATA_PREFIXES)
            }
    if isinstance(cleaned.get("sweeps"), dict):
        cleaned["sweeps"] = {
            key: {
                option: option_value
                for option, option_value in values.items()
                if not option.startswith(_METADATA_PREFIXES)
            }
            if isinstance(values, dict)
            else values
            for key, values in cleaned["sweeps"].items()
            if not key.startswith(_METADATA_PREFIXES)
        }
    return cleaned


class VariantConfig(_Forbid):
    enforce_thresholds: bool = True
    threshold_json: str
    ib_hca_devices: Union[Literal["auto"], List[str], None] = "auto"
    ib_netdev: Optional[str] = None
    paths: Paths
    container: ContainerConfig
    server_params: ServerParams
    benchmark_params: BenchmarkParams = Field(default_factory=BenchmarkParams)
    sweeps: Dict[str, Dict[str, Any]]
    runs: List[str]
    thresholds: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    accuracy: AccuracyConfig = Field(default_factory=AccuracyConfig)
    _effective_topology: Any = PrivateAttr(default=None)
    _run_cells: Dict[str, RunCell] = PrivateAttr(default_factory=dict)

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
            _validate_cli_option_map(overrides, section=f"sweeps.{cell_key}")
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
    def model_id(self) -> str:
        return self.server_params.model

    def bind_effective_topology(self, topology) -> None:
        self._effective_topology = topology

    def cell(self, key: str) -> RunCell:
        return self._run_cells[key]

    def expected_cells(self) -> List[str]:
        return list(self.runs)

    def cell_key(self, isl, osl, concurrency, **_unused) -> str:
        return (
            f"ISL={isl},OSL={osl},TP={self.server_params.tensor_parallel_size},"
            f"PP={self.server_params.pipeline_parallel_size},CONC={concurrency}"
        )

    def resolved_runs(self) -> List[ResolvedRun]:
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


def load_variant(config_path, cluster_dict):
    """Load a vLLM config, threshold file, placeholders, and selected runs."""
    raw, thresholds = substitute_config(config_path, cluster_dict)
    raw = _strip_metadata(raw)
    raw["thresholds"] = thresholds
    return VariantConfig(**raw)
