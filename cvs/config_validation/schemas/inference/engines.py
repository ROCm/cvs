'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Engine configs on ``LlmServingConfig``: ``VllmConfig``, ``SglangConfig`` and
``AtomConfig``, each with the one top-level block only its engine reads.

``vllm serve`` flags go in ``server_params.engine_args`` and ``vllm bench
serve`` flags in ``benchmark_params.client_args``, in snake_case, the spelling
the commands are built from. Neither may name a field the harness sets.
``SglangConfig`` and ``AtomConfig`` hold only their blocks and the checks every
engine without an implementation needs; their ports add the rest.
'''

import copy
import json
from collections import namedtuple
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal

from pydantic import AfterValidator, BeforeValidator, ConfigDict, Field, PrivateAttr, ValidationError, model_validator

from cvs.config_validation.schemas.base import ConfigStr, StrictBlock, drop_comment_keys, problem_lines
from cvs.config_validation.schemas.common_training_inference import ContainerSpec, RuntimeArgs, RuntimeSpec
from cvs.config_validation.schemas.inference.llm_serving import (
    LlmServingConfig,
    ServingBenchmarkParams,
    ServingServerParams,
    check_flags,
)
from cvs.config_validation.schemas.sweeps import Axis, parse_cell_key
from cvs.config_validation.schemas.thresholds import Threshold, ThresholdFile
from cvs.lib.conf_lib.loader import ConfigError
from cvs.lib.inference.utils.vllm_metrics import METRICS_BY_NAME
from cvs.lib.utils.config_loader import substitute_placeholders

ISL = Axis("ISL", "isl")
OSL = Axis("OSL", "osl")
TP = Axis("TP", "tp")
PP = Axis("PP", "pp")
CONC = Axis("CONC", "concurrency")
_AXES = (ISL, OSL, TP, PP, CONC)

_SOCKET_ENV = {"NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME"}
_METADATA_PREFIXES = ("_comment", "_example")

# A server_params field, a former field name, or a flag the launcher sets; as a vllm
# serve flag, each would repeat or contradict the harness.
SERVER_RESERVED = frozenset(
    {
        "backend",
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
        "max_model_len",
        "max_model_length",
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


def _refuse_unimplemented(config, engine):
    """Fail when ``config`` turns on an accuracy check or metric ``engine`` does not implement."""
    unimplemented = [
        name
        for name, on in (
            ("accuracy.quant_parity", config.accuracy.quant_parity.enabled),
            ("accuracy.mtp_quality", config.accuracy.mtp_quality.enabled),
            (
                "benchmark_params.scaling_baseline_output_throughput",
                config.benchmark_params.scaling_baseline_output_throughput is not None,
            ),
        )
        if on
    ]
    if unimplemented:
        raise ValueError(f"not implemented for {engine}: {', '.join(unimplemented)}")


def serialize_cli_options(options):
    """The command-line arguments for a snake_case flag map such as ``engine_args`` or ``client_args``.

    A None value is left out, True writes the bare flag, a list writes its
    items after the flag and a dict writes compact JSON.
    """
    check_flags(options)
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


def _deep_merge(base, override):
    if not (isinstance(base, dict) and isinstance(override, dict)):
        return override
    merged = dict(base)
    for key, value in override.items():
        merged[key] = _deep_merge(base[key], value) if key in base else value
    return merged


class VllmRuntimeArgs(RuntimeArgs):
    model_config = ConfigDict(extra="forbid")

    network: str = "host"
    ipc: str = "host"
    privileged: bool = True
    volumes: List[ConfigStr] = Field(default_factory=list)
    devices: List[str] = Field(default_factory=list)


class VllmRuntime(RuntimeSpec):
    model_config = ConfigDict(extra="forbid")

    name: Literal["docker"] = "docker"
    args: VllmRuntimeArgs = Field(default_factory=VllmRuntimeArgs)


class VllmContainerSpec(ContainerSpec):
    runtime: VllmRuntime = Field(default_factory=VllmRuntime)

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
    """The shared server fields plus the old server wait, a warmup then readiness polls."""

    server_poll_iterations: int = 60
    server_poll_wait_s: int = 60
    server_warmup_wait_s: int = 330


class VllmBenchmarkParams(ServingBenchmarkParams):
    """The shared client fields plus the old client wait, an initial wait then polls."""

    client_poll_iterations: int = 20
    client_poll_wait_s: int = 60
    client_initial_wait_s: int = 120

    def options(self):
        """The fields and ``client_args`` as one map, as the client command reads them."""
        return {**self.model_dump(exclude={"client_args"}), **self.client_args}


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


def _written_section(written, checked):
    return {metric: written[metric] for metric in checked}


def _thresholds_as_written(value):
    """``value`` checked as a ``VllmThresholdFile``, returned as written without its comment keys.

    The verdict and report code read cells and specs as dicts, and a spec keeps
    the number it was written with, so an int value stays an int. ``accuracy``
    keeps only its non-empty sections.
    """
    checked = VllmThresholdFile.model_validate(value)
    written = {key: _written_section(value[key], cell) for key, cell in checked.model_extra.items()}
    sections = value.get("accuracy", {}) if isinstance(value, dict) else {}
    accuracy = {}
    for name in ("lm_eval", "custom_eval"):
        by_id = getattr(checked.accuracy, name)
        if by_id:
            accuracy[name] = {key: _written_section(sections[name][key], cell) for key, cell in by_id.items()}
    for name in ("quant_parity", "mtp_quality"):
        cell = getattr(checked.accuracy, name)
        if cell:
            accuracy[name] = _written_section(sections[name], cell)
    if accuracy:
        written["accuracy"] = accuracy
    return written


VllmThresholds = Annotated[Dict[str, Dict[str, Any]], BeforeValidator(_thresholds_as_written)]

ResolvedRun = namedtuple("ResolvedRun", "cell benchmark_params")


class VllmBlock(StrictBlock):
    """The keys only vLLM reads."""

    distributed_executor_backend: Literal["mp", "ray"] = "ray"


class VllmConfig(LlmServingConfig):
    """A vLLM suite config: the shared serving blocks plus ``vllm``.

    vLLM has no data or expert parallel fields, so ``dp`` and ``ep`` stay 1;
    their vLLM flags go in ``engine_args``. ``engine_args`` and ``client_args``
    name no field the harness sets, and every threshold metric is a registered
    vLLM metric with its direction or ``info``.

    ``thresholds`` is the threshold file as written, without comment keys: one
    dict of metric specs per cell key, plus ``accuracy``.

    The suite runs in mode ``single`` or ``distributed`` on the hosts the shared
    host rule picks, and launches its container in its own tests. ``read_raw``
    fills placeholders such as ``{user-id}`` and ``{shared_fs}`` before the
    loader's checks.
    """

    suite = "vllm"
    modes = {"_single": "single", "_distributed": "distributed"}
    launch_in_fixture = False
    axes = _AXES

    container: VllmContainerSpec
    server_params: VllmServerParams
    benchmark_params: VllmBenchmarkParams
    thresholds: VllmThresholds = Field(default_factory=dict)
    vllm: VllmBlock

    _effective_topology = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check_vllm(self):
        server = self.server_params
        if (server.dp, server.ep) != (1, 1):
            raise ValueError("vLLM runs with server_params.dp and server_params.ep of 1; set its flags in engine_args")
        for block, flags, reserved in (
            ("server_params.engine_args", server.engine_args, SERVER_RESERVED),
            ("benchmark_params.client_args", self.benchmark_params.client_args, CLIENT_RESERVED),
        ):
            collisions = sorted(reserved & set(flags))
            if collisions:
                raise ValueError(f"{block} cannot set harness fields: {collisions}")
        # A cell may change a shared client field such as num_prompts, but not a flag the harness sets.
        harness_flags = CLIENT_RESERVED - set(ServingBenchmarkParams.model_fields)
        for key, overrides in self.sweeps.items():
            collisions = sorted(harness_flags & set(overrides))
            if collisions:
                raise ValueError(f"sweeps.{key} cannot set harness fields: {collisions}")
        _refuse_unimplemented(self, "vLLM")
        return self

    @classmethod
    def read_raw(cls, config_file, cluster):
        return substitute_placeholders(config_file, cluster, all_placeholders=True)

    @classmethod
    def old_keys(cls, raw):
        # Imported here because the map imports this module for its schema.
        from cvs.config_validation.migrate import engine
        from cvs.config_validation.migrate.maps import vllm

        return engine.old_paths(raw, vllm.CONFIG)

    @classmethod
    def validate_config(cls, raw, config_file):
        try:
            return cls.model_validate(raw, context={"placeholders": "reject", "config_path": Path(config_file)})
        except ValidationError as exc:
            raise ConfigError(config_file, problem_lines(exc)) from None

    @classmethod
    def orch_inputs(cls, raw, config, cluster):
        """The container orchestrator, with the config's ``container`` merged onto the cluster file's.

        The config wins on shared keys and replaces lists such as volumes. The
        merged block is a copy, so changing it does not change the cluster.
        """
        container = copy.deepcopy(_deep_merge(cluster.get("container") or {}, config.container.model_dump()))
        return {"orchestrator": "container", "container": container}

    @property
    def model_id(self):
        return self.server_params.model

    def bind_effective_topology(self, topology):
        self._effective_topology = topology

    def cell_key(self, isl, osl, concurrency, **_unused):
        return f"ISL={isl},OSL={osl},TP={self.server_params.tp},PP={self.server_params.pp},CONC={concurrency}"

    def resolved_runs(self):
        """One ``ResolvedRun`` per run, in ``runs`` order: ``benchmark_params``, then the cell's own overrides."""
        cells = self.cells()
        runs = []
        for key in self.runs:
            cell = cells[key]
            options = {
                **self.benchmark_params.options(),
                **self.sweeps[key],
                "random_input_len": cell.isl,
                "random_output_len": cell.osl,
                "max_concurrency": cell.concurrency,
            }
            runs.append(ResolvedRun(cell, options))
        return runs


class SglangBlock(StrictBlock):
    pass


class SglangConfig(LlmServingConfig):
    """An SGLang suite config: the shared serving blocks plus ``sglang``, which has no keys and marks the engine."""

    axes = _AXES

    sglang: SglangBlock

    @model_validator(mode="after")
    def _check_sglang(self):
        _refuse_unimplemented(self, "SGLang")
        return self


class AtomBlock(StrictBlock):
    """The keys only ATOM reads: ``driver`` picks ATOM's coordinator and its native client."""

    driver: Literal["atom", "vllm", "vllm_atom", "sglang"] = "vllm"


class AtomConfig(LlmServingConfig):
    """An ATOM suite config: the shared serving blocks plus ``atom``."""

    axes = _AXES

    atom: AtomBlock

    @model_validator(mode="after")
    def _check_atom(self):
        return self
