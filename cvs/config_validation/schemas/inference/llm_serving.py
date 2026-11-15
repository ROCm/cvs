'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

``LlmServingConfig``, the base that vLLM, SGLang and ATOM configs extend, with
its server and benchmark blocks.

The server and benchmark blocks are shared by every engine as they are; an
engine adds one top-level block of its own. ``engine_args`` and ``client_args``
hold the server's and the benchmark client's own flags in snake_case, kept as
written for the engine to render; ``_`` keys inside them are not dropped.

Rules every engine follows are checks here: ``container.env`` is the only
environment block and holds no Hugging Face token, a per-cell override is a
client flag that names no harness field, every cell and threshold cell has the
TP and PP of ``server_params``, and with ``enforce_thresholds`` every run has a
threshold cell. ``select_hosts`` is the one host rule for every engine.
'''

import json
import re
from typing import Annotated, Any, Dict, Optional, Union

from pydantic import AfterValidator, Field, field_validator, model_validator

from cvs.config_validation.schemas.base import ConfigStr, SecretInputError, StrictBlock
from cvs.config_validation.schemas.common_training_inference import WorkloadConfig
from cvs.config_validation.schemas.inference.accuracy import Accuracy
from cvs.config_validation.schemas.sweeps import SweepsAndRuns, parse_cell_key
from cvs.config_validation.schemas.thresholds import ThresholdFile, Thresholds
from cvs.lib.conf_lib.config_blocks import PLACEHOLDER_MESSAGE, has_placeholder, placeholders_rejected

_FLAG_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_HF_TOKEN_ENV = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}

# Client flags every engine's harness sets itself, so a cell may not override them.
RESERVED_OVERRIDES = frozenset(
    {
        "model",
        "backend",
        "base_url",
        "max_concurrency",
        "random_input_len",
        "random_output_len",
        "result_dir",
        "result_filename",
        "percentile_metrics",
        "metric_percentiles",
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


def check_flags(flags, reserved=frozenset()):
    """``flags`` if every name is snake_case and its value renders as a flag, and none is in ``reserved``.

    True writes the bare flag; False is refused as ambiguous. A list holds
    scalars only.
    """
    for name, value in flags.items():
        if not _FLAG_NAME_RE.fullmatch(name):
            raise ValueError(f"flag names must be snake_case: {name!r}")
        if value is False:
            raise ValueError(f"{name}=false is ambiguous; omit it or use the flag's negative form")
        if isinstance(value, list) and any(isinstance(item, (list, dict)) for item in value):
            raise ValueError(f"{name}: lists may contain scalar values only")
    collisions = sorted(reserved & set(flags))
    if collisions:
        raise ValueError(f"cannot set harness fields: {collisions}")
    return flags


class ContainerEnvTokenError(SecretInputError):
    """A Hugging Face token name in container.env."""


class LongContextFlags(StrictBlock):
    """Server flags an engine adds only when ``niah`` evals run."""

    chunked_prefill_size: Optional[int] = None
    max_prefill_tokens: Optional[int] = None


class ServingServerParams(StrictBlock):
    """The server fields every engine reads; ``engine_args`` holds every other server flag.

    ``nnodes`` is the node count of one job; null means 1, single only.
    ``prefill_nnodes`` is for disaggregated runs; null means half of ``nnodes``.
    """

    model: ConfigStr
    tp: int
    pp: int = 1
    dp: int = 1
    ep: int = 1
    port: int = 8888
    master_port: int = 29501
    nnodes: Optional[int] = None
    prefill_nnodes: Optional[int] = None
    max_model_length: Optional[int] = None
    long_context: LongContextFlags = Field(default_factory=LongContextFlags)
    engine_args: Dict[str, FlagValue] = Field(default_factory=dict)

    @field_validator("engine_args")
    @classmethod
    def _check_engine_args(cls, engine_args):
        return check_flags(engine_args)


class ServingBenchmarkParams(StrictBlock):
    """The client fields every engine reads; ``client_args`` holds every other client flag."""

    num_prompts: int
    random_range_ratio: float = 0.0
    seed: int = 0
    request_rate: Union[str, float] = "inf"
    burstiness: float = 1.0
    random_prefix_len: int = 0
    tokenizer_mode: str = "auto"
    scaling_baseline_output_throughput: Optional[float] = None
    client_args: Dict[str, FlagValue] = Field(default_factory=dict)

    @field_validator("client_args")
    @classmethod
    def _check_client_args(cls, client_args):
        return check_flags(client_args)


Seconds = Annotated[int, Field(gt=0)]


class Timeouts(StrictBlock):
    """How long each wait may take, in seconds; None uses the engine's default.

    ``server_ready_s`` covers server launch to ready, and ``client_s`` one
    benchmark cell's client run.
    """

    server_ready_s: Optional[Seconds] = None
    client_s: Optional[Seconds] = None


def _cluster_hosts(cluster):
    nodes = cluster.get("node_dict") or {}
    return list(nodes) if isinstance(nodes, dict) else [node.get("mgmt_ip") for node in nodes]


class LlmServingConfig(Accuracy, SweepsAndRuns, Thresholds, WorkloadConfig):
    """An LLM serving suite's config. An engine subclass sets ``axes`` and adds its own block.

    The mixins come before ``WorkloadConfig``, so a loader hook one of them
    defines overrides ``SuiteConfig``'s default. The loader fills ``thresholds``
    from the file ``threshold_json`` names.
    """

    thresholds: ThresholdFile = Field(default_factory=ThresholdFile)
    server_params: ServingServerParams
    benchmark_params: ServingBenchmarkParams
    timeouts: Timeouts = Field(default_factory=Timeouts)

    # Before validation, so a pydantic error never carries the env values.
    @model_validator(mode="before")
    @classmethod
    def _reject_hf_token_env(cls, data):
        container = data.get("container") if isinstance(data, dict) else None
        env = container.get("env") if isinstance(container, dict) else None
        tokens = sorted(str(name) for name in env or {} if str(name).upper() in _HF_TOKEN_ENV)
        if tokens:
            raise ContainerEnvTokenError(
                f"container.env must not set {tokens}: container.env values become docker run -e arguments. "
                "Put the Hugging Face token in the file named by paths.hf_token_file instead."
            )
        return data

    @model_validator(mode="after")
    def _check_serving_rules(self):
        if self.container is not None and self.container.runtime.args.env is not None:
            raise ValueError("container.runtime.args.env is not read; set these variables in container.env")
        for key, overrides in self.sweeps.items():
            try:
                check_flags(overrides, RESERVED_OVERRIDES)
            except ValueError as exc:
                raise ValueError(f"sweeps.{key}: {exc}") from None
        server = (self.server_params.tp, self.server_params.pp)
        for cell in self.cells().values():
            if (getattr(cell, "tp", server[0]), getattr(cell, "pp", server[1])) != server:
                raise ValueError(f"cell {cell.key} conflicts with server_params.tp or server_params.pp")
        threshold_cells = self.threshold_cell_keys()
        for key in threshold_cells:
            cell = parse_cell_key(key, self.axes)
            if (getattr(cell, "tp", server[0]), getattr(cell, "pp", server[1])) != server:
                raise ValueError(f"threshold cell {key} conflicts with server_params.tp or server_params.pp")
        if self.enforce_thresholds:
            missing = [key for key in self.runs if key not in threshold_cells]
            if missing:
                raise ValueError(f"runs missing threshold coverage: {missing}")
        return self

    def threshold_cell_keys(self):
        """The cell keys of the threshold file, in file order."""
        if isinstance(self.thresholds, ThresholdFile):
            return list(self.thresholds.model_extra)
        return [key for key in self.thresholds if key != "accuracy"]

    @classmethod
    def select_hosts(cls, config, cluster, mode):
        """The hosts one job runs on, in the cluster file's order.

        Single takes the first node and distributed the first
        ``server_params.nnodes``, at least 2; extra nodes are left out. Any
        other mode, such as disaggregated, keeps every node for its engine to
        split.
        """
        hosts = _cluster_hosts(cluster)
        if mode == "single":
            wanted = 1
        elif mode == "distributed":
            wanted = config.server_params.nnodes
            if wanted is None or wanted < 2:
                raise ValueError("distributed runs need server_params.nnodes of 2 or more")
        else:
            return None
        if len(hosts) < wanted:
            raise ValueError(f"the cluster file has {len(hosts)} node(s); {mode} needs {wanted}")
        return hosts[:wanted]
