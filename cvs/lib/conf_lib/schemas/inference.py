'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Shared base for inference suite schemas.

``InferenceConfig`` holds what every inference suite shares: vLLM, SGLang, ATOM
and xDiT. ``ServingConfig`` adds what the LLM serving engines share: the
parallelism fields, sweeps, runs, the cell key and accuracy. An engine schema
subclasses one of them and adds only the fields CVS reads for that engine.

A field belongs to CVS only if CVS reads it. Settings CVS only passes to the
engine go in ``engine_args``, and each engine renders them its own way.
'''

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Optional

from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

from cvs.lib.conf_lib.config_blocks import (
    PLACEHOLDER_MESSAGE,
    ConfigStr,
    ContainerSpec,
    Paths,
    RuntimeArgs,
    RuntimeSpec,
    ThresholdFile,
    _drop_comment_keys,
    _Forbid,
    has_placeholder,
    placeholders_rejected,
)
from cvs.lib.inference.utils.accuracy_config import AccuracyConfig
from cvs.lib.utils.config_loader import attach_threshold_file

_HF_TOKEN_ENV = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}
_ENGINE_ARG_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*$")
_N = r"[1-9][0-9]*"
_CELL_RE = re.compile(
    rf"^ISL=(?P<isl>{_N}),OSL=(?P<osl>{_N}),TP=(?P<tp>{_N}),PP=(?P<pp>{_N})"
    rf"(?:,DP=(?P<dp>{_N}))?(?:,EP=(?P<ep>{_N}))?,CONC=(?P<concurrency>{_N})$"
)
_THRESHOLD_COMMENT_PREFIXES = ("_comment", "_example")
_CELL_FORMAT = "ISL=<n>,OSL=<n>,TP=<n>,PP=<n>[,DP=<n>][,EP=<n>],CONC=<n>"

Port = Annotated[int, Field(ge=1, le=65535)]


def _problems(exc):
    """``loc: msg`` text for a nested ValidationError, without the input values."""
    lines = []
    for error in exc.errors(include_input=False):
        location = ".".join(str(part) for part in error["loc"])
        message = error["msg"].removeprefix("Value error, ")
        lines.append(f"{location}: {message}" if location else message)
    return "; ".join(lines)


def _reject_underscore_metrics(cells):
    # A misspelled metric such as _output_throughput would otherwise be dropped
    # as a comment, and the cell would still count as covered.
    for cell_key, cell in cells.items():
        if not isinstance(cell, dict):
            continue
        for metric, spec in cell.items():
            if metric.startswith(_THRESHOLD_COMMENT_PREFIXES):
                continue
            if metric.startswith("_"):
                raise ValueError(f"thresholds.{cell_key}: unknown metric {metric!r}")
            extra = sorted(key for key in spec if key.startswith("_")) if isinstance(spec, dict) else []
            if extra:
                raise ValueError(f"thresholds.{cell_key}.{metric}: unknown threshold fields {extra}")


def _reject_option_placeholder(value, info):
    if not placeholders_rejected(info):
        return value
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if has_placeholder(text):
        raise ValueError(PLACEHOLDER_MESSAGE)
    return value


EngineArgValue = Annotated[Any, AfterValidator(_reject_option_placeholder)]
EngineArgs = Annotated[Dict[str, EngineArgValue], BeforeValidator(_drop_comment_keys)]


def _engine_arg_key(name):
    return name.replace("-", "_")


def _merge_engine_args(base, override):
    """``base`` with ``override`` on top, where ``a-b`` and ``a_b`` name the same option."""
    overridden = {_engine_arg_key(name) for name in override}
    return {**{name: value for name, value in base.items() if _engine_arg_key(name) not in overridden}, **override}


def _validate_engine_args(args, reserved):
    seen = {}
    for name, value in args.items():
        if not _ENGINE_ARG_NAME_RE.fullmatch(name):
            raise ValueError(f"engine_args names must be lowercase snake_case or kebab-case: {name!r}")
        if _engine_arg_key(name) in seen:
            raise ValueError(f"engine_args names {seen[_engine_arg_key(name)]!r} and {name!r} set the same option")
        seen[_engine_arg_key(name)] = name
        if value is False:
            raise ValueError(f"engine_args.{name}=false is ambiguous; omit it or use the option's negative flag")
        if isinstance(value, list) and any(isinstance(item, (list, dict)) for item in value):
            raise ValueError(f"engine_args.{name} lists may contain scalar values only")
        if isinstance(value, dict):
            try:
                json.dumps(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"engine_args.{name} must be JSON serializable") from exc
    collisions = sorted(name for name in args if _engine_arg_key(name) in reserved)
    if collisions:
        raise ValueError(f"engine_args cannot set fields CVS renders: {collisions}")


class ContainerEnvTokenError(Exception):
    """A Hugging Face token name in container.env.

    Not a ValueError, so pydantic raises it as is instead of wrapping it in a
    ValidationError whose str(), errors() and json() would carry the token value.
    """


class InferenceRuntimeArgs(RuntimeArgs):
    model_config = ConfigDict(extra="forbid")

    # container.env is the one env block, so the Hugging Face token check
    # cannot be bypassed through docker run -e arguments built from here.
    @model_validator(mode="before")
    @classmethod
    def _reject_env(cls, data):
        if isinstance(data, dict) and "env" in data:
            raise ValueError("runtime.args.env is not read; set these variables in container.env")
        return data


class InferenceRuntime(RuntimeSpec):
    model_config = ConfigDict(extra="forbid")

    args: InferenceRuntimeArgs = Field(default_factory=InferenceRuntimeArgs)


class InferenceContainerSpec(ContainerSpec):
    runtime: InferenceRuntime

    @model_validator(mode="before")
    @classmethod
    def _reject_hf_token_env(cls, data):
        env = data.get("env") if isinstance(data, dict) else None
        tokens = sorted(name for name in env if str(name).upper() in _HF_TOKEN_ENV) if isinstance(env, dict) else []
        if tokens:
            raise ContainerEnvTokenError(
                f"container.env must not set {tokens}: container.env values become docker run -e arguments. "
                "Put the Hugging Face token in the file named by paths.hf_token_file instead."
            )
        return data


class EngineParams(_Forbid):
    """CVS fields plus ``engine_args``, the settings CVS passes to the engine as they are."""

    engine_args: EngineArgs = Field(default_factory=dict)

    @classmethod
    def reserved_engine_args(cls):
        """Names ``engine_args`` may not set, with ``-`` written as ``_``.

        Every field of the block is reserved. A subclass adds its engine's
        spellings of the flags CVS renders.
        """
        return frozenset(cls.model_fields) - {"engine_args"}

    @model_validator(mode="after")
    def _check_engine_args(self):
        _validate_engine_args(self.engine_args, self.reserved_engine_args())
        return self


class InferenceServerParams(EngineParams):
    model: ConfigStr

    @classmethod
    def reserved_engine_args(cls):
        return super().reserved_engine_args() | {
            "model",
            "host",
            "port",
            "nnodes",
            "node_rank",
            "master_addr",
            "master_port",
        }


class ServingServerParams(InferenceServerParams):
    """The server fields every LLM serving engine reads. Unset fields take the engine's default."""

    tp: PositiveInt
    pp: PositiveInt = 1
    dp: PositiveInt = 1
    ep: PositiveInt = 1
    port: Optional[Port] = None
    dist_init_port: Optional[Port] = None
    max_model_len: Optional[PositiveInt] = None
    server_precheck_wait_s: Optional[NonNegativeInt] = None
    server_warmup_wait_s: Optional[NonNegativeInt] = None
    server_poll_iterations: Optional[PositiveInt] = None
    server_poll_wait_s: Optional[NonNegativeInt] = None

    @field_validator("ep")
    @classmethod
    def _reject_ep(cls, value):
        if value != 1:
            raise ValueError("ep is reserved; no suite supports expert parallelism yet")
        return value


class ServingBenchmarkParams(EngineParams):
    """The benchmark client fields every LLM serving engine reads. Unset fields take the engine's default."""

    dataset_name: Optional[ConfigStr] = None
    num_prompts: Optional[PositiveInt] = None
    random_range_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    client_initial_wait_s: Optional[NonNegativeInt] = None
    client_poll_iterations: Optional[PositiveInt] = None
    client_poll_wait_s: Optional[NonNegativeInt] = None

    @classmethod
    def reserved_engine_args(cls):
        # The cell key sets input and output length and concurrency for each run.
        return super().reserved_engine_args() | {
            "model",
            "host",
            "port",
            "base_url",
            "max_concurrency",
            "random_input_len",
            "random_output_len",
        }


def format_cell_key(isl, osl, tp, pp, concurrency, dp=1, ep=1):
    """The cell key for one run. DP and EP appear only when they are 2 or more."""
    parts = [f"ISL={isl}", f"OSL={osl}", f"TP={tp}", f"PP={pp}"]
    if dp > 1:
        parts.append(f"DP={dp}")
    if ep > 1:
        parts.append(f"EP={ep}")
    parts.append(f"CONC={concurrency}")
    return ",".join(parts)


@dataclass(frozen=True)
class RunCell:
    key: str
    isl: int
    osl: int
    tp: int
    pp: int
    dp: int
    ep: int
    concurrency: int

    @classmethod
    def parse(cls, value):
        match = _CELL_RE.fullmatch(value)
        if not match:
            raise ValueError(f"cell key must be {_CELL_FORMAT}")
        # One cell has one spelling, so DP=1 is left out of the key.
        if match["dp"] == "1":
            raise ValueError("omit DP=1 from the cell key")
        if match["ep"] is not None:
            raise ValueError("EP is reserved; no suite supports expert parallelism yet")
        values = {name: int(number or 1) for name, number in match.groupdict().items()}
        return cls(value, **values)


class InferenceConfig(_Forbid):
    """The fields every inference suite shares.

    ``thresholds`` comes from the file ``threshold_json`` names; a config that
    names none runs record-only. Its shape is the engine's.
    """

    enforce_thresholds: bool = True
    threshold_json: str = ""
    thresholds: Dict[str, Any] = Field(default_factory=dict)
    paths: Paths
    container: InferenceContainerSpec
    server_params: InferenceServerParams

    @model_validator(mode="before")
    @classmethod
    def _attach_threshold_file(cls, data, info):
        return attach_threshold_file(data, info.context)


class ServingConfig(InferenceConfig):
    """What every LLM serving engine shares: parallelism, sweeps, runs, cell keys and accuracy.

    A ``sweeps`` value is a partial ``benchmark_params`` for that cell. Threshold
    cells match sweep keys exactly; the other top-level threshold keys are
    ``non_cell_threshold_keys``.
    """

    server_params: ServingServerParams
    benchmark_params: ServingBenchmarkParams = Field(default_factory=ServingBenchmarkParams)
    sweeps: Dict[str, Dict[str, Any]]
    runs: List[str]
    accuracy: AccuracyConfig = Field(default_factory=AccuracyConfig)
    _run_cells: Dict[str, RunCell] = PrivateAttr(default_factory=dict)
    _run_benchmark_params: Dict[str, ServingBenchmarkParams] = PrivateAttr(default_factory=dict)

    @classmethod
    def non_cell_threshold_keys(cls):
        """Top-level threshold keys that are not cells. A subclass adds its engine's own."""
        return frozenset({"accuracy"})

    @field_validator("sweeps", mode="before")
    @classmethod
    def _drop_sweep_comments(cls, value):
        if not isinstance(value, dict):
            return value
        return {key: _drop_comment_keys(entry) for key, entry in value.items() if not key.startswith("_")}

    @model_validator(mode="after")
    def _check_runs_and_thresholds(self, info):
        if not self.runs:
            raise ValueError("runs must be a nonempty explicit list")
        if len(set(self.runs)) != len(self.runs):
            raise ValueError("runs contains duplicate cells")
        unknown_runs = sorted(set(self.runs) - set(self.sweeps))
        if unknown_runs:
            raise ValueError(f"runs reference unknown sweeps: {unknown_runs}")

        server = self.server_params
        cells = {}
        for key in self.sweeps:
            try:
                cell = RunCell.parse(key)
            except ValueError as exc:
                raise ValueError(f"sweeps.{key}: {exc}") from None
            if (cell.tp, cell.pp, cell.dp, cell.ep) != (server.tp, server.pp, server.dp, server.ep):
                raise ValueError(f"{key} does not match server_params tp, pp, dp and ep")
            cells[key] = cell

        params_cls = type(self.benchmark_params)
        base = self.benchmark_params.model_dump(exclude_unset=True)
        resolved = {}
        for key, override in self.sweeps.items():
            merged = {**base, **override}
            if "engine_args" in override and isinstance(override["engine_args"], dict):
                merged["engine_args"] = _merge_engine_args(self.benchmark_params.engine_args, override["engine_args"])
            try:
                resolved[key] = params_cls.model_validate(merged, context=info.context)
            except ValidationError as exc:
                raise ValueError(f"sweeps.{key}: {_problems(exc)}") from None

        non_cell = self.non_cell_threshold_keys()
        orphans = sorted(key for key in self.thresholds if key not in non_cell and key not in cells)
        if orphans:
            raise ValueError(f"threshold keys match no sweep cell: {orphans}")
        threshold_cells = {key: spec for key, spec in self.thresholds.items() if key in cells}
        _reject_underscore_metrics(threshold_cells)
        try:
            validated = ThresholdFile.model_validate(threshold_cells).cells()
        except ValidationError as exc:
            raise ValueError(f"thresholds: {_problems(exc)}") from None
        # Cells keep the specs as written, minus comment keys, for code that reads them as dicts.
        self.thresholds = {
            **{key: value for key, value in self.thresholds.items() if key in non_cell},
            **{key: {metric: threshold_cells[key][metric] for metric in cell} for key, cell in validated.items()},
        }
        if self.enforce_thresholds:
            missing = [key for key in self.runs if key not in self.thresholds]
            if missing:
                raise ValueError(f"selected runs missing threshold coverage: {missing}")

        self._run_cells = cells
        self._run_benchmark_params = resolved
        return self

    def cell(self, key):
        return self._run_cells[key]

    def run_benchmark_params(self, key):
        """``benchmark_params`` with the ``sweeps`` override for cell ``key`` applied."""
        return self._run_benchmark_params[key]

    def cell_key(self, isl, osl, concurrency):
        server = self.server_params
        return format_cell_key(isl, osl, server.tp, server.pp, concurrency, dp=server.dp, ep=server.ep)
