'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Config blocks every CVS suite schema is built from.

Keys that start with ``_`` are comments. Each model drops them at its own level
only, so a ``_`` key inside a free-form map (``env``, a CLI option payload)
reaches the suite unchanged.

Shipped configs are templates: ``<changeme>`` and ``<changeme-...>`` mark the
values a user fills in for their setup. A ``ConfigStr`` field rejects them only
when validated with ``context={"placeholders": "reject"}``, which
``cvs.lib.conf_lib.loader.load_config`` always passes.
'''

from __future__ import annotations

import re
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_serializer,
    model_validator,
)

_PLACEHOLDER_RE = re.compile(r"<changeme(?:-[^<>]*)?>", re.IGNORECASE)
PLACEHOLDER_MESSAGE = "still a <changeme> placeholder; replace it with the value for your setup"


def _drop_comment_keys(value):
    if isinstance(value, dict):
        return {key: item for key, item in value.items() if not (isinstance(key, str) and key.startswith("_"))}
    return value


def has_placeholder(value):
    return isinstance(value, str) and bool(_PLACEHOLDER_RE.search(value))


def placeholders_rejected(info):
    """True when the caller validates with ``context={"placeholders": "reject"}``."""
    return bool(info.context) and info.context.get("placeholders") == "reject"


def _reject_placeholder(value, info):
    if placeholders_rejected(info) and has_placeholder(value):
        raise ValueError(PLACEHOLDER_MESSAGE)
    return value


ConfigStr = Annotated[str, AfterValidator(_reject_placeholder)]


class _Forbid(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _drop_comments(cls, data):
        return _drop_comment_keys(data)


class _Allow(_Forbid):
    model_config = ConfigDict(extra="allow")


class Paths(_Forbid):
    shared_fs: ConfigStr
    models_dir: ConfigStr
    log_dir: ConfigStr
    hf_token_file: ConfigStr
    # Host-user-namespaced scratch (jaxmaxtext launchers/yml). Optional so
    # inference configs that omit it still load; jaxmaxtext configs set it to
    # /tmp/{user-id}/jaxmaxtext so container-root /tmp/root is never used.
    temp_dir: ConfigStr = ""


class ModelSpec(_Forbid):
    id: str
    remote: Literal[0, 1]
    precision: str = ""


class RuntimeSpec(_Allow):
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class ContainerSpec(_Forbid):
    lifetime: Literal["no_launch", "per_run", "persistent"] = "per_run"
    name: ConfigStr
    image: ConfigStr
    runtime: RuntimeSpec
    # Environment variables passed to the container at `docker run` time (as
    # `-e KEY=VALUE`, merged with the orchestrator defaults). The container
    # orchestrator reads `container_config['env']`; carrying it on the spec means
    # `container.model_dump()` forwards it verbatim to the runtime.
    env: Dict[str, ConfigStr] = Field(default_factory=dict)


class RuntimeArgs(_Allow):
    """The ``runtime.args`` keys the Docker runtime reads.

    An unset key is left out of the dump, so the runtime falls back to its own
    default exactly as if the config had not named it.
    """

    volumes: Optional[List[ConfigStr]] = None
    devices: Optional[List[str]] = None
    env: Optional[Dict[str, ConfigStr]] = None
    cap_add: Optional[List[str]] = None
    security_opt: Optional[List[str]] = None
    group_add: Optional[List[Union[str, int]]] = None
    ulimit: Optional[List[str]] = None
    network: Optional[str] = None
    ipc: Optional[str] = None
    user: Optional[Union[str, int]] = None
    shm_size: Optional[Union[str, int]] = None
    privileged: Optional[bool] = None
    registry: Optional[Dict[str, str]] = None

    @model_serializer(mode="wrap")
    def _drop_unset(self, handler):
        return {key: value for key, value in handler(self).items() if value is not None}


FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class ValueThreshold(_Forbid):
    kind: Literal["min", "max", "max_ms", "min_tok_s", "info"]
    value: FiniteNumber
    optional: bool = False


class WithinThreshold(_Forbid):
    kind: Literal["within"]
    value: FiniteNumber
    tolerance_pct: FiniteNumber
    optional: bool = False


class MinRatioThreshold(_Forbid):
    kind: Literal["min_ratio"]
    value: FiniteNumber
    reference: str
    optional: bool = False


ThresholdSpec = Annotated[
    Union[ValueThreshold, WithinThreshold, MinRatioThreshold],
    Field(discriminator="kind"),
]
ThresholdCell = Annotated[Dict[str, ThresholdSpec], BeforeValidator(_drop_comment_keys)]


class ThresholdFile(_Allow):
    """A threshold file: one cell of metric specs per run cell, plus ``accuracy``.

    Accuracy thresholds are keyed by task and read by the accuracy harness, so
    they are not checked here.
    """

    accuracy: Dict[str, Any] = Field(default_factory=dict)
    __pydantic_extra__: Dict[str, ThresholdCell]

    def cells(self):
        return dict(self.model_extra or {})
