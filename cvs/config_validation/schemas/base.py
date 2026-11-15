'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Base models, strings and load hooks every opt-in suite schema is built from.

Keys that start with ``_`` are comments. Each block drops them at its own level
only, so a ``_`` key inside a free-form map such as ``env`` reaches the suite
unchanged.

A ``ConfigStr`` field rejects ``<changeme>`` and ``<changeme-...>`` only when
validated with ``context={"placeholders": "reject"}``. It uses the detection in
``cvs.lib.conf_lib.config_blocks``, so the old and new loaders refuse the same
forms.
'''

import json
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, model_validator

from cvs.lib.conf_lib.config_blocks import PLACEHOLDER_MESSAGE, has_placeholder, placeholders_rejected
from cvs.lib.conf_lib.loader import ConfigError


def drop_comment_keys(value):
    if isinstance(value, dict):
        return {key: item for key, item in value.items() if not (isinstance(key, str) and key.startswith("_"))}
    return value


def problem_lines(exc):
    """One ``location: problem`` line per error in a ``ValidationError``, never with the input value."""
    lines = []
    for error in exc.errors(include_input=False):
        location = ".".join(str(part) for part in error["loc"]) or "(config)"
        lines.append(f"{location}: {error['msg'].removeprefix('Value error, ')}")
    return lines


def reject_placeholder(value, info):
    if placeholders_rejected(info) and has_placeholder(value):
        raise ValueError(PLACEHOLDER_MESSAGE)
    return value


ConfigStr = Annotated[str, AfterValidator(reject_placeholder)]
Port = Annotated[int, Field(ge=1, le=65535)]


class StrictBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _drop_comments(cls, data):
        return drop_comment_keys(data)


class OpenBlock(StrictBlock):
    model_config = ConfigDict(extra="allow")


class SuiteConfig(StrictBlock):
    """A suite's top-level config and the hooks its loader calls.

    The defaults read the file as written, report no old keys, validate without
    rejecting placeholders, and give the orchestrator the raw config, as the
    shared ``orch`` fixture does for a suite without a schema. A suite overrides
    only the hooks it needs.

    ``validate_config`` raises ``ConfigError`` with one line per problem that
    names the field and never its value, which can be a credential.
    """

    @classmethod
    def read_raw(cls, config_file, cluster):
        with open(config_file) as fp:
            return json.load(fp)

    @classmethod
    def old_keys(cls, raw):
        return []

    @classmethod
    def validate_config(cls, raw, config_file):
        try:
            return cls.model_validate(raw, context={"config_path": Path(config_file)})
        except ValidationError as exc:
            raise ConfigError(config_file, problem_lines(exc)) from None

    @classmethod
    def orch_inputs(cls, raw, config, cluster):
        return raw
