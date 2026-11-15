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
from typing import Annotated, ClassVar, Dict, Optional

from pydantic import AfterValidator, BaseModel, ConfigDict, ValidationError, model_validator

from cvs.lib.conf_lib.config_blocks import PLACEHOLDER_MESSAGE, has_placeholder, placeholders_rejected
from cvs.lib.conf_lib.loader import ConfigError


class SecretInputError(Exception):
    """A config problem a validator raises when the input may be a secret; its message names fields only.

    Not a ValueError, so pydantic raises it as is instead of wrapping it in a
    ValidationError whose str(), errors() and json() would carry the input.
    """


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

    ``suite`` is the name ``cvs config migrate`` takes for this schema.
    ``modes`` maps a test module name's suffix to the suite's mode, and
    ``default_mode`` is the mode when no suffix matches.
    """

    suite: ClassVar[Optional[str]] = None
    modes: ClassVar[Dict[str, str]] = {}
    default_mode: ClassVar[Optional[str]] = None

    @classmethod
    def read_raw(cls, config_file, cluster):
        """The config to check and validate; by default, the file as written.

        The loader looks for old keys and ``<changeme>`` in what this returns,
        so an override may substitute values and drop ``_`` comment keys but
        keeps every other key. The ``_unmapped`` check reads the file itself.
        """
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
