"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import getpass
import json
import os
import re
from pathlib import Path
from typing import Dict, Type

import yaml
from pydantic import ValidationError

from cvs.lib.config.base import BaseTestConfig

# framework literal -> concrete config class. Populated by @register_config when
# each framework module under cvs/lib/config/frameworks/ is imported.
CONFIG_REGISTRY: Dict[str, Type[BaseTestConfig]] = {}

_ENV_REF_RE = re.compile(r"^\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}$")
_SENTINEL = "<changeme>"


class ConfigError(Exception):
    """Raised when a config cannot be parsed, dispatched, or validated."""


def register_config(framework: str):
    """Decorator: register a concrete config class under its ``framework`` key."""

    def _wrap(cls: Type[BaseTestConfig]) -> Type[BaseTestConfig]:
        CONFIG_REGISTRY[framework] = cls
        return cls

    return _wrap


def _ensure_frameworks_loaded() -> None:
    # Lazy import so loader has no import-time dependency on the framework
    # modules (which import this module for @register_config).
    import cvs.lib.config.frameworks  # noqa: F401


def _resolve_placeholders(obj):
    """Resolve ``{user-id}`` and ``${env:VAR}`` in string leaves; reject sentinels.

    B3: an ``${env:VAR}`` whose variable is *absent* from the environment fails
    closed with a :class:`ConfigError` -- distinguishing an unset required value
    from a deliberately-empty one (``export VAR=`` resolves to ``""`` and is
    allowed). This stops a missing credential from silently becoming ``""`` and
    surfacing far downstream.
    """
    if isinstance(obj, dict):
        return {k: _resolve_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_placeholders(v) for v in obj]
    if isinstance(obj, str):
        if _SENTINEL in obj:
            raise ConfigError(f"unresolved '{_SENTINEL}' sentinel in config value: {obj!r}")
        env_match = _ENV_REF_RE.match(obj)
        if env_match:
            var = env_match.group(1)
            if var not in os.environ:
                raise ConfigError(f"required env var '${{env:{var}}}' is not set")
            return os.environ[var]
        # An embedded or malformed ``${env:...}`` would otherwise slip through
        # unresolved (and unchecked for unset), defeating the fail-closed intent.
        if "${env:" in obj:
            raise ConfigError(
                f"malformed env reference {obj!r}: only a whole-value '${{env:VAR}}' is supported, not an embedded one"
            )
        return obj.replace("{user-id}", getpass.getuser())
    return obj


def _resolve_class(raw: dict) -> Type[BaseTestConfig]:
    """Validate the config root and dispatch on ``framework`` to its class."""
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    framework = raw.get("framework")
    if framework is None:
        raise ConfigError("config is missing required 'framework' field")
    # Guard the type before the registry lookup: an unhashable ``framework``
    # (list/dict) would make ``CONFIG_REGISTRY.get(framework)`` raise a raw
    # TypeError that escapes the ConfigError contract, and a non-str (int/bool)
    # can never match a registered key anyway.
    if not isinstance(framework, str):
        raise ConfigError(f"'framework' must be a string, got {type(framework).__name__}")
    cls = CONFIG_REGISTRY.get(framework)
    if cls is None:
        known = ", ".join(sorted(CONFIG_REGISTRY)) or "<none>"
        raise ConfigError(f"unknown framework {framework!r}; registered frameworks: {known}")
    return cls


def parse_config(raw: dict) -> BaseTestConfig:
    """Dispatch on ``framework``, resolve placeholders, and validate."""
    _ensure_frameworks_loaded()
    cls = _resolve_class(raw)
    resolved = _resolve_placeholders(raw)
    try:
        return cls.model_validate(resolved)
    except ValidationError as exc:
        raise ConfigError(f"invalid {raw.get('framework')} config: {exc}") from exc


def load_config_file(path) -> BaseTestConfig:
    """Load a v2 YAML (or JSON) config file into its typed config object."""
    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        raw = yaml.safe_load(text)
    elif p.suffix == ".json":
        raw = json.loads(text)
    else:
        raise ConfigError(f"unsupported config extension {p.suffix!r} (use .yaml or .json)")
    return parse_config(raw)
