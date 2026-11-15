'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Loads a suite config through its schema's hooks, and picks the suite mode from
the test module's name.

``load_suite_config`` raises one ``ConfigError`` for the first check that
fails, in this order: an old config format, an ``_unmapped`` block left by
``cvs config migrate``, a ``<changeme>`` placeholder, then the schema. Each
line names a field and never its value. The threshold file a config names gets
the same four checks before the config's schema runs, so the schema's
validators see the file. Its old format is an old threshold kind, or an
``accuracy`` key outside ``lm_eval``, ``custom_eval``, ``quant_parity`` and
``mtp_quality``, such as a task id the old layout put there directly.

The ``_unmapped`` check reads the config file as written; the others read what
the schema's ``read_raw`` returns.

Keys that start with ``_`` count as comments at any depth for the
``<changeme>`` and old-kind checks. Free-form maps such as ``env`` keep their
``_`` keys, so a suite that needs those checked validates with
``context={"placeholders": "reject"}``.
'''

import json
from collections import namedtuple
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from cvs.config_validation.schemas.base import SecretInputError, problem_lines
from cvs.config_validation.schemas.thresholds import ACCURACY_SECTIONS, Thresholds
from cvs.lib.conf_lib.config_blocks import PLACEHOLDER_MESSAGE, has_placeholder
from cvs.lib.conf_lib.loader import ConfigError
from cvs.lib.utils.config_loader import _find_threshold_file
from cvs.lib.utils_lib import resolve_cluster_config_placeholders

LoadedSuiteConfig = namedtuple("LoadedSuiteConfig", "schema config cluster orch_inputs")

OLD_THRESHOLD_KINDS = ("max_ms", "min_tok_s", "within", "min_ratio")
UNMAPPED = "_unmapped"
UNMAPPED_FIELD_MESSAGE = "has no place in the new format; move or delete it, then delete the _unmapped block"
UNMAPPED_BLOCK_MESSAGE = "left by cvs config migrate; delete it once its fields are moved"
PLACEHOLDER_KEY_MESSAGE = f"this key is {PLACEHOLDER_MESSAGE}"
MISSING_THRESHOLD_MESSAGE = "names no file next to the config and no packaged threshold file"


def read_cluster(cluster_file):
    """The cluster file with ``{user-id}`` resolved, as the shared ``orch`` fixture reads it.

    A ``<changeme>`` left in the cluster file exits, as it does for every suite.
    """
    with open(cluster_file) as fp:
        return resolve_cluster_config_placeholders(json.load(fp))


def _read_json(path):
    """The JSON in ``path``; a file that is not JSON raises ``ConfigError`` naming it and the position."""
    try:
        with open(path) as fp:
            return json.load(fp)
    except json.JSONDecodeError as exc:
        raise ConfigError(path, [f"not valid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"]) from None


def load_suite_config(schema, config_file, cluster_file, log):
    """Load ``config_file`` with the ``SuiteConfig`` subclass ``schema`` and return a ``LoadedSuiteConfig``.

    A schema with the ``Thresholds`` fields also declares ``thresholds``. The
    file ``threshold_json`` names, found next to the config or as its packaged
    copy, is validated as that field's type and given to ``validate_config`` in
    ``thresholds``. An empty ``threshold_json`` records metrics without gating:
    ``validate_config`` gets an empty file and ``enforce_thresholds`` false.
    Unlike the older loaders, it never falls back to a lone ``*threshold.json``
    beside the config. ``orch_inputs`` gets the config as ``read_raw`` returned it.
    """
    return load_suite_config_with_cluster(schema, config_file, read_cluster(cluster_file), log)


def load_suite_config_with_cluster(schema, config_file, cluster, log):
    """``load_suite_config`` with the cluster file already read into ``cluster``.

    With ``{}``, a suite's collection hook can run every check before any
    fixture reads the cluster file.
    """
    written = _read_json(config_file)
    raw = schema.read_raw(config_file, cluster)
    _refuse_before_schema(schema, config_file, schema.old_keys(raw), written, raw)
    config_input = _with_thresholds(schema, raw, config_file, log) if issubclass(schema, Thresholds) else raw
    try:
        config = schema.validate_config(config_input, config_file)
    except SecretInputError as exc:
        raise ConfigError(config_file, [str(exc)]) from None
    return LoadedSuiteConfig(schema, config, cluster, schema.orch_inputs(raw, config, cluster))


def mode_from_module(schema, module_name):
    """The suite mode for a test module: the mode of the longest ``schema.modes`` suffix its name ends with.

    With no match the mode is ``schema.default_mode``. A schema that declares
    modes and no default refuses a module that matches none.
    """
    for suffix in sorted(schema.modes, key=len, reverse=True):
        if module_name.endswith(suffix):
            return schema.modes[suffix]
    if schema.modes and schema.default_mode is None:
        raise ValueError(
            f"{module_name} matches no {schema.__name__} mode; "
            f"the module name must end with one of: {', '.join(schema.modes)}"
        )
    return schema.default_mode


def _with_thresholds(schema, raw, config_file, log):
    """``raw`` with its threshold file in ``thresholds``.

    A config that is not an object, or whose ``threshold_json`` is not a
    string, is returned unchanged for ``validate_config`` to refuse.
    """
    field = schema.model_fields.get("thresholds")
    if field is None:
        raise TypeError(f"{schema.__name__} has the Thresholds fields but no thresholds field for its threshold file")
    name = raw.get("threshold_json", "") if isinstance(raw, dict) else None
    if not isinstance(name, str):
        return raw
    if not name.strip():
        log.warning("%s names no threshold_json; metrics are recorded without gating", config_file)
        return {**raw, "thresholds": {}, "enforce_thresholds": False}
    try:
        path = _find_threshold_file(raw, Path(config_file))
    except FileNotFoundError:
        raise ConfigError(config_file, [f"threshold_json: {MISSING_THRESHOLD_MESSAGE}"]) from None
    written = _read_json(path)
    _refuse_before_schema(schema, path, old_threshold_paths(written), written, written)
    try:
        return {**raw, "thresholds": TypeAdapter(field.rebuild_annotation()).validate_python(written)}
    except ValidationError as exc:
        raise ConfigError(path, problem_lines(exc)) from None


def _refuse_before_schema(schema, path, old, written, raw):
    if old:
        suite = schema.suite or "<suite>"
        raise ConfigError(
            path, [f"old config format: {', '.join(old)}; run cvs config migrate {suite} {path} <new config>"]
        )
    unmapped = _unmapped_lines(written)
    if unmapped:
        raise ConfigError(path, unmapped)
    placeholders = _placeholder_lines(raw)
    if placeholders:
        raise ConfigError(path, placeholders)


def _is_comment(key):
    return isinstance(key, str) and key.startswith("_")


def _children(node, path):
    if isinstance(node, dict):
        items = node.items()
    elif isinstance(node, list):
        items = enumerate(node)
    else:
        return []
    return [(key, value, f"{path}.{key}" if path else str(key)) for key, value in items]


def _unmapped_lines(node, path=""):
    lines = []
    for key, value, here in _children(node, path):
        if key != UNMAPPED:
            lines += _unmapped_lines(value, here)
        elif isinstance(value, dict) and value:
            lines += [f"{here}.{field}: {UNMAPPED_FIELD_MESSAGE}" for field in value]
        else:
            lines.append(f"{here}: {UNMAPPED_BLOCK_MESSAGE}")
    return lines


def _placeholder_lines(node, path=""):
    if has_placeholder(node):
        return [f"{path or '(config)'}: {PLACEHOLDER_MESSAGE}"]
    lines = []
    for key, value, here in _children(node, path):
        if _is_comment(key):
            continue
        if has_placeholder(key):
            lines.append(f"{here}: {PLACEHOLDER_KEY_MESSAGE}")
        lines += _placeholder_lines(value, here)
    return lines


def old_threshold_paths(written):
    """The paths in a threshold file that only the old format has: accuracy sections by task id, then old kinds."""
    accuracy = written.get("accuracy") if isinstance(written, dict) else None
    sections = [
        f"accuracy.{key}"
        for key in (accuracy if isinstance(accuracy, dict) else {})
        if not _is_comment(key) and key not in ACCURACY_SECTIONS
    ]
    return sections + _old_kind_paths(written)


def _old_kind_paths(node, path=""):
    if isinstance(node, dict) and node.get("kind") in OLD_THRESHOLD_KINDS:
        return [path or "(config)"]
    paths = []
    for key, value, here in _children(node, path):
        if not _is_comment(key):
            paths += _old_kind_paths(value, here)
    return paths
