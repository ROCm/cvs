'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Converts an old suite config or threshold file to the new format through the
suite's map in ``cvs.config_validation.migrate.maps``. The input is never
changed.

``migrate`` converts a file in six steps:

1. Read it as written, keeping comments and ``{user-id}``-style placeholders.
2. Stop unless it holds one of the map's old keys, or for a threshold file an
   old kind: it is already in the new format, or another suite's map matches.
3. Move each renamed key, and each field a ``FIELDS`` function places. Two keys
   that reach one path with different values stop it.
4. Copy every other key to the same path, except an empty ``{}`` or ``[]``
   whose new default is the same. A block whose keys all moved is left out with
   its comments. When the schema has a block named for the suite, such as
   ``vllm``, and the input has none, add it empty, since that block names the
   engine. When the map sets ``FILL_THRESHOLD_FROM_SIBLING``, fill an empty
   ``threshold_json`` with the one ``*threshold.json`` beside the input.
5. Validate as ``load_suite_config`` does, with placeholders allowed: a schema
   with the ``Thresholds`` fields gets the threshold file the config names,
   found as the loader finds it from the input's folder, or records only when it
   names none. A refused key moves to ``_unmapped`` under its old dotted path, a
   missing required field is written as ``<changeme>``, and any other problem,
   including one in the threshold file, is named.
6. Write the output through a temporary file beside it, ``_unmapped`` first,
   holding any ``_unmapped`` block the input had, and print one line per field
   to fix.

A threshold file holds metric specs, objects with a ``kind``, in its top-level
sections and in the sections of its ``accuracy``. Its old kinds are renamed
through the map's ``THRESHOLDS``; a spec with no new kind, and a top-level entry
that is not a section, go to ``_unmapped``. An ``accuracy`` section keyed by
task id, the old layout, moves under ``accuracy.lm_eval``. It is validated as
the schema's ``thresholds`` type, as the loader validates it, or as a
``ThresholdFile`` when the schema has none.

``flags_from_list`` turns a flag list as old configs write it into a flag map,
for a map's ``FIELDS`` function.

``migrate`` returns 0 when it writes a clean file, or writes nothing because
the input is already in the new format; 1 when it writes a file with fields to
fix; and 2 when it stops for any other reason, writing nothing. Its messages
name fields, never their values.
'''

import contextlib
import copy
import importlib
import json
import os
import shlex
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel, TypeAdapter, ValidationError

from cvs.config_validation.loaders.suite_config import (
    MISSING_THRESHOLD_MESSAGE,
    OLD_THRESHOLD_KINDS,
    UNMAPPED,
    UNMAPPED_FIELD_MESSAGE,
    _with_thresholds,
)
from cvs.config_validation.migrate import maps
from cvs.config_validation.schemas.thresholds import ACCURACY_SECTIONS, ThresholdFile, Thresholds
from cvs.lib.conf_lib.config_blocks import has_placeholder
from cvs.lib.conf_lib.loader import ConfigError

TO_UNMAPPED = object()
CHANGEME = "<changeme>"
MISSING_MESSAGE = "required; written as <changeme>, replace it with the value for your setup"

_MISSING = object()
# The loader's warning that an empty threshold_json records only is for a test run, not a conversion.
_QUIET_LOG = SimpleNamespace(warning=lambda *args: None)


class _Stop(Exception):
    """Stops a conversion before anything is written; the message is the line to print."""

    def __init__(self, message, code=2):
        super().__init__(message)
        self.code = code


class _SuiteMaps(Mapping):
    """Suite name to map module, importing a map only when it is looked up."""

    def __init__(self, names):
        self._names = list(names)

    def __getitem__(self, suite):
        if suite not in self._names:
            raise KeyError(suite)
        return importlib.import_module(f"{maps.__name__}.{suite}")

    def __contains__(self, suite):
        return suite in self._names

    def __iter__(self):
        return iter(self._names)

    def __len__(self):
        return len(self._names)


def suite_maps():
    """Each suite's map module in ``cvs.config_validation.migrate.maps``, by suite name, imported on lookup."""
    return _SuiteMaps(suite_names())


def suite_names():
    """The suites ``migrate`` converts, one per map module, sorted, without importing them."""
    return maps.suite_names()


def old_paths(raw, renames):
    """The dotted paths in ``raw`` that ``renames``, a map's ``CONFIG``, renames or leaves out, in file order."""
    found = []

    def visit(node, path):
        if path and _first_match(renames, path) is not None:
            found.append(_dotted(path))
            return
        for key, value in _children(node):
            visit(value, path + (key,))

    visit(raw, ())
    return found


def migrate(suite, input_path, output_path, force=False, out=print):
    """Convert ``input_path`` for ``suite``, write it to ``output_path`` and return the exit code.

    ``output_path`` may be an existing directory, which gets the input's file
    name. An existing output file is replaced only with ``force``. ``out`` is
    called with each line to print.
    """
    found = suite_maps()
    if suite not in found:
        out(f"unknown suite {suite}; the suites with a map are: {', '.join(sorted(found)) or 'none'}")
        return 2
    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.is_dir():
        output_path = output_path / input_path.name
    try:
        conversion = _convert(suite, found, _read(input_path), input_path, output_path)
        lines = conversion.settle()
    except _Stop as stop:
        out(str(stop))
        return stop.code
    if output_path.exists() and output_path.samefile(input_path):
        out(f"{output_path} is the input; write the new file elsewhere")
        return 2
    if output_path.exists() and not force:
        out(f"{output_path} already exists; use --force to overwrite it")
        return 2
    try:
        _write(output_path, json.dumps(conversion.output(), indent=2, ensure_ascii=False) + "\n")
    except OSError as exc:
        out(f"cannot write {output_path}: {exc.strerror}")
        return 2
    out(f"wrote {output_path}; fix these fields before using it:" if lines else f"wrote {output_path}")
    for line in lines:
        out(f"  {line}")
    return 1 if lines else 0


def migrate_many(suite, input_paths, output_path, force=False, out=print):
    """``migrate`` for each of ``input_paths``; return the highest of their exit codes.

    With one input, ``output_path`` is as ``migrate`` takes it. With more, it
    must be an existing folder, each file is written there under its own name,
    and the first line printed for each file names it. Threshold files go
    first, so a config passed with its threshold file is checked against the
    converted one. Inputs that share a file name stop the run before anything
    is written.
    """
    input_paths, output_path = [Path(path) for path in input_paths], Path(output_path)
    if len(input_paths) == 1:
        return migrate(suite, input_paths[0], output_path, force=force, out=out)
    if not output_path.is_dir():
        out(f"{output_path} is not a folder; with several input files, the output must be an existing folder")
        return 2
    names = [path.name for path in input_paths]
    shared = sorted({name for name in names if names.count(name) > 1})
    if shared:
        out(f"several input files are named {', '.join(shared)}; they would be written to one file")
        return 2
    codes = []
    for path in sorted(input_paths, key=lambda path: not _holds_thresholds(path)):
        lines = []
        codes.append(migrate(suite, path, output_path, force=force, out=lines.append))
        first, rest = (lines[0], lines[1:]) if lines else ("", [])
        out(first if str(path) in first or str(output_path) in first else f"{path}: {first}")
        for line in rest:
            out(line)
    return max(codes)


def _holds_thresholds(path):
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(raw, dict) and _is_threshold_file(raw)


def _read(path):
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _Stop(f"cannot read {path}: {exc.strerror}") from None
    except UnicodeDecodeError:
        raise _Stop(f"{path} is not UTF-8 text") from None
    if not text.strip():
        raise _Stop(f"{path} is empty")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _Stop(f"{path} is not valid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}") from None
    if not isinstance(raw, dict):
        raise _Stop(f"{path} must hold a JSON object")
    if not raw:
        raise _Stop(f"{path} is empty")
    return raw


def _write(path, text):
    """Write ``text`` to ``path`` through a new file beside it.

    A failed write leaves ``path`` as it was and removes the folders it made.
    """
    created = []
    folder = path.parent
    while not folder.exists():
        created.append(folder)
        folder = folder.parent
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    opened = written = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp, "x", encoding="utf-8") as fp:
            opened = True
            fp.write(text)
        os.replace(temp, path)
        written = True
    finally:
        if not written:
            if opened:
                temp.unlink(missing_ok=True)
            for folder in created:
                with contextlib.suppress(OSError):
                    folder.rmdir()


def _convert(suite, found, raw, input_path, output_path=None):
    suite_map = found[suite]
    if _is_threshold_file(raw):
        tree, old, unmapped = _rename_kinds(raw, suite_map.THRESHOLDS)
        tree, moved = _nest_accuracy_sections(tree)
        old += moved
        if not old:
            raise _already_new(input_path)
        field = suite_map.SCHEMA.model_fields.get("thresholds")
        schema = field.rebuild_annotation() if field is not None else ThresholdFile
        conversion = _Conversion(schema, {}, {}, tree, unmapped=unmapped)
        conversion.walk()
        return conversion
    if not old_paths(raw, suite_map.CONFIG):
        others = [name for name, other in sorted(found.items()) if name != suite and old_paths(raw, other.CONFIG)]
        if others:
            raise _Stop(
                f"{input_path} looks like {' or '.join(others)}, not {suite}; "
                f"run cvs config migrate {others[0]} {input_path} <output>"
            )
        raise _already_new(input_path)
    fields = getattr(suite_map, "FIELDS", {})
    conversion = _Conversion(
        suite_map.SCHEMA, suite_map.CONFIG, fields, raw, config_path=input_path, output_path=output_path
    )
    conversion.walk()
    conversion.set_engine(suite)
    if getattr(suite_map, "FILL_THRESHOLD_FROM_SIBLING", False):
        conversion.fill_threshold_json(input_path)
    return conversion


def _already_new(input_path):
    return _Stop(f"{input_path} is already in the new format; nothing to convert", code=0)


class _Conversion:
    """The new tree for one input, built key by key, remembering where each moved key came from."""

    def __init__(self, schema, renames, fields, raw, unmapped=None, config_path=None, output_path=None):
        """``config_path`` is the input path when it is a config, which ``settle`` validates as the loader would.

        It is None for a threshold file. A threshold file the config names that
        is already written beside ``output_path`` is the one it is checked
        against, so a config converted with its threshold file sees the new one.
        """
        self.schema = schema
        self.renames = renames
        self.fields = fields
        self.raw = raw
        self.config_path = config_path
        self.output_path = output_path
        self.tree = {}
        self.origins = {}
        self.slots = {}
        earlier = raw.get(UNMAPPED, {})
        self.unmapped = dict(earlier) if isinstance(earlier, dict) else {UNMAPPED: earlier}
        for path, value in (unmapped or {}).items():
            self._unmap(path, value)

    def walk(self):
        for key, value in self.raw.items():
            if key != UNMAPPED:
                self._walk(value, (key,))

    def _walk(self, node, path):
        """Place ``node`` and return True when a rename or field function took it or any key under it."""
        match = _first_match(self.fields, path)
        if match is not None:
            self._apply_field(match, node, path)
            return True
        match = _first_match(self.renames, path)
        if match is not None:
            target = self.renames[match[0]]
            if target is not None:
                self._move(self._target(target, match[1], path), node, path)
            return True
        children = [] if _is_comment(path[-1]) else _children(node)
        if not children:
            if not self._is_empty_default(node, path):
                self._place(path, node, path)
            return False
        moved = False
        for key, value in children:
            moved = self._walk(value, path + (key,)) or moved
        if moved and isinstance(node, dict):
            self._drop_if_only_comments(path)
        return moved

    def _apply_field(self, match, node, path):
        pattern, captures = match
        result = self.fields[pattern](node, self.raw)
        if result is TO_UNMAPPED:
            self._unmap(_dotted(path), node)
            return
        for target, value in result.items():
            self._move(self._target(target, captures, path), value, path)

    def _target(self, pattern, captures, origin):
        """``pattern`` with each ``*`` filled with the key of ``origin`` at the matching position in ``captures``.

        A list item stays at its index when it lands in the same list and is
        appended, in order, to any other list, so no gap is left there.
        """
        path = []
        positions = list(captures)
        for part in pattern.split("."):
            if part != "*":
                path.append(part)
                continue
            if not positions:
                raise ValueError(f"map target {pattern} has more * than the old path it comes from")
            position = positions.pop(0)
            key = origin[position]
            if isinstance(key, int) and tuple(path) != origin[:position]:
                slots = self.slots.setdefault(tuple(path), {})
                key = slots.setdefault(origin[: position + 1], len(slots))
            path.append(key)
        return tuple(path)

    def _unmap(self, path, value):
        if path in self.unmapped and not _same(self.unmapped[path], value):
            raise _Stop(
                f"{path} is both in the input's {UNMAPPED} block and set in the input, with different values; "
                "keep one, then run cvs config migrate again"
            )
        self.unmapped[path] = copy.deepcopy(value)

    def _move(self, target, value, origin):
        self._place(target, value, origin)
        if target != origin:
            self.origins[target] = origin

    def _place(self, path, value, origin):
        parent = self.tree
        for depth in range(1, len(path)):
            parent = self._container(parent, path[:depth], path[depth], origin)
        existing = _get(parent, path[-1])
        if existing is _MISSING:
            _put(parent, path[-1], copy.deepcopy(value))
        elif not _merge(existing, value):
            raise self._collision(path, origin)

    def _container(self, parent, path, next_key, origin):
        kind = list if isinstance(next_key, int) else dict
        child = _get(parent, path[-1])
        if child is _MISSING:
            child = kind()
            _put(parent, path[-1], child)
        elif not isinstance(child, kind):
            raise self._collision(path, origin)
        return child

    def _collision(self, path, origin):
        existing = self._origin(path)
        if existing == path:
            existing = next((old for new, old in self.origins.items() if new[: len(path)] == path), path)
        return _Stop(
            f"{_dotted(path)} is set by both {_dotted(existing)} and {_dotted(origin)} "
            "with different values; keep one, then run cvs config migrate again"
        )

    def _origin(self, path):
        for end in range(len(path), 0, -1):
            if path[:end] in self.origins:
                return self.origins[path[:end]] + path[end:]
        return path

    def _is_empty_default(self, node, path):
        if not isinstance(node, (dict, list)) or node:
            return False
        field = _field_at(self.schema, path)
        if field is None or field.is_required():
            return False
        default = field.get_default(call_default_factory=True)
        return type(default) is type(node) and default == node

    def _drop_if_only_comments(self, path):
        parent = _lookup(self.tree, path[:-1])
        if isinstance(parent, dict) and isinstance(parent.get(path[-1]), dict):
            if all(_is_comment(key) for key in parent[path[-1]]):
                del parent[path[-1]]

    def set_engine(self, suite):
        if _field_at(self.schema, (suite,)) is None or suite in self.tree:
            return
        self.tree[suite] = {}

    def fill_threshold_json(self, input_path):
        if _field_at(self.schema, ("threshold_json",)) is None:
            return
        name = self.tree.get("threshold_json", "")
        if not isinstance(name, str) or name.strip():
            return
        siblings = sorted(input_path.parent.glob("*threshold.json"))
        if len(siblings) == 1:
            self.tree["threshold_json"] = siblings[0].name

    def settle(self):
        """Validate the tree as the loader would, with placeholders allowed, and return each field to fix.

        A refused key moves to ``_unmapped`` and a missing field is written as ``<changeme>``.
        """
        data, threshold_problems = self._validation_input()
        context = None if self.config_path is None else {"config_path": self.config_path}
        try:
            TypeAdapter(self.schema).validate_python(data, context=context)
            errors = []
        except ValidationError as exc:
            errors = exc.errors()
        problems = []
        for error in errors:
            loc = tuple(error["loc"])
            parent = _lookup(self.tree, loc[:-1])
            if error["type"] == "extra_forbidden" and isinstance(parent, dict) and loc[-1] in parent:
                self._unmap(_dotted(self._origin(loc)), parent.pop(loc[-1]))
            elif error["type"] == "missing" and isinstance(parent, dict):
                parent[loc[-1]] = CHANGEME
                problems.append(f"{_dotted(loc)}: {MISSING_MESSAGE}")
            elif not has_placeholder(error.get("input")):
                problems.append(f"{_dotted(loc) or '(config)'}: {error['msg'].removeprefix('Value error, ')}")
        unmapped = [f"{UNMAPPED}.{path}: {UNMAPPED_FIELD_MESSAGE}" for path in self.unmapped]
        return unmapped + problems + threshold_problems

    def _validation_input(self):
        """The tree as ``load_suite_config`` gives it to the schema, and the threshold file's problems."""
        if self.config_path is None or not issubclass(self.schema, Thresholds):
            return self.tree, []
        name = self.tree.get("threshold_json", "")
        problems = []
        if not has_placeholder(name):
            try:
                return _with_thresholds(self.schema, self.tree, self._threshold_lookup(name), _QUIET_LOG), []
            except ConfigError as exc:
                missing = [f"threshold_json: {MISSING_THRESHOLD_MESSAGE}"]
                problems = exc.problems if exc.problems == missing else [f"{name}: {line}" for line in exc.problems]
        # A placeholder or refused threshold file can't be read, so the rest is checked as a config that records only.
        record_only = {**self.tree, "threshold_json": ""}
        return _with_thresholds(self.schema, record_only, self.config_path, _QUIET_LOG), problems

    def _threshold_lookup(self, name):
        """The path whose folder the named threshold file is looked up from: the output's when it is written there."""
        if self.output_path is not None and isinstance(name, str) and name.strip():
            if (self.output_path.parent / name).is_file() and not Path(name).is_absolute():
                return self.output_path
        return self.config_path

    def output(self):
        return {UNMAPPED: self.unmapped, **self.tree} if self.unmapped else self.tree


def _rename_kinds(raw, kinds):
    """Return the threshold file with old kinds renamed, the paths of old specs, and what goes to ``_unmapped``."""
    old, unmapped = [], {}

    def convert(node, path):
        if path and isinstance(node, dict) and node.get("kind") in OLD_THRESHOLD_KINDS:
            old.append(path)
            if node["kind"] in kinds:
                return {**node, "kind": kinds[node["kind"]]}
            unmapped[_dotted(path)] = node
            return _MISSING
        if not isinstance(node, dict):
            return node
        converted = {}
        for key, value in node.items():
            if _is_comment(key):
                converted[key] = value
            elif not path and not isinstance(value, dict):
                unmapped[key] = value
            else:
                value = convert(value, path + (key,))
                if value is not _MISSING:
                    converted[key] = value
        return converted

    return convert(raw, ()), old, unmapped


def _nest_accuracy_sections(tree):
    """``tree`` with each ``accuracy`` section keyed by task id moved under ``lm_eval``, and the moved paths."""
    accuracy = tree.get("accuracy")
    if not isinstance(accuracy, dict):
        return tree, []
    old = [key for key, value in accuracy.items() if not _is_comment(key) and key not in ACCURACY_SECTIONS]
    if not old:
        return tree, []
    lm_eval = accuracy.get("lm_eval", {})
    if not isinstance(lm_eval, dict):
        raise _Stop("accuracy.lm_eval is not an object; fix it before converting")
    clashing = sorted(key for key in old if key in lm_eval)
    if clashing:
        raise _Stop(f"accuracy.lm_eval and accuracy hold the same task id: {', '.join(clashing)}")
    nested = {}
    for key, value in accuracy.items():
        if key in old or key == "lm_eval":
            sections = nested.setdefault("lm_eval", dict(lm_eval))
            if key in old:
                sections[key] = value
        else:
            nested[key] = value
    return {**tree, "accuracy": nested}, [("accuracy", key) for key in old]


def flags_from_list(flags):
    """A flag list such as ``["--attention-backend aiter", "--kv-cache-dtype", "fp8"]`` as a snake_case flag map.

    Each item is split as a shell would split it. ``--name`` starts a flag and
    the words after it, up to the next ``--name``, are its value: none makes it
    True, one a string, more a list. ``--name=value`` works too. A flag given
    twice raises ValueError naming it; a word before the first flag raises
    ValueError without the word, which may be a secret.
    """
    converted, name = {}, None
    for item in flags:
        for word in shlex.split(str(item)):
            if word.startswith("--") and len(word) > 2:
                flag, has_value, value = word[2:].partition("=")
                name = flag.replace("-", "_")
                if name in converted:
                    raise ValueError(f"--{flag} is given more than once")
                converted[name] = [value] if has_value else []
            elif name is None:
                raise ValueError("a value comes before any --flag")
            else:
                converted[name].append(word)
    return {key: True if not words else words[0] if len(words) == 1 else words for key, words in converted.items()}


def _is_threshold_file(raw):
    sections = _entries(raw)
    accuracy = raw.get("accuracy")
    for key, value in accuracy.items() if isinstance(accuracy, dict) else ():
        if _is_comment(key) or not isinstance(value, dict):
            continue
        sections += _entries(value) if key in ("lm_eval", "custom_eval") else [value]
    return any(
        isinstance(spec, dict) and "kind" in spec
        for section in sections
        if isinstance(section, dict)
        for spec in _entries(section)
    )


def _entries(node):
    return [value for key, value in node.items() if not _is_comment(key)]


def _first_match(patterns, path):
    for pattern in patterns:
        captures = _captures(pattern, path)
        if captures is not None:
            return pattern, captures
    return None


def _captures(pattern, path):
    """The positions in ``path`` that the ``*`` parts of ``pattern`` match, or None when it does not match."""
    parts = pattern.split(".")
    if len(parts) != len(path):
        return None
    captures = []
    for position, (part, key) in enumerate(zip(parts, path)):
        if part == "*" and not _is_comment(key):
            captures.append(position)
        elif part != key:
            return None
    return captures


def _field_at(model, path):
    field = None
    for key in path:
        if not _is_model(model) or not isinstance(key, str):
            return None
        field = model.model_fields.get(key)
        if field is None:
            return None
        model = _model_of(field.annotation)
    return field


def _model_of(annotation):
    if _is_model(annotation):
        return annotation
    args = getattr(annotation, "__args__", ())
    models = [arg for arg in args if _is_model(arg)]
    return models[0] if type(None) in args and len(models) == 1 else None


def _is_model(value):
    return BaseModel in getattr(value, "__mro__", ())


def _is_comment(key):
    return isinstance(key, str) and key.startswith("_")


def _dotted(path):
    return ".".join(str(key) for key in path)


def _children(node):
    if isinstance(node, dict):
        return list(node.items())
    if isinstance(node, list):
        return list(enumerate(node))
    return []


def _lookup(node, path):
    for key in path:
        children = dict(_children(node))
        if key not in children:
            return _MISSING
        node = children[key]
    return node


def _get(parent, key):
    if isinstance(parent, dict):
        return parent.get(key, _MISSING)
    return parent[key] if key < len(parent) else _MISSING


def _put(parent, key, value):
    if isinstance(parent, dict):
        parent[key] = value
        return
    parent.extend({} for _ in range(key - len(parent)))
    parent.append(value)


def _merge(existing, value):
    if isinstance(existing, dict) and isinstance(value, dict):
        for key, item in value.items():
            if key not in existing:
                existing[key] = copy.deepcopy(item)
            elif not _merge(existing[key], item):
                return False
        return True
    return _same(existing, value)


def _same(first, second):
    return json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
