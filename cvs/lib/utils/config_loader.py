'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Framework-agnostic config machinery shared by every CVS suite.

Holds `load_config`, which turns a config file and a cluster file into a
validated config, the resolved cluster, and the merged container block; the
3-pass placeholder substitution; the `enforce_thresholds` gate carried on
`BaseVariantConfig`; and the `substitute_config` helper that reads a variant
`config.json` + sibling `*threshold.json` and resolves placeholders. The
schema blocks live in `cvs.lib.utils.config_blocks` and are re-exported here.
A per-framework module subclasses `BaseVariantConfig` or composes the blocks
directly and adds its own params, sweep and `cell_key`.

3-pass placeholder substitution:
  1. cluster placeholders (`{user-id}`) anywhere
  2. self-reference within `paths` (e.g. `{shared_fs}`)
  3. cross-block (`{paths.models_dir}`, etc.) into the rest of the doc

A loaded variant's `container` field (`lifetime`, `name`, `image`, `runtime`)
matches the dict shape that `cvs.core.orchestrators.factory.OrchestratorConfig`
already understands. `runtime` is a nested `RuntimeSpec`, not a flat dict;
`container.model_dump()` serialises it to the `runtime: {name, args}` shape the
factory consumes; `load_config` merges that dump onto the cluster file's
container block. The loaded variant also carries a `thresholds` field with the
parsed threshold contents.

`model.remote=1` raises NotImplementedError for suites other than xDiT.
xDiT uses it as an opt-in for runtime Hugging Face snapshot download.

'''

from __future__ import annotations

import copy
import getpass
import json
import re
import warnings
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict

from pydantic import Field, ValidationError, model_validator
from typing_extensions import Literal

from cvs.lib.utils.config_blocks import (  # noqa: F401 - re-exported for suite loaders
    ContainerSpec,
    ModelSpec,
    Paths,
    RuntimeSpec,
    _Allow,
    _Forbid,
)
from cvs.lib.utils_lib import resolve_cluster_config_placeholders


# ---------- pydantic models (generic) ----------


class BaseVariantConfig(_Forbid):
    """The framework-agnostic skeleton of a variant config.

    Carries the fields every suite shares (schema/paths/model/image/container/
    thresholds + the enforce gate) and the remote-not-implemented guard.
    Per-framework subclasses add their own `framework`/`Params`/`Sweep` and the
    `cell_key`/coverage-check pair that depend on them.
    """

    schema_version: Literal[1]
    # When false, the threshold-coverage gate warns instead of raising and the
    # test records metrics without asserting pass/fail (record-only). Use for
    # un-calibrated shapes (e.g. a throughput characterization whose published
    # numbers are curves, not tabulated values). Default true keeps the gate
    # strict for calibrated configs -- no regression to the remediation work.
    enforce_thresholds: bool = True
    threshold_json: str = ""
    paths: Paths
    model: ModelSpec
    # The container image is declared once, on container.image (ContainerSpec).
    # There is no separate top-level image block.
    container: ContainerSpec
    thresholds: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # pydantic runs @model_validator(mode="after") hooks in definition order,
    # parent-class hooks before subclass hooks. This remote check is intentionally
    # the first to run: an unimplemented remote config fails fast
    # (NotImplementedError) before any subclass's threshold-coverage check runs,
    # which is meaningless for a config we are going to reject anyway.
    @model_validator(mode="after")
    def _check_remote_not_implemented(self):
        if self.model.remote == 1 and getattr(self, "framework", None) != "xdit":
            raise NotImplementedError(
                "model.remote=1 (remote model download) is not implemented in the PoC. "
                "Port from cvs-dtni-v1/resource_resolver.py before enabling."
            )
        return self


# ---------- placeholder substitution ----------

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_.\-]+)\}")


def _walk_substitute(node, mapping):
    if isinstance(node, str):

        def repl(m):
            key = m.group(1)
            if key in mapping:
                return str(mapping[key])
            return m.group(0)

        return _PLACEHOLDER_RE.sub(repl, node)
    if isinstance(node, list):
        return [_walk_substitute(x, mapping) for x in node]
    if isinstance(node, dict):
        return {k: _walk_substitute(v, mapping) for k, v in node.items()}
    return node


def _flatten_paths(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_paths(v, key))
        elif isinstance(v, (str, int, float)):
            out[key] = str(v)
    return out


def _resolve_cluster_mapping(cluster_dict):
    raw = cluster_dict.get("username") or "{user-id}"
    user = getpass.getuser() if raw == "{user-id}" else raw
    return {"user-id": user}


# ---------- public API (generic) ----------


def _find_packaged_threshold(name):
    """Locate a threshold file by name in the packaged ``input/config_file`` tree.

    Fallback for when a config is run from a copied location that no longer has
    its sibling threshold file: the shipped threshold still lives under the cvs
    package's ``input/config_file/**`` (config_loader.py is at
    ``cvs/lib/utils/``, so ``parents[2]`` is the package root). Returns the path,
    or ``None`` when there is no packaged match.
    """
    try:
        pkg_config_root = Path(__file__).resolve().parents[2] / "input" / "config_file"
    except IndexError:
        return None
    if not pkg_config_root.is_dir():
        return None
    matches = sorted(pkg_config_root.rglob(name))
    return matches[0] if matches else None


def substitute_config(config_path, cluster_dict):
    """Read a variant config + sibling threshold file and resolve placeholders.

    Returns `(raw_dict, thresholds)`: the substituted config dict (NOT yet
    validated -- the caller's per-framework `VariantConfig(**raw)` does that)
    and the parsed, comment-stripped threshold dict.

    Threshold discovery supports both layouts:
    - ``threshold_json`` in the config (literal path; vllm_single style), or
    - a sole ``*threshold.json`` sibling next to the config (atom style).

    This is the framework-neutral body of the old `load_variant`: file read +
    3-pass substitution + threshold read. Per-framework loaders call it, attach
    ``thresholds``, then build their typed config.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"variant config not found: {config_path}")

    raw = json.loads(config_path.read_text())

    threshold_json = (raw.get("threshold_json") or "").strip()
    if "<changeme>" in threshold_json.lower():
        raise ValueError(
            f"threshold_json is still a placeholder ('{threshold_json}'). Set it to the correct "
            "threshold file for your setup (e.g. the GPU-specific *_threshold.json) before running."
        )
    if threshold_json:
        threshold_path = Path(threshold_json)
        if not threshold_path.is_absolute():
            threshold_path = (config_path.parent / threshold_path).resolve()
        if not threshold_path.is_file():
            # The config may be run from a copied location that lacks its sibling
            # threshold file; fall back to the shipped copy under the packaged
            # input/config_file tree (matched by filename).
            packaged = _find_packaged_threshold(threshold_path.name)
            if packaged is None:
                raise FileNotFoundError(
                    f"threshold_json not found: {threshold_path} "
                    f"(and no packaged fallback named '{threshold_path.name}')"
                )
            warnings.warn(
                f"threshold_json '{threshold_path.name}' not found next to the config "
                f"({config_path.parent}); using the packaged copy at {packaged}",
                stacklevel=2,
            )
            threshold_path = packaged
    else:
        threshold_candidates = sorted(config_path.parent.glob("*threshold.json"))
        if not threshold_candidates:
            raise FileNotFoundError(f"no *threshold.json next to config: {config_path.parent}")
        if len(threshold_candidates) > 1:
            raise ValueError(f"multiple *threshold.json files next to config (ambiguous): {threshold_candidates}")
        threshold_path = threshold_candidates[0]
    thresholds = json.loads(threshold_path.read_text())

    # Pass 1: cluster placeholders ({user-id}) everywhere.
    cluster_map = _resolve_cluster_mapping(cluster_dict)
    raw = _walk_substitute(raw, cluster_map)

    # Pass 2: self-reference within paths ({shared_fs} inside paths.*).
    paths_block = raw.get("paths", {})
    if isinstance(paths_block, dict):
        for _ in range(len(paths_block) + 1):
            new = {
                k: _walk_substitute(v, {pk: pv for pk, pv in paths_block.items() if isinstance(pv, str)})
                for k, v in paths_block.items()
            }
            if new == paths_block:
                break
            paths_block = new
        raw["paths"] = paths_block

    # Pass 3: cross-block ({paths.models_dir} -> anywhere else).
    flat_map = _flatten_paths({"paths": raw.get("paths", {})})
    raw = _walk_substitute(raw, flat_map)

    # Drop comment keys (e.g. "_comment") before framework/threshold validation.
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    thresholds = {k: v for k, v in thresholds.items() if not k.startswith("_")}

    return raw, thresholds


class ConfigError(ValueError):
    """A config file that does not match its suite's schema.

    The message has one ``location: problem`` line per error and never the
    offending values, which can be credentials.
    """

    def __init__(self, config_path, problems):
        self.problems = problems
        super().__init__(f"{config_path} is not a valid config:\n" + "\n".join(f"  {p}" for p in problems))


LoadedConfig = namedtuple("LoadedConfig", "config cluster container")


def _deep_merge(base, override):
    """Merge `override` onto `base`: dicts merge by key, lists and scalars are replaced."""
    if not (isinstance(base, dict) and isinstance(override, dict)):
        return override
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(base[k], v) if k in base else v
    return out


def _problem_lines(exc):
    lines = []
    for error in exc.errors(include_input=False):
        location = ".".join(str(part) for part in error["loc"]) or "(config)"
        lines.append(f"{location}: {error['msg'].removeprefix('Value error, ')}")
    return lines


def load_config(model_cls, config_path, cluster):
    """Load and validate a suite config against its cluster.

    `cluster` is a cluster file path or an already-parsed cluster dict. A
    `<changeme>` left in the cluster file exits, as for every suite; one left in
    the config raises `ConfigError` listing each field that still has one.

    Returns `LoadedConfig(config, cluster, container)`. `container` is the
    cluster file's container block with the config's `container.model_dump()`
    merged onto it, the config winning on shared keys, or None when the model
    has no container. Cluster list values such as volumes are replaced here;
    the container orchestrator adds its own defaults to them. The merged
    container is a copy, so changing it does not change the cluster.
    """
    if not isinstance(cluster, dict):
        with open(cluster) as fp:
            cluster = json.load(fp)
    cluster = resolve_cluster_config_placeholders(cluster)
    raw, thresholds = substitute_config(config_path, cluster)
    if "thresholds" in model_cls.model_fields:
        raw["thresholds"] = thresholds
    try:
        config = model_cls.model_validate(raw, context={"placeholders": "reject"})
    except ValidationError as exc:
        raise ConfigError(config_path, _problem_lines(exc)) from None
    spec = getattr(config, "container", None)
    if spec is None:
        container = None
    else:
        container = copy.deepcopy(_deep_merge(cluster.get("container") or {}, spec.model_dump()))
    return LoadedConfig(config, cluster, container)
