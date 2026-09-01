'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Framework-agnostic config machinery shared by every CVS suite.

Holds placeholder substitution, threshold file discovery, and
``substitute_config`` — the I/O half of variant loading. Pydantic models
(``Paths``, ``BaseVariantConfig``, etc.) live in ``cvs.schema.common.base``.

3-pass placeholder substitution:
  1. cluster placeholders (`{user-id}`) anywhere
  2. self-reference within `paths` (e.g. `{shared_fs}`)
  3. cross-block (`{paths.models_dir}`, etc.) into the rest of the doc

A loaded variant's `container` field (`lifetime`, `name`, `image`, `runtime`)
matches the dict shape that `cvs.core.orchestrators.factory.OrchestratorConfig`
already understands. `runtime` is a nested `RuntimeSpec`, not a flat dict;
`container.model_dump()` serialises it to the `runtime: {name, args}` shape the
factory consumes (the vllm conftest does exactly this before building the
orchestrator). The loaded variant also carries a `thresholds` field with the
parsed threshold contents.

`model.remote=1` raises NotImplementedError -- schema is present, but the
download/resolve logic lives in cvs-dtni-v1's `resource_resolver.py` and is
out of scope for this PoC.
'''

from __future__ import annotations

import getpass
import json
import re
import warnings
from pathlib import Path

from cvs.schema.base import _Allow, _Forbid
from cvs.schema.common.base import (
    BaseVariantConfig,
    ContainerSpec,
    ModelSpec,
    Paths,
    RuntimeSpec,
)

__all__ = [
    "BaseVariantConfig",
    "ContainerSpec",
    "ModelSpec",
    "Paths",
    "RuntimeSpec",
    "_Allow",
    "_Forbid",
    "substitute_config",
]


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
    - ``threshold_json`` in the config (literal path), or
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
