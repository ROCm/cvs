'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Framework-agnostic config machinery shared by every CVS suite.

Holds the generic half of what used to be one monolithic loader: the
container/paths/model/image schema, the 3-pass placeholder substitution, the
`enforce_thresholds` gate carried on `BaseVariantConfig`, and the
`substitute_config` helper that reads a variant `config.json` + sibling
`*threshold.json` and resolves placeholders. A per-framework module subclasses
`BaseVariantConfig` and adds its own `Params`/`Sweep`/`cell_key` (see
`cvs.lib.inference.utils.inferencing_config_loader` for the vllm flavour).

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
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Literal


# ---------- pydantic models (generic) ----------


class _Forbid(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Allow(BaseModel):
    model_config = ConfigDict(extra="allow")


class Paths(_Forbid):
    shared_fs: str
    models_dir: str
    log_dir: str
    hf_token_file: str


class ModelSpec(_Forbid):
    id: str
    remote: Literal[0, 1]
    precision: str = ""


class RuntimeSpec(_Allow):
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class ContainerSpec(_Forbid):
    lifetime: Literal["no_launch", "per_run", "persistent"] = "per_run"
    name: str
    image: str
    runtime: RuntimeSpec


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
        if self.model.remote == 1:
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


def substitute_config(config_path, cluster_dict):
    """Read a variant config + sibling threshold file and resolve placeholders.

    Returns `(raw_dict, thresholds)`: the substituted config dict (NOT yet
    validated -- the caller's per-framework `VariantConfig(**raw)` does that)
    and the parsed, comment-stripped threshold dict.

    Threshold discovery supports both layouts:
    - ``threshold_json`` in the config (literal path; vllm_single style), or
    - a sole ``*threshold.json`` sibling next to the config (inferencex_atom style).

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
        if not threshold_path.is_file():
            raise FileNotFoundError(f"threshold_json not found: {threshold_path}")
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
