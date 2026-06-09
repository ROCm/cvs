'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Typed config loader for the dtni vllm_single PoC.

Loads a per-variant `config.json` + sibling `threshold.json`, validates
shape with pydantic v2 (extra="forbid"), and runs a 3-pass placeholder
substitution:
  1. cluster placeholders (`{user-id}`) anywhere
  2. self-reference within `paths` (e.g. `{shared_fs}`)
  3. cross-block (`{paths.models_dir}`, etc.) into the rest of the doc

A loaded variant is returned as a `VariantConfig` instance whose `container`
field matches the dict shape that `cvs.core.orchestrators.factory.OrchestratorConfig`
already understands (`enabled`, `launch`, `name`, `image`, `runtime{name,args}`),
plus a `thresholds` field carrying the parsed threshold.json contents.

`model.remote=1` raises NotImplementedError -- schema is present, but the
download/resolve logic lives in cvs-dtni-v1's `resource_resolver.py` and
is out of scope for this PoC.
'''

from __future__ import annotations

import getpass
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from typing_extensions import Literal


# ---------- pydantic models ----------

class _Forbid(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Allow(BaseModel):
    model_config = ConfigDict(extra="allow")


class Paths(_Forbid):
    shared_fs: str
    models_dir: str
    log_dir: str
    benchmark_scripts_dir: str
    hf_token_file: str


class ModelSpec(_Forbid):
    id: str
    remote: Literal[0, 1]
    precision: str


class ImageSpec(_Forbid):
    tag: str
    remote: Literal[0, 1]


class RuntimeSpec(_Allow):
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class ContainerSpec(_Forbid):
    enabled: bool
    launch: bool
    name: str
    image: str
    runtime: RuntimeSpec


class RoleServer(_Forbid):
    server_script: str


class Roles(_Forbid):
    server: RoleServer


class SeqCombo(_Forbid):
    isl: str
    osl: str
    name: str


class Sweep(_Forbid):
    sequence_combinations: List[SeqCombo]
    concurrency_levels: List[int]


class Params(_Forbid):
    backend: str = "vllm"
    base_url: str = "http://0.0.0.0"
    port_no: str = "8888"
    dataset_name: str = "random"
    burstiness: str = "1.0"
    seed: str = "0"
    request_rate: str = "inf"
    max_model_length: str = "9216"
    random_range_ratio: str = "0.8"
    random_prefix_len: str = "0"
    tensor_parallelism: str = "1"
    tokenizer_mode: str = "auto"
    percentile_metrics: str = "ttft,tpot,itl,e2el"
    metric_percentiles: str = "99"
    bench_serv_script: str = "benchmark_serving.py"
    num_prompts: str = "3200"


class BenchmarkParams(_Allow):
    pass


class VariantConfig(_Forbid):
    schema_version: Literal[1]
    framework: Literal["vllm_single"]
    gpu_arch: str
    paths: Paths
    model: ModelSpec
    image: ImageSpec
    container: ContainerSpec
    roles: Roles
    params: Params
    benchmark_params: Dict[str, Any] = Field(default_factory=dict)
    sweep: Sweep
    thresholds: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

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
    user = cluster_dict.get("username") or getpass.getuser()
    if user == "{user-id}":
        user = getpass.getuser()
    return {"user-id": user}


# ---------- public API ----------

def load_variant(config_path, cluster_dict):
    """Load and validate a single variant config + its sibling threshold.json."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise AssertionError(f"variant config not found: {config_path}")

    raw = json.loads(config_path.read_text())
    threshold_path = config_path.parent / "threshold.json"
    if not threshold_path.is_file():
        raise AssertionError(f"threshold.json missing next to config: {threshold_path}")
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

    # Validate sibling thresholds, attach, build VariantConfig.
    raw["thresholds"] = thresholds
    return VariantConfig(**raw)


def enumerate_variants(root_dir):
    """Walk cvs/input/dtni/vllm_single/*/config.json and return sorted list of config paths."""
    root = Path(root_dir)
    if not root.is_dir():
        return []
    return sorted(p / "config.json" for p in root.iterdir() if p.is_dir() and (p / "config.json").is_file())
