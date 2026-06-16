'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Typed config loader for the dtni vllm_single PoC.

Loads a per-variant `config.json` + sibling `*_threshold.json`, validates
shape with pydantic v2 (extra="forbid"), and runs a 3-pass placeholder
substitution:
  1. cluster placeholders (`{user-id}`) anywhere
  2. self-reference within `paths` (e.g. `{shared_fs}`)
  3. cross-block (`{paths.models_dir}`, etc.) into the rest of the doc

A loaded variant is returned as a `VariantConfig` instance whose `container`
field (`lifetime`, `name`, `image`, `runtime`) matches the dict shape that
`cvs.core.orchestrators.factory.OrchestratorConfig` already understands.
`runtime` is a nested `RuntimeSpec`, not a flat dict; `container.model_dump()`
serialises it to the `runtime: {name, args}` shape the factory consumes (the
vllm conftest does exactly this before building the orchestrator). The result
also carries a `thresholds` field with the parsed threshold contents.

`model.remote=1` raises NotImplementedError -- schema is present, but the
download/resolve logic lives in cvs-dtni-v1's `resource_resolver.py` and
is out of scope for this PoC.

GENERALIZATION SEAM (do this on the SECOND consumer, not speculatively):
`Paths`/`ModelSpec`/`ImageSpec`/`ContainerSpec`/`thresholds`, the placeholder
substitution, the `enforce_thresholds` gate, and `load_variant` are
framework-agnostic. `Params`, `Sweep`, `Roles`, and `cell_key`/`expected_cells`
(the ISL/OSL/TP/CONC key shape) are vllm_single-specific. When the next
framework lands (sglang-disagg, distributed, or a training suite), split this
into a `BaseVariantConfig` (the generic half) + per-framework subclasses
carrying their own `Params`/`Sweep`/`cell_key`. Extracting now -- with one
consumer -- would be guessing the seam.
'''

from __future__ import annotations

import getpass
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, model_validator
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
    lifetime: Literal["no_launch", "per_run", "persistent"] = "per_run"
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
    # Completion-poll budget for the bench client = client_poll_count * 60s
    # (plus a 120s initial wait). Large-output cells (high osl) need a bigger
    # budget; the poll loop exits as soon as the client finishes, so raising
    # this never slows down fast cells. See regressions REG-20260609-001.
    client_poll_count: str = "20"


class VariantConfig(_Forbid):
    schema_version: Literal[1]
    framework: Literal["vllm_single"]
    gpu_arch: str
    # When false, the threshold-coverage gate warns instead of raising and the
    # test records metrics without asserting pass/fail (record-only). Use for
    # un-calibrated shapes (e.g. a throughput characterization whose published
    # numbers are curves, not tabulated values). Default true keeps the gate
    # strict for calibrated configs -- no regression to the remediation work.
    enforce_thresholds: bool = True
    paths: Paths
    model: ModelSpec
    image: ImageSpec
    container: ContainerSpec
    roles: Roles
    params: Params
    sweep: Sweep
    thresholds: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # pydantic runs @model_validator(mode="after") hooks in definition order.
    # This remote check is intentionally first: an unimplemented remote config
    # fails fast (NotImplementedError) before _check_thresholds_cover_sweep runs,
    # which is meaningless for a config we are going to reject anyway.
    @model_validator(mode="after")
    def _check_remote_not_implemented(self):
        if self.model.remote == 1:
            raise NotImplementedError(
                "model.remote=1 (remote model download) is not implemented in the PoC. "
                "Port from cvs-dtni-v1/resource_resolver.py before enabling."
            )
        return self

    def cell_key(self, isl, osl, concurrency):
        """The canonical threshold key for one sweep cell.

        Single source of truth shared by the loader's coverage check and the
        test's verdict lookup -- so the two can never drift on whitespace,
        ordering, or field names.
        """
        return f"ISL={isl},OSL={osl},TP={self.params.tensor_parallelism},CONC={concurrency}"

    def expected_cells(self):
        """Every (isl, osl, conc) cell the sweep will parametrize."""
        return [
            self.cell_key(combo.isl, combo.osl, conc)
            for combo in self.sweep.sequence_combinations
            for conc in self.sweep.concurrency_levels
        ]

    @model_validator(mode="after")
    def _check_thresholds_cover_sweep(self):
        """Fail at load time if any sweep cell lacks a threshold entry.

        Without this, a mistyped/whitespaced threshold key (or a new sweep
        entry without a matching threshold) makes the test silently skip its
        verdict and report a green PASS with zero assertions.

        When `enforce_thresholds` is false the same mismatch is reported as a
        warning rather than an error -- the config loads as a record-only
        scaffold (metrics captured, no assertions) instead of failing.
        """
        expected = set(self.expected_cells())
        present = set(self.thresholds.keys())
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        problems = []
        if missing:
            problems.append(f"sweep cells with no threshold entry: {missing}")
        if extra:
            problems.append(f"threshold keys matching no sweep cell (typo?): {extra}")
        if problems:
            msg = "threshold.json does not match the sweep matrix; " + "; ".join(problems)
            if self.enforce_thresholds:
                raise ValueError(msg)
            warnings.warn(f"{msg} (enforce_thresholds=false -> record-only)", stacklevel=2)
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


# ---------- public API ----------


def load_variant(config_path, cluster_dict):
    """Load and validate a single variant config + its sibling threshold file.

    The threshold file is the sole `*threshold.json` next to the config (e.g.
    `w1_..._threshold.json` beside `w1_..._config.json`), so config and
    threshold can share a descriptive per-variant prefix.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"variant config not found: {config_path}")

    raw = json.loads(config_path.read_text())

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

    # Drop threshold-file comment keys (e.g. "_comment") before coverage check.
    thresholds = {k: v for k, v in thresholds.items() if not k.startswith("_")}

    # Validate sibling thresholds, attach, build VariantConfig.
    raw["thresholds"] = thresholds
    return VariantConfig(**raw)
