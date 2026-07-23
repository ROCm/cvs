'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

SGLang single-node config loader for ContainerOrchestrator suites.

Supports two on-disk layouts:

1. **Legacy** (existing ``mi30x_sglang_*.json``):
   top-level ``config`` + ``benchmark_params`` + per-variant ``threshold_file``.

2. **Unified** (vLLM-style, optional future configs):
   ``schema_version: 1``, ``framework: "sglang_single"``, ``paths`` / ``container`` /
   ``model`` / ``threshold_json``.

``load_variant()`` is the single entry point for ``sglang_single`` conftest and
produces both:
- typed fields for ``OrchestratorFactory`` (``container``, ``paths``, ``model``)
- legacy dicts (``inference``, ``benchmark_params``) for ``SglangSingle``
'''

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from pydantic import Field, model_validator
from typing_extensions import Literal

from cvs.lib import globals
from cvs.lib.utils.config_loader import (
    BaseVariantConfig,
    ContainerSpec,
    RuntimeSpec,
    _Forbid,
    substitute_config,
)
from cvs.lib.utils_lib import resolve_test_config_placeholders

log = globals.log

_LEGACY_FRAMEWORK = "sglang_single"
_UNIFIED_FRAMEWORK = "sglang_single"

_PERF_CELL_RE = re.compile(
    r"^ISL=(?P<isl>\d+),OSL=(?P<osl>\d+),TP=(?P<tp>\d+),PP=(?P<pp>\d+),CONC=(?P<conc>\d+)$"
)


# ---------- threshold / variant helpers (moved out of conftest) ----------


def resolve_benchmark_variant_key(root: Mapping[str, Any], config_path: str) -> str:
    """Pick which ``benchmark_params`` entry to run."""
    env_key = (os.environ.get("SGLANG_BENCHMARK_KEY") or "").strip()
    bp = root.get("benchmark_params") or {}
    if not isinstance(bp, dict) or not bp:
        raise ValueError(f"benchmark_params missing or empty in {config_path!r}")

    if env_key:
        if env_key not in bp:
            raise ValueError(
                f"SGLANG_BENCHMARK_KEY={env_key!r} not found in benchmark_params "
                f"({config_path}); valid: {sorted(bp)!r}"
            )
        log.info("Using benchmark variant from env SGLANG_BENCHMARK_KEY=%r", env_key)
        return env_key

    explicit = root.get("active_benchmark")
    if explicit is not None:
        if explicit not in bp:
            raise ValueError(
                f"active_benchmark={explicit!r} not found in benchmark_params "
                f"({config_path}); valid: {sorted(bp)!r}"
            )
        log.info("Using benchmark variant from active_benchmark=%r", explicit)
        return str(explicit)

    if len(bp) == 1:
        only = next(iter(bp))
        log.info("Single benchmark_params entry; using %r", only)
        return str(only)

    raise ValueError(
        f"Multiple benchmark_params keys in {config_path!r}: {sorted(bp)!r}. "
        'Set top-level "active_benchmark" to one of them, or export SGLANG_BENCHMARK_KEY.'
    )


def flat_expected_from_specs(specs: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric, spec in specs.items():
        if isinstance(spec, dict) and "value" in spec:
            out[metric] = float(spec["value"])
        else:
            out[metric] = float(spec)
    return out


def perf_cell_key(bp_dict: Mapping[str, Any]) -> str:
    bench = (bp_dict.get("inference_tests") or {}).get("bench_serv_random") or {}
    return (
        f"ISL={bench.get('input_length', '-')},"
        f"OSL={bench.get('output_length', '-')},"
        f"TP={bp_dict.get('tensor_parallelism', '8')},"
        f"PP={bp_dict.get('pipeline_parallelism', '1')},"
        f"CONC={bp_dict.get('max_concurrency', '-')}"
    )


def bench_cell_key(bench_name: str) -> str:
    return f"BENCH={bench_name}"


def perf_cells_from_thresholds(thresholds: Mapping[str, Any]) -> list[dict[str, Any]]:
    cells = []
    for cell_key, specs in thresholds.items():
        if str(cell_key).startswith("_") or str(cell_key).startswith("BENCH="):
            continue
        m = _PERF_CELL_RE.match(str(cell_key))
        if not m:
            continue
        cells.append({
            "cell_key": cell_key,
            "isl": m.group("isl"),
            "osl": m.group("osl"),
            "tp": m.group("tp"),
            "conc": m.group("conc"),
            "specs": specs,
        })
    cells.sort(key=lambda c: (int(c["isl"]), int(c["osl"]), int(c["conc"])))
    return cells


def _resolve_threshold_path(threshold_path: str, *, config_path: Path) -> Path:
    path = Path(threshold_path)
    if path.is_absolute():
        return path
    # Legacy configs often use repo-relative paths like cvs/input/...
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.is_file():
        return cwd_candidate
    return (config_path.parent / path).resolve()


def _load_thresholds_file(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fp:
        raw = json.load(fp)
    if not isinstance(raw, dict):
        raise ValueError(f"threshold file must be a JSON object: {path}")
    return {k: v for k, v in raw.items() if not str(k).startswith("_")}


def _threshold_file_path(bp_dict: Mapping[str, Any]) -> str | None:
    path = bp_dict.get("threshold_file")
    return str(path).strip() if path else None


def _inject_thresholds_into_bp_dict(bp_dict: dict[str, Any], thresholds: Mapping[str, Any]) -> None:
    inference_tests = bp_dict.setdefault("inference_tests", {})

    perf_key = perf_cell_key(bp_dict)
    perf_specs = thresholds.get(perf_key)
    if perf_specs:
        bench = inference_tests.setdefault("bench_serv_random", {})
        expected = bench.setdefault("expected_results", {})
        expected["auto"] = flat_expected_from_specs(perf_specs)
        log.info("Loaded performance thresholds from cell %r", perf_key)
    else:
        log.warning("No performance thresholds for cell %r in threshold file", perf_key)

    for bench_name in ("lm_eval_hellaswag", "lm_eval_gsm8k", "lm_eval_mmlu"):
        cell = bench_cell_key(bench_name)
        acc_specs = thresholds.get(cell)
        if not acc_specs:
            continue
        bench = inference_tests.setdefault(bench_name, {})
        expected = bench.setdefault("expected_results", {})
        task_key = bench_name.removeprefix("lm_eval_")
        expected[task_key] = flat_expected_from_specs(acc_specs)
        log.info("Loaded accuracy thresholds from cell %r", cell)


def load_perf_cells_for_collection(config_file: str) -> list[dict[str, Any]]:
    """Collection-time loader (no fixtures yet)."""
    variant = load_variant(config_file, cluster_dict={})
    cells = perf_cells_from_thresholds(variant.thresholds)
    if not cells:
        raise ValueError(f"No ISL=... performance cells in thresholds for {config_file!r}")
    return cells


# ---------- legacy → ContainerOrchestrator bridge ----------


def _volume_dict_to_mounts(volume_dict: Mapping[str, Any]) -> list[str]:
    mounts: list[str] = []
    for host, container in volume_dict.items():
        mounts.append(f"{host}:{container}")
    return mounts


def _infer_models_dir(inference: Mapping[str, Any]) -> str:
    volume_dict = ((inference.get("container_config") or {}).get("volume_dict") or {})
    for host, container in volume_dict.items():
        host_s, container_s = str(host), str(container)
        if "models" in host_s.lower() or "models" in container_s.lower():
            return host_s
    # Fallback: sibling of log_dir
    log_dir = str(inference.get("log_dir") or "").rstrip("/")
    if log_dir:
        return str(Path(log_dir).parent / "models")
    raise ValueError(
        "cannot infer models_dir from legacy config; add a models volume mount "
        "or migrate to unified paths.models_dir"
    )


def _infer_shared_fs(inference: Mapping[str, Any]) -> str:
    log_dir = str(inference.get("log_dir") or "").rstrip("/")
    if log_dir:
        return str(Path(log_dir).parent)
    token = str(inference.get("hf_token_file") or "")
    if token:
        return str(Path(token).parent.parent)
    raise ValueError("cannot infer shared_fs from legacy config")


def _legacy_server_env(inference: Mapping[str, Any], bp: Mapping[str, Any]) -> dict[str, str]:
    """NCCL / runtime env merged into container env for ContainerOrchestrator."""
    env: dict[str, str] = {}

    def _put(key: str, src_key: str) -> None:
        val = inference.get(src_key)
        if val is not None and str(val).strip():
            env[key] = str(val)

    _put("NCCL_DEBUG", "nccl_debug")
    _put("NCCL_IB_HCA", "nccl_ib_hca")
    _put("NCCL_IB_GID_INDEX", "nccl_ib_gid_index")
    _put("NCCL_SOCKET_IFNAME", "nccl_socket_ifname")
    _put("GLOO_SOCKET_IFNAME", "gloo_socket_ifname")
    _put("GLOO_TCP_IFNAME", "gloo_tcp_ifname")

    cc_env = ((inference.get("container_config") or {}).get("env_dict") or {})
    for k, v in cc_env.items():
        if v is not None:
            env[str(k)] = str(v)

    for entry in bp.get("add_export_env") or []:
        line = str(entry).strip()
        if not line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()

    return env


def legacy_container_block_from_inference(inference: Mapping[str, Any]) -> dict[str, Any]:
    """Build a ``ContainerSpec``-compatible dict from legacy ``config``."""
    cc = inference.get("container_config") or {}
    runtime_args: dict[str, Any] = {
        "network": "host",
        "ipc": "host",
        "privileged": True,
        "volumes": _volume_dict_to_mounts(cc.get("volume_dict") or {}),
        "devices": list(cc.get("device_list") or []),
    }
    shm = inference.get("shm_size")
    if shm:
        runtime_args["shm_size"] = str(shm)

    return {
        "lifetime": inference.get("container_lifetime", "per_run"),
        "name": inference["container_name"],
        "image": inference["container_image"],
        "runtime": {
            "name": "docker",
            "args": runtime_args,
        },
    }


def legacy_paths_from_inference(inference: Mapping[str, Any]) -> dict[str, str]:
    shared_fs = _infer_shared_fs(inference)
    return {
        "shared_fs": shared_fs,
        "models_dir": _infer_models_dir(inference),
        "log_dir": str(inference["log_dir"]),
        "hf_token_file": str(inference["hf_token_file"]),
    }


def _is_legacy_root(raw: Mapping[str, Any]) -> bool:
    return "benchmark_params" in raw and ("config" in raw or "container_image" in raw)


# ---------- typed config ----------


class SglangRoleServer(_Forbid):
    env: Dict[str, str] = Field(default_factory=dict)
    serve_port: str = "8000"


class SglangRoles(_Forbid):
    server: SglangRoleServer = Field(default_factory=SglangRoleServer)


class SglangSingleVariantConfig(BaseVariantConfig):
    """Typed config for ``sglang_single`` + ContainerOrchestrator."""

    framework: Literal["sglang_single"]
    gpu_arch: str
    variant_key: str = ""
    config_path: str = ""

    # Legacy blocks kept for ``SglangSingle`` until that lib is refactored.
    inference: Dict[str, Any] = Field(default_factory=dict)
    benchmark_params: Dict[str, Any] = Field(default_factory=dict)

    roles: SglangRoles = Field(default_factory=SglangRoles)

    def cell_key(self, isl, osl, concurrency) -> str:
        tp = self.benchmark_params.get("tensor_parallelism", "-")
        pp = self.benchmark_params.get("pipeline_parallelism", "-")
        return f"ISL={isl},OSL={osl},TP={tp},PP={pp},CONC={concurrency}"

    def perf_cell_key(self) -> str:
        return perf_cell_key(self.benchmark_params)

    @property
    def hf_token_file(self) -> str:
        return self.paths.hf_token_file

    @model_validator(mode="after")
    def _sync_legacy_inference_container_name(self):
        """Keep legacy inference dict aligned with orchestrator container name."""
        if self.inference and self.container.name:
            self.inference["container_name"] = self.container.name
            self.inference["container_image"] = self.container.image
        return self


# ---------- public API ----------


def orchestrator_container_from_variant(variant: SglangSingleVariantConfig) -> Dict[str, Any]:
    """``container`` block for ``OrchestratorConfig`` (includes server env)."""
    block = variant.container.model_dump()
    server_env = variant.roles.server.env
    if server_env:
        block = {**block, "env": dict(server_env)}
    return block


def _load_legacy_variant(config_path: str, cluster_dict: Mapping[str, Any]) -> SglangSingleVariantConfig:
    path = Path(config_path)
    with open(path, encoding="utf-8") as fp:
        root = json.load(fp)

    variant_key = resolve_benchmark_variant_key(root, config_path)
    cfg = root["config"] if isinstance(root.get("config"), dict) else root

    inference = resolve_test_config_placeholders(cfg, cluster_dict)
    bp_all = resolve_test_config_placeholders(root["benchmark_params"], cluster_dict)
    bp = dict(bp_all[variant_key])

    threshold_path_str = _threshold_file_path(bp)
    if not threshold_path_str:
        raise ValueError(
            f"benchmark_params[{variant_key!r}] missing 'threshold_file' in {config_path!r}"
        )

    threshold_path = _resolve_threshold_path(threshold_path_str, config_path=path)
    thresholds = _load_thresholds_file(threshold_path)
    log.info("Loaded thresholds from %s (%d cells)", threshold_path, len(thresholds))
    _inject_thresholds_into_bp_dict(bp, thresholds)

    container_raw = legacy_container_block_from_inference(inference)
    paths_raw = legacy_paths_from_inference(inference)
    server_env = _legacy_server_env(inference, bp)

    raw: dict[str, Any] = {
        "schema_version": 1,
        "framework": _LEGACY_FRAMEWORK,
        "gpu_arch": str(root.get("gpu_arch") or "mi30x"),
        "enforce_thresholds": bool(root.get("enforce_thresholds", True)),
        "threshold_json": str(threshold_path),
        "paths": paths_raw,
        "model": {
            "id": str(bp["model"]),
            "remote": int(root.get("model_remote", bp.get("model_remote", 0))),
        },
        "container": container_raw,
        "thresholds": thresholds,
        "variant_key": variant_key,
        "config_path": str(path.resolve()),
        "inference": dict(inference),
        "benchmark_params": bp,
        "roles": {
            "server": {
                "env": server_env,
                "serve_port": str(
                    inference.get("proxy_router_serv_port")
                    or inference.get("proxy_router_port")
                    or "8000"
                ),
            }
        },
    }
    return SglangSingleVariantConfig(**raw)


def _load_unified_variant(config_path: str, cluster_dict: Mapping[str, Any]) -> SglangSingleVariantConfig:
    raw, thresholds = substitute_config(config_path, cluster_dict)
    raw["thresholds"] = thresholds
    raw["config_path"] = str(Path(config_path).resolve())

    if not raw.get("variant_key"):
        if "benchmark_params" in raw:
            raw["variant_key"] = resolve_benchmark_variant_key(raw, config_path)
        else:
            raw["variant_key"] = raw.get("active_benchmark") or "default"

    # Optional embedded legacy blocks in unified configs.
    if "config" in raw and not raw.get("inference"):
        inference = resolve_test_config_placeholders(raw["config"], cluster_dict)
        raw["inference"] = dict(inference)
    if "benchmark_params" in raw and not raw.get("benchmark_params"):
        bp_all = resolve_test_config_placeholders(raw["benchmark_params"], cluster_dict)
        raw["benchmark_params"] = dict(bp_all[raw["variant_key"]])

    known = {k: v for k, v in raw.items() if k in SglangSingleVariantConfig.model_fields}
    return SglangSingleVariantConfig(**known)


def load_variant(config_path: str, cluster_dict: Mapping[str, Any]) -> SglangSingleVariantConfig:
    """Load and validate an ``sglang_single`` variant config + thresholds."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"variant config not found: {path}")

    with open(path, encoding="utf-8") as fp:
        peek = json.load(fp)

    if _is_legacy_root(peek):
        return _load_legacy_variant(config_path, cluster_dict)

    if peek.get("framework") not in (None, _UNIFIED_FRAMEWORK):
        raise ValueError(
            f"unsupported framework {peek.get('framework')!r} in {config_path!r}; "
            f"expected {_UNIFIED_FRAMEWORK!r}"
        )

    return _load_unified_variant(config_path, cluster_dict)