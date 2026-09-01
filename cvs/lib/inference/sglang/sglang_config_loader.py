'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

SGLang config loader for single-node, distributed, and disaggregated suites.

``load_variant()`` is the single entry point for ``sglang_single`` conftest and
produces both:
- typed fields for ``OrchestratorFactory`` (``container``, ``paths``, ``model``)
- controller dictionaries (``inference``, ``benchmark_params``) used by the
  existing SGLang job classes
'''

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator
from typing_extensions import Literal

from cvs.lib import globals
from cvs.lib.inference.sglang.sglang_common import perf_enforce_thresholds
from cvs.lib.utils.config_loader import (
    BaseVariantConfig,
    _Allow,
    _Forbid,
    substitute_config,
)
from cvs.lib.utils_lib import resolve_test_config_placeholders

log = globals.log

_LEGACY_FRAMEWORK = "sglang_single"
_UNIFIED_FRAMEWORK = "sglang_single"

_PERF_CELL_RE = re.compile(r"^ISL=(?P<isl>\d+),OSL=(?P<osl>\d+),TP=(?P<tp>\d+),PP=(?P<pp>\d+),CONC=(?P<conc>\d+)$")


# ---------- threshold / variant helpers (moved out of conftest) ----------


def resolve_benchmark_variant_key(root: Mapping[str, Any], config_path: str) -> str:
    """Pick which ``benchmark_params`` entry to run.

    Resolution order:
    1. Environment ``SGLANG_BENCHMARK_KEY`` (override for CI matrices).
    2. If ``benchmark_params`` has exactly one key, use it.

    ``root`` is the full JSON object loaded from ``--config_file`` (not only ``config``).
    """
    env_key = (os.environ.get("SGLANG_BENCHMARK_KEY") or "").strip()
    bp = root.get("benchmark_params") or {}
    if not isinstance(bp, dict) or not bp:
        raise ValueError(f"benchmark_params missing or empty in {config_path!r}")

    if env_key:
        if env_key not in bp:
            raise ValueError(
                f"SGLANG_BENCHMARK_KEY={env_key!r} not found in benchmark_params ({config_path}); valid: {sorted(bp)!r}"
            )
        log.info("Using benchmark variant from env SGLANG_BENCHMARK_KEY=%r", env_key)
        return env_key

    if len(bp) == 1:
        only = next(iter(bp))
        log.info("Single benchmark_params entry; using %r", only)
        return str(only)

    raise ValueError(
        f"Multiple benchmark_params keys in {config_path!r}: {sorted(bp)!r}. Export SGLANG_BENCHMARK_KEY to select one."
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
        cells.append(
            {
                "cell_key": cell_key,
                "isl": m.group("isl"),
                "osl": m.group("osl"),
                "tp": m.group("tp"),
                "conc": m.group("conc"),
                "specs": specs,
            }
        )
    cells.sort(key=lambda c: (int(c["isl"]), int(c["osl"]), int(c["conc"])))
    return cells


def perf_specs_for_cell(thresholds: Mapping[str, Any], isl, osl, conc) -> dict[str, float]:
    """Flattened threshold gates for one perf cell, or ``{}`` when the cell has none.

    Perf cells are parametrized on ISL/OSL/CONC alone (see ``perf_cells_from_thresholds``),
    so TP/PP in the cell key are descriptive and deliberately excluded from the match: a
    threshold file written for TP=8,PP=1 still gates a run whose config sets PP=2.
    """
    target = (str(isl), str(osl), str(conc))
    for cell in perf_cells_from_thresholds(thresholds):
        if (cell["isl"], cell["osl"], cell["conc"]) != target:
            continue
        specs = {metric: spec for metric, spec in (cell["specs"] or {}).items() if spec is not None}
        return flat_expected_from_specs(specs)
    return {}


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
        raise TypeError(f"threshold file must be a JSON object: {path}")
    return {k: v for k, v in raw.items() if not str(k).startswith("_")}


def _threshold_file_path(bp_dict: Mapping[str, Any]) -> str | None:
    path = bp_dict.get("threshold_file")
    return str(path).strip() if path else None


def _inject_thresholds_into_bp_dict(
    bp_dict: dict[str, Any],
    thresholds: Mapping[str, Any],
    *,
    inject_current_perf: bool = True,
) -> None:
    inference_tests = bp_dict.setdefault("inference_tests", {})

    if inject_current_perf:
        perf_key = perf_cell_key(bp_dict)
        perf_specs = thresholds.get(perf_key)
        if perf_specs:
            bench = inference_tests.setdefault("bench_serv_random", {})
            expected = bench.setdefault("expected_results", {})
            expected["auto"] = flat_expected_from_specs(perf_specs)
            log.info("Loaded performance thresholds from cell %r", perf_key)
        else:
            log.warning("No performance thresholds for cell %r in threshold file", perf_key)

    for bench_name in ("lm_eval_hellaswag", "lm_eval_gsm8k"):
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
    volume_dict = (inference.get("container_config") or {}).get("volume_dict") or {}
    for host, container in volume_dict.items():
        host_s, container_s = str(host), str(container)
        if "models" in host_s.lower() or "models" in container_s.lower():
            return host_s
    # Fallback: sibling of log_dir
    log_dir = str(inference.get("log_dir") or "").rstrip("/")
    if log_dir:
        return str(Path(log_dir).parent / "models")
    raise ValueError(
        "cannot infer models_dir from legacy config; add a models volume mount or migrate to unified paths.models_dir"
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

    cc_env = (inference.get("container_config") or {}).get("env_dict") or {}
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


class SglangRoleServer(_Allow):
    env: dict[str, str] = Field(default_factory=dict)
    serve_port: str = ""


class SglangRoles(_Forbid):
    server: SglangRoleServer = Field(default_factory=SglangRoleServer)


class SglangParams(_Allow):
    """SGLang runtime parameters passed to the existing job controllers."""

    inference_tests: dict[str, Any] = Field(default_factory=dict)
    add_export_env: list[str] = Field(default_factory=list)
    add_flags: list[str] = Field(default_factory=list)


class SglangAccuracyTask(_Allow):
    """One named lm-eval task from the unified ``accuracy.tasks`` block."""

    id: str


class SglangAccuracy(_Forbid):
    tasks: list[SglangAccuracyTask] = Field(default_factory=list)


class SglangSingleVariantConfig(BaseVariantConfig):
    """Typed config shared by all SGLang topologies."""

    framework: Literal["sglang", "sglang_single"]
    gpu_arch: str
    topology: Literal["single", "distributed", "disaggregated"] = "single"
    variant_key: str = ""
    config_path: str = ""
    params: SglangParams = Field(default_factory=SglangParams)
    accuracy: SglangAccuracy = Field(default_factory=SglangAccuracy)

    # Legacy blocks kept for ``SglangSingle`` until that lib is refactored.
    inference: dict[str, Any] = Field(default_factory=dict)
    benchmark_params: dict[str, Any] = Field(default_factory=dict)

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

    @model_validator(mode="after")
    def _validate_topology_fields(self):
        required = {
            "single": ("benchmark_serv_node",),
            "distributed": ("server_node_list", "benchmark_serv_node", "dist_init_port"),
            "disaggregated": (
                "prefill_node_list",
                "decode_node_list",
                "proxy_router_node",
                "benchmark_serv_node",
                "prefill_coordinator_addr",
                "decode_coordinator_addr",
            ),
        }[self.topology]
        missing = [key for key in required if not self.inference.get(key)]
        if missing:
            raise ValueError(f"{self.topology} SGLang config missing required role fields: {missing}")

        if self.topology == "distributed":
            nodes = self.inference["server_node_list"]
            node_count = len(nodes) if isinstance(nodes, list) else 1
            nnodes = int(self.inference.get("nnodes") or node_count)
            if nnodes != node_count:
                raise ValueError(f"distributed SGLang nnodes={nnodes} must match server_node_list length {node_count}")
        return self


# ---------- public API ----------


def orchestrator_container_from_variant(variant: SglangSingleVariantConfig) -> dict[str, Any]:
    """``container`` block for ``OrchestratorConfig`` (includes server env)."""
    block = variant.container.model_dump()
    server_env = variant.roles.server.env
    if server_env:
        block = {**block, "env": dict(server_env)}
    return block


def _mounts_to_volume_dict(mounts: list[Any]) -> dict[str, str]:
    volumes: dict[str, str] = {}
    for mount in mounts:
        host, separator, container = str(mount).partition(":")
        if separator and host and container:
            volumes[host] = container
    return volumes


def _accuracy_tasks_to_inference_tests(accuracy: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    inference_tests: dict[str, dict[str, Any]] = {}
    for raw_task in accuracy.get("tasks") or []:
        task = dict(raw_task)
        task_id = str(task.pop("id", "")).strip()
        if not task_id:
            raise ValueError("each accuracy.tasks entry requires a non-empty 'id'")
        if task_id in inference_tests:
            raise ValueError(f"duplicate accuracy task id: {task_id!r}")
        inference_tests[task_id] = task
    return inference_tests


def _unified_runtime_views(raw: Mapping[str, Any], thresholds: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    """Build legacy controller views from the unified SGLang schema."""
    paths = dict(raw.get("paths") or {})
    container = dict(raw.get("container") or {})
    runtime = dict(container.get("runtime") or {})
    runtime_args = dict(runtime.get("args") or {})
    server = dict((raw.get("roles") or {}).get("server") or {})
    params = dict(raw.get("params") or {})

    inference_tests = dict(params.get("inference_tests") or {})
    inference_tests.update(_accuracy_tasks_to_inference_tests(raw.get("accuracy") or {}))
    params["inference_tests"] = inference_tests
    params["model"] = str((raw.get("model") or {}).get("id") or "")
    params["threshold_file"] = str(raw.get("threshold_json") or "")

    inference: dict[str, Any] = {
        "container_image": container.get("image"),
        "container_name": container.get("name"),
        "container_lifetime": container.get("lifetime", "per_run"),
        "hf_token_file": paths.get("hf_token_file"),
        "log_dir": paths.get("log_dir"),
        "shm_size": runtime_args.get("shm_size"),
        "container_config": {
            "device_list": list(runtime_args.get("devices") or []),
            "volume_dict": _mounts_to_volume_dict(list(runtime_args.get("volumes") or [])),
            "env_dict": dict(runtime_args.get("env") or {}),
        },
    }
    for key, value in server.items():
        if key.startswith("_") or key in ("env", "serve_port"):
            continue
        inference[key] = value
    if server.get("serve_port") and "proxy_router_serv_port" not in inference:
        inference["proxy_router_serv_port"] = server["serve_port"]

    _inject_thresholds_into_bp_dict(params, thresholds, inject_current_perf=False)
    server["env"] = {
        **_legacy_server_env(inference, params),
        **dict(server.get("env") or {}),
    }
    return inference, params, server


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
        raise ValueError(f"benchmark_params[{variant_key!r}] missing 'threshold_file' in {config_path!r}")

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
        "topology": (
            "disaggregated"
            if inference.get("prefill_node_list") or inference.get("decode_node_list")
            else "distributed"
            if inference.get("server_node_list")
            else "single"
        ),
        "enforce_thresholds": perf_enforce_thresholds(bp),
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
                    inference.get("proxy_router_serv_port") or inference.get("proxy_router_port") or "8000"
                ),
            }
        },
    }
    return SglangSingleVariantConfig(**raw)


def _load_unified_variant(config_path: str, cluster_dict: Mapping[str, Any]) -> SglangSingleVariantConfig:
    raw, thresholds = substitute_config(config_path, cluster_dict)
    raw = resolve_test_config_placeholders(raw, cluster_dict)
    raw["thresholds"] = thresholds
    raw["config_path"] = str(Path(config_path).resolve())

    raw["variant_key"] = raw.get("variant_key") or "default"

    inference, benchmark_params, server = _unified_runtime_views(raw, thresholds)
    raw["inference"] = inference
    raw["benchmark_params"] = benchmark_params
    raw.setdefault("roles", {})["server"] = server
    raw["enforce_thresholds"] = perf_enforce_thresholds(benchmark_params)

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

    if peek.get("framework") not in (None, "sglang", _UNIFIED_FRAMEWORK):
        raise ValueError(f"unsupported framework {peek.get('framework')!r} in {config_path!r}; expected 'sglang'")

    return _load_unified_variant(config_path, cluster_dict)
