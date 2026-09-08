'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Convert unified inference serving configs (server_params, benchmark_params,
sweeps, runs) into ATOM AtomVariantConfig-compatible dicts.
'''

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

_PERF_CELL_RE = re.compile(r"^ISL=(?P<isl>\d+),OSL=(?P<osl>\d+),TP=(?P<tp>\d+),PP=(?P<pp>\d+),CONC=(?P<conc>\d+)$")


def is_serving_config(raw: Mapping[str, Any]) -> bool:
    return (
        isinstance(raw.get("server_params"), dict)
        and isinstance(raw.get("benchmark_params"), dict)
        and isinstance(raw.get("runs"), list)
        and isinstance(raw.get("sweeps"), dict)
    )


def parse_perf_cell_key(cell_key: str) -> dict[str, str]:
    m = _PERF_CELL_RE.match(str(cell_key).strip())
    if not m:
        raise ValueError(f"invalid performance cell key: {cell_key!r}")
    return {k: m.group(k) for k in ("isl", "osl", "tp", "pp", "conc")}


def filter_thresholds_by_runs(thresholds: Mapping[str, Any], runs: list[str]) -> dict[str, Any]:
    if not runs:
        return dict(thresholds)
    allowed = set(runs)
    out: dict[str, Any] = {}
    for key, value in thresholds.items():
        if str(key).startswith("ISL=") and key not in allowed:
            continue
        out[key] = value
    return out


def atom_sweep_to_serving(sweep: Mapping[str, Any], tp: str, pp: str) -> tuple[dict[str, dict], list[str]]:
    by_name = {c["name"]: c for c in sweep.get("sequence_combinations") or []}
    sweeps: dict[str, dict] = {}
    runs: list[str] = []
    for run in sweep.get("runs") or []:
        combo = by_name[run["combo"]]
        key = f"ISL={combo['isl']},OSL={combo['osl']},TP={tp},PP={pp},CONC={run['concurrency']}"
        sweeps[key] = {}
        runs.append(key)
    return sweeps, runs


def _container_runtime_env(container: Mapping[str, Any]) -> dict[str, str]:
    runtime = container.get("runtime") or {}
    args = runtime.get("args") or {}
    env = args.get("env") or {}
    return {str(k): str(v) for k, v in env.items() if v is not None}


def serving_runs_to_atom_sweep(runs: list[str]) -> dict[str, Any]:
    sequence_combinations: list[dict[str, Any]] = []
    sweep_runs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cell_key in runs:
        parts = parse_perf_cell_key(cell_key)
        name = f"isl{parts['isl']}_osl{parts['osl']}"
        if name not in seen:
            sequence_combinations.append({"name": name, "isl": parts["isl"], "osl": parts["osl"]})
            seen.add(name)
        sweep_runs.append({"combo": name, "concurrency": int(parts["conc"])})
    return {"sequence_combinations": sequence_combinations, "runs": sweep_runs}


def _resolve_atom_driver(server: Mapping[str, Any]) -> str:
    backend = str(server.get("backend", "atom")).lower()
    if backend in ("vllm", "vllm_atom"):
        return "vllm_atom"
    if backend == "sglang":
        return "sglang"
    if backend == "atom":
        return "atom"
    raise ValueError(f"unsupported server_params.backend for atom suite: {backend!r}")


def serving_to_atom_variant_raw(raw: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    """Build an ATOM ``AtomVariantConfig``-compatible dict from serving schema."""
    server = dict(raw["server_params"])
    bench = dict(raw["benchmark_params"])
    container = dict(raw.get("container") or {})
    runs = list(raw.get("runs") or [])
    driver = _resolve_atom_driver(server)

    roles_server: dict[str, Any] = {"env": dict(server.get("env") or {})}
    roles_server["env"].update(_container_runtime_env(container))
    if driver == "vllm_atom":
        roles_server["serve_args"] = dict(server.get("serve_args") or {})
    elif driver == "sglang":
        roles_server["sglang_args"] = list(server.get("add_flags") or [])
    if server.get("ib_hca_devices") is not None:
        roles_server["ib_hca_devices"] = server["ib_hca_devices"]
    if server.get("ib_netdev") is not None:
        roles_server["ib_netdev"] = server["ib_netdev"]

    params: dict[str, Any] = {
        "driver": driver,
        "port_no": str(server.get("proxy_router_serv_port", server.get("port_no", "8000"))),
        "tensor_parallelism": str(server.get("tensor_parallelism", "8")),
        "pipeline_parallel_size": str(server.get("pipeline_parallelism", "1")),
        "nnodes": str(server.get("nnodes", "1")),
        "dataset_name": str(bench.get("data_set_name", "random")),
        "random_range_ratio": str(bench.get("random_range_ratio", "0.8")),
        "num_prompts": str(bench.get("num_prompts", "1000")),
        "metric_percentiles": str(bench.get("metric_percentiles", "95,99")),
        "client_poll_count": str(bench.get("client_poll_count", "80")),
        "client_poll_wait_time": str(bench.get("client_poll_wait_time", "60")),
        "reuse_server_across_sweep": str(bench.get("reuse_server_across_sweep", "true")),
    }
    if server.get("context_length") or bench.get("max_model_length"):
        params["max_model_length"] = str(server.get("context_length") or bench.get("max_model_length"))
    if server.get("master_addr"):
        params["master_addr"] = str(server["master_addr"])
    if server.get("master_port") or server.get("dist_init_port"):
        params["master_port"] = str(server.get("master_port") or server.get("dist_init_port"))
    if bench.get("scaling_baseline_output_throughput"):
        params["scaling_baseline_output_throughput"] = str(bench["scaling_baseline_output_throughput"])

    model = dict(raw.get("model") or {})
    if not model.get("id"):
        model["id"] = str(server.get("model") or "")

    return {
        "schema_version": 1,
        "framework": "atom",
        "gpu_arch": str(raw.get("gpu_arch") or "mi3xx"),
        "enforce_thresholds": bool(raw.get("enforce_thresholds", True)),
        "threshold_json": raw.get("threshold_json", ""),
        "paths": dict(raw.get("paths") or {}),
        "model": model,
        "container": container,
        "roles": {"server": roles_server},
        "params": params,
        "sweep": serving_runs_to_atom_sweep(runs),
        "accuracy": deepcopy(raw.get("accuracy") or {"tasks": []}),
        "functional": deepcopy(raw.get("functional") or {"api_smoke": True, "health_check": True}),
        "quant_parity": deepcopy(raw.get("quant_parity") or {"enabled": False}),
        "long_context_accuracy": deepcopy(raw.get("long_context_accuracy") or {"cells": []}),
        "platform": deepcopy(raw.get("platform") or {}),
        "thresholds": filter_thresholds_by_runs(thresholds, runs),
    }
