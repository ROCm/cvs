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
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cvs.lib import globals
from cvs.lib.inference.sglang.sglang_common import (
    DEFAULT_SGLANG_DECODE_COORD_PORT,
    DEFAULT_SGLANG_DECODE_SERV_PORT,
    DEFAULT_SGLANG_PREFILL_COORD_PORT,
    DEFAULT_SGLANG_PREFILL_SERV_PORT,
    perf_enforce_thresholds,
    resolve_disagg_execution_roles,
    resolve_distributed_execution_hosts,
    resolve_single_execution_hosts,
    stamp_disagg_roles,
)
from cvs.lib.utils.config_loader import substitute_config
from cvs.lib.utils_lib import resolve_test_config_placeholders
from cvs.parsers.schemas import (  # noqa: F401 - re-exported for the SGLang conftest and job classes
    SGLANG_FIRST_POLL_WAIT_S,
    SGLANG_POLL_PERIOD_S,
    SGLANG_POLL_TOTAL_S,
    SglangAccuracy,
    SglangAccuracyTask,
    SglangConfig,
    SglangParams,
    SglangRoles,
    SglangRoleServer,
    SglangSingleVariantConfig,
    resolve_old_sglang_keys,
    resolve_old_sglang_threshold_names,
)

log = globals.log

_LEGACY_FRAMEWORK = "sglang_single"

_LEGACY_ACC_CELL_RE = re.compile(r"^ACC_ISL=(\d+),OSL=(\d+)$")
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
                "pp": m.group("pp"),
                "conc": m.group("conc"),
                "specs": specs,
            }
        )
    cells.sort(key=lambda c: (int(c["isl"]), int(c["osl"]), int(c["conc"])))
    return cells


def selected_sweep_combo_keys(
    sweep: Mapping[str, Any] | None,
    threshold_cells: list[Mapping[str, Any]],
) -> list[str]:
    """Return combo keys to run from ``sweep.runs``, or every threshold cell if ``runs`` is empty."""
    runs = (sweep or {}).get("runs") or []
    if not runs:
        return [str(cell["cell_key"]) for cell in threshold_cells]

    combo_keys = []
    for run in runs:
        combo = run.get("combo") if isinstance(run, Mapping) else run
        combo_keys.append(str(combo or "").strip())
    return combo_keys


def perf_cells_for_variant(variant: "SglangSingleVariantConfig") -> list[dict[str, Any]]:
    """Return configured performance cells with per-cell benchmark overrides."""
    threshold_cells = perf_cells_from_thresholds(variant.thresholds)
    by_key = {str(cell["cell_key"]): cell for cell in threshold_cells}
    by_shape: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for cell in threshold_cells:
        by_shape.setdefault((cell["isl"], cell["osl"], cell["conc"]), []).append(cell)
    base_benchmark = dict((variant.benchmark_params.get("inference_tests") or {}).get("bench_serv_random") or {})
    combo_keys = selected_sweep_combo_keys(variant.sweep, threshold_cells)

    cells = []
    seen = set()
    for combo_key in combo_keys:
        match = _PERF_CELL_RE.fullmatch(combo_key)
        if not match:
            raise ValueError(f"invalid SGLang sweep combo key: {combo_key!r}")
        shape = (match.group("isl"), match.group("osl"), match.group("conc"))
        threshold_cell = by_key.get(combo_key)
        candidates = by_shape.get(shape) or []
        if threshold_cell is None and len(candidates) == 1:
            threshold_cell = candidates[0]
        if threshold_cell is None and not candidates:
            raise ValueError(f"SGLang sweep combo {combo_key!r} has no matching threshold cell")
        if threshold_cell is None:
            raise ValueError(
                f"SGLang sweep combo {combo_key!r} matches multiple threshold cells; use an exact TP/PP combo key"
            )
        if combo_key in seen:
            raise ValueError(f"duplicate SGLang sweep combo: {combo_key!r}")
        seen.add(combo_key)

        overrides = variant.sweeps.get(combo_key)
        if overrides is None:
            matching_override_keys = [
                key
                for key in variant.sweeps
                if (override_match := _PERF_CELL_RE.fullmatch(str(key)))
                and (override_match.group("isl"), override_match.group("osl"), override_match.group("conc")) == shape
            ]
            if len(matching_override_keys) == 1:
                overrides = variant.sweeps[matching_override_keys[0]]
            elif len(matching_override_keys) > 1:
                raise ValueError(
                    f"SGLang sweep combo {combo_key!r} matches multiple sweeps overrides; use an exact TP/PP combo key"
                )
        overrides = overrides or {}
        if not isinstance(overrides, Mapping):
            raise TypeError(f"SGLang sweeps[{combo_key!r}] must be a JSON object")
        cells.append(
            {
                **threshold_cell,
                "cell_key": combo_key,
                "tp": match.group("tp"),
                "pp": match.group("pp"),
                "benchmark_overrides": dict(overrides),
                "benchmark_params": {**base_benchmark, **dict(overrides)},
            }
        )
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
    variant = load_variant(config_file, cluster_dict={}, warn=False)
    cells = perf_cells_for_variant(variant)
    if not cells:
        raise ValueError(f"No selected ISL=... performance cells for {config_file!r}")
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
    _put("HSA_FORCE_FINE_GRAIN_PCIE", "hsa_force_fine_grain_pcie")

    cc_env = (inference.get("container_config") or {}).get("env_dict") or {}
    for k, v in cc_env.items():
        # Container env holds single ``docker run -e KEY=VALUE`` scalars. A leftover
        # ``ADD_EXPORT_ENV`` list would stringify to a Python repr whose spaces
        # break the command; those entries are expanded into real keys below.
        if v is None or isinstance(v, (list, dict, tuple)):
            continue
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

    hsa = str(env.get("HSA_FORCE_FINE_GRAIN_PCIE") or "").strip()
    env["HSA_FORCE_FINE_GRAIN_PCIE"] = hsa or "1"
    env.pop("HF_TOKEN", None)
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


# ---------- public API ----------


def orchestrator_container_from_variant(variant: SglangSingleVariantConfig) -> dict[str, Any]:
    """``container`` block for ``OrchestratorConfig`` (includes server env).

    ``runtime.args.env`` is dropped: the container runtime renders ``-e`` flags
    only from the top-level ``env`` here. Scalar ``runtime.args.env`` entries
    (and a leftover ``ADD_EXPORT_ENV`` list) are already flattened into
    ``roles.server.env`` by the loader.
    """
    block = variant.container.model_dump()
    runtime = dict(block.get("runtime") or {})
    runtime_args = {key: value for key, value in dict(runtime.get("args") or {}).items() if key != "env"}
    runtime["args"] = runtime_args
    block["runtime"] = runtime

    server_env = variant.roles.server.env
    if server_env:
        block["env"] = dict(server_env)
    return block


def _mounts_to_volume_dict(mounts: list[Any]) -> dict[str, str]:
    volumes: dict[str, str] = {}
    for mount in mounts:
        host, separator, container = str(mount).partition(":")
        if separator and host and container:
            volumes[host] = container
    return volumes


_RUNTIME_ENV_TO_INFERENCE = {
    "NCCL_IB_HCA": "nccl_ib_hca",
    "NCCL_SOCKET_IFNAME": "nccl_socket_ifname",
    "GLOO_SOCKET_IFNAME": "gloo_socket_ifname",
    "GLOO_TCP_IFNAME": "gloo_tcp_ifname",
    "NCCL_IB_GID_INDEX": "nccl_ib_gid_index",
    "NCCL_DEBUG": "nccl_debug",
    "HSA_FORCE_FINE_GRAIN_PCIE": "hsa_force_fine_grain_pcie",
}


def _add_export_env_from_runtime(runtime_env, existing=None):
    """KEY=VALUE lines for in-container ``export`` from extra env (and legacy ADD_EXPORT_ENV)."""
    entries = []
    seen = set()

    def _add(line):
        line = str(line).strip()
        if not line:
            return
        if line.startswith("export "):
            line = line[7:].strip()
        if line.split("=", 1)[0].strip() == "HF_TOKEN":
            return
        if line in seen:
            return
        seen.add(line)
        entries.append(line)

    if existing is not None:
        items = existing if isinstance(existing, list) else [str(existing)]
        for item in items:
            _add(item)

    raw = runtime_env.get("ADD_EXPORT_ENV")
    if raw is not None:
        items = raw if isinstance(raw, list) else [str(raw)]
        for item in items:
            _add(item)

    for key, value in runtime_env.items():
        if key == "ADD_EXPORT_ENV" or key in _RUNTIME_ENV_TO_INFERENCE:
            continue
        if value is None or isinstance(value, (list, dict, tuple)):
            continue
        _add(f"{key}={value}")
    return entries


def _view_number(value):
    return int(value) if isinstance(value, float) and value.is_integer() else value


def engine_args_to_flags(engine_args):
    """``launch_server`` flag lines for ``server_params.engine_args``; ``True`` writes a bare flag."""
    flags = []
    for name, value in engine_args.items():
        flag = "--" + name.replace("_", "-")
        if value is True:
            flags.append(flag)
        elif isinstance(value, (list, tuple)):
            flags.append(f"{flag} {' '.join(str(item) for item in value)}")
        elif value is not False and value is not None:
            flags.append(f"{flag} {_view_number(value)}")
    return flags


def _lm_eval_task_view(task):
    view = {
        key: _view_number(value)
        for key, value in task.model_dump(exclude_unset=True, exclude={"id", "task", "backend"}).items()
    }
    if isinstance(view.get("tasks"), list):
        view["tasks"] = ",".join(view["tasks"])
    return view


def _threshold_cells_view(config):
    return {
        key: {metric: spec.model_dump(exclude_defaults=True) for metric, spec in cell.items()}
        for key, cell in config.thresholds.cells().items()
    }


def _client_poll_view(client_s):
    """The poll count that gives ``client_s``, the inverse of the old-key rule.

    The total stays main's 3600 s default up to 40 polls; past that it is one
    poll period beyond the polls, so the time each log fetch takes never drops
    the last poll.
    """
    polling = max(client_s - SGLANG_FIRST_POLL_WAIT_S, SGLANG_POLL_PERIOD_S)
    polls = -(-polling // SGLANG_POLL_PERIOD_S)
    total = (
        SGLANG_POLL_TOTAL_S if polls * SGLANG_POLL_PERIOD_S <= SGLANG_POLL_TOTAL_S else polling + SGLANG_POLL_PERIOD_S
    )
    return {"inference_poll_iterations": str(polls), "inference_poll_total_timeout_sec": str(total)}


def _runtime_views(config):
    """Build the ``inference``, ``benchmark_params`` and ``roles.server`` dicts the job classes read."""
    server = config.server_params
    runtime_args = config.container.runtime.args.model_dump()
    runtime_env = dict(config.container.env)
    engine_args = dict(server.engine_args)
    memory_fraction = engine_args.pop("mem_fraction_static", None)
    log_level = engine_args.pop("log_level", None)

    benchmark = {
        "backend": "sglang",
        "num_prompts": config.benchmark_params.num_prompts,
        "random_range_ratio": _view_number(config.benchmark_params.random_range_ratio),
        "enforce_thresholds": config.enforce_thresholds,
    }
    inference_tests = {"bench_serv_random": benchmark}
    if config.accuracy.custom_eval:
        inference_tests["long_ctx_niah"] = {}
    for task in config.accuracy.lm_eval.tasks:
        inference_tests[task.id] = _lm_eval_task_view(task)

    params = {
        "model": server.model,
        "tensor_parallelism": str(server.tp),
        "pipeline_parallelism": str(server.pp),
        "inference_tests": inference_tests,
        "threshold_file": config.threshold_json,
    }
    if server.nnodes is not None:
        params["nnodes"] = str(server.nnodes)
    if memory_fraction is not None:
        params["memory_fraction"] = str(_view_number(memory_fraction))
    flags = engine_args_to_flags(engine_args)
    if flags:
        params["add_flags"] = flags
    if config.accuracy.custom_eval:
        params["lng_ctx_activate"] = True
        params["context_length"] = str(server.max_model_length)
        params["chunked_prefill_size"] = str(server.long_context.chunked_prefill_size)
        if server.long_context.max_prefill_tokens is not None:
            params["max_prefill_tokens"] = str(server.long_context.max_prefill_tokens)
    if config.timeouts.client_s is not None:
        params.update(_client_poll_view(config.timeouts.client_s))
    add_export_env = _add_export_env_from_runtime(runtime_env)
    if add_export_env:
        params["add_export_env"] = add_export_env

    inference = {
        "container_image": config.container.image,
        "container_name": config.container.name,
        "container_lifetime": config.container.lifetime,
        "hf_token_file": config.paths.hf_token_file,
        "log_dir": config.paths.log_dir,
        "shm_size": runtime_args.get("shm_size"),
        "container_config": {
            "device_list": list(runtime_args.get("devices") or []),
            "volume_dict": _mounts_to_volume_dict(list(runtime_args.get("volumes") or [])),
            "env_dict": dict(runtime_env),
        },
        "tensor_parallelism": params["tensor_parallelism"],
        "pipeline_parallelism": params["pipeline_parallelism"],
    }
    if server.nnodes is not None:
        inference["nnodes"] = params["nnodes"]
    if log_level is not None:
        inference["log_level"] = str(log_level)
    for env_key, inference_key in _RUNTIME_ENV_TO_INFERENCE.items():
        if env_key in runtime_env:
            inference[inference_key] = runtime_env[env_key]
    hsa_force = str(inference.get("hsa_force_fine_grain_pcie") or "").strip()
    inference["hsa_force_fine_grain_pcie"] = hsa_force or "1"
    inference["proxy_router_serv_port"] = str(server.port)
    inference["proxy_router_port"] = str(server.port)
    inference["dist_init_port"] = str(server.master_port)
    inference["prefill_serv_port"] = DEFAULT_SGLANG_PREFILL_SERV_PORT
    inference["decode_serv_port"] = DEFAULT_SGLANG_DECODE_SERV_PORT
    inference["prefill_coordinator_port"] = DEFAULT_SGLANG_PREFILL_COORD_PORT
    inference["decode_coordinator_port"] = DEFAULT_SGLANG_DECODE_COORD_PORT

    _inject_typed_accuracy_thresholds(params, config)
    # The token is read from hf_token_file inside the container. A literal
    # HF_TOKEN here becomes `docker run -e`, which the orchestrator logs.
    env = {**runtime_env, **_legacy_server_env(inference, params)}
    env.pop("HF_TOKEN", None)
    role_server = {"env": env, "serve_port": str(server.port)}
    return inference, params, role_server


def _inject_typed_accuracy_thresholds(bp_dict, config):
    inference_tests = bp_dict["inference_tests"]
    for task_id, specs in config.thresholds.accuracy.lm_eval.items():
        if task_id not in inference_tests:
            continue
        expected = inference_tests[task_id].setdefault("expected_results", {})
        expected[task_id.removeprefix("lm_eval_")] = flat_expected_from_specs(
            {metric: spec.model_dump() for metric, spec in specs.items()}
        )
        log.info("Loaded accuracy thresholds for %r", task_id)


def _legacy_acc_isl(key):
    match = _LEGACY_ACC_CELL_RE.match(str(key))
    return int(match.group(1)) if match else 0


def acc_cells_for_variant(variant):
    """One cell per ``niah`` eval, with its threshold specs and run options.

    A legacy-layout variant has no validated config; its cells are the
    ``ACC_ISL=<n>,OSL=<n>`` sections of its threshold file.
    """
    if variant.config is None:
        niah = (variant.benchmark_params.get("inference_tests") or {}).get("long_ctx_niah") or {}
        options = {"num_prompts": niah.get("num_prompts", 16), "seed": niah.get("seed", 42)}
        return [
            {"cell_key": key, "id": key, "isl": match.group(1), "osl": match.group(2), **options, "specs": specs}
            for key, specs in sorted(variant.thresholds.items(), key=lambda item: _legacy_acc_isl(item[0]))
            if (match := _LEGACY_ACC_CELL_RE.match(str(key)))
        ]
    cells = []
    for niah in variant.config.accuracy.custom_eval:
        specs = variant.config.thresholds.accuracy.custom_eval.get(niah.id, {})
        cells.append(
            {
                "cell_key": f"ACC_ISL={niah.isl},OSL={niah.osl}",
                "id": niah.id,
                "isl": str(niah.isl),
                "osl": str(niah.osl),
                "num_prompts": niah.num_prompts,
                "seed": niah.seed,
                "specs": {metric: spec.model_dump() for metric, spec in specs.items()},
            }
        )
    return cells


def _warn_old_keys(message, config_path):
    if message:
        where = f"{config_path}: " if config_path else ""
        warnings.warn(where + message, FutureWarning, stacklevel=3)


def validate_sglang_config(raw, *, config_path="", warn=True):
    """Resolve old keys, then validate ``raw`` (thresholds under ``thresholds``).

    With ``warn``, an old config raises one ``FutureWarning`` naming the file.
    """
    converted, message = resolve_old_sglang_keys(dict(raw))
    if warn:
        _warn_old_keys(message, config_path)
    return SglangConfig.model_validate(converted, context={"placeholders": "reject"})


def _load_legacy_variant(config_path, cluster_dict, warn):
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
    thresholds, message = resolve_old_sglang_threshold_names(_load_thresholds_file(threshold_path))
    if warn:
        _warn_old_keys(message, config_path)
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


def _infer_topology(config):
    return "distributed" if (config.server_params.nnodes or 1) > 1 else "single"


def _load_unified_variant(config_path, cluster_dict, topology, warn):
    raw, thresholds = substitute_config(config_path, cluster_dict)
    raw = resolve_test_config_placeholders(raw, cluster_dict)
    config = validate_sglang_config({**raw, "thresholds": thresholds}, config_path=str(config_path), warn=warn)
    topology = topology or _infer_topology(config)

    inference, benchmark_params, server = _runtime_views(config)
    if topology == "single":
        if cluster_dict.get("node_dict"):
            hosts = resolve_single_execution_hosts(cluster_dict)
            inference["_execution_hosts"] = list(hosts)
            inference["benchmark_serv_node"] = hosts[0]
    elif topology == "distributed":
        if cluster_dict.get("node_dict"):
            hosts = resolve_distributed_execution_hosts(cluster_dict, inference)
            inference["_execution_hosts"] = list(hosts)
            inference["server_node_list"] = list(hosts)
            inference["benchmark_serv_node"] = hosts[0]
    elif cluster_dict.get("node_dict"):
        stamp_disagg_roles(inference, resolve_disagg_execution_roles(cluster_dict, inference))

    return SglangSingleVariantConfig(
        schema_version=1,
        framework="sglang",
        topology=topology,
        enforce_thresholds=config.enforce_thresholds,
        threshold_json=config.threshold_json,
        paths=config.paths,
        model={"id": config.server_params.model, "remote": 0},
        container=config.container.model_dump(),
        thresholds=_threshold_cells_view(config),
        variant_key="default",
        config_path=str(Path(config_path).resolve()),
        config=config,
        inference=inference,
        benchmark_params=benchmark_params,
        params=benchmark_params,
        sweeps=config.sweeps,
        sweep={"runs": list(config.runs)},
        roles={"server": server},
    )


def load_variant(config_path, cluster_dict, topology=None, warn=True):
    """Load and validate an SGLang config and its thresholds.

    ``topology`` is ``single``, ``distributed`` or ``disaggregated``, from the
    test module; without it, ``server_params.nnodes`` above 1 means distributed.
    Collection passes ``warn=False``, so an old config warns once, on the
    fixture's load.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"variant config not found: {path}")

    with open(path, encoding="utf-8") as fp:
        peek = json.load(fp)

    if _is_legacy_root(peek):
        return _load_legacy_variant(config_path, cluster_dict, warn)

    return _load_unified_variant(config_path, cluster_dict, topology, warn)
