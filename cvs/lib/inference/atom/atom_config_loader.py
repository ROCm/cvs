'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

ATOM suite config loading (``atom``).

Users write :class:`cvs.parsers.schemas.AtomConfig`. :func:`load_variant`
converts an old layout through :func:`cvs.parsers.schemas.resolve_old_atom_config`
with one ``FutureWarning``, validates it, and builds the
:class:`~cvs.parsers.schemas.AtomVariantConfig` view the suite runs.
'''

from __future__ import annotations

import json
import math
import re
import shlex
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from cvs.lib.utils.config_loader import substitute_config
from cvs.parsers.schemas import (
    ATOM_DRIVERS,
    ATOM_PP_DRIVERS,
    AtomBlock,
    AtomConfig,
    AtomParams,
    AtomRoles,
    AtomRoleServer,
    AtomRunCard,
    AtomVariantConfig,
    ContainerEnvTokenError,
    OLD_KEYS_QUIET,
    MtpQualityConfig,
    QuantParityConfig,
    merge_mxfp4_triton_env,
    parse_inference_cell_key,
    resolve_old_atom_config,
    validate_sweep_selector,
    validation_problem_lines,
    warn_old_keys,
)

__all__ = [
    "ATOM_DRIVERS",
    "ATOM_PP_DRIVERS",
    "AtomBlock",
    "AtomConfig",
    "AtomParams",
    "AtomRoleServer",
    "AtomRoles",
    "AtomRunCard",
    "AtomVariantConfig",
    "MtpQualityConfig",
    "QuantParityConfig",
    "merge_mxfp4_triton_env",
]

# ATOM's openai_server spells these flags with underscores and accepts no
# dashed alias (atom/model_engine/arg_utils.py); every other flag uses dashes.
ATOM_UNDERSCORE_FLAGS = frozenset(
    {"enable_prefix_caching", "kv_cache_dtype", "load_dummy", "enable_chunked_prefill", "online_quant_config"}
)
_POLL_WAIT_S = 60


def resolved_atom_image_pin(variant):
    rc = getattr(variant, "run_card", None)
    pin = (getattr(rc, "atom_image_pin", None) or "").strip() if rc is not None else ""
    if pin:
        return pin
    container = getattr(variant, "container", None)
    return str(getattr(container, "image", "") or "").strip()


def _flag_name(name, underscore_flags):
    if name in underscore_flags:
        return f"--{name}"
    if name.startswith("no_") and name[3:] in underscore_flags:
        return f"--no-{name[3:]}"
    return "--" + name.replace("_", "-")


def _flag_value(value):
    return json.dumps(value) if isinstance(value, dict) else str(value)


def args_to_flag_tokens(args, underscore_flags=frozenset()):
    """Render ``{"flag": value}`` as CLI tokens: true is a bare flag, a list repeats the flag, a dict is JSON."""
    tokens = []
    for name, value in args.items():
        flag = _flag_name(name, underscore_flags)
        for item in value if isinstance(value, list) else [value]:
            tokens.extend([flag] if item is True else [flag, _flag_value(item)])
    return tokens


def _server_timing(total):
    if total is None:
        return {}
    precheck = min(30, total)
    warmup = min(330, total - precheck)
    polls = max(1, math.ceil((total - precheck - warmup) / _POLL_WAIT_S))
    return {
        "server_precheck_wait_s": str(precheck),
        "server_warmup_wait_s": str(warmup),
        "server_poll_count": str(polls),
        "server_poll_wait_time": str(_POLL_WAIT_S),
    }


def _client_timing(total):
    if total is None:
        return {}
    initial = min(120, total)
    polls = max(1, math.ceil((total - initial) / _POLL_WAIT_S))
    return {
        "client_initial_wait_s": str(initial),
        "client_poll_count": str(polls),
        "client_poll_wait_time": str(_POLL_WAIT_S),
    }


_PLAIN_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _plain_names(value):
    """Names in an NCCL device or interface list; NCCL's ``^``, ``=`` and ``:port`` forms give none."""
    names = [name.strip() for name in str(value or "").split(",")]
    return names if names and all(_PLAIN_NAME_RE.fullmatch(name) for name in names) else []


def _sweep_from_cell_keys(keys):
    sequence_combinations = []
    runs = []
    seen = set()
    for key in keys:
        axes = parse_inference_cell_key(key)
        name = f"isl{axes['isl']}_osl{axes['osl']}"
        if name not in seen:
            sequence_combinations.append({"name": name, "isl": str(axes["isl"]), "osl": str(axes["osl"])})
            seen.add(name)
        runs.append({"combo": name, "concurrency": axes["concurrency"]})
    return {"sequence_combinations": sequence_combinations, "runs": runs}


def _threshold_view(thresholds):
    view = {key: {name: spec.model_dump() for name, spec in cell.items()} for key, cell in thresholds.cells().items()}
    accuracy = thresholds.accuracy
    sections = (
        ("accuracy", accuracy.lm_eval, True),
        ("long_context_accuracy", accuracy.custom_eval, True),
        ("quant_parity", accuracy.quant_parity, False),
        ("mtp_quality", accuracy.mtp_quality, False),
    )
    for name, specs, keyed in sections:
        if not specs:
            continue
        if keyed:
            view[name] = {
                key: {metric: spec.model_dump() for metric, spec in cell.items()} for key, cell in specs.items()
            }
        else:
            view[name] = {metric: spec.model_dump() for metric, spec in specs.items()}
    return view


def atom_variant_raw(config, gpu_arch):
    """Build the ``AtomVariantConfig`` fields the suite reads from a validated ``AtomConfig``."""
    server = config.server_params
    bench = config.benchmark_params
    driver = config.atom.driver
    env = dict(config.container.env)
    roles_server = {"env": env}
    hcas = _plain_names(env.get("NCCL_IB_HCA"))
    if hcas:
        roles_server["ib_hca_devices"] = hcas
    netdevs = _plain_names(env.get("NCCL_SOCKET_IFNAME"))
    if len(netdevs) == 1:
        roles_server["ib_netdev"] = netdevs[0]
    if driver == "atom":
        roles_server["atom_args"] = [
            "-tp",
            str(server.tp),
            *args_to_flag_tokens(server.engine_args, ATOM_UNDERSCORE_FLAGS),
        ]
    elif driver == "sglang":
        roles_server["sglang_args"] = args_to_flag_tokens(server.engine_args)
    else:
        roles_server["serve_args"] = {
            name.replace("_", "-"): json.dumps(value) if isinstance(value, dict) else value
            for name, value in server.engine_args.items()
        }
    params = {
        "driver": driver,
        "port_no": str(server.port),
        "tensor_parallelism": str(server.tp),
        "pipeline_parallel_size": str(server.pp),
        "nnodes": str(server.nnodes or 1),
        "master_port": str(server.master_port),
        "num_prompts": str(bench.num_prompts),
        "random_range_ratio": str(bench.random_range_ratio),
        "seed": str(bench.seed),
        "request_rate": str(bench.request_rate),
        "burstiness": str(bench.burstiness),
        "random_prefix_len": str(bench.random_prefix_len),
        "tokenizer_mode": bench.tokenizer_mode,
        "bench_extra_args": shlex.join(args_to_flag_tokens(bench.client_args)),
        "reuse_server_across_sweep": "true",
        **_server_timing(config.timeouts.server_ready_s),
        **_client_timing(config.timeouts.client_s),
    }
    if server.max_model_length is not None:
        params["max_model_length"] = str(server.max_model_length)
    if bench.scaling_baseline_output_throughput is not None:
        params["scaling_baseline_output_throughput"] = str(bench.scaling_baseline_output_throughput)
    container = config.container.model_dump()
    return {
        "schema_version": 1,
        "framework": "atom",
        "gpu_arch": gpu_arch,
        "enforce_thresholds": config.enforce_thresholds,
        "threshold_json": config.threshold_json,
        "paths": config.paths.model_dump(),
        "model": {"id": server.model, "remote": 0, "precision": ""},
        "container": container,
        "roles": {"server": roles_server},
        "params": params,
        "sweep": _sweep_from_cell_keys(config.runs),
        "accuracy": config.accuracy.lm_eval.model_dump(),
        "quant_parity": config.accuracy.quant_parity.model_dump(),
        "mtp_quality": config.accuracy.mtp_quality.model_dump(),
        "long_context_accuracy": {"cells": [cell.model_dump(exclude={"type"}) for cell in config.accuracy.custom_eval]},
        "functional": {"api_smoke": True, "health_check": True},
        "platform": {"gpu_metrics_poll": True, "dmesg_scan": True},
        "thresholds": _threshold_view(config.thresholds),
    }


_CONFIG_STEM_GPU_ARCH = re.compile(r"^([^_]+)_atom_")


def gpu_arch_from_config_path(config_path) -> str:
    """Infer ``gpu_arch`` from ``{gpu}_atom_*.json`` config stems."""
    stem = Path(config_path).stem
    match = _CONFIG_STEM_GPU_ARCH.match(stem)
    if not match:
        raise ValueError(
            f"config filename must be {{gpu}}_atom_*.json to infer gpu_arch (got {Path(config_path).name})"
        )
    return match.group(1)


def _config_gpu_arch(raw, config_path):
    try:
        return gpu_arch_from_config_path(config_path)
    except ValueError:
        old_arch = str(raw.get("gpu_arch") or "").strip() if "atom" not in raw else ""
        if not old_arch:
            raise
        return old_arch


def atom_config_from_raw(raw, thresholds, config_path, profile=None, context=None):
    """Convert old keys and validate; returns ``(AtomConfig, gpu_arch)``.

    Old keys raise one ``FutureWarning`` and log it once, unless ``context``
    holds ``old_keys=OLD_KEYS_QUIET``.
    """
    gpu_arch = _config_gpu_arch(raw, config_path)
    try:
        raw, thresholds, message = resolve_old_atom_config(raw, thresholds, gpu_arch, profile)
    except ValueError as exc:
        raise ValueError(f"{Path(config_path).name}: {exc}") from None
    if message and (context or {}).get("old_keys") != OLD_KEYS_QUIET:
        warn_old_keys(
            f"{Path(config_path).name} uses old ATOM config keys: {message}. Rewrite it in the current format "
            "(see the shipped configs in cvs/input/config_file/inference/atom); old keys will be removed in a "
            "future release."
        )
    try:
        config = AtomConfig.model_validate({**raw, "thresholds": thresholds}, context=context)
    except ValidationError as exc:
        problems = "\n".join(f"  {line}" for line in validation_problem_lines(exc))
        raise ValueError(f"{Path(config_path).name} is not a valid ATOM config:\n{problems}") from None
    except ContainerEnvTokenError as exc:
        raise ValueError(f"{Path(config_path).name}: {exc}") from None
    return config, gpu_arch


def load_variant(config_path, cluster_dict, profile=None) -> AtomVariantConfig:
    raw, thresholds = substitute_config(config_path, cluster_dict)
    config, gpu_arch = atom_config_from_raw(raw, thresholds, config_path, profile, context={"placeholders": "reject"})
    return AtomVariantConfig(**atom_variant_raw(config, gpu_arch))


def collection_variant_raw(config_file, profile=None):
    """The view fields pytest collection parametrizes on, read before the cluster file is known."""
    config_path = Path(config_file)
    raw, thresholds = substitute_config(config_path, {})
    config, gpu_arch = atom_config_from_raw(raw, thresholds, config_path, profile, context={"old_keys": OLD_KEYS_QUIET})
    return atom_variant_raw(config, gpu_arch)


def expand_sweep(sweep):
    """Expand a sweep into ``(cases, ids)`` for pytest parametrization."""
    if hasattr(sweep, "sequence_combinations"):
        combos = [c.model_dump() for c in sweep.sequence_combinations]
        runs = [r.model_dump() for r in sweep.runs]
    else:
        combos = sweep.get("sequence_combinations", [])
        runs = sweep.get("runs", [])
    validate_sweep_selector([c["name"] for c in combos], [r["combo"] for r in runs])
    by_name = {c["name"]: c for c in combos}
    cases = []
    ids = []
    for run in runs:
        combo = by_name[run["combo"]]
        conc = run["concurrency"]
        cases.append((combo, conc))
        ids.append(f"{run['combo']}-conc{conc}")
    return cases, ids


def reuse_server_flag(params) -> bool:
    """Return True when ``params.reuse_server_across_sweep`` is a truthy string."""
    raw = str(getattr(params, "reuse_server_across_sweep", "false")).strip().lower()
    return raw in ("true", "1", "yes")


def server_session_key(variant_config, isl, osl):
    """Stable key for server reuse across sweep cells with identical model/shape."""
    p = variant_config.params
    roles = variant_config.roles.server
    if p.driver == "atom":
        server_tokens = tuple(roles.atom_args)
    elif p.driver == "sglang":
        server_tokens = tuple(roles.sglang_args)
    else:
        server_tokens = tuple(sorted(roles.serve_args.items()))
    return (
        variant_config.model.id,
        p.driver,
        str(isl),
        str(osl),
        server_tokens,
        p.tensor_parallelism,
        p.nnodes,
        p.pipeline_parallel_size,
        p.master_addr,
        p.master_port,
    )


def expand_sweep_parametrize(sweep, fixturenames):
    """Build pytest parametrize args for inference or metric-tier collection."""
    from cvs.lib.inference.atom.atom_parsing import METRIC_TIER_ORDER

    cases, ids = expand_sweep(sweep)
    if "metric_tier" in fixturenames:
        if not cases:
            return None
        tier_cases = []
        tier_ids = []
        for (combo, c), cid in zip(cases, ids):
            for tier in METRIC_TIER_ORDER:
                tier_cases.append((combo, c, tier))
                tier_ids.append(f"{cid}-{tier}")
        return ("seq_combo,concurrency,metric_tier", tier_cases, tier_ids)
    if "seq_combo" in fixturenames and "concurrency" in fixturenames and cases:
        return ("seq_combo,concurrency", cases, ids)
    return None


def placeholder_gated_threshold_cell(
    *,
    output_throughput_min: float = 0,
    total_token_throughput_min: float = 0,
    per_gpu_throughput_min: float = 0,
    output_tput_per_gpu_min: float = 0,
    mean_ttft_max_ms: float = 1_000_000,
    p99_ttft_max_ms: float = 1_000_000,
    mean_tpot_max_ms: float = 1_000_000,
    p95_tpot_max_ms: float = 1_000_000,
    failed_max: int = 1_000_000_000,
    success_rate_min: float = 0,
) -> dict[str, Any]:
    """Return one sweep cell's bare-metric specs covering every ``GATED_METRICS`` member."""
    loose_ms = {"kind": "max", "value": 1_000_000}
    return {
        "total_token_throughput": {"kind": "min", "value": total_token_throughput_min},
        "output_throughput": {"kind": "min", "value": output_throughput_min},
        "per_gpu_throughput": {"kind": "min", "value": per_gpu_throughput_min},
        "output_tput_per_gpu": {"kind": "min", "value": output_tput_per_gpu_min},
        "mean_ttft_ms": {"kind": "max", "value": mean_ttft_max_ms},
        "median_ttft_ms": loose_ms,
        "p90_ttft_ms": loose_ms,
        "p95_ttft_ms": loose_ms,
        "p99_ttft_ms": {"kind": "max", "value": p99_ttft_max_ms},
        "mean_tpot_ms": {"kind": "max", "value": mean_tpot_max_ms},
        "median_tpot_ms": loose_ms,
        "p90_tpot_ms": loose_ms,
        "p95_tpot_ms": {"kind": "max", "value": p95_tpot_max_ms},
        "p99_tpot_ms": loose_ms,
        "mean_itl_ms": loose_ms,
        "median_itl_ms": loose_ms,
        "p95_itl_ms": loose_ms,
        "p99_itl_ms": loose_ms,
        "mean_e2el_ms": loose_ms,
        "median_e2el_ms": loose_ms,
        "p90_e2el_ms": loose_ms,
        "p95_e2el_ms": loose_ms,
        "p99_e2el_ms": loose_ms,
        "success_rate": {"kind": "min", "value": success_rate_min},
        "failed": {"kind": "max", "value": failed_max},
    }


CONTAINER_HF_TOKEN_PATH = "/run/secrets/hf_token"
SECRET_ENV_KEYS = frozenset({"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"})


def _without_secret_env(env):
    if not isinstance(env, dict):
        return env
    return {key: value for key, value in env.items() if key not in SECRET_ENV_KEYS}


def _deep_merge(base, override):
    """Merge ``override`` onto ``base``. Dicts merge key-wise; lists are replaced."""
    if not (isinstance(base, dict) and isinstance(override, dict)):
        return override
    out = dict(base)
    for key, value in override.items():
        out[key] = _deep_merge(base[key], value) if key in base else value
    return out


def _token_volume_parts(entry):
    parts = str(entry).split(":")
    if len(parts) >= 2 and parts[1] == CONTAINER_HF_TOKEN_PATH:
        return parts
    return None


def bind_hf_token_volume(block, host_path):
    """Bind ``host_path`` at the container token path, always read-only.

    An existing mount of that path is rewritten to ``:ro`` so a writable mapping
    cannot leave the host credential modifiable from inside the container.
    """
    host_path = str(host_path or "").strip()
    if not host_path or not isinstance(block, dict):
        return block
    runtime = block.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        return block
    args = runtime.setdefault("args", {})
    if not isinstance(args, dict):
        return block
    volumes = []
    saw_token_mount = False
    for entry in list(args.get("volumes") or []):
        parts = _token_volume_parts(entry)
        if parts is None:
            volumes.append(entry)
            continue
        saw_token_mount = True
        options = parts[2].split(",") if len(parts) >= 3 else []
        if "ro" in options:
            volumes.append(str(entry))
            continue
        host = parts[0] or host_path
        volumes.append(f"{host}:{CONTAINER_HF_TOKEN_PATH}:ro")
    if not saw_token_mount:
        volumes.append(f"{host_path}:{CONTAINER_HF_TOKEN_PATH}:ro")
    args["volumes"] = volumes
    return block


def orchestrator_container_from_variant(variant: AtomVariantConfig) -> dict[str, Any]:
    """``container`` block for :class:`OrchestratorConfig` (includes server env)."""
    block = variant.container.model_dump()
    server_env = variant.roles.server.env
    if server_env:
        block = {**block, "env": dict(server_env)}
    if isinstance(block.get("env"), dict):
        block["env"] = _without_secret_env(block["env"])
    return block


def atom_container_block(cluster_container, variant):
    """Merge cluster and variant container config, then bind the token file.

    Secret env keys are removed after the merge. Stripping the variant first
    would leave a token set in the cluster ``container.env`` or in
    ``runtime.args.env``, and both of those become ``docker -e`` flags.
    The token volume is appended after the merge. Putting it on the variant
    block first would replace cluster fallback volumes, because list values
    are replaced rather than concatenated.
    """
    merged = _deep_merge(cluster_container or {}, orchestrator_container_from_variant(variant))
    blocks = [merged]
    runtime = merged.get("runtime") if isinstance(merged, dict) else None
    args = runtime.get("args") if isinstance(runtime, dict) else None
    if isinstance(args, dict):
        blocks.append(args)
    for block in blocks:
        if isinstance(block, dict) and isinstance(block.get("env"), dict):
            block["env"] = _without_secret_env(block["env"])
    return bind_hf_token_volume(merged, variant.paths.hf_token_file)
