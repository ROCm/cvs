'''Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

from __future__ import annotations

import json

# Human-readable derived metrics exposed as HTML rows (one row per entry per cell).
# These are computed in vllm_single.py from the raw amd-smi snapshots and stored
# under "gpu.<short>" keys in inf_res_dict.
GPU_METRICS: list[tuple[str, str]] = [
    ("peak_gpu_memory_mb", "MB"),
    ("model_load_memory_mb", "MB"),
    ("model_load_s", "s"),
    ("gpu_bandwidth_util_pct", "%"),
    ("gpu_compute_util_pct", "%"),
]
GPU_METRIC_UNITS: dict[str, str] = {k: u for k, u in GPU_METRICS}

# Raw amd-smi field keys emitted by parse_gpu_metrics(). Not used as test rows.
_RAW_GPU_FIELDS: list[tuple[str, str]] = [
    ("gfx_activity", "%"),
    ("umc_activity", "%"),
    ("mm_activity", "%"),
    ("total_vram", "MB"),
    ("used_vram", "MB"),
    ("free_vram", "MB"),
    ("energy_j", "J"),
]
_RAW_GPU_FIELD_UNITS: dict[str, str] = {k: u for k, u in _RAW_GPU_FIELDS}


def _safe_get(d, *keys, default=None):
    """Navigate nested dicts safely; return default on missing key or 'N/A' value."""
    cur = d
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key, default)
        if cur is default:
            return default
    if cur == "N/A":
        return default
    return cur


def parse_usage(gpu_entry: dict) -> dict:
    """Extract activity metrics from one GPU entry dict.

    Returns: {"gpu.gfx_activity", "gpu.umc_activity", "gpu.mm_activity"}
    Values are int or None; never raises.
    """
    fields = ("gfx_activity", "umc_activity", "mm_activity")
    result = {}
    for field in fields:
        val = _safe_get(gpu_entry, "usage", field, "value")
        result[f"gpu.{field}"] = val
    return result


def parse_mem_usage(gpu_entry: dict) -> dict:
    """Extract memory usage metrics from one GPU entry dict.

    Returns: {"gpu.total_vram", "gpu.used_vram", "gpu.free_vram"}
    Values are int or None; never raises.
    """
    fields = ("total_vram", "used_vram", "free_vram")
    result = {}
    for field in fields:
        val = _safe_get(gpu_entry, "mem_usage", field, "value")
        result[f"gpu.{field}"] = val
    return result


def parse_energy(gpu_entry: dict) -> dict:
    """Extract energy consumption from one GPU entry dict.

    Returns: {"gpu.energy_j"}
    Value is float or None; never raises.
    """
    val = _safe_get(gpu_entry, "energy", "total_energy_consumption", "value")
    if val is not None:
        val = float(val)
    return {"gpu.energy_j": val}


def parse_gpu_metrics(raw: list) -> dict:
    """Aggregate all GPU entries from one host's amd-smi --json output.

    raw: the parsed JSON list (one dict per GPU per host).
    Activity metrics (%) -> averaged across GPUs (only non-None values counted).
    Memory / energy metrics -> summed across GPUs (only non-None values counted).
    Empty/missing -> all None.
    """
    all_none = {f"gpu.{k}": None for k, _u in _RAW_GPU_FIELDS}
    if not raw:
        return all_none

    activity_keys = ("gpu.gfx_activity", "gpu.umc_activity", "gpu.mm_activity")
    vram_keys = ("gpu.total_vram", "gpu.used_vram", "gpu.free_vram")
    energy_key = "gpu.energy_j"

    # Accumulators: sum and count per field (None excluded from both)
    activity_sums: dict[str, float] = {k: 0.0 for k in activity_keys}
    activity_counts: dict[str, int] = {k: 0 for k in activity_keys}
    vram_sums: dict[str, int | None] = {k: None for k in vram_keys}
    energy_sum: float | None = None

    for entry in raw:
        usage = parse_usage(entry)
        mem = parse_mem_usage(entry)
        eng = parse_energy(entry)

        for key in activity_keys:
            val = usage[key]
            if val is not None:
                activity_sums[key] += val
                activity_counts[key] += 1

        for key in vram_keys:
            val = mem[key]
            if val is not None:
                if vram_sums[key] is None:
                    vram_sums[key] = val
                else:
                    vram_sums[key] += val

        e = eng[energy_key]
        if e is not None:
            if energy_sum is None:
                energy_sum = e
            else:
                energy_sum += e

    result = {}
    for key in activity_keys:
        count = activity_counts[key]
        result[key] = (activity_sums[key] / count) if count > 0 else None

    for key in vram_keys:
        result[key] = vram_sums[key]

    result[energy_key] = energy_sum
    return result


def _try_parse(text: str) -> list:
    """Parse JSON text; return [] on empty/None/invalid JSON or non-list result.

    Accepts both bare-list format and the {"gpu_data": [...]} envelope that
    amd-smi metric --json emits on ROCm 6.x nodes.
    """
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError):
        return []
    if isinstance(parsed, dict):
        parsed = parsed.get("gpu_data", [])
    if not isinstance(parsed, list):
        return []
    return parsed


def capture_gpu_metrics(orch) -> dict:
    """One amd-smi exec on the host node. Returns flat {gpu.* metrics} dict.

    orch: ContainerOrchestrator (has .exec_on_head(cmd) -> {host: str}).
    amd-smi is a host-side tool and must run outside the container.
    Degrades gracefully on empty/unparseable JSON per host (returns all-None dict).
    """
    out = orch.exec_on_head("amd-smi metric --json")
    all_entries = []
    for _host, text in out.items():
        all_entries.extend(_try_parse(text))
    return parse_gpu_metrics(all_entries)


def _mean(values: list) -> "float | None":
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def agg_readings(readings: list) -> dict:
    """Aggregate poll readings into derived metrics.
    Returns dict with peak_gpu_memory_mb, gpu_compute_util_pct, gpu_bandwidth_util_pct.
    Any metric is None if no valid readings exist for it.

    Readings are raw snapshot dicts from capture_gpu_metrics (keys use gpu.* prefix).
    """
    used_vrams = [r.get("gpu.used_vram") for r in readings if r.get("gpu.used_vram") is not None]
    gfx_vals = [r.get("gpu.gfx_activity") for r in readings if r.get("gpu.gfx_activity") is not None]
    umc_vals = [r.get("gpu.umc_activity") for r in readings if r.get("gpu.umc_activity") is not None]
    return {
        "peak_gpu_memory_mb": max(used_vrams) if used_vrams else None,
        "gpu_compute_util_pct": _mean(gfx_vals),
        "gpu_bandwidth_util_pct": _mean(umc_vals),
    }


def poll_gpu_metrics(
    orch,
    is_done_fn,
    poll_interval_s: float = 15,
    label: str = "poll",
    log_path=None,
    max_consecutive_failures: int = 3,
    model_load_s=None,
    model_load_memory_mb=None,
) -> list:
    """Poll GPU metrics while a vLLM client is running.

    Calls capture_gpu_metrics repeatedly until is_done_fn() returns True
    or max_consecutive_failures consecutive exceptions are raised.
    Returns list of raw snapshot dicts (failed polls excluded).
    Never raises. Writes per-poll lines + summary to log_path if given.
    """
    import time
    import logging

    log = logging.getLogger(__name__)
    readings: list = []
    log_lines: list = []
    poll_n = 0
    consecutive_failures = 0

    while True:
        poll_n += 1
        try:
            snap = capture_gpu_metrics(orch)
            consecutive_failures = 0
            readings.append(snap)
            used = snap.get("gpu.used_vram")
            gfx = snap.get("gpu.gfx_activity")
            umc = snap.get("gpu.umc_activity")
            mm = snap.get("gpu.mm_activity")
            done = is_done_fn()
            done_tag = "  [done]" if done else ""
            line = f"[gpu {label} {poll_n}/?] used_vram={used} MB  gfx={gfx}%  umc={umc}%  mm={mm}%{done_tag}"
            log_lines.append(line)
            if done:
                break
        except Exception as exc:
            consecutive_failures += 1
            line = (
                f"[gpu {label} {poll_n}/?] FAILED"
                f" [{consecutive_failures}/{max_consecutive_failures} consecutive]:"
                f" {type(exc).__name__}: {exc} (skipped)"
            )
            log_lines.append(line)
            if consecutive_failures >= max_consecutive_failures:
                log.warning(
                    "poll_gpu_metrics: %d consecutive failures, stopping early",
                    consecutive_failures,
                )
                break

        time.sleep(poll_interval_s)

    # Build summary
    agg = agg_readings(readings)
    n_failed = poll_n - len(readings)
    failed_note = f" ({n_failed} failed, excluded)" if n_failed else ""
    peak = agg.get("peak_gpu_memory_mb")
    compute = agg.get("gpu_compute_util_pct")
    bw = agg.get("gpu_bandwidth_util_pct")
    ml_mem = f"{model_load_memory_mb:.0f}" if model_load_memory_mb is not None else "-"
    ml_s = f"{model_load_s:.1f}" if model_load_s is not None else "-"
    compute_s = f"{compute:.1f}" if compute is not None else "-"
    bw_s = f"{bw:.1f}" if bw is not None else "-"
    peak_s = f"{peak:.0f}" if peak is not None else "-"

    summary_lines = [
        "",
        "--- summary ---",
        f"samples:              {poll_n}{failed_note}",
        f"peak_gpu_memory_mb:   {peak_s} MB",
        f"model_load_memory_mb: {ml_mem} MB",
        f"model_load_s:         {ml_s} s",
        f"gpu_compute_util_pct:  {compute_s} %",
        f"gpu_bandwidth_util_pct: {bw_s} %",
    ]
    log_lines.extend(summary_lines)

    if log_path is not None:
        try:
            import pathlib

            pathlib.Path(log_path).write_text("\n".join(log_lines) + "\n")
        except Exception as exc:
            log.warning("poll_gpu_metrics: failed to write log %s: %s", log_path, exc)

    log.info(
        "poll_gpu_metrics: %d readings (%d failed) | peak_vram=%s MB compute=%s%% bw=%s%%",
        len(readings),
        n_failed,
        peak_s,
        compute_s,
        bw_s,
    )
    return readings
