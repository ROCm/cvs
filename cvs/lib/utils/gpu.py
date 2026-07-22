'''Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

from __future__ import annotations

import json
import logging
import pathlib
import re
import shlex
import time
from dataclasses import dataclass

# Sentinel line delimiting per-iteration amd-smi chunks in the remote poller's
# output file (raw amd-smi --json output is multi-line/pretty-printed, not NDJSON).
_RECORD_SEP = "===GPU_POLL_RECORD_SEP==="

# Human-readable derived metrics exposed as HTML rows (one row per entry per cell).
# These are computed by the calling suite from the raw amd-smi snapshots and stored
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


def capture_gpu_metrics(orch, nodes=None, timeout_s=None) -> dict:
    """One amd-smi exec on the node(s). Returns flat {gpu.* metrics} dict.

    amd-smi runs fine from inside the benchmark container -- GPU device
    files (/dev/kfd, /dev/dri) are passed through, so this uses the same
    orch.exec_on_head()/orch.exec() calls as every other command in the
    suite (server launch, client run, log tailing, etc.), with no special
    host-vs-container routing needed.

    Single-node (nodes=None): orch must have .exec_on_head(cmd) -> {host: str}.
    Multi-node (nodes provided, incl. []): nodes is a list of (label, hosts)
    pairs where hosts is a list of hostnames passed to
    orch.exec(cmd, hosts=hosts) -> {host: str}. nodes=[] is a valid
    "zero nodes" case: no exec call is made and all fields come back None, the
    same no-op result an empty raw list produces. All nodes' GPU entries are
    merged before aggregation. Return type is identical in both cases.

    timeout_s: optional timeout (seconds) passed through to orch.exec/
    exec_on_head. None means no timeout (blocks until the remote call
    returns), matching this function's historical behavior.

    Exceptions from exec calls (including a timeout firing) propagate to the
    caller.
    """
    all_entries = []
    if nodes is None:
        kwargs = {"timeout": timeout_s} if timeout_s is not None else {}
        out = orch.exec_on_head("amd-smi metric --json", **kwargs)
        for _host, text in out.items():
            all_entries.extend(_try_parse(text))
    else:
        kwargs = {"timeout": timeout_s} if timeout_s is not None else {}
        for _label, hosts in nodes:
            out = orch.exec("amd-smi metric --json", hosts=hosts, **kwargs)
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


def _node_label_tag(nodes) -> str:
    """Return '+'-joined node labels for log line tagging, or empty string."""
    if not nodes:
        return ""
    return "[" + "+".join(lbl for lbl, _hosts in nodes) + "] "


def _capture_multi_node(orch, nodes, timeout_s=None) -> "tuple[dict, dict[str, int | None]]":
    """One amd-smi exec per (label, hosts) pair.

    Returns (merged_snapshot, per_node_vram) computed from a single exec round:
    merged_snapshot is parse_gpu_metrics() over every node's GPU entries combined
    (same shape as capture_gpu_metrics), per_node_vram is {label: used_vram_mb}.

    Degrades per label: if orch.exec raises (including a timeout_s
    firing) for a node, that label's entries are excluded from the merge and
    its per-node VRAM is None.
    """
    all_entries = []
    per_node_vram: "dict[str, int | None]" = {}
    kwargs = {"timeout": timeout_s} if timeout_s is not None else {}
    for label, hosts in nodes:
        try:
            out = orch.exec("amd-smi metric --json", hosts=hosts, **kwargs)
            node_entries = []
            for _host, text in out.items():
                node_entries.extend(_try_parse(text))
            all_entries.extend(node_entries)
            snap = parse_gpu_metrics(node_entries)
            per_node_vram[label] = snap.get("gpu.used_vram")
        except Exception:
            per_node_vram[label] = None
    return parse_gpu_metrics(all_entries), per_node_vram


@dataclass
class GpuPollerHandle:
    """Handle returned by start_gpu_poller; opaque to callers other than
    passing it back into stop_and_collect_gpu_poller."""

    run_id: str
    marker: str
    nodes: "list[str] | None"
    paths: "str | dict[str, str]"


def _sanitize_run_id(run_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)


def _poller_script(marker: str, poll_interval_s: float, max_iterations: int) -> str:
    log_path = f"/tmp/{marker}.log"
    return (
        "#!/bin/bash\n"
        f"for i in $(seq 1 {max_iterations}); do\n"
        f"  amd-smi metric --json >> {shlex.quote(log_path)} 2>/dev/null\n"
        f"  echo {shlex.quote(_RECORD_SEP)} >> {shlex.quote(log_path)}\n"
        f"  sleep {poll_interval_s}\n"
        "done\n"
    )


def start_gpu_poller(
    orch,
    run_id: str,
    poll_interval_s: float = 15,
    nodes: "list[str] | None" = None,
    hard_cap_s: float = 14400,
) -> GpuPollerHandle:
    """Launch a detached remote background script that repeatedly snapshots
    amd-smi metrics to a file on each node, avoiding a second OS thread
    sharing the orchestrator's SSH transport with the main polling thread.

    run_id: arbitrary string (e.g. a pytest node id); sanitized into the
    marker/file name. poll_interval_s: seconds between amd-smi calls.
    nodes: None for single-node (head only, via orch.exec_on_head); a list
    of hostnames for multi-node (one script launched per host via
    orch.exec(cmd, hosts=[host])). hard_cap_s: orphan-safety backstop -- the
    remote script self-terminates after hard_cap_s // poll_interval_s
    iterations even if stop_and_collect_gpu_poller is never called.

    Raises if the launch exec call(s) raise.
    """
    sanitized = _sanitize_run_id(run_id)
    marker = f"cvs_gpu_poll_{sanitized}"
    max_iterations = int(hard_cap_s // poll_interval_s)
    script = _poller_script(marker, poll_interval_s, max_iterations)
    script_path = f"/tmp/{marker}.sh"
    write_cmd = "bash -c " + shlex.quote(f"printf '%s' {shlex.quote(script)} > {script_path}")
    launch_cmd = "bash -c " + shlex.quote(f"nohup bash {script_path} > /dev/null 2>&1 &")

    if nodes is None:
        orch.exec_on_head(write_cmd)
        orch.exec_on_head(launch_cmd)
        paths: "str | dict[str, str]" = f"/tmp/{marker}.log"
    else:
        for host in nodes:
            orch.exec(write_cmd, hosts=[host])
            orch.exec(launch_cmd, hosts=[host])
        paths = {host: f"/tmp/{marker}.log" for host in nodes}

    return GpuPollerHandle(run_id=sanitized, marker=marker, nodes=nodes, paths=paths)


def _split_chunks(text: str) -> list:
    """Split raw poller-log text on _RECORD_SEP, stripping exactly one
    well-terminated trailing phantom chunk if the file ends with the
    separator."""
    if not text:
        return []
    chunks = text.split(_RECORD_SEP)
    if chunks and chunks[-1].strip() == "":
        chunks = chunks[:-1]
    return chunks


def stop_and_collect_gpu_poller(
    orch,
    handle: GpuPollerHandle,
    log_path=None,
    model_load_s=None,
    model_load_memory_mb=None,
) -> list:
    """Stop the remote poller launched by start_gpu_poller and collect its
    readings.

    Never raises for orch-transport failures (the stop broadcast or the
    file read-back) -- those are caught, logged, and degrade the affected
    host's contribution rather than propagating, so this is safe to call
    from a `finally` block during in-flight exception handling. Returns a
    list of raw snapshot dicts in the same shape capture_gpu_metrics()
    returns (failed/malformed polls excluded). Writes a summary block
    (compatible with agg_readings()) to log_path if given.
    """
    log = logging.getLogger(__name__)
    pkill_cmd = "bash -c " + shlex.quote(f"pkill -f {handle.marker} || true")
    try:
        if handle.nodes is None:
            orch.exec_on_head(pkill_cmd)
        else:
            orch.exec(pkill_cmd, hosts=handle.nodes)
    except Exception as exc:
        log.warning("stop_and_collect_gpu_poller: pkill broadcast failed: %s", exc)

    log_lines: list = []
    node_tag = _node_label_tag([(h, [h]) for h in handle.nodes] if handle.nodes else None)

    if handle.nodes is None:
        text = None
        try:
            out = orch.exec_on_head(f"cat {shlex.quote(handle.paths)}")
            text = next(iter(out.values()), "")
        except Exception as exc:
            log.warning("stop_and_collect_gpu_poller: read-back failed: %s", exc)
            text = ""

        chunks = _split_chunks(text)
        poll_n = len(chunks)
        readings: list = []
        for i, chunk in enumerate(chunks, start=1):
            entries = _try_parse(chunk)
            if not entries:
                log_lines.append(f"[gpu poll {i}/{poll_n}] FAILED: empty/malformed chunk (skipped)")
                continue
            snap = parse_gpu_metrics(entries)
            readings.append(snap)
            used = snap.get("gpu.used_vram")
            gfx = snap.get("gpu.gfx_activity")
            umc = snap.get("gpu.umc_activity")
            mm = snap.get("gpu.mm_activity")
            log_lines.append(f"[gpu poll {i}/{poll_n}] used_vram={used} MB  gfx={gfx}%  umc={umc}%  mm={mm}%")
        n_failed = poll_n - len(readings)
        node_last_vram: "dict[str, int | None]" = {}
    else:
        host_chunks: "dict[str, list]" = {}
        for host in handle.nodes:
            path = handle.paths[host] if isinstance(handle.paths, dict) else handle.paths
            text = ""
            try:
                out = orch.exec(f"cat {shlex.quote(path)}", hosts=[host])
                text = next(iter(out.values()), "")
            except Exception as exc:
                log.warning("stop_and_collect_gpu_poller: read-back failed for %s: %s", host, exc)
                text = ""
            host_chunks[host] = _split_chunks(text)

        n_rounds = max((len(v) for v in host_chunks.values()), default=0)
        readings = []
        node_last_vram = {host: None for host in handle.nodes}
        n_failed = 0
        for i in range(n_rounds):
            round_entries: list = []
            round_hosts: list = []
            for host in handle.nodes:
                chunks = host_chunks[host]
                if i >= len(chunks):
                    continue
                entries = _try_parse(chunks[i])
                if entries:
                    round_entries.extend(entries)
                    round_hosts.append(host)
            if not round_entries:
                n_failed += 1
                log_lines.append(f"[gpu poll {i + 1}/{n_rounds}] {node_tag}FAILED: all nodes malformed (skipped)")
                continue
            snap = parse_gpu_metrics(round_entries)
            readings.append(snap)
            for host in round_hosts:
                host_entries = _try_parse(host_chunks[host][i])
                host_snap = parse_gpu_metrics(host_entries)
                vram = host_snap.get("gpu.used_vram")
                if vram is not None:
                    node_last_vram[host] = vram
            used = snap.get("gpu.used_vram")
            gfx = snap.get("gpu.gfx_activity")
            umc = snap.get("gpu.umc_activity")
            mm = snap.get("gpu.mm_activity")
            log_lines.append(
                f"[gpu poll {i + 1}/{n_rounds}] {node_tag}used_vram={used} MB  gfx={gfx}%  umc={umc}%  mm={mm}%"
            )
        poll_n = n_rounds

    agg = agg_readings(readings)
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
    if handle.nodes:
        summary_lines.append("--- per-node vram (last reading) ---")
        for host in handle.nodes:
            vram = node_last_vram.get(host)
            vram_s = f"{vram}" if vram is not None else "-"
            summary_lines.append(f"node_vram_mb [{host}]: {vram_s} MB")
    log_lines.extend(summary_lines)

    if log_path is not None:
        try:
            pathlib.Path(log_path).write_text("\n".join(log_lines) + "\n")
        except Exception as exc:
            log.warning("stop_and_collect_gpu_poller: failed to write log %s: %s", log_path, exc)

    log.info(
        "stop_and_collect_gpu_poller: %d readings (%d failed) | peak_vram=%s MB compute=%s%% bw=%s%%",
        len(readings),
        n_failed,
        peak_s,
        compute_s,
        bw_s,
    )
    return readings
