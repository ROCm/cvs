'''Shared helpers for SGLang single-node and disaggregated inference libs.'''

from __future__ import annotations

import json
import re
import shlex
from typing import Any, Mapping, Callable

from cvs.lib import globals

log = globals.log

DEFAULT_GPU_MEM_THRESHOLD_MB = 5000
AMD_SMI_METRIC_CMD = "sudo amd-smi metric --json"

_SERVER_READY_RE = re.compile(
    r"server is fired up and ready to roll",
    re.I,
)


def textwrap_for_yml(msg_string: str) -> str:
    return '\n'.join([m.lstrip() for m in msg_string.split('\n')])


def as_node_list(value) -> list:
    """Normalize cluster JSON node field to a list of host strings."""
    if isinstance(value, str):
        return [value]
    return list(value)


def resolve_server_node_list(inf_dict: Mapping[str, Any]) -> list[str]:
    """Hosts for unified multi-node SGLang (not PD disagg).

    Resolution order:
    1. ``server_node_list`` when set.
    2. Union of ``prefill_node_list`` and ``decode_node_list`` (stable order).
    """
    explicit = inf_dict.get('server_node_list')
    if explicit:
        return as_node_list(explicit)
    seen: list[str] = []
    for key in ('prefill_node_list', 'decode_node_list'):
        for host in as_node_list(inf_dict.get(key) or []):
            if host not in seen:
                seen.append(host)
    if not seen:
        raise ValueError(
            'sglang_distributed requires server_node_list or at least one of '
            'prefill_node_list / decode_node_list in the inference config'
        )
    return seen


def resolve_distributed_client_host(
    inf_dict: Mapping[str, Any],
    *,
    rank0_node: str,
    benchmark_serv_node: str,
) -> str:
    """HTTP target for bench/smoke/lm-eval when the unified server spans multiple nodes."""
    explicit = inf_dict.get('client_host')
    if explicit:
        return str(explicit)
    if benchmark_serv_node == rank0_node:
        return '127.0.0.1'
    return rank0_node


def resolve_client_host(inf_dict: Mapping[str, Any], *, unified_server: bool = False) -> str:
    """HTTP target for smoke/bench/lm-eval clients running inside a container."""
    explicit = inf_dict.get('client_host')
    if explicit:
        return str(explicit)
    if unified_server:
        return '127.0.0.1'
    proxy = as_node_list(inf_dict['proxy_router_node'])[0]
    bench = as_node_list(inf_dict['benchmark_serv_node'])[0]
    if proxy == bench:
        return '127.0.0.1'
    return proxy


def _normalize_key_value_list(raw: Any, field_name: str) -> list[str]:
    """Normalize ``add_export_env`` entries to ``KEY=VALUE`` strings."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [f'{k}={v}' for k, v in raw.items()]
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            line = str(item).strip()
            if not line:
                continue
            if line.startswith('export '):
                line = line[7:].strip()
            out.append(line)
        return out
    raise ValueError(f'{field_name} must be a list or dict, got {type(raw).__name__}')


def _normalize_cli_flags(raw: Any) -> list[str]:
    """Normalize ``add_flags`` entries to extra ``launch_server`` CLI tokens."""
    if raw is None:
        return []
    if isinstance(raw, str):
        line = raw.strip()
        return [line] if line else []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    raise ValueError(f'add_flags must be a list or str, got {type(raw).__name__}')


def add_export_env_block(bp_dict: Mapping[str, Any], indent: str = '                      ') -> str:
    """Shell ``export`` lines from ``bp_dict['add_export_env']``."""
    env = _normalize_key_value_list(bp_dict.get('add_export_env'), 'add_export_env')
    return '\n'.join(f'{indent}export {entry}' for entry in env)


def add_cli_flags_block(bp_dict: Mapping[str, Any], indent: str = '                              ') -> str:
    """Extra ``launch_server`` CLI flag lines from ``bp_dict['add_flags']``."""
    flags = _normalize_cli_flags(bp_dict.get('add_flags'))
    if not flags:
        return ''
    return '\n'.join(f'{indent}{flag} \\' for flag in flags)


def first_float(pattern: str, text: str):
    m = re.search(pattern, text, re.I)
    return m.group(1) if m else None


def _is_sglang_latency_metric(metric_name: str) -> bool:
    name = metric_name.lower()
    return 'ms' in name or 'latency' in name


def _is_sglang_higher_is_better_metric(metric_name: str) -> bool:
    if _is_sglang_latency_metric(metric_name):
        return False
    name = metric_name.lower()
    return any(
        token in name
        for token in (
            'throughput',
            'goodput',
            'mfu',
            'request_throughput',
        )
    )


def normalize_sglang_threshold_spec(metric_name: str, spec: Any) -> dict[str, Any]:
    """Map threshold JSON specs (or legacy flat floats) to evaluate_all kinds."""
    if isinstance(spec, dict) and spec.get('kind'):
        return spec
    value = float(spec['value'] if isinstance(spec, dict) and 'value' in spec else spec)
    if _is_sglang_latency_metric(metric_name):
        return {'kind': 'max_ms', 'value': value}
    if _is_sglang_higher_is_better_metric(metric_name):
        kind = 'min_tok_s' if 'throughput' in metric_name.lower() else 'min'
        return {'kind': kind, 'value': value}
    return {'kind': 'min', 'value': value}


def coerce_sglang_actual(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_log_dir_cleanup_cmd(log_dir: str, user: str) -> str:
    """Shell command: rm -rf, recreate, chown (host namespace, not in-container)."""
    if not log_dir or not str(log_dir).strip():
        raise ValueError("log_dir must be a non-empty path")
    log_dir = str(log_dir).strip()
    quser = shlex.quote(str(user))
    qdir = shlex.quote(log_dir)
    return f"sudo rm -rf {qdir} && sudo mkdir -p {qdir} && sudo chown -R {quser}:{quser} {qdir}"


def cleanup_sglang_log_dir(
    orch: Any,
    log_dir: str,
    *,
    all_nodes: bool | None = None,
    timeout: int = 60,
) -> None:
    """Reset log root on cluster hosts via baremetal SSH (``orch.head`` / ``orch.all``)."""
    if all_nodes is None:
        all_nodes = len(orch.hosts) > 1
    cmd = build_log_dir_cleanup_cmd(log_dir, orch.user)
    if all_nodes:
        orch.all.exec(cmd, timeout=timeout)
    else:
        orch.head.exec(cmd, timeout=timeout)


LM_EVAL_SPECS = {
    'lm_eval_hellaswag': {
        'display': 'HellaSwag',
        'default_metric': 'acc_norm',
        'default_metric_key': 'acc_norm,none',
        'default_num_concurrent': '1',
    },
    'lm_eval_gsm8k': {
        'display': 'GSM8K',
        'default_metric': 'exact_match',
        'default_metric_key': 'exact_match,flexible-extract',
        'default_num_concurrent': '4',
    },
}


def _parse_amd_smi_gpu_entries(payload: str | None) -> list[dict]:
    """Unwrap amd-smi --json (list or {"gpu_data": [...]}) -> GPU entry list."""
    try:
        entries = json.loads((payload or "").strip())
    except (json.JSONDecodeError, AttributeError, TypeError):
        return []
    if isinstance(entries, dict):
        entries = entries.get("gpu_data", [])
    return entries if isinstance(entries, list) else []


def count_occupied_gpus_on_node(
    payload: str | None,
    *,
    mem_threshold_mb: int = DEFAULT_GPU_MEM_THRESHOLD_MB,
) -> int:
    count = 0
    for g in _parse_amd_smi_gpu_entries(payload):
        used_mb = g.get("mem_usage", {}).get("used_vram", {}).get("value", 0)
        if used_mb > mem_threshold_mb:
            count += 1
    return count


def count_occupied_gpus_per_node(
    out_dict: Mapping[str, str | None],
    *,
    mem_threshold_mb: int = DEFAULT_GPU_MEM_THRESHOLD_MB,
) -> dict[str, int]:
    per_node: dict[str, int] = {}
    for node, payload in out_dict.items():
        if payload is None:
            log.warning("No amd-smi output on node %s", node)
            per_node[node] = 0
            continue
        try:
            per_node[node] = count_occupied_gpus_on_node(payload, mem_threshold_mb=mem_threshold_mb)
        except (TypeError, ValueError, AttributeError):
            log.warning("Failed to parse amd-smi JSON on node %s", node)
            per_node[node] = 0
    return per_node


def collect_sglang_gpu_topology(
    host_exec: Callable[..., dict[str, str | None]],
    groups: Mapping[str, list[str]],
    *,
    mem_threshold_mb: int = DEFAULT_GPU_MEM_THRESHOLD_MB,
    amd_smi_cmd: str = AMD_SMI_METRIC_CMD,
    timeout: int | None = None,
) -> dict[str, Any]:
    """
    groups: e.g. {"server": [...]} or {"prefill": [...], "decode": [...]}
    host_exec: suite _host_exec(cmd, hosts=..., timeout=...)
    """
    group_stats: dict[str, dict[str, Any]] = {}
    total = 0

    for name, hosts in groups.items():
        if not hosts:
            group_stats[name] = {"per_node": {}, "total": 0}
            continue
        per_node = count_occupied_gpus_per_node(
            host_exec(amd_smi_cmd, hosts=hosts, timeout=timeout),
            mem_threshold_mb=mem_threshold_mb,
        )
        group_total = sum(per_node.values())
        group_stats[name] = {"per_node": per_node, "total": group_total}
        total += group_total

    return {"groups": group_stats, "total_occupied_gpus": total}


def format_sglang_gpu_topology_lines(
    *,
    configured_tp: int,
    configured_pp: int,
    groups: Mapping[str, dict[str, Any]],
    configured_nnodes: int | None = None,
) -> list[str]:
    lines = ["", f"Configured TP: {configured_tp}", f"Configured PP: {configured_pp}"]
    if configured_nnodes is not None:
        lines.append(f"Configured nnodes: {configured_nnodes}")
    for label, stats in groups.items():
        lines.extend(["", f"{label.title()}:"])
        for node, count in stats["per_node"].items():
            lines.append(f"  {node}: {count} occupied GPUs")
        lines.append(f"  Total: {stats['total']} occupied GPUs")
    lines.extend(["", "Total hardware GPUs consumed:", f"  {sum(s['total'] for s in groups.values())}"])
    return lines
