'''Shared helpers for SGLang single-node and disaggregated inference libs.'''

from __future__ import annotations

import re
from typing import Any, Mapping


def textwrap_for_yml(msg_string: str) -> str:
    return '\n'.join([m.lstrip() for m in msg_string.split('\n')])


def as_node_list(value) -> list:
    """Normalize cluster JSON node field to a list of host strings."""
    if isinstance(value, str):
        return [value]
    return list(value)


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
