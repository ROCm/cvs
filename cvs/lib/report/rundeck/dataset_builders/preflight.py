'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

Preflight operator-summary dataset builder.
'''

from cvs.lib.preflight.report import preflight_check_display_name
from cvs.lib.report.rundeck.dataset_builders.registry import register_dataset_builder


_FAILED_STATUSES = {'FAIL', 'BLOCKED', 'ERROR', 'UNKNOWN'}
_WARNING_STATUSES = {'WARN', 'WARNING'}
_STATUS_ORDER = {
    'FAIL': 0,
    'BLOCKED': 1,
    'ERROR': 2,
    'UNKNOWN': 3,
    'WARNING': 4,
    'WARN': 4,
}


def _normalize_status(value):
    status = str(value or 'UNKNOWN').strip().upper()
    if status == 'OK':
        return 'PASS'
    if status == 'WARN':
        return 'WARNING'
    return status


def _context_value(context, key, default=None):
    if isinstance(context, dict):
        return context.get(key, default)
    return getattr(context, key, default)


def _result_for_check(results, check_name):
    aliases = {
        'ssh_reachability': 'node_reachability',
        'node_smoke_tier2': 'node_smoke_tier1',
    }
    result_key = aliases.get(check_name, check_name)
    result = results.get(result_key)
    if result is None and check_name == 'node_smoke_tier1':
        result = results.get('node_smoke')
    if result is None and check_name == 'node_smoke_tier3':
        result = results.get('tier3_info')
    return result or {}


def _node_result_map(check_name, result):
    if check_name in {'node_health', 'node_smoke_tier1', 'node_smoke_tier2', 'node_smoke_tier3'}:
        return result.get('node_results') or {}
    if check_name == 'ifoe_l2_connectivity':
        return result.get('node_results') or {}
    if check_name == 'transferbench_smoke':
        return result.get('nodes') or {}
    if check_name == 'rdma_connectivity':
        return result.get('node_status') or {}
    if check_name in {'gid_consistency', 'rocm_versions', 'interface_names'}:
        return result
    return {}


def _rdma_node_status(node_result):
    status = node_result.get('status')
    if status:
        return _normalize_status(status)
    if int(node_result.get('failed_tests') or 0) > 0:
        return 'FAIL'
    if int(node_result.get('successful_tests') or 0) > 0:
        return 'PASS'
    return 'NOT RUN'


def _node_status(check_name, result, node, overall_status):
    if overall_status in {'SKIPPED', 'BLOCKED'}:
        return overall_status
    if check_name == 'ssh_reachability':
        unreachable = set(result.get('unreachable_nodes') or [])
        return 'FAIL' if node in unreachable else 'PASS'

    node_results = _node_result_map(check_name, result)
    node_result = node_results.get(node)
    if not isinstance(node_result, dict):
        return 'NOT RUN' if node_results else overall_status
    if check_name == 'rdma_connectivity':
        return _rdma_node_status(node_result)
    return _normalize_status(node_result.get('status'))


def _short_reason(check_summary):
    reason = ' '.join(str(check_summary.get('summary') or 'No summary available').split())
    if len(reason) <= 180:
        return reason
    return reason[:177].rstrip() + '...'


def _affected_nodes(check_summary):
    nodes = []
    for key in ('failed_nodes', 'missing_nodes', 'incomplete_nodes', 'unknown_nodes', 'warning_nodes'):
        nodes.extend(check_summary.get(key) or [])
    return ', '.join(dict.fromkeys(str(node) for node in nodes)) or '—'


def _headline_table(checks):
    statuses = [_normalize_status(check.get('status')) for check in checks.values()]
    passed = statuses.count('PASS')
    failed = sum(status in _FAILED_STATUSES for status in statuses)
    skipped = statuses.count('SKIPPED')
    warnings = sum(status in _WARNING_STATUSES for status in statuses)
    return {
        'headers': ['Passed', 'Failed', 'Skipped', 'Warnings', 'Total'],
        'rows': [[passed, failed, skipped, warnings, len(statuses)]],
    }


def _matrix_table(results, checks, nodes):
    headers = ['Check', 'Overall', *nodes]
    rows = []
    for check_name, check_summary in checks.items():
        overall_status = _normalize_status(check_summary.get('status'))
        result = _result_for_check(results, check_name)
        statuses = [_node_status(check_name, result, node, overall_status) for node in nodes]
        rows.append([preflight_check_display_name(check_name), overall_status, *statuses])
    return {'headers': headers, 'rows': rows}


def _failures_table(checks):
    rows = []
    for check_name, check_summary in checks.items():
        status = _normalize_status(check_summary.get('status'))
        if status not in _FAILED_STATUSES | _WARNING_STATUSES:
            continue
        rows.append(
            [
                status,
                preflight_check_display_name(check_name),
                _affected_nodes(check_summary),
                _short_reason(check_summary),
            ]
        )
    rows.sort(key=lambda row: (_STATUS_ORDER.get(row[0], 99), row[1]))
    return {
        'headers': ['Status', 'Check', 'Affected nodes', 'Reason'],
        'rows': rows,
    }


@register_dataset_builder('preflight')
def build_preflight_datasets(sources, _profile):
    results = sources.get('results') or sources.get('cvs_results_dict') or {}
    context = sources.get('variant') or {}
    summary = results.get('summary') or {}
    checks = summary.get('checks') or {}
    nodes = sorted(str(node) for node in (_context_value(context, 'cluster_nodes', []) or []))
    overall = _normalize_status(summary.get('overall_status'))
    overall_status = 'pass' if overall == 'PASS' else ('fail' if overall == 'FAIL' else 'na')

    return {
        'overall_status': overall_status,
        'headline': _headline_table(checks),
        'matrix': _matrix_table(results, checks, nodes),
        'failures': _failures_table(checks),
    }
