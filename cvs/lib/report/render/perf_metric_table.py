'''Pytest-html extras: collapsible per-metric pass/fail list for benchmark tests.'''

from __future__ import annotations

import html
from typing import Any, Mapping, Sequence

from cvs.lib.report.formatting import fmt_num

_BENCHMARK_METRICS_WRAP = 'cvs-benchmark-metrics-wrap'

MetricColumn = tuple[str, str | None]


def metric_display_label(
    metric_key: str,
    columns: Sequence[MetricColumn] = (),
) -> str:
    for label, key in columns:
        if key == metric_key:
            return label
    if not metric_key:
        return ''
    return metric_key[0].upper() + metric_key[1:]


def dedupe_metric_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Keep one metric verdict row per (node, metric) pair."""
    seen: set[tuple[str, str]] = set()
    out: list[Mapping[str, Any]] = []
    for row in rows:
        key = (str(row.get('node') or ''), str(row.get('metric') or ''))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def render_benchmark_metrics_html(
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[MetricColumn] = (),
) -> str:
    """Render a collapsible pass/fail/skip/record metric-verdict table."""

    def _display_value(value: Any, unit: Any) -> str:
        if value is None:
            return '—'
        suffix = f' {unit}' if unit and unit != '-' else ''
        return f'{fmt_num(value)}{suffix}'

    def _display_gate(spec: Any, *, enforced: bool) -> str:
        if not isinstance(spec, Mapping):
            return '—'
        kind = spec.get('kind', '')
        value = spec.get('value')
        rendered = str(kind) if value is None else f'{kind} {fmt_num(value)}'
        return rendered if enforced else f'reference: {rendered}'

    body_rows = []
    for row in dedupe_metric_rows(rows):
        status = str(row.get('status') or '').lower()
        outcome, outcome_cls = {
            'pass': ('Passed', 'passed'),
            'skip': ('Skipped', 'skipped'),
            'record': ('Recorded', 'record'),
        }.get(status, ('Failed', 'failed'))
        label = str(row.get('label') or metric_display_label(str(row.get('metric') or ''), columns))
        node = row.get('node') or row.get('host')
        if node:
            label = f'{node}: {label}'
        label = html.escape(label)
        actual = html.escape(_display_value(row.get('actual'), row.get('unit')))
        gate = html.escape(_display_gate(row.get('spec'), enforced=bool(row.get('enforced', status != 'record'))))
        reason = html.escape(str(row.get('reason') or ''))
        body_rows.append(
            f"<tr class='cvs-benchmark-metric-row cvs-benchmark-metric-{outcome_cls} {outcome_cls}'>"
            f"<td class='col-result'>{outcome}</td>"
            f"<td class='col-testId'>{label}</td>"
            f"<td class='col-actual'>{actual}</td>"
            f"<td class='col-gate'>{gate}</td>"
            f"<td class='col-reason'>{reason}</td>"
            f'</tr>'
        )
    return (
        f"<table class='cvs-benchmark-metrics-table {_BENCHMARK_METRICS_WRAP}'>"
        "<thead><tr><th>Result</th><th>Metric</th><th>Actual</th><th>Gate / reference</th><th>Note</th></tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )


def is_benchmark_metrics_extra(extra: object) -> bool:
    """True when an pytest-html extra dict carries benchmark metric markup."""
    if not isinstance(extra, dict) or extra.get('format_type') != 'html':
        return False
    content = extra.get('content') or extra.get('content_raw') or ''
    return _BENCHMARK_METRICS_WRAP in str(content)
