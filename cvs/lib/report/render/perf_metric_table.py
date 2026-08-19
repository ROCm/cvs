'''Pytest-html extras: collapsible per-metric pass/fail list for benchmark tests.'''

from __future__ import annotations

import html
from typing import Any, Mapping, Sequence

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
    """Keep one pass/fail row per (node, metric) pair."""
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
    """Table rows aligned with pytest-html Result / Test / Duration / Links columns."""
    body_rows = []
    for row in dedupe_metric_rows(rows):
        status = str(row.get('status') or '').lower()
        passed = status == 'pass'
        label = html.escape(metric_display_label(str(row.get('metric') or ''), columns))
        outcome = 'Passed' if passed else 'Failed'
        outcome_cls = 'passed' if passed else 'failed'
        body_rows.append(
            f"<tr class='cvs-benchmark-metric-row cvs-benchmark-metric-{outcome_cls} {outcome_cls}'>"
            f"<td class='col-result'>{outcome}</td>"
            f"<td class='col-testId'>{label}</td>"
            f"<td class='col-duration'></td>"
            f"<td class='col-links'></td>"
            f'</tr>'
        )
    return (
        f"<table class='cvs-benchmark-metrics-table {_BENCHMARK_METRICS_WRAP}'>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )


def is_benchmark_metrics_extra(extra: object) -> bool:
    """True when an pytest-html extra dict carries benchmark metric markup."""
    if not isinstance(extra, dict) or extra.get('format_type') != 'html':
        return False
    content = extra.get('content') or extra.get('content_raw') or ''
    return _BENCHMARK_METRICS_WRAP in str(content)
