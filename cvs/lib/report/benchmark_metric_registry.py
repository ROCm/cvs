'''Shared storage for benchmark metric pass/fail rows used by pytest-html hooks.

Metric rows must live in a normal import path (``cvs.lib...``), not only on the
sglang ``conftest`` module. Pytest loads directory ``conftest.py`` files under a
different module name than ``cvs.tests.inference.sglang.conftest``, so a dict on
the conftest module is invisible to hooks while ``record_benchmark_metric_rows()``
writes to the package import path.
'''

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Sequence

from _pytest.stash import StashKey

from cvs.lib.report.render.perf_metric_table import (
    MetricColumn,
    dedupe_metric_rows,
    is_benchmark_metrics_extra,
    render_benchmark_metrics_html,
)

BENCHMARK_METRIC_ROWS_KEY = StashKey[list[dict[str, Any]]]()
BENCHMARK_METRIC_ROWS_USER_PROPERTY = 'cvs_benchmark_metric_rows'
DEFAULT_BENCHMARK_TEST_NAME = 'test_run_performance_benchmark_test'

_ROWS_BY_NODEID: dict[str, list[dict[str, Any]]] = {}
_COLUMNS_BY_NODEID: dict[str, Sequence[MetricColumn]] = {}
_SUBTEST_SUMMARY = {'failed': 0, 'passed': 0, 'skipped': 0}
_SUBTEST_SUMMARY_COUNTED: set[str] = set()


def record_benchmark_metric_rows(
    node,
    rows: list[dict[str, Any]],
    *,
    columns: Sequence[MetricColumn] = (),
) -> None:
    """Persist metric pass/fail rows for pytest-html (read back after the test call)."""
    deduped = dedupe_metric_rows(rows)
    node.stash[BENCHMARK_METRIC_ROWS_KEY] = deduped
    _ROWS_BY_NODEID[node.nodeid] = deduped
    if columns:
        _COLUMNS_BY_NODEID[node.nodeid] = tuple(columns)


def benchmark_metric_columns_for_nodeid(nodeid: str) -> Sequence[MetricColumn]:
    return _COLUMNS_BY_NODEID.get(nodeid) or ()


def benchmark_metric_rows_for_nodeid(nodeid: str) -> list[dict[str, Any]]:
    return list(_ROWS_BY_NODEID.get(nodeid) or [])


def all_benchmark_metric_rows() -> dict[str, list[dict[str, Any]]]:
    return {nodeid: list(rows) for nodeid, rows in _ROWS_BY_NODEID.items() if rows}


def benchmark_metric_rows_from_report(report) -> list[dict[str, Any]]:
    for key, value in getattr(report, 'user_properties', ()) or ():
        if key == BENCHMARK_METRIC_ROWS_USER_PROPERTY and value:
            return list(value)
    return benchmark_metric_rows_for_nodeid(report.nodeid)


def benchmark_metric_rows_from_item(item) -> list[dict[str, Any]]:
    rows = item.stash.get(BENCHMARK_METRIC_ROWS_KEY, None)
    if rows:
        return list(rows)
    return benchmark_metric_rows_for_nodeid(item.nodeid)


def stamp_benchmark_metric_rows_on_report(report, rows: list[dict[str, Any]]) -> None:
    """Copy rows onto the call report so pytest-html row hooks can read them."""
    deduped = dedupe_metric_rows(rows)
    props = [(k, v) for k, v in report.user_properties if k != BENCHMARK_METRIC_ROWS_USER_PROPERTY]
    props.append((BENCHMARK_METRIC_ROWS_USER_PROPERTY, deduped))
    report.user_properties[:] = props


def record_benchmark_metric_summary(nodeid: str, rows: list[dict[str, Any]]) -> None:
    if nodeid in _SUBTEST_SUMMARY_COUNTED:
        return
    _SUBTEST_SUMMARY_COUNTED.add(nodeid)
    for row in dedupe_metric_rows(rows):
        status = str(row.get('status') or '').lower()
        if status == 'pass':
            _SUBTEST_SUMMARY['passed'] += 1
        elif status == 'skip':
            _SUBTEST_SUMMARY['skipped'] += 1
        else:
            _SUBTEST_SUMMARY['failed'] += 1


def benchmark_subtest_summary() -> tuple[int, int, int, int]:
    failed = _SUBTEST_SUMMARY['failed']
    passed = _SUBTEST_SUMMARY['passed']
    skipped = _SUBTEST_SUMMARY['skipped']
    return failed + passed + skipped, failed, passed, skipped


def benchmark_metrics_extra(
    rows: list[dict[str, Any]],
    *,
    columns: Sequence[MetricColumn] = (),
) -> dict[str, Any]:
    from pytest_html import extras

    return extras.html(render_benchmark_metrics_html(rows, columns=columns))


def _is_full_log_extra(extra: object) -> bool:
    return isinstance(extra, dict) and extra.get('format_type') == 'url' and extra.get('name') == 'Full Log'


def _result_cell_html(entry: dict[str, Any]) -> str:
    rows = entry.get('resultsTableRow') or []
    return str(rows[0]) if rows else ''


def _entry_test_id(entry: dict[str, Any]) -> str:
    rows = entry.get('resultsTableRow') or []
    if len(rows) < 2:
        return ''
    match = re.search(r'class="col-testId">([^<]+)', str(rows[1]))
    return html.unescape(match.group(1)) if match else ''


def _is_main_call_entry(entry: dict[str, Any], nodeid: str) -> bool:
    test_id = _entry_test_id(entry)
    return test_id == nodeid


def _entry_outcome(entry: dict[str, Any]) -> str:
    cell = _result_cell_html(entry)
    if 'Failed' in cell:
        return 'failed'
    if 'Passed' in cell:
        return 'passed'
    if 'Skipped' in cell:
        return 'skipped'
    return 'other'


def _mark_collapsible_result_cell(result_cell: str) -> str:
    if 'cvs-benchmark-collapsible' in result_cell:
        return result_cell
    if 'class="col-result' in result_cell:
        return result_cell.replace(
            'class="col-result',
            'class="col-result cvs-benchmark-collapsible',
            1,
        )
    return result_cell


def mark_collapsible_result_cell(result_cell: str) -> str:
    return _mark_collapsible_result_cell(result_cell)


def _apply_benchmark_entry_patch(
    entry: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    columns: Sequence[MetricColumn] = (),
) -> None:
    extras: list[dict[str, Any]] = []
    for extra in entry.get('extras') or []:
        if _is_full_log_extra(extra):
            extras.append(extra)
        elif is_benchmark_metrics_extra(extra):
            extras.append(extra)
    if not any(is_benchmark_metrics_extra(e) for e in extras):
        extras.append(benchmark_metrics_extra(rows, columns=columns))
    entry['extras'] = extras
    entry['log'] = ''

    result_cell = _result_cell_html(entry)
    if result_cell:
        entry['resultsTableRow'][0] = _mark_collapsible_result_cell(result_cell)


def _dedupe_benchmark_call_entries(
    entries: list[dict[str, Any]],
    nodeid: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    main_calls = [entry for entry in entries if _is_main_call_entry(entry, nodeid)]
    if len(main_calls) <= 1:
        return entries, {}

    failed = [entry for entry in main_calls if _entry_outcome(entry) == 'failed']
    passed = [entry for entry in main_calls if _entry_outcome(entry) == 'passed']
    skipped = [entry for entry in main_calls if _entry_outcome(entry) == 'skipped']
    retained = failed[:1] or passed[:1] or skipped[:1] or main_calls[:1]
    retained_ids = set(map(id, retained))
    dropped = [entry for entry in main_calls if id(entry) not in retained_ids]
    if not dropped:
        return entries, {}
    dropped_ids = set(map(id, dropped))
    counts = {}
    for entry in dropped:
        outcome = _entry_outcome(entry)
        counts[outcome] = counts.get(outcome, 0) + 1
    return [entry for entry in entries if id(entry) not in dropped_ids], counts


def _adjust_outcome_counts_in_html(content: str, decrements: dict[str, int]) -> str:
    """Drop report counts for subtest rows removed from pytest-html's result data."""
    total = sum(decrements.values())
    if total <= 0:
        return content

    for outcome, label in (
        ('passed', 'Passed'),
        ('failed', 'Failed'),
        ('skipped', 'Skipped'),
    ):
        decrement = decrements.get(outcome, 0)
        if decrement <= 0:
            continue

        def _decrement(match: re.Match[str], decrement: int = decrement, label: str = label) -> str:
            value = max(0, int(match.group(1)) - decrement)
            return f'<span class="{outcome}">{value} {label},</span>'

        content = re.sub(
            rf'<span class="{outcome}">(\d+) {label},</span>',
            _decrement,
            content,
            count=1,
        )

    def _dec_run_count(match: re.Match[str]) -> str:
        value = max(0, int(match.group(1)) - total)
        label = 'tests' if value != 1 else 'test'
        return f'<p class="run-count">{value} {label} took {match.group(3)}</p>'

    return re.sub(
        r'<p class="run-count">(\d+) (tests|test) took ([^<]+)</p>',
        _dec_run_count,
        content,
        count=1,
    )


def _subtests_filter_summary_html(total: int, failed: int, passed: int, skipped: int) -> str:
    failed_cls = 'failed' if failed else 'filter'
    passed_cls = 'passed' if passed else 'filter'
    skipped_cls = 'skipped' if skipped else 'filter'
    return (
        '<span class="filter"> | </span>'
        f'<span class="filter cvs-subtests-count">{total} subtests,</span>'
        f'<span class="{failed_cls}"> {failed} Failed,</span>'
        f'<span class="{passed_cls}"> {passed} Passed,</span>'
        f'<span class="{skipped_cls}"> {skipped} Skipped</span>'
    )


def _strip_legacy_subtest_summary(content: str) -> str:
    content = re.sub(
        r'<span class="filter"> \| \d+ subtests ran</span>.*?passed</span>',
        '',
        content,
    )
    content = re.sub(
        r'<span class="filter"> \| </span>\s*'
        r'<span class="filter cvs-subtests-count">\d+ subtests,</span>.*?Passed</span>',
        '',
        content,
    )
    content = re.sub(
        r'<span class="filter"> \| </span>\s*'
        r'<span class="filter">\d+ subtests,</span>.*?Passed</span>',
        '',
        content,
    )
    return content


def _inject_subtest_summary_into_filters(content: str, total: int, failed: int, passed: int, skipped: int) -> str:
    content = _strip_legacy_subtest_summary(content)
    summary_html = _subtests_filter_summary_html(total, failed, passed, skipped)
    filters_match = re.search(r'(<div class="filters">.*?)(</div>\s*<div class="collapse">)', content, re.DOTALL)
    if not filters_match:
        return content
    if summary_html.strip() in filters_match.group(1):
        return content
    return content.replace(filters_match.group(0), f'{filters_match.group(1)}{summary_html}{filters_match.group(2)}', 1)


def _strip_filter_metrics_summary(content: str) -> str:
    return _strip_legacy_subtest_summary(content)


def _nodeid_test_name(nodeid: str) -> str:
    return nodeid.rsplit('::', 1)[-1].split('[', 1)[0]


def patch_benchmark_metrics_into_html(
    html_path: Path | str,
    *,
    benchmark_test_name: str = DEFAULT_BENCHMARK_TEST_NAME,
) -> bool:
    """Patch pytest-html JSON so benchmark rows expose the collapsible metric panel."""
    path = Path(html_path)
    if not path.is_file():
        return False

    content = path.read_text(encoding='utf-8')
    json_pattern = r'data-jsonblob="([^"]*)"'
    match = re.search(json_pattern, content)
    if not match:
        return False

    data = json.loads(html.unescape(match.group(1)))
    tests: dict[str, list[dict[str, Any]]] = data.get('tests') or {}
    patched = False
    dropped_outcomes: dict[str, int] = {}

    for nodeid, rows in all_benchmark_metric_rows().items():
        if _nodeid_test_name(nodeid) != benchmark_test_name:
            continue
        entries = tests.get(nodeid)
        if not entries:
            continue

        record_benchmark_metric_summary(nodeid, rows)
        entries, counts = _dedupe_benchmark_call_entries(entries, nodeid)
        for outcome, count in counts.items():
            dropped_outcomes[outcome] = dropped_outcomes.get(outcome, 0) + count
        tests[nodeid] = entries

        for entry in entries:
            if not _is_main_call_entry(entry, nodeid):
                continue
            _apply_benchmark_entry_patch(
                entry,
                rows,
                columns=benchmark_metric_columns_for_nodeid(nodeid),
            )
            patched = True

    data['tests'] = tests
    total, failed, passed, skipped = benchmark_subtest_summary()
    if total:
        content = _inject_subtest_summary_into_filters(content, total, failed, passed, skipped)
        patched = True

    if dropped_outcomes:
        content = _adjust_outcome_counts_in_html(content, dropped_outcomes)
        patched = True

    if not patched:
        return False

    encoded = html.escape(json.dumps(data), quote=True)
    updated = re.sub(json_pattern, lambda _m: f'data-jsonblob="{encoded}"', content)
    path.write_text(updated, encoding='utf-8')
    return True
