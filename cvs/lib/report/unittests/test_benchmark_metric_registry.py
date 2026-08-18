'''Unit tests for cvs/lib/report/benchmark_metric_registry.py.'''

import html
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cvs.lib.report.benchmark_metric_registry as registry


class _FakeStash(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class TestBenchmarkMetricRegistry(unittest.TestCase):
    def setUp(self):
        registry._ROWS_BY_NODEID.clear()
        registry._SUBTEST_SUMMARY_COUNTED.clear()
        registry._SUBTEST_SUMMARY['failed'] = 0
        registry._SUBTEST_SUMMARY['passed'] = 0

    def test_record_and_fetch_benchmark_metric_rows(self):
        node = SimpleNamespace(
            stash=_FakeStash(),
            nodeid='cvs/tests/x.py::test_run_performance_benchmark_test',
        )
        rows = [
            {'node': 'n1', 'metric': 'mean_ttft_ms', 'status': 'pass'},
            {'node': 'n1', 'metric': 'mean_ttft_ms', 'status': 'fail'},
        ]
        registry.record_benchmark_metric_rows(node, rows)
        stored = registry.benchmark_metric_rows_for_nodeid(node.nodeid)
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]['metric'], 'mean_ttft_ms')

    def test_stamp_benchmark_metric_rows_on_report(self):
        report = SimpleNamespace(user_properties=[])
        rows = [{'node': 'n1', 'metric': 'goodput', 'status': 'pass'}]
        registry.stamp_benchmark_metric_rows_on_report(report, rows)
        props = dict(report.user_properties)
        self.assertIn(registry.BENCHMARK_METRIC_ROWS_USER_PROPERTY, props)
        self.assertEqual(
            props[registry.BENCHMARK_METRIC_ROWS_USER_PROPERTY][0]['metric'],
            'goodput',
        )

    def test_benchmark_subtest_summary_counts_once_per_nodeid(self):
        nodeid = 'cvs/tests/x.py::test_run_performance_benchmark_test'
        rows = [
            {'node': 'n1', 'metric': 'mean_ttft_ms', 'status': 'pass'},
            {'node': 'n1', 'metric': 'goodput', 'status': 'fail'},
        ]
        registry.record_benchmark_metric_summary(nodeid, rows)
        registry.record_benchmark_metric_summary(nodeid, rows)
        total, failed, passed = registry.benchmark_subtest_summary()
        self.assertEqual(total, 2)
        self.assertEqual(failed, 1)
        self.assertEqual(passed, 1)

    def test_mark_collapsible_result_cell_adds_class(self):
        cell = '<td class="col-result">Passed</td>'
        out = registry.mark_collapsible_result_cell(cell)
        self.assertIn('cvs-benchmark-collapsible', out)

    def test_patch_benchmark_metrics_into_html(self):
        nodeid = 'cvs/tests/inference/sglang/sglang_single.py::test_run_performance_benchmark_test'
        rows = [{'node': 'n1', 'metric': 'mean_ttft_ms', 'status': 'pass'}]
        registry._ROWS_BY_NODEID[nodeid] = rows

        payload = {
            'tests': {
                nodeid: [
                    {
                        'resultsTableRow': [
                            '<td class="col-result">Passed</td>',
                            f'<td class="col-testId">{html.escape(nodeid)}</td>',
                        ],
                        'extras': [],
                        'log': 'raw log',
                    }
                ]
            }
        }
        blob = html.escape(json.dumps(payload), quote=True)
        with tempfile.TemporaryDirectory() as tmp:
            html_path = Path(tmp) / 'report.html'
            html_path.write_text(
                '<html><body>'
                '<div class="filters"></div><div class="collapse"></div>'
                f'<div data-jsonblob="{blob}"></div>'
                '</body></html>',
                encoding='utf-8',
            )
            self.assertTrue(registry.patch_benchmark_metrics_into_html(html_path))
            updated = html_path.read_text(encoding='utf-8')
            self.assertIn('cvs-benchmark-metrics-table', updated)
            self.assertIn('cvs-subtests-count', updated)


if __name__ == '__main__':
    unittest.main()