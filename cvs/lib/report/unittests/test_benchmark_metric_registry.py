'''Unit tests for cvs/lib/report/benchmark_metric_registry.py.'''

import html
import json
import re
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
        registry._COLUMNS_BY_NODEID.clear()
        registry._SUBTEST_SUMMARY_COUNTED.clear()
        registry._SUBTEST_SUMMARY['failed'] = 0
        registry._SUBTEST_SUMMARY['passed'] = 0
        registry._SUBTEST_SUMMARY['skipped'] = 0
        registry._SUBTEST_SUMMARY['recorded'] = 0

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

    def test_record_benchmark_metric_rows_stores_columns(self):
        node = SimpleNamespace(
            stash=_FakeStash(),
            nodeid='cvs/tests/x.py::test_run_performance_benchmark_test',
        )
        columns = (('Mean TTFT (ms)', 'mean_ttft_ms'),)
        rows = [{'node': 'n1', 'metric': 'mean_ttft_ms', 'status': 'pass'}]
        registry.record_benchmark_metric_rows(node, rows, columns=columns)
        self.assertEqual(
            registry.benchmark_metric_columns_for_nodeid(node.nodeid),
            columns,
        )

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
            {'node': 'n1', 'metric': 'queue_time_p95_ms', 'status': 'skip'},
            {'node': 'n1', 'metric': 'output_throughput', 'status': 'record'},
        ]
        registry.record_benchmark_metric_summary(nodeid, rows)
        registry.record_benchmark_metric_summary(nodeid, rows)
        total, failed, passed, skipped, recorded = registry.benchmark_subtest_summary()
        self.assertEqual(total, 4)
        self.assertEqual(failed, 1)
        self.assertEqual(passed, 1)
        self.assertEqual(skipped, 1)
        self.assertEqual(recorded, 1)

    def test_mark_collapsible_result_cell_adds_class(self):
        cell = '<td class="col-result">Passed</td>'
        out = registry.mark_collapsible_result_cell(cell)
        self.assertIn('cvs-benchmark-collapsible', out)

    def test_patch_benchmark_metrics_into_html(self):
        nodeid = 'cvs/tests/inference/sglang/sglang_single.py::test_run_performance_benchmark_test'
        rows = [{'node': 'n1', 'metric': 'mean_ttft_ms', 'status': 'pass'}]
        columns = (('Mean TTFT (ms)', 'mean_ttft_ms'),)
        registry._ROWS_BY_NODEID[nodeid] = rows
        registry._COLUMNS_BY_NODEID[nodeid] = columns

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
            self.assertIn('Mean TTFT (ms)', updated)
            self.assertIn('cvs-subtests-count', updated)

    def test_patch_accepts_vllm_verification_parent(self):
        nodeid = 'cvs/tests/inference/vllm/vllm_single.py::test_verify_cell_metrics[1k1k-conc16]'
        registry._ROWS_BY_NODEID[nodeid] = [
            {
                'node': 'n1',
                'metric': 'client.output_throughput',
                'status': 'record',
                'actual': 99,
                'spec': {'kind': 'min_tok_s', 'value': 100},
                'enforced': False,
            }
        ]
        full_log = {'format_type': 'url', 'name': 'Full Log', 'content': 'test_html/full.html'}
        payload = {
            'tests': {
                nodeid: [
                    {
                        'resultsTableRow': [
                            '<td class="col-result">Skipped</td>',
                            f'<td class="col-testId">{html.escape(nodeid)}</td>',
                        ],
                        'extras': [full_log, dict(full_log)],
                        'log': 'raw log',
                    }
                ]
            }
        }
        blob = html.escape(json.dumps(payload), quote=True)
        with tempfile.TemporaryDirectory() as tmp:
            html_path = Path(tmp) / 'report.html'
            html_path.write_text(
                '<html><body><div class="filters"></div><div class="collapse"></div>'
                f'<div data-jsonblob="{blob}"></div></body></html>',
                encoding='utf-8',
            )
            self.assertTrue(
                registry.patch_benchmark_metrics_into_html(
                    html_path,
                    benchmark_test_name='test_verify_cell_metrics',
                )
            )
            self.assertTrue(
                registry.patch_benchmark_metrics_into_html(
                    html_path,
                    benchmark_test_name='test_verify_cell_metrics',
                )
            )
            updated = html_path.read_text(encoding='utf-8')
            self.assertIn('Recorded', updated)
            self.assertIn('0 Failed', updated)
            self.assertIn('1 Recorded', updated)
            self.assertIn('cvs-subtests-count', updated)
            self.assertEqual(updated.count('cvs-subtests-count'), 1)

            match = re.search(r'data-jsonblob="([^"]*)"', updated)
            patched_payload = json.loads(html.unescape(match.group(1)))
            entry = patched_payload['tests'][nodeid][0]
            self.assertEqual(entry['log'], '')
            self.assertEqual(
                sum(extra.get('name') == 'Full Log' for extra in entry['extras']),
                1,
            )
            self.assertEqual(
                sum(registry.is_benchmark_metrics_extra(extra) for extra in entry['extras']),
                1,
            )


if __name__ == '__main__':
    unittest.main()
