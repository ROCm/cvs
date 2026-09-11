'''Unit tests for cvs/lib/report/render/perf_metric_table.py.'''

import unittest

from cvs.lib.report.render.perf_metric_table import (
    dedupe_metric_rows,
    is_benchmark_metrics_extra,
    metric_display_label,
    render_benchmark_metrics_html,
)


class TestPerfMetricTable(unittest.TestCase):
    def test_dedupe_metric_rows_keeps_first_per_node_metric(self):
        rows = [
            {'node': 'n1', 'metric': 'mean_ttft_ms', 'status': 'pass'},
            {'node': 'n1', 'metric': 'mean_ttft_ms', 'status': 'fail'},
            {'node': 'n1', 'metric': 'goodput', 'status': 'pass'},
        ]
        out = dedupe_metric_rows(rows)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]['status'], 'pass')

    def test_dedupe_metric_rows_keeps_same_metric_from_multiple_nodes(self):
        rows = [
            {'node': 'head', 'metric': 'client.output_throughput', 'status': 'pass'},
            {'node': 'worker', 'metric': 'client.output_throughput', 'status': 'pass'},
        ]

        self.assertEqual(len(dedupe_metric_rows(rows)), 2)

    def test_metric_display_label_uses_columns_when_provided(self):
        columns = (('Mean TTFT (ms)', 'mean_ttft_ms'),)
        self.assertEqual(metric_display_label('mean_ttft_ms', columns), 'Mean TTFT (ms)')

    def test_metric_display_label_prettifies_unknown_keys(self):
        self.assertEqual(metric_display_label('goodput'), 'Goodput')

    def test_render_benchmark_metrics_html_includes_metric_statuses(self):
        columns = (('Mean TTFT (ms)', 'mean_ttft_ms'), ('Goodput', 'goodput'))
        html_out = render_benchmark_metrics_html(
            [
                {'node': 'n1', 'metric': 'mean_ttft_ms', 'status': 'pass'},
                {'node': 'n1', 'metric': 'goodput', 'status': 'fail'},
                {
                    'node': 'n1',
                    'metric': 'gpu.gpu_compute_util_pct',
                    'status': 'skip',
                    'actual': None,
                    'reason': 'metric unavailable',
                },
                {
                    'node': 'n1',
                    'metric': 'client.output_throughput',
                    'status': 'record',
                    'actual': 99,
                    'spec': {'kind': 'min_tok_s', 'value': 100},
                    'enforced': False,
                    'reason': 'threshold enforcement disabled; no threshold asserted',
                },
            ],
            columns=columns,
        )
        self.assertIn('cvs-benchmark-metrics-table', html_out)
        self.assertIn('Mean TTFT (ms)', html_out)
        self.assertNotIn('n1:', html_out)
        self.assertIn('Passed', html_out)
        self.assertIn('Failed', html_out)
        self.assertIn('Skipped', html_out)
        self.assertIn('Recorded', html_out)
        self.assertIn('reference: min_tok_s 100', html_out)
        self.assertIn('metric unavailable', html_out)

    def test_is_benchmark_metrics_extra_detects_wrapped_table(self):
        html_out = render_benchmark_metrics_html([{'node': 'n1', 'metric': 'goodput', 'status': 'pass'}])
        extra = {'format_type': 'html', 'content': html_out}
        self.assertTrue(is_benchmark_metrics_extra(extra))
        self.assertFalse(is_benchmark_metrics_extra({'format_type': 'html', 'content': '<p>x</p>'}))


if __name__ == '__main__':
    unittest.main()
