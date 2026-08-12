'''Unit tests for cvs/lib/report/render/perf_metric_table.py.'''

import unittest

from cvs.lib.report.render.perf_metric_table import (
    dedupe_metric_rows,
    is_benchmark_metrics_extra,
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

    def test_render_benchmark_metrics_html_includes_pass_and_fail_rows(self):
        html_out = render_benchmark_metrics_html(
            [
                {'node': 'n1', 'metric': 'mean_ttft_ms', 'status': 'pass'},
                {'node': 'n1', 'metric': 'goodput', 'status': 'fail'},
            ]
        )
        self.assertIn('cvs-benchmark-metrics-table', html_out)
        self.assertIn('Mean TTFT (ms)', html_out)
        self.assertIn('Passed', html_out)
        self.assertIn('Failed', html_out)

    def test_is_benchmark_metrics_extra_detects_wrapped_table(self):
        html_out = render_benchmark_metrics_html(
            [{'node': 'n1', 'metric': 'goodput', 'status': 'pass'}]
        )
        extra = {'format_type': 'html', 'content': html_out}
        self.assertTrue(is_benchmark_metrics_extra(extra))
        self.assertFalse(is_benchmark_metrics_extra({'format_type': 'html', 'content': '<p>x</p>'}))


if __name__ == '__main__':
    unittest.main()
