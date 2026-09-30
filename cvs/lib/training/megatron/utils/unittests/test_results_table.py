'''Unit tests for the Megatron metric-results table renderer.'''

import unittest

from cvs.lib.training.megatron.utils.results_table import (
    build_benchmark_metric_row,
    build_metric_row,
    format_expected,
    format_value,
    metric_display_name,
    metric_unit,
    render_metric_results_html,
)


class TestResultsTable(unittest.TestCase):
    def test_format_expected_kinds(self):
        self.assertEqual(format_expected(None), "-")
        self.assertEqual(format_expected({"kind": "min", "value": 260}), ">= 260")
        self.assertEqual(format_expected({"kind": "max", "value": 15}), "<= 15")
        self.assertEqual(format_expected({"kind": "max_ms", "value": 3600000}), "<= 3600000 ms")
        self.assertEqual(format_expected({"kind": "info", "value": 100}), "info (100)")
        self.assertEqual(
            format_expected({"kind": "within", "value": 1.0, "tolerance_pct": 5}), "1.0 +/-5%"
        )
        self.assertEqual(
            format_expected({"kind": "min_ratio", "value": 0.8, "reference": "base"}), ">= 0.8 x base"
        )

    def test_format_value(self):
        self.assertEqual(format_value(None), "None")
        self.assertEqual(format_value(1.23456), "1.2346")
        self.assertEqual(format_value(9), "9")

    def test_metric_display_name_strips_prefix(self):
        self.assertEqual(metric_display_name("training.throughput_per_gpu"), "throughput_per_gpu")
        self.assertEqual(metric_display_name("throughput_per_gpu"), "throughput_per_gpu")

    def test_metric_unit_lookup(self):
        self.assertEqual(metric_unit("training.throughput_per_gpu"), "TFLOP/s/GPU")
        self.assertEqual(metric_unit("training.tokens_per_gpu"), "tok/s/GPU")
        self.assertEqual(metric_unit("training.unknown_metric"), "-")

    def test_build_metric_row(self):
        row = build_metric_row(
            "BF16", "training.throughput_per_gpu", {"kind": "min", "value": 260}, 483.24, "PASS"
        )
        self.assertEqual(
            row,
            {
                "sweep": "BF16",
                "metric": "throughput_per_gpu",
                "expected": ">= 260",
                "actual": "483.2400",
                "unit": "TFLOP/s/GPU",
                "status": "PASS",
            },
        )

    def test_render_metric_results_html_contains_rows_and_colors(self):
        rows = [
            build_metric_row("BF16", "training.throughput_per_gpu", {"kind": "min", "value": 260}, 483.24, "PASS"),
            build_metric_row("FP8", "training.tokens_per_gpu", {"kind": "min", "value": 1836}, 1689.61, "FAIL"),
        ]
        html = render_metric_results_html(rows, "Training Metric Results (distributed)")
        self.assertIn("<h2>Training Metric Results (distributed)</h2>", html)
        self.assertIn("<th>Sweep</th>", html)
        self.assertIn("throughput_per_gpu", html)
        self.assertIn("483.2400", html)
        self.assertIn("#2e7d32", html)  # PASS color
        self.assertIn("#c62828", html)  # FAIL color

    def test_build_benchmark_metric_row(self):
        spec = {"kind": "min", "value": 260}
        row = build_benchmark_metric_row(
            "training.throughput_per_gpu", spec, 483.24, "pass", reason="", enforced=True
        )
        self.assertEqual(
            row,
            {
                "node": "",
                "metric": "throughput_per_gpu",
                "label": "throughput_per_gpu",
                "status": "pass",
                "actual": 483.24,
                "unit": "TFLOP/s/GPU",
                "spec": spec,
                "reason": "",
                "enforced": True,
            },
        )

    def test_build_benchmark_metric_row_failure_carries_reason(self):
        row = build_benchmark_metric_row(
            "training.tokens_per_gpu", {"kind": "min", "value": 1836}, 1689.6, "fail", reason="below min", enforced=True
        )
        self.assertEqual(row["status"], "fail")
        self.assertEqual(row["reason"], "below min")
        self.assertEqual(row["unit"], "tok/s/GPU")

    def test_render_escapes_html(self):
        rows = [build_metric_row("<b>", "training.x", None, "<script>", "N/A")]
        html = render_metric_results_html(rows, "t")
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)


if __name__ == "__main__":
    unittest.main()
