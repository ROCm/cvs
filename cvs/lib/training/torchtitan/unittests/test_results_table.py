'''Unit tests for the TorchTitan metric-results table renderer.'''

import unittest

from cvs.lib.training.torchtitan.utils.results_table import (
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
        self.assertEqual(format_expected({"kind": "min", "value": 5000}), ">= 5000")
        self.assertEqual(format_expected({"kind": "max", "value": 15}), "<= 15")
        self.assertEqual(format_expected({"kind": "max_ms", "value": 3600000}), "<= 3600000 ms")
        self.assertEqual(format_expected({"kind": "info", "value": 100}), "info (100)")
        self.assertEqual(format_expected({"kind": "within", "value": 1.0, "tolerance_pct": 5}), "1.0 +/-5%")
        self.assertEqual(format_expected({"kind": "min_ratio", "value": 0.8, "reference": "base"}), ">= 0.8 x base")

    def test_format_value(self):
        self.assertEqual(format_value(None), "None")
        self.assertEqual(format_value(1.23456), "1.2346")
        self.assertEqual(format_value(9), "9")

    def test_metric_display_name_strips_prefix(self):
        self.assertEqual(metric_display_name("training.tokens_per_sec"), "tokens_per_sec")
        self.assertEqual(metric_display_name("tokens_per_sec"), "tokens_per_sec")

    def test_metric_unit_lookup(self):
        self.assertEqual(metric_unit("training.tokens_per_sec"), "tok/s/device")
        self.assertEqual(metric_unit("training.tflops"), "TFLOP/s/device")
        self.assertEqual(metric_unit("training.unknown_metric"), "-")

    def test_build_metric_row(self):
        row = build_metric_row("BF16", "training.tokens_per_sec", {"kind": "min", "value": 5000}, 12000.5, "PASS")
        self.assertEqual(
            row,
            {
                "sweep": "BF16",
                "metric": "tokens_per_sec",
                "expected": ">= 5000",
                "actual": "12000.5000",
                "unit": "tok/s/device",
                "status": "PASS",
            },
        )

    def test_render_metric_results_html_contains_rows_and_colors(self):
        rows = [
            build_metric_row("BF16", "training.tokens_per_sec", {"kind": "min", "value": 5000}, 12000.5, "PASS"),
            build_metric_row("FP8", "training.tflops", {"kind": "min", "value": 180}, 150.0, "FAIL"),
        ]
        html = render_metric_results_html(rows, "Training Metric Results (distributed)")
        self.assertIn("<h2>Training Metric Results (distributed)</h2>", html)
        self.assertIn("<th>Sweep</th>", html)
        self.assertIn("tokens_per_sec", html)
        self.assertIn("12000.5000", html)
        self.assertIn("#2e7d32", html)
        self.assertIn("#c62828", html)

    def test_build_benchmark_metric_row(self):
        spec = {"kind": "min", "value": 5000}
        row = build_benchmark_metric_row("training.tokens_per_sec", spec, 12000.5, "pass", reason="", enforced=True)
        self.assertEqual(
            row,
            {
                "node": "",
                "metric": "tokens_per_sec",
                "label": "tokens_per_sec",
                "status": "pass",
                "actual": 12000.5,
                "unit": "tok/s/device",
                "spec": spec,
                "reason": "",
                "enforced": True,
            },
        )

    def test_build_benchmark_metric_row_failure_carries_reason(self):
        row = build_benchmark_metric_row(
            "training.tflops", {"kind": "min", "value": 180}, 150.0, "fail", reason="below min", enforced=True
        )
        self.assertEqual(row["status"], "fail")
        self.assertEqual(row["reason"], "below min")
        self.assertEqual(row["unit"], "TFLOP/s/device")

    def test_render_escapes_html(self):
        rows = [build_metric_row("<b>", "training.x", None, "<script>", "N/A")]
        html = render_metric_results_html(rows, "t")
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)


if __name__ == "__main__":
    unittest.main()
