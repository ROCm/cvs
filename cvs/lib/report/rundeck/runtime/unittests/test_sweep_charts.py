'''Regression tests for inference bars and RCCL line charts.'''

import unittest

from cvs.lib.report.rundeck.runtime.sweep_charts import SweepChartRenderer
from cvs.lib.report.rundeck.runtime.theme import report_css


class TestSweepChartRenderer(unittest.TestCase):
    def setUp(self):
        self.renderer = SweepChartRenderer()

    def test_inference_bar_defaults_still_require_two_concurrency_points(self):
        self.assertEqual(self.renderer.render_bar_chart("throughput", [(1, 2)], "tok/s"), "")
        doc = self.renderer.render_bar_chart("throughput", [(1, 2), (2, 3)], "tok/s")
        self.assertIn("C=1", doc)
        self.assertIn("C=2", doc)
        self.assertIn("chart-bar-accent", doc)
        self.assertNotIn("<polyline", doc)

    def test_dense_bar_chart_drops_ticks_and_keeps_hover(self):
        points = [(index, float(index + 1)) for index in range(9)]
        doc = self.renderer.render_bar_chart("TFLOP/s/GPU", points, "TFLOP/s/GPU")
        self.assertNotIn("chart-xlbl", doc)
        self.assertIn("chart-bars-dense", doc)
        self.assertIn("C=1: 2.0 TFLOP/s/GPU", doc)
        self.assertEqual(doc.count("chart-bar "), 9)

    def test_custom_bar_labels_and_single_point(self):
        doc = self.renderer.render_bar_chart("size", [("1K", 2)], "GB/s", x_label="{x}", min_points=1)
        self.assertIn("1K: 2.0 GB/s", doc)
        self.assertNotIn("C=", doc)

    def test_series_lines_and_single_point(self):
        self.assertEqual(self.renderer.render_series_chart("size", [], "GB/s"), "")
        for points in ([("1K", 2)], [("1K", 2), ("1M", 3)]):
            with self.subTest(points=points):
                doc = self.renderer.render_series_chart("size", points, "GB/s")
                self.assertIn("<polyline", doc)
                self.assertEqual(doc.count("<circle"), len(points))
                self.assertIn("1K: 2.0 GB/s", doc)
                self.assertNotIn("C=", doc)

    def test_chart_text_is_escaped_and_invalid_label_format_falls_back(self):
        for label_format in ("{x}", "{missing}"):
            with self.subTest(label_format=label_format):
                doc = self.renderer.render_series_chart("<title>", [("<size>", 0)], "<unit>", x_label=label_format)
                self.assertIn("&lt;size&gt;", doc)
                self.assertIn("&lt;unit&gt;", doc)
                self.assertNotIn("<size>", doc)

    def test_short_chart_tick_lines_still_wrap(self):
        css = report_css()
        start = css.index(".chart-xlbl-line")
        rule = css[start : css.index("}", start)]
        self.assertIn("overflow-wrap: anywhere", rule)
        self.assertNotIn("text-overflow", rule)


if __name__ == "__main__":
    unittest.main()
