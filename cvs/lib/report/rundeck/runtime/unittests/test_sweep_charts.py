'''Unit tests for Run Deck bar-chart x labels and point floors.'''

import unittest

from cvs.lib.report.rundeck.runtime.sweep_charts import SweepChartRenderer


class TestSweepCharts(unittest.TestCase):
    def test_inference_floor_stays_two_points(self):
        html = SweepChartRenderer().render_bar_chart("tput", [(1, 10.0)], "tok/s")
        self.assertEqual(html, "")

    def test_series_allows_single_point_without_c_prefix(self):
        html = SweepChartRenderer().render_series_chart("tput", [("MBS=4", 12.0)], "TFLOP/s", x_label="{x}")
        self.assertIn("MBS=4", html)
        self.assertNotIn("C=", html)
        self.assertIn("chart-bar", html)


if __name__ == "__main__":
    unittest.main()
