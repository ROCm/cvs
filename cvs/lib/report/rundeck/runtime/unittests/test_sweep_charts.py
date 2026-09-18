"""Unit tests for Run Deck chart rendering."""

import unittest

from cvs.lib.report.rundeck.runtime.sweep_charts import SweepChartRenderer


class TestSweepChartRenderer(unittest.TestCase):
    def test_series_chart_renders_single_point_with_axis_label(self):
        document = SweepChartRenderer().render_series_chart(
            "FLUX",
            [(1.048576, 1.25)],
            "s/output",
            x_label="Resolution (MP)",
        )

        self.assertIn("FLUX", document)
        self.assertIn("Resolution (MP)=1.0", document)
        self.assertIn("1.2 s/output", document)

    def test_empty_series_does_not_render(self):
        self.assertEqual(SweepChartRenderer().render_series_chart("empty", [], "s"), "")


if __name__ == "__main__":
    unittest.main()
