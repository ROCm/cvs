'''Unit tests for Run Deck sweep chart rendering.'''

import unittest

from cvs.lib.report.rundeck.runtime.sweep_charts import SweepChartRenderer


class TestSweepChartRenderer(unittest.TestCase):
    def test_single_named_cell_renders_without_concurrency_label(self):
        cell = "MBS=2,GBS=32,PRECISION=BF16"

        rendered = SweepChartRenderer().render_bar_chart(
            "Training throughput vs cells",
            [(cell, 1200.0)],
            "tok/s",
        )

        self.assertIn("Training throughput vs cells", rendered)
        self.assertIn(cell, rendered)
        self.assertNotIn(f"C={cell}", rendered)

    def test_single_inference_concurrency_still_needs_comparison(self):
        rendered = SweepChartRenderer().render_bar_chart(
            "Output throughput",
            [(128, 1200.0)],
            "tok/s",
        )

        self.assertEqual(rendered, "")


if __name__ == "__main__":
    unittest.main()
