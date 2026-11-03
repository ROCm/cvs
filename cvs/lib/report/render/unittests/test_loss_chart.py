'''Unit tests for the static Run Deck loss-curve chart.'''

import unittest

from cvs.lib.report.render.loss_chart import LossChartRenderer, loss_chart_css, render_loss_chart_html
from cvs.lib.report.rundeck.runtime.theme import report_css


class TestLossChartRenderer(unittest.TestCase):
    _CELLS = [
        {
            "cell_id": "MBS=4,GBS=128,PRECISION=FP8",
            "subtitle": "MBS=4 GBS=128 · FP8",
            "loss_curve": [[0, 2.5], [10, 2.1], [20, 1.8]],
        },
        {
            "cell_id": "MBS=4,GBS=128,PRECISION=BF16",
            "subtitle": "MBS=4 GBS=128 · BF16",
            "loss_curve": [[0, 2.6], [10, 2.2], [20, 1.9]],
        },
    ]

    def setUp(self):
        self.renderer = LossChartRenderer()

    def test_series_per_cell_with_points(self):
        series = self.renderer.build_series(self._CELLS)
        self.assertEqual([s["label"] for s in series], ["MBS=4 GBS=128 · FP8", "MBS=4 GBS=128 · BF16"])
        self.assertNotEqual(series[0]["color"], series[1]["color"])

    def test_cells_without_usable_points_are_skipped(self):
        cells = [
            {"cell_id": "no-curve"},
            {"cell_id": "single-point", "loss_curve": [[0, 2.0]]},
            {"cell_id": "junk", "loss_curve": [["a", "b"], [1, 2], [2, 1]]},
        ]
        series = self.renderer.build_series(cells)
        self.assertEqual([s["label"] for s in series], ["junk"])
        self.assertEqual(series[0]["points"], [(1.0, 2.0), (2.0, 1.0)])

    def test_render_draws_one_polyline_per_series_with_legend(self):
        markup = self.renderer.render(self._CELLS)
        self.assertEqual(markup.count("<polyline class='loss-line'"), 2)
        self.assertIn("MBS=4 GBS=128 · FP8", markup)
        self.assertIn("lm_loss versus step", markup)
        self.assertIn("loss-legend", markup)

    def test_render_is_empty_without_loss_data(self):
        self.assertEqual(self.renderer.render([{"cell_id": "c1"}]), "")
        self.assertEqual(render_loss_chart_html([]), "")

    def test_flat_curve_still_produces_a_line(self):
        markup = self.renderer.render([{"cell_id": "flat", "loss_curve": [[0, 2.0], [10, 2.0]]}])
        self.assertIn("<polyline class='loss-line'", markup)

    def test_labels_are_escaped(self):
        markup = self.renderer.render([{"cell_id": "<script>", "loss_curve": [[0, 2.0], [1, 1.0]]}])
        self.assertIn("&lt;script&gt;", markup)
        self.assertNotIn("<script>", markup)

    def test_report_css_embeds_loss_chart_rules(self):
        self.assertIn(".loss-line", loss_chart_css())
        self.assertIn(".loss-line", report_css())


if __name__ == "__main__":
    unittest.main()
