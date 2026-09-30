"""Unit tests for sweep bar-chart hover tooltip CSS."""

import unittest

from cvs.lib.report.render.sweep_charts import chart_tooltip_css
from cvs.lib.report.rundeck.runtime.theme import report_css


class TestChartTooltipCss(unittest.TestCase):
    """Long combo-key tooltips must stay readable instead of hiding behind a neighbour panel."""

    def setUp(self):
        self.css = chart_tooltip_css()

    def test_tooltip_wraps_instead_of_running_off_the_panel(self):
        self.assertIn("white-space: normal;", self.css)
        self.assertIn("max-width: 180px;", self.css)
        self.assertNotIn("white-space: nowrap;", self.css)

    def test_hovered_panel_is_lifted_above_sibling_panels(self):
        self.assertIn(".chart-panel:hover, .chart-panel:focus-within", self.css)
        self.assertIn("z-index: 30;", self.css)

    def test_edge_columns_anchor_tooltip_inside_the_plot(self):
        self.assertIn(".chart-col:first-child .chart-has-tip::before { left: 0; transform: none; }", self.css)
        self.assertIn(
            ".chart-col:last-child .chart-has-tip::before { left: auto; right: 0; transform: none; }",
            self.css,
        )

    def test_report_css_embeds_tooltip_rules(self):
        self.assertIn(".chart-panel:hover, .chart-panel:focus-within", report_css())


if __name__ == "__main__":
    unittest.main()
