'''Regression tests for Run Deck line-chart panel rendering.'''

import unittest

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html
from cvs.lib.report.rundeck.runtime.cards import DeckCardRenderer
from cvs.lib.report.rundeck.runtime.theme import report_css
from cvs.lib.report.rundeck.unittests.test_jaxmaxtext_deck import _flat, _variant


def _series_data(names):
    return {"charts": {"bus_bw": {name: [{"label": name, "points": [("1K", 1.0), ("1M", 2.0)]}] for name in names}}}


class TestRenderLineChart(unittest.TestCase):
    def setUp(self):
        self.renderer = DeckCardRenderer()
        self.card = {"title": "Bus bandwidth", "series": {"y_field": "bus_bw", "unit": "GB/s"}}

    def test_no_truncation_under_default_cap(self):
        names = [f"collective-{i}" for i in range(5)]
        html_out = self.renderer.render_line_chart({}, self.card, _series_data(names))
        self.assertNotIn("Showing", html_out)
        self.assertEqual(html_out.count("chart-panel"), 5)

    def test_truncates_and_banners_past_default_cap(self):
        names = [f"collective-{i:02d}" for i in range(63)]
        html_out = self.renderer.render_line_chart({}, self.card, _series_data(names))
        self.assertIn(f"Showing {DeckCardRenderer.DEFAULT_MAX_LINE_CHART_SERIES} of 63 series charts", html_out)
        self.assertEqual(html_out.count("chart-panel"), DeckCardRenderer.DEFAULT_MAX_LINE_CHART_SERIES)
        self.assertIn("results table and JSON export", html_out)

    def test_max_series_override_from_profile(self):
        names = [f"collective-{i}" for i in range(5)]
        card = {"title": "Bus bandwidth", "series": {"y_field": "bus_bw", "unit": "GB/s", "max_series": 2}}
        html_out = self.renderer.render_line_chart({}, card, _series_data(names))
        self.assertIn("Showing 2 of 5 series charts", html_out)
        self.assertEqual(html_out.count("chart-panel"), 2)

    def test_entries_sorted_by_label_for_deterministic_truncation(self):
        names = ["zeta", "alpha", "mid"]
        card = {"title": "Bus bandwidth", "series": {"y_field": "bus_bw", "unit": "GB/s", "max_series": 2}}
        html_out = self.renderer.render_line_chart({}, card, _series_data(names))
        self.assertLess(html_out.index("alpha"), html_out.index("mid"))
        self.assertNotIn("zeta", html_out)

    def test_no_series_data_returns_muted_message(self):
        html_out = self.renderer.render_line_chart({}, self.card, {"charts": {}})
        self.assertEqual(html_out, "<p class='muted'>No series data.</p>")


class TestMetricChartsHeatmapHeading(unittest.TestCase):
    DATA = {"heatmaps": [{"name": "per rank", "rows": ["a"], "cols": ["0"], "values": [[1.0]]}]}

    def test_heatmap_heading_false_drops_the_subheading(self):
        self.assertIn(">Heatmaps</h3>", DeckCardRenderer().render_metric_charts({}, {}, self.DATA))
        self.assertNotIn("Heatmaps", DeckCardRenderer().render_metric_charts({}, {"heatmap_heading": False}, self.DATA))


class TestRenderTable(unittest.TestCase):
    TABLE = {"headers": ["Test", "BW"], "rows": [["write", 1.0]]}

    def test_max_height_scrolls_under_a_sticky_header(self):
        out = DeckCardRenderer.render_table({}, {"max_height": "480px"}, self.TABLE)
        self.assertIn("<div class='results-wrap results-scroll' style='max-height:480px;overflow-y:auto'>", out)
        self.assertIn(".results-scroll .results-table th { position: sticky; top: 0;", report_css())

    def test_decks_without_max_height_do_not_scroll(self):
        self.assertEqual(
            DeckCardRenderer.render_table({}, {}, self.TABLE).split("<table")[0], "<div class='results-wrap'>"
        )
        payload = build_rundeck_payload(
            profile=load_json_profile("jaxmaxtext"),
            store={"cvs_results_dict": _flat(), "variant_config": _variant(), "lifecycle_report": {}},
        )
        document = render_rundeck_html(payload)
        self.assertIn("Full results", document)
        self.assertNotIn("results-scroll'", document)
        self.assertNotIn("max-height:", document)
        self.assertNotIn("class='th-unit'", document)


if __name__ == "__main__":
    unittest.main()
