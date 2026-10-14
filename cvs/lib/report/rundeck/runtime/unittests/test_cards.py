'''Regression tests for Run Deck line-chart panel rendering.'''

import unittest

from cvs.lib.report.rundeck.runtime.cards import DeckCardRenderer


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


class TestMetricChartOverlay(unittest.TestCase):
    def test_log_x_zero_based_panel_overlays_band_dash_marker_and_single_point(self):
        axes = {"log_x": True, "y_min": 0, "x_ticks": [[2, "2"], [8192, "8K"]], "x_title": "Message size (bytes)"}
        series = [
            {
                **axes,
                "name": "ib_write_bw",
                "unit": "Gb/s",
                "label": "QP 8 median",
                "y_title": "Gb/s per NIC",
                "points": [[2, 1.0], [1024, 150.0], [8192, 380.0]],
                "band": [[2, 0.5, 1.5], [1024, 140.0, 160.0], [8192, 190.0, 392.0]],
                "markers": [[8192, 300.0, "threshold 300"]],
            },
            {"name": "ib_write_bw", "unit": "Gb/s", "label": "QP 8 slowest", "dash": True, "points": [[2, 0.5]]},
            {"name": "ib_write_bw", "unit": "Gb/s", "label": "QP 16 median", "points": [[8192, 390.0]]},
        ]

        out = DeckCardRenderer().render_metric_charts({}, {}, {"series": series})

        self.assertEqual(out.count("class='chart-panel'"), 1)
        self.assertIn("cx='310.00'", out)
        for text in (">0</text>", ">400</text>", ">8K</text>", "Message size (bytes)", "Gb/s per NIC"):
            self.assertIn(text, out)
        self.assertIn("<polygon", out)
        self.assertIn("stroke-dasharray='4 3'", out)
        self.assertIn("<title>threshold 300</title>", out)
        self.assertIn("QP 16 median · 8,192: 390.0 Gb/s", out)
        self.assertIn("max-width:640px", out)
        self.assertNotIn("chart-unit", out)

    def test_log_y_ticks_are_decades(self):
        series = [{"name": "lat", "unit": "us", "log_x": True, "log_y": True, "points": [[2, 2.0], [65536, 820.0]]}]
        out = DeckCardRenderer().render_metric_charts({}, {}, {"series": series})
        for text in (">1</text>", ">10</text>", ">100</text>", ">1,000</text>"):
            self.assertIn(text, out)

    def test_heatmap_cell_dict_sets_label_status_and_tip(self):
        heat = {
            "name": "bw",
            "rows": ["node-a"],
            "cols": ["NIC 0", "NIC 1"],
            "values": [[{"value": 49.0, "label": "49%", "status": "fail", "tip": "node-a NIC 0"}, 99.0]],
            "threshold": 90,
            "direction": "higher",
        }
        out = DeckCardRenderer().render_metric_charts({}, {}, {"heatmaps": [heat]})
        self.assertNotIn("Heatmaps", out)
        self.assertIn("<td class='hm-cell hm-fail' style='' title='node-a NIC 0'>49%</td>", out)
        self.assertIn("<td class='hm-cell hm-pass' style=''>99.0</td>", out)


class TestSummaryAndCollapse(unittest.TestCase):
    def test_summary_cards_show_verdict_and_tone(self):
        data = {
            "status": "fail",
            "verdict": "1 of 4 checks failed",
            "cards": [{"label": "Slowest NIC", "value": 193, "unit": "Gb/s", "meta": "node-a NIC 3", "tone": "fail"}],
        }
        out = DeckCardRenderer.render_summary_cards({}, {}, data)
        self.assertIn("<span class='chip chip-fail'>fail</span> 1 of 4 checks failed", out)
        self.assertIn("<div class='summary-stat summary-stat-fail'>193 <span class='headline-unit'>Gb/s</span>", out)
        self.assertEqual(DeckCardRenderer.render_summary_cards({}, {}, {"cards": []}), "")

    def test_collapsed_card_folds_body_into_details(self):
        payload = {"tbl": {"headers": ["a"], "rows": [[1]]}}
        _sid, out, _nav = DeckCardRenderer().render_card(
            payload, {"type": "table", "id": "t", "bind": "tbl", "collapsed": "Bandwidth rows"}
        )
        self.assertIn("<details class='card-collapse'><summary>Bandwidth rows</summary>", out)
        self.assertNotIn("<details", DeckCardRenderer().render_card(payload, {"type": "table", "bind": "tbl"})[1])

    def test_table_card_empty_message(self):
        out = DeckCardRenderer.render_table({}, {"empty_message": "No weak NICs."}, {"headers": ["a"], "rows": []})
        self.assertIn("No weak NICs.", out)


if __name__ == "__main__":
    unittest.main()
