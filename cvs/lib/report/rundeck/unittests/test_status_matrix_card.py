'''Unit tests for the ``status_matrix`` card renderer.'''

import unittest

from cvs.lib.report.rundeck.runtime.cards import DeckCardRenderer
from cvs.lib.report.rundeck.runtime.theme import report_css


def _dataset():
    return {
        "nodes": ["n1", "n2"],
        "groups": ["cpu_sanity", "hbm_lvl3"],
        "grid": {
            "n1": {
                "cpu_sanity": {
                    "status": "pass",
                    "items_summary": "Items: 1",
                    "items": [{"name": "a", "status": "pass"}],
                },
                "hbm_lvl3": {"status": "na", "items": []},
            },
            "n2": {
                "cpu_sanity": {"status": "pass", "items": []},
                "hbm_lvl3": {
                    "status": "fail",
                    "items_summary": "Items: 2 Total | 0 PASSED, 2 FAILED",
                    "items": [{"name": "ecc", "status": "fail", "message": "812 < 900"}],
                    "errors_json_href": "n2_errors.json",
                    "log_tarball_href": "n2_logs.tar.gz",
                },
            },
        },
    }


class TestStatusMatrixCard(unittest.TestCase):
    def setUp(self):
        self.renderer = DeckCardRenderer()

    def test_renders_table_with_group_headers(self):
        html = self.renderer.render_status_matrix({}, {}, _dataset())
        self.assertIn("status-matrix", html)
        self.assertIn("cpu_sanity", html)
        self.assertIn("hbm_lvl3", html)

    def test_fail_cell_has_details_and_items(self):
        html = self.renderer.render_status_matrix({}, {}, _dataset())
        self.assertIn("sm-fail", html)
        self.assertIn("<details", html)
        self.assertIn("ecc", html)
        self.assertIn("812 &lt; 900", html)  # message HTML-escaped

    def test_fail_cell_shows_artifact_links(self):
        html = self.renderer.render_status_matrix({}, {}, _dataset())
        self.assertIn("n2_errors.json", html)
        self.assertIn("n2_logs.tar.gz", html)

    def test_fail_cell_count_uses_summary_not_failure_len(self):
        # items holds only failures; the cell count must come from items_summary,
        # not len(items) -- otherwise a 10-passed/2-failed cell reads as "0/2".
        cell = {
            "status": "fail",
            "items_summary": "Items: 12 Total | 10 PASSED, 2 FAILED",
            "items": [
                {"name": "a", "status": "fail", "message": "x"},
                {"name": "b", "status": "fail", "message": "y"},
            ],
        }
        html = DeckCardRenderer._status_cell_html(cell)
        self.assertIn("10 PASSED, 2 FAILED", html)
        self.assertNotIn("0/2 items", html)

    def test_na_cell_is_not_expandable(self):
        html = self.renderer.render_status_matrix({}, {}, _dataset())
        self.assertIn("chip-na", html)
        self.assertIn("group not on node", html)

    def test_empty_dataset_message(self):
        html = self.renderer.render_status_matrix({}, {}, {"nodes": [], "groups": [], "grid": {}})
        self.assertIn("No node results", html)

    def test_item_message_only_on_fail(self):
        item_pass = DeckCardRenderer._status_item_html({"name": "x", "status": "pass", "message": "should hide"})
        self.assertNotIn("should hide", item_pass)
        item_fail = DeckCardRenderer._status_item_html({"name": "y", "status": "fail", "message": "shown"})
        self.assertIn("shown", item_fail)

    def test_card_registered_in_renderer_map(self):
        self.assertIn("status_matrix", self.renderer.card_renderers())

    def test_render_card_wraps_in_titled_section(self):
        payload = {
            "deck_profile": {
                "cards": [
                    {
                        "type": "status_matrix",
                        "id": "results",
                        "title": "Full results",
                        "bind": "datasets.status_matrix",
                    }
                ]
            },
            "datasets": {"status_matrix": _dataset()},
        }
        card = payload["deck_profile"]["cards"][0]
        section_id, html, in_nav = self.renderer.render_card(payload, card)
        self.assertEqual(section_id, "results")
        self.assertIn("Full results", html)
        self.assertTrue(in_nav)

    def test_default_hint_is_generic(self):
        html = self.renderer.render_status_matrix({}, {}, _dataset())
        self.assertIn("item breakdown", html)
        self.assertNotIn("ANC", html)

    def test_profile_hint_is_escaped(self):
        html = self.renderer.render_status_matrix({}, {"hint": "<b>Preset items</b>"}, _dataset())
        self.assertIn("&lt;b&gt;Preset items&lt;/b&gt;", html)
        self.assertNotIn("item breakdown", html)

    def test_status_overview_shows_pass_rate_and_failures(self):
        overview = {
            "pass_rate": 0.75,
            "counts": {"pass": 3, "fail": 1, "na": 2},
            "failures_by_node": [{"node": "n2", "fail": 1, "pass": 1, "na": 0}],
            "failures_by_group": [{"group": "hbm_lvl3", "fail": 1, "pass": 1, "na": 0}],
        }
        html = self.renderer.render_status_overview({}, {}, overview)
        self.assertIn("75%", html)
        self.assertIn("n2", html)
        self.assertIn("hbm_lvl3", html)
        self.assertIn("overview-table", html)

    def test_status_overview_empty(self):
        html = self.renderer.render_status_overview({}, {}, {"counts": {"pass": 0, "fail": 0, "na": 0}})
        self.assertIn("No node results", html)

    def test_metric_charts_threshold_series_and_heatmap(self):
        data = {
            "metrics": [
                {
                    "name": "rtotal",
                    "group": "a2a",
                    "unit": "GB/s",
                    "threshold": 400,
                    "direction": "higher",
                    "points": [
                        {"node": "n1", "value": 420, "status": "pass"},
                        {"node": "n2", "value": 10, "status": "fail"},
                    ],
                }
            ],
            "series": [
                {
                    "name": "power",
                    "node": "n1",
                    "unit": "W",
                    "points": [{"x": "GPU0", "y": 300}, {"x": "GPU1", "y": 280}],
                }
            ],
            "heatmaps": [
                {
                    "name": "xgmi",
                    "node": "n1",
                    "unit": "GB/s",
                    "rows": ["GPU0"],
                    "cols": ["GPU1"],
                    "values": [[48.2]],
                    "threshold": 40,
                    "direction": "higher",
                }
            ],
        }
        html = self.renderer.render_metric_charts({}, {}, data)
        self.assertIn("chart-bar-pass", html)
        self.assertIn("chart-bar-fail", html)
        self.assertIn("chart-threshold", html)
        self.assertIn("higher is better", html)
        self.assertIn("power", html)
        self.assertIn("hm-pass", html)
        self.assertIn("xgmi", html)

    def test_metric_charts_empty_message(self):
        html = self.renderer.render_metric_charts({}, {}, {"metrics": [], "series": [], "heatmaps": []})
        self.assertIn("No metric data", html)

    def test_metric_charts_hide_when_empty(self):
        card = {
            "type": "metric_charts",
            "id": "metrics",
            "title": "Metrics",
            "bind": "datasets.status_matrix.metric_charts",
            "when_empty": "hide",
        }
        payload = {"datasets": {"status_matrix": {"metric_charts": {"metrics": [], "series": [], "heatmaps": []}}}}
        section_id, html, in_nav = self.renderer.render_card(payload, card)
        self.assertEqual(section_id, "")
        self.assertEqual(html, "")
        self.assertFalse(in_nav)

    def test_health_cards_registered(self):
        renderers = self.renderer.card_renderers()
        self.assertIn("status_overview", renderers)
        self.assertIn("metric_charts", renderers)

    def test_theme_includes_overview_and_heatmap_rules(self):
        css = report_css()
        self.assertIn(".overview-split", css)
        self.assertIn(".chart-bar-pass", css)
        self.assertIn(".chart-threshold", css)
        self.assertIn(".hm-pass", css)
        self.assertIn("grid-template-columns: 1fr", css)


if __name__ == "__main__":
    unittest.main()
