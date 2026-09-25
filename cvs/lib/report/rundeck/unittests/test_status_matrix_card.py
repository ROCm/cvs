'''Unit tests for the ``status_matrix`` card renderer.'''

import unittest

from cvs.lib.report.rundeck.runtime.cards import DeckCardRenderer


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


if __name__ == "__main__":
    unittest.main()
