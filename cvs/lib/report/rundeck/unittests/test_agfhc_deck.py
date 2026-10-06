'''
Payload/render test for the AGFHC status_matrix deck.

A registered agfhc_cvs profile must render the run card, health overview, and
the node x recipe matrix, hide measurements when a run has none, and must not
use inference gate vocabulary.
'''

import unittest

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.config_adapter import resolve_report_config
from cvs.lib.report.rundeck.payload import SummaryMetaApplier, build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html

_AGFHC_RESULTS = {
    "_meta": {
        "cluster": "helios",
        "version": "1.24.1",
        "version_label": "AGFHC version",
        "suite": "agfhc_cvs",
        "generated_at": "t0",
    },
    "groups": {
        "hbm": {
            "nodes": {
                "node-a": {
                    "status": "fail",
                    "items_summary": "0 pass, 1 fail, 0 skipped",
                    "items": [
                        {
                            "name": "hbm_bw",
                            "status": "fail",
                            "message": "hbm_bw                                  failed",
                        }
                    ],
                    "errors_json_href": "",
                    "log_tarball_href": "",
                }
            }
        },
    },
}


class TestAgfhcDeck(unittest.TestCase):
    def setUp(self):
        self.profile = load_json_profile("agfhc_cvs")
        store = {"cvs_results_dict": _AGFHC_RESULTS, "inf_res_dict": _AGFHC_RESULTS}
        self.payload = SummaryMetaApplier(resolve_report_config(self.profile)).apply(
            build_rundeck_payload(profile=self.profile, store=store, provenance={}, cvs_version="dev")
        )
        self.html = render_rundeck_html(self.payload)

    def test_profile_cards_overview_and_viewer(self):
        types = [card["type"] for card in self.profile["cards"]]
        self.assertEqual(
            types,
            ["run_card", "lifecycle_timeline", "status_overview", "metric_charts", "status_matrix"],
        )
        metrics = next(card for card in self.profile["cards"] if card["type"] == "metric_charts")
        self.assertEqual(metrics["bind"], "datasets.status_matrix.metric_charts")
        self.assertEqual(metrics["when_empty"], "hide")
        self.assertTrue(self.profile["interactive_viewer"])
        self.assertEqual(self.profile["sources"]["results"], "agfhc_res_dict")
        self.assertEqual(self.profile["sources"]["lifecycle"], "lifecycle")
        self.assertEqual(self.profile["dataset_builder"], "status_matrix")
        self.assertIn("Pass rate", self.html)
        self.assertIn("Columns are executed AGFHC recipes", self.html)
        self.assertIn("agfhc_run_deck_viewer.html", self.html)
        self.assertNotIn("Measurements", self.html)
        self.assertNotIn("item breakdown", self.html)

    def test_lifecycle_draws_recorded_recipes(self):
        store = {
            "cvs_results_dict": _AGFHC_RESULTS,
            "inf_res_dict": _AGFHC_RESULTS,
            "lifecycle_report": {"health": [("hbm", 40.0, "s"), ("xgmi_lvl1", 12.5, "s"), ("all_lvl5", 0.0, "s")]},
        }
        payload = build_rundeck_payload(profile=self.profile, store=store, provenance={}, cvs_version="dev")
        html = render_rundeck_html(payload)
        self.assertIn("Lifecycle timeline", html)
        self.assertIn("40.0s", html)
        self.assertIn("xgmi lvl1", html)
        self.assertNotIn("all lvl5", html)

    def test_overall_status_is_fail(self):
        self.assertEqual(self.payload["overall_status"], "fail")

    def test_run_card_names_agfhc_version(self):
        self.assertIn("AGFHC version", self.html)
        self.assertIn("1.24.1", self.html)
        self.assertIn("helios", self.html)
        self.assertNotIn("ANC version", self.html)

    def test_status_matrix_shows_failed_recipe(self):
        self.assertIn("status-matrix", self.html)
        self.assertIn("hbm", self.html)
        self.assertIn("node-a", self.html)
        self.assertIn("hbm_bw", self.html)

    def test_no_inference_gate_vocabulary(self):
        self.assertNotIn("Gate matrix", self.html)
        self.assertNotIn("hover for tier status", self.html)


if __name__ == "__main__":
    unittest.main()
