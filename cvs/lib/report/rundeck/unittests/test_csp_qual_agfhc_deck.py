'''
Payload/render test for the CSP Qual AGFHC status_matrix deck.

A registered csp_qual_agfhc profile must render the run card, health overview,
and the node x recipe matrix, hide measurements when a run has none, and must
not use inference gate vocabulary.
'''

import unittest

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.config_adapter import resolve_report_config
from cvs.lib.report.rundeck.payload import SummaryMetaApplier, build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html

_CSP_RESULTS = {
    "_meta": {
        "cluster": "helios",
        "version": "1.24.1",
        "version_label": "AGFHC version",
        "suite": "csp_qual_agfhc",
        "generated_at": "t0",
    },
    "groups": {
        "hbm_lvl5": {
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


class TestCspQualAgfhcDeck(unittest.TestCase):
    def setUp(self):
        self.profile = load_json_profile("csp_qual_agfhc")
        store = {"cvs_results_dict": _CSP_RESULTS, "inf_res_dict": _CSP_RESULTS}
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
        self.assertEqual(metrics["when_empty"], "hide")
        self.assertTrue(self.profile["interactive_viewer"])
        self.assertEqual(self.profile["sources"]["results"], "agfhc_res_dict")
        self.assertEqual(self.profile["dataset_builder"], "status_matrix")
        self.assertEqual(self.profile["suite_id"], "csp_qual_agfhc")
        self.assertIn("Pass rate", self.html)
        self.assertIn("Columns are executed CSP qualification steps", self.html)
        self.assertIn("csp_qual_agfhc_run_deck_viewer.html", self.html)
        self.assertNotIn("Measurements", self.html)

    def test_lifecycle_draws_recorded_recipes(self):
        store = {
            "cvs_results_dict": _CSP_RESULTS,
            "inf_res_dict": _CSP_RESULTS,
            "lifecycle_report": {
                "health": [("version_check", 1.5, "s"), ("hbm_lvl5", 7200.0, "s"), ("all_perf", 0.0, "s")]
            },
        }
        payload = build_rundeck_payload(profile=self.profile, store=store, provenance={}, cvs_version="dev")
        html = render_rundeck_html(payload)
        self.assertIn("Lifecycle timeline", html)
        self.assertIn("7200.0s", html)
        self.assertIn("version check", html)
        self.assertNotIn("all perf", html)

    def test_overall_status_is_fail(self):
        self.assertEqual(self.payload["overall_status"], "fail")

    def test_run_card_names_agfhc_version(self):
        self.assertIn("AGFHC version", self.html)
        self.assertIn("1.24.1", self.html)
        self.assertIn("csp_qual_agfhc", self.html)
        self.assertNotIn("ANC version", self.html)

    def test_status_matrix_shows_failed_recipe(self):
        self.assertIn("status-matrix", self.html)
        self.assertIn("hbm_lvl5", self.html)
        self.assertIn("node-a", self.html)
        self.assertIn("hbm_bw", self.html)

    def test_no_inference_gate_vocabulary(self):
        self.assertNotIn("Gate matrix", self.html)
        self.assertNotIn("hover for tier status", self.html)


if __name__ == "__main__":
    unittest.main()
