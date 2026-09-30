'''
Payload/render test for the RVS status_matrix deck.

A registered rvs_cvs profile must render the run card, health overview, and
the node x module matrix, hide measurements when a run has none, and must not
use inference gate vocabulary.
'''

import copy
import unittest

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.config_adapter import resolve_report_config
from cvs.lib.report.rundeck.payload import SummaryMetaApplier, build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html

_RVS_RESULTS = {
    "_meta": {
        "cluster": "helios",
        "version": "1.3.0",
        "version_label": "RVS version",
        "suite": "rvs_cvs",
        "generated_at": "t0",
    },
    "groups": {
        "gst_single": {
            "nodes": {
                "node-a": {
                    "status": "fail",
                    "items_summary": "1 failure pattern(s)",
                    "items": [{"name": r"\[ERROR\s*\]", "status": "fail", "message": "matched failure pattern"}],
                    "errors_json_href": "",
                    "log_tarball_href": "",
                }
            }
        },
    },
}


class TestRvsDeck(unittest.TestCase):
    def setUp(self):
        self.profile = load_json_profile("rvs_cvs")
        store = {"cvs_results_dict": _RVS_RESULTS, "inf_res_dict": _RVS_RESULTS}
        self.payload = build_rundeck_payload(profile=self.profile, store=store, provenance={}, cvs_version="dev")
        self.html = render_rundeck_html(self.payload)

    def test_profile_cards_overview_and_viewer(self):
        types = [card["type"] for card in self.profile["cards"]]
        self.assertEqual(types, ["run_card", "status_overview", "metric_charts", "status_matrix"])
        metrics = next(card for card in self.profile["cards"] if card["type"] == "metric_charts")
        self.assertEqual(metrics["bind"], "datasets.status_matrix.metric_charts")
        self.assertEqual(metrics["when_empty"], "hide")
        self.assertTrue(self.profile["interactive_viewer"])
        self.assertEqual(self.profile["sources"]["results"], "rvs_res_dict")
        self.assertEqual(self.profile["dataset_builder"], "status_matrix")
        self.assertIn("Pass rate", self.html)
        self.assertIn("LEVEL run", self.html)
        linked_payload = SummaryMetaApplier(resolve_report_config(self.profile)).apply(copy.deepcopy(self.payload))
        linked = render_rundeck_html(linked_payload)
        self.assertIn("rvs_run_deck_viewer.html", linked)
        self.assertNotIn("No metric data recorded", self.html)
        self.assertNotIn("Measurements", self.html)
        self.assertNotIn("item breakdown", self.html)

    def test_overall_status_is_fail(self):
        self.assertEqual(self.payload["overall_status"], "fail")

    def test_run_card_names_rvs_version(self):
        self.assertIn("RVS version", self.html)
        self.assertIn("1.3.0", self.html)
        self.assertIn("helios", self.html)
        self.assertNotIn("ANC version", self.html)

    def test_status_matrix_shows_failed_module(self):
        self.assertIn("status-matrix", self.html)
        self.assertIn("gst_single", self.html)
        self.assertIn("node-a", self.html)
        self.assertIn("matched failure pattern", self.html)

    def test_no_inference_gate_vocabulary(self):
        self.assertNotIn("Gate matrix", self.html)
        self.assertNotIn("hover for tier status", self.html)

    def test_measurements_render_when_present(self):
        results = {
            "_meta": _RVS_RESULTS["_meta"],
            "groups": {
                "gst_single": {
                    "nodes": {
                        "node-a": {
                            "status": "pass",
                            "items_summary": "passed",
                            "items": [],
                            "metrics": [
                                {
                                    "name": "fp8",
                                    "value": 1299650,
                                    "unit": "GFLOPS",
                                    "threshold": 983000,
                                    "direction": "higher",
                                    "status": "pass",
                                    "group": "gst",
                                }
                            ],
                            "heatmaps": [
                                {
                                    "name": "xgmi",
                                    "unit": "GB/s",
                                    "rows": ["GPU2", "GPU5"],
                                    "cols": ["GPU2", "GPU5"],
                                    "values": [[None, 101.231], [99.1, None]],
                                    "direction": "higher",
                                }
                            ],
                        }
                    }
                }
            },
        }
        store = {"cvs_results_dict": results, "inf_res_dict": results}
        payload = build_rundeck_payload(profile=self.profile, store=store, provenance={}, cvs_version="dev")
        html = render_rundeck_html(payload)
        self.assertIn("Measurements", html)
        self.assertIn("fp8", html)
        self.assertIn("GFLOPS", html)
        self.assertIn("xgmi", html)
        self.assertNotIn("Gate matrix", html)


if __name__ == "__main__":
    unittest.main()
