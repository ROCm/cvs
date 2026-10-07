'''
Payload/render test for the host config status_matrix deck.

A registered host_configs_cvs profile must render the run card, health overview,
and the node x check matrix, hide measurements when a run has none, and must not
use inference gate vocabulary.
'''

import unittest

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.config_adapter import resolve_report_config
from cvs.lib.report.rundeck.payload import SummaryMetaApplier, build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html

_HOST_RESULTS = {
    "_meta": {
        "cluster": "helios",
        "version": "7.0.2",
        "version_label": "ROCm version",
        "suite": "host_configs_cvs",
        "generated_at": "t0",
    },
    "groups": {
        "os_release": {
            "nodes": {
                "node-a": {
                    "status": "fail",
                    "items_summary": "22.04.1",
                    "items": [
                        {
                            "name": "os_version",
                            "status": "fail",
                            "message": "Installed OS Version 22.04.1 not matching expected version 24.04 on node node-a",
                        }
                    ],
                    "errors_json_href": "",
                    "log_tarball_href": "",
                }
            }
        },
    },
}


class TestHostConfigsDeck(unittest.TestCase):
    def setUp(self):
        self.profile = load_json_profile("host_configs_cvs")
        store = {"cvs_results_dict": _HOST_RESULTS, "inf_res_dict": _HOST_RESULTS}
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
        self.assertEqual(self.profile["sources"]["results"], "host_res_dict")
        self.assertEqual(self.profile["sources"]["lifecycle"], "lifecycle")
        self.assertEqual(self.profile["dataset_builder"], "status_matrix")
        self.assertEqual(self.profile["report_basename"], "host_configs_run_deck")
        self.assertIn("Pass rate", self.html)
        self.assertIn("Columns are host configuration checks", self.html)
        self.assertIn("host_configs_run_deck_viewer.html", self.html)
        self.assertNotIn("Measurements", self.html)
        self.assertNotIn("item breakdown", self.html)

    def test_lifecycle_draws_recorded_checks(self):
        store = {
            "cvs_results_dict": _HOST_RESULTS,
            "inf_res_dict": _HOST_RESULTS,
            "lifecycle_report": {
                "health": [("os_release", 1.5, "s"), ("rocm_version", 2.0, "s"), ("nic_pcie", 0.0, "s")]
            },
        }
        payload = build_rundeck_payload(profile=self.profile, store=store, provenance={}, cvs_version="dev")
        html = render_rundeck_html(payload)
        self.assertIn("Lifecycle timeline", html)
        self.assertIn("1.5s", html)
        self.assertIn("rocm version", html)
        self.assertNotIn("nic pcie", html)

    def test_overall_status_is_fail(self):
        self.assertEqual(self.payload["overall_status"], "fail")

    def test_run_card_names_rocm_version(self):
        self.assertIn("ROCm version", self.html)
        self.assertIn("7.0.2", self.html)
        self.assertIn("helios", self.html)
        self.assertNotIn("ANC version", self.html)

    def test_status_matrix_shows_failed_check(self):
        self.assertIn("status-matrix", self.html)
        self.assertIn("os_release", self.html)
        self.assertIn("node-a", self.html)
        self.assertIn("os_version", self.html)

    def test_no_inference_gate_vocabulary(self):
        self.assertNotIn("Gate matrix", self.html)
        self.assertNotIn("hover for tier status", self.html)


if __name__ == "__main__":
    unittest.main()
