'''
Payload/render test for the RVS status_matrix deck.

A registered rvs_cvs profile must render the run card and the node x module
matrix, including a failed module's pattern, and must not use inference gate
vocabulary.
'''

import unittest

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.payload import build_rundeck_payload
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

    def test_profile_uses_run_card_and_status_matrix_only(self):
        types = [card["type"] for card in self.profile["cards"]]
        self.assertEqual(types, ["run_card", "status_matrix"])
        self.assertEqual(self.profile["sources"]["results"], "rvs_res_dict")
        self.assertEqual(self.profile["dataset_builder"], "status_matrix")

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


if __name__ == "__main__":
    unittest.main()
