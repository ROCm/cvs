'''
Payload/render test for the TransferBench status_matrix deck.

A registered transferbench_cvs profile must render the run card and the
node x preset matrix, including a measured bandwidth that missed its threshold.
'''

import unittest

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html

_TB_RESULTS = {
    "_meta": {
        "cluster": "helios",
        "version": "—",
        "version_label": "TransferBench",
        "suite": "transferbench_cvs",
        "generated_at": "t0",
    },
    "groups": {
        "a2a": {
            "nodes": {
                "node-a": {
                    "status": "fail",
                    "items_summary": "min RTotal 90.0 GB/s",
                    "items": [
                        {
                            "name": "GPU0",
                            "status": "fail",
                            "message": "90.0 GB/s (threshold 100)",
                        }
                    ],
                    "errors_json_href": "",
                    "log_tarball_href": "",
                }
            }
        },
    },
}


class TestTransferBenchDeck(unittest.TestCase):
    def setUp(self):
        self.profile = load_json_profile("transferbench_cvs")
        store = {"cvs_results_dict": _TB_RESULTS, "inf_res_dict": _TB_RESULTS}
        self.payload = build_rundeck_payload(profile=self.profile, store=store, provenance={}, cvs_version="dev")
        self.html = render_rundeck_html(self.payload)

    def test_profile_uses_run_card_and_status_matrix_only(self):
        types = [card["type"] for card in self.profile["cards"]]
        self.assertEqual(types, ["run_card", "status_matrix"])
        self.assertEqual(self.profile["sources"]["results"], "transferbench_res_dict")
        self.assertEqual(self.profile["dataset_builder"], "status_matrix")

    def test_overall_status_is_fail(self):
        self.assertEqual(self.payload["overall_status"], "fail")

    def test_run_card_names_transferbench(self):
        self.assertIn("TransferBench", self.html)
        self.assertIn("helios", self.html)
        self.assertNotIn("ANC version", self.html)

    def test_status_matrix_shows_missed_bandwidth(self):
        self.assertIn("status-matrix", self.html)
        self.assertIn("a2a", self.html)
        self.assertIn("node-a", self.html)
        self.assertIn("90.0 GB/s", self.html)

    def test_no_inference_gate_vocabulary(self):
        self.assertNotIn("Gate matrix", self.html)
        self.assertNotIn("hover for tier status", self.html)


if __name__ == "__main__":
    unittest.main()
