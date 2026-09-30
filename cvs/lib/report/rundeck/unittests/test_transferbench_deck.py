'''
Payload/render test for the TransferBench status_matrix deck.

A registered transferbench_cvs profile renders the run card, health overview,
bandwidth highlights, and the node x preset matrix. The bandwidth card hides
when a node recorded no charts.
'''

import copy
import html
import unittest

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.config_adapter import resolve_report_config
from cvs.lib.report.rundeck.payload import SummaryMetaApplier, build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html

_TB_RESULTS = {
    "_meta": {
        "cluster": "helios",
        "version": "1.67.00",
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
                    "metrics": [
                        {
                            "name": "GPU00",
                            "value": 90.0,
                            "unit": "GB/s",
                            "threshold": 100.0,
                            "direction": "higher",
                            "status": "fail",
                        }
                    ],
                    "series": [
                        {
                            "name": "RTotal",
                            "unit": "GB/s",
                            "points": [{"x": "GPU00", "y": 90.0}],
                        }
                    ],
                    "heatmaps": [
                        {
                            "name": "XGMI",
                            "unit": "GB/s",
                            "rows": ["GPU00", "GPU01"],
                            "cols": ["GPU00", "GPU01"],
                            "values": [[None, 40.3], [41.2, None]],
                            "threshold": 43.65,
                            "direction": "higher",
                        }
                    ],
                }
            }
        },
    },
}


class TestTransferBenchDeck(unittest.TestCase):
    def setUp(self):
        self.profile = load_json_profile("transferbench_cvs")
        store = {"cvs_results_dict": _TB_RESULTS, "inf_res_dict": _TB_RESULTS}
        self.payload = SummaryMetaApplier(resolve_report_config(self.profile)).apply(
            build_rundeck_payload(profile=self.profile, store=store, provenance={}, cvs_version="dev")
        )
        self.html = render_rundeck_html(self.payload)

    def test_profile_cards_and_viewer(self):
        types = [card["type"] for card in self.profile["cards"]]
        self.assertEqual(types, ["run_card", "status_overview", "metric_charts", "status_matrix"])
        self.assertEqual(self.profile["cards"][1]["bind"], "datasets.status_matrix.overview")
        self.assertEqual(self.profile["cards"][2]["bind"], "datasets.status_matrix.metric_charts")
        self.assertEqual(self.profile["cards"][2]["when_empty"], "hide")
        self.assertEqual(self.profile["sources"]["results"], "transferbench_res_dict")
        self.assertEqual(self.profile["dataset_builder"], "status_matrix")
        self.assertTrue(self.profile["interactive_viewer"])

    def test_overall_status_is_fail(self):
        self.assertEqual(self.payload["overall_status"], "fail")

    def test_run_card_names_transferbench(self):
        self.assertIn("TransferBench", self.html)
        self.assertIn("helios", self.html)
        self.assertIn("1.67.00", self.html)
        self.assertNotIn("ANC version", self.html)

    def test_overview_and_bandwidth_highlights(self):
        self.assertIn("Health overview", self.html)
        self.assertIn("Pass rate", self.html)
        self.assertIn("Bandwidth highlights", self.html)
        self.assertIn("GPU00", self.html)
        self.assertIn("higher is better", self.html)
        self.assertIn("RTotal", self.html)
        self.assertIn("XGMI", self.html)
        self.assertIn("transferbench_run_deck_viewer.html", self.html)

    def test_status_matrix_shows_missed_bandwidth(self):
        self.assertIn("status-matrix", self.html)
        self.assertIn("a2a", self.html)
        self.assertIn("node-a", self.html)
        self.assertIn("90.0 GB/s", self.html)
        self.assertIn(html.escape(self.profile["cards"][-1]["hint"]), self.html)

    def test_bandwidth_card_hides_without_charts(self):
        bare = copy.deepcopy(_TB_RESULTS)
        node = bare["groups"]["a2a"]["nodes"]["node-a"]
        for key in ("metrics", "series", "heatmaps"):
            node.pop(key, None)
        store = {"cvs_results_dict": bare, "inf_res_dict": bare}
        payload = build_rundeck_payload(profile=self.profile, store=store, provenance={}, cvs_version="dev")
        html = render_rundeck_html(payload)
        self.assertNotIn("Bandwidth highlights", html)
        self.assertIn("Health overview", html)
        self.assertIn("90.0 GB/s", html)

    def test_no_inference_gate_vocabulary(self):
        self.assertNotIn("Gate matrix", self.html)
        self.assertNotIn("hover for tier status", self.html)


if __name__ == "__main__":
    unittest.main()
