'''
Payload/render integration test for the ANC status_matrix deck.

Guards the regression that would otherwise ship a blank deck: a status_matrix
profile must build a populated payload and render the run card + status matrix
(with the real failure detail and artifact link), and must NOT carry the
inference "gate" vocabulary.
'''

import unittest

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html

_ANC_RESULTS = {
    "_meta": {"cluster": "helios", "version": "1.7.0", "suite": "anc_test_gpu", "generated_at": "t0"},
    "groups": {
        "hbm_lvl1": {
            "nodes": {
                "10.0.0.1_n1": {
                    "status": "fail",
                    "items_summary": "Items: 6 Total | 5 PASSED, 1 FAILED",
                    "items": [{"name": "oblex_remix2", "status": "fail", "message": "ANC_PDC_DMESG_ERROR: PDC Failed"}],
                    "errors_json_href": "n1_test_hbm_lvl1_errors.json",
                    "log_tarball_href": "n1_test_hbm_lvl1_anc_logs.tar.gz",
                }
            }
        },
    },
}


class TestStatusMatrixDeck(unittest.TestCase):
    def setUp(self):
        profile = load_json_profile("anc_test_gpu")
        store = {"cvs_results_dict": _ANC_RESULTS, "inf_res_dict": _ANC_RESULTS}
        self.payload = build_rundeck_payload(profile=profile, store=store, provenance={}, cvs_version="dev")
        self.html = render_rundeck_html(self.payload)

    def test_profile_uses_run_card_and_status_matrix_only(self):
        types = [c["type"] for c in load_json_profile("anc_test_gpu")["cards"]]
        self.assertEqual(types, ["run_card", "status_matrix"])

    def test_overall_status_is_fail(self):
        self.assertEqual(self.payload["overall_status"], "fail")

    def test_run_card_bound_from_dataset(self):
        # Run card values come off datasets.status_matrix.run_card_display.
        self.assertIn("helios", self.html)
        self.assertIn("1.7.0", self.html)
        self.assertIn("Run card", self.html)

    def test_status_matrix_shows_failure_detail_and_link(self):
        self.assertIn("status-matrix", self.html)
        self.assertIn("Full results", self.html)
        self.assertIn("oblex_remix2", self.html)
        self.assertIn("ANC_PDC_DMESG_ERROR", self.html)
        self.assertIn("n1_test_hbm_lvl1_errors.json", self.html)

    def test_no_inference_gate_vocabulary(self):
        self.assertNotIn("Gate matrix", self.html)
        self.assertNotIn("hover for tier status", self.html)
        self.assertNotIn(">Cell<", self.html)

    def test_deck_is_not_blank(self):
        # The regression: a missing profile/binding would produce an empty deck.
        self.assertIn("10.0.0.1_n1", self.html)
        self.assertGreater(len(self.html), 2000)


if __name__ == "__main__":
    unittest.main()
