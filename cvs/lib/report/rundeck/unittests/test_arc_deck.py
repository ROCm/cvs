'''Render the ARC example-failure Run Deck through the actual profile.'''

import unittest

from cvs.lib.arc_rundeck import build_node_record, make_meta
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html
from cvs.lib.unittests.test_arc_rundeck import EXAMPLE_FAILURE


class TestArcDeck(unittest.TestCase):
    def test_failed_example_renders_arc_version_item_and_json_link(self):
        profile = load_json_profile("arc_test_examples")
        results = {
            "_meta": make_meta({"name": "rack"}, {"arc": {"arc_version": "1.8.1"}}, "arc_test_examples", "ts"),
            "groups": {
                "example_failure": {
                    "nodes": {
                        "controller": build_node_record(
                            status="fail",
                            summary=EXAMPLE_FAILURE,
                            errors_json_href="controller_errors.json",
                            log_tarball_href="logs.tar.gz",
                        )
                    }
                }
            },
        }
        payload = build_rundeck_payload(
            profile=profile,
            store={"arc_res_dict": results, "cvs_results_dict": results, "inf_res_dict": results},
            provenance={},
            cvs_version="dev",
        )
        html = render_rundeck_html(payload)
        self.assertEqual(payload["overall_status"], "fail")
        for text in ("ARC version", "1.8.1", "Mock_FailingTest", "controller_errors.json"):
            self.assertIn(text, html)
        self.assertNotIn("ANC version", html)
