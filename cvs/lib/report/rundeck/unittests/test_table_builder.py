'''Unit tests for the table dataset builder and RVS Run Deck profile.'''

import unittest
from types import SimpleNamespace

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.profiles.hooks.rvs_run_card import rvs_run_card_display
from cvs.lib.report.rundeck.config_adapter import build_inference_config_from_profile
from cvs.lib.report.rundeck.dataset_builders import table  # noqa: F401
from cvs.lib.report.rundeck.dataset_builders.registry import build_datasets
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html


class TestTableBuilder(unittest.TestCase):
    def test_flattens_node_results_into_test_node_result_rows(self):
        sources = {
            "results": {
                "gpu_burn": {"node-b": "fail", "node-a": "pass"},
                "smqt": {"node-a": "pass"},
            }
        }
        datasets = build_datasets("table", sources, {})
        self.assertEqual(
            datasets["results_table"]["headers"],
            ["Test", "Node", "Result"],
        )
        self.assertEqual(
            datasets["results_table"]["rows"],
            [
                ["gpu_burn", "node-a", "pass"],
                ["gpu_burn", "node-b", "fail"],
                ["smqt", "node-a", "pass"],
            ],
        )

    def test_custom_headers_from_profile_table(self):
        profile = {"table": {"headers": ["Suite", "Host", "Status"]}}
        sources = {"results": {"gpu_burn": {"node-a": "pass"}}}
        datasets = build_datasets("table", sources, profile)
        self.assertEqual(
            datasets["results_table"]["headers"],
            ["Suite", "Host", "Status"],
        )
        self.assertEqual(
            datasets["results_table"]["rows"],
            [["gpu_burn", "node-a", "pass"]],
        )

    def test_pass_through_when_results_already_has_headers_and_rows(self):
        sources = {
            "results": {
                "headers": ["Col A", "Col B"],
                "rows": [["x", "y"], ["1", "2"]],
            }
        }
        datasets = build_datasets("table", sources, {})
        self.assertEqual(
            datasets["results_table"],
            {
                "headers": ["Col A", "Col B"],
                "rows": [["x", "y"], ["1", "2"]],
            },
        )

    def test_empty_or_missing_results_yield_empty_rows(self):
        self.assertEqual(
            build_datasets("table", {}, {})["results_table"]["rows"],
            [],
        )
        self.assertEqual(
            build_datasets("table", {"results": None}, {})["results_table"]["rows"],
            [],
        )
        self.assertEqual(
            build_datasets("table", {"cvs_results_dict": {}}, {})["results_table"]["rows"],
            [],
        )

    def test_rvs_profile_builds_payload_and_renders_run_deck(self):
        profile = load_json_profile("rvs_cvs")
        variant = SimpleNamespace(
            rvs_version="2.0.0",
            rvs_test_level="1",
            rvs_path="/opt/rvs",
        )
        store = {
            "cvs_results_dict": {
                "gpu_burn": {"node-a": "pass", "node-b": "fail"},
            },
            "variant_config": variant,
        }
        payload = build_rundeck_payload(profile=profile, store=store, cvs_version="9.9.9")
        self.assertEqual(payload["report"]["title"], "RVS Run Deck")
        self.assertEqual(payload["suite_id"], "rvs")
        self.assertIn(
            ["gpu_burn", "node-a", "pass"],
            payload["results_table"]["rows"],
        )
        self.assertIn(
            ["gpu_burn", "node-b", "fail"],
            payload["results_table"]["rows"],
        )
        doc = render_rundeck_html(payload)
        self.assertIn("RVS Run Deck", doc)
        self.assertIn("RVS version", doc)
        self.assertIn("2.0.0", doc)
        self.assertIn("node-a", doc)
        self.assertIn("pass", doc)

    def test_rvs_profile_inference_config(self):
        profile = load_json_profile("rvs_cvs")
        config = build_inference_config_from_profile(profile)
        self.assertEqual(config.suite_id, "rvs")
        self.assertFalse(config.interactive_viewer)
        self.assertIs(config.run_card_display_builder, rvs_run_card_display)


if __name__ == "__main__":
    unittest.main()
