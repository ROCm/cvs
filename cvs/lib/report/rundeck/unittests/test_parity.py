'''Parity and integration tests for the unified Run Deck engine.'''

import unittest

from cvs.lib.report.inference import build_inference_report_payload, render_report_html
from cvs.lib.report.rundeck.config_adapter import build_inference_config_from_profile
from cvs.lib.report.rundeck.dataset_builders.registry import build_datasets
from cvs.lib.report.rundeck.payload import RundeckPayloadBuilder, RundeckPayloadContext, build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html
from cvs.lib.report.testing.fixtures import generic_sweep_profile, generic_variant, two_cell_inf_res


class TestRundeckParity(unittest.TestCase):
    def test_json_profile_resolves_expected_config(self):
        profile = generic_sweep_profile()
        cfg = build_inference_config_from_profile(profile)
        self.assertEqual(cfg.suite_id, "test_inference_suite")
        self.assertEqual(cfg.report_basename, "test_inference_suite_run_deck")
        self.assertEqual(cfg.metric_tier_order, ("throughput", "record"))
        self.assertEqual(len(cfg.chart_series), 3)

    def test_rundeck_payload_matches_inference_builder(self):
        profile = generic_sweep_profile()
        store = {
            "inf_res_dict": two_cell_inf_res(),
            "variant_config": generic_variant(),
            "lifecycle_report": {},
        }
        config = build_inference_config_from_profile(profile)
        rundeck_payload = build_rundeck_payload(profile=profile, store=store, cvs_version="1.0.0")
        inference_payload = build_inference_report_payload(
            config=config,
            variant_config=generic_variant(),
            inf_res_dict=two_cell_inf_res(),
            lifecycle_report={},
            cvs_version="1.0.0",
        )
        self.assertEqual(len(rundeck_payload["cells"]), len(inference_payload["cells"]))
        self.assertEqual(
            rundeck_payload["results_table"]["headers"],
            inference_payload["results_table"]["headers"],
        )
        self.assertEqual(len(rundeck_payload["gate_matrix"]), len(inference_payload["gate_matrix"]))
        self.assertEqual(rundeck_payload["chart_series"].keys(), inference_payload["chart_series"].keys())
        self.assertIn("viewer_config", rundeck_payload)
        self.assertEqual(rundeck_payload["viewer_config"]["group_by"], ["isl", "osl"])

    def test_rundeck_payload_builder_class(self):
        profile = generic_sweep_profile()
        ctx = RundeckPayloadContext(
            profile=profile,
            store={
                "inf_res_dict": two_cell_inf_res(),
                "variant_config": generic_variant(),
                "lifecycle_report": {},
            },
            cvs_version="1.0.0",
        )
        payload = RundeckPayloadBuilder(ctx).build()
        self.assertEqual(payload["suite_id"], "test_inference_suite")
        self.assertIn("viewer_config", payload)

    def test_rundeck_render_contains_core_panels(self):
        profile = generic_sweep_profile()
        payload = build_rundeck_payload(
            profile=profile,
            store={
                "inf_res_dict": two_cell_inf_res(),
                "variant_config": generic_variant(),
                "lifecycle_report": {},
            },
            cvs_version="1.0.0",
        )
        doc = render_rundeck_html(payload)
        self.assertIn("Test Sweep Run Deck", doc)
        self.assertIn("report-nav", doc)
        self.assertIn("Gate matrix", doc)
        self.assertIn("Full results", doc)
        self.assertIn("Sweep analytics", doc)

    def test_interactivity_viewer_card_links_to_viewer(self):
        profile = generic_sweep_profile()
        payload = build_rundeck_payload(
            profile=profile,
            store={
                "inf_res_dict": two_cell_inf_res(),
                "variant_config": generic_variant(),
                "lifecycle_report": {},
            },
            cvs_version="1.0.0",
        )
        payload["summary"] = {"viewer_html": "test_inference_suite_run_deck_viewer.html"}
        doc = render_rundeck_html(payload)
        self.assertIn("Open interactivity chart", doc)
        self.assertIn("test_inference_suite_run_deck_viewer.html#interactivity-panel", doc)

    def test_inference_render_path_uses_unified_runtime(self):
        config = build_inference_config_from_profile(generic_sweep_profile())
        payload = build_inference_report_payload(
            config=config,
            variant_config=generic_variant(),
            inf_res_dict=two_cell_inf_res(),
            lifecycle_report={},
            cvs_version="1.0.0",
        )
        doc = render_report_html(payload)
        self.assertIn("Gate matrix", doc)
        self.assertIn("Full results", doc)

    def test_series_builder_from_nested_graph_shape(self):
        graph = {
            "all_reduce": {
                "8": {"bus_bw": 12.5, "alg_bw": 11.0, "time": 100},
                "64": {"bus_bw": 45.0, "alg_bw": 40.0, "time": 200},
            }
        }
        profile = {
            "dataset_builder": "series",
            "series": {"x_field": "size", "y_fields": ["bus_bw", "alg_bw"]},
        }
        datasets = build_datasets("series", {"results": graph}, profile)
        self.assertIn("bus_bw", datasets["charts"])
        self.assertTrue(datasets["results_table"]["rows"])


if __name__ == "__main__":
    unittest.main()
