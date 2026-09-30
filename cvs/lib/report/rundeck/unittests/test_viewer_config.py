'''Tests for profile-driven interactive viewer configuration.'''

import unittest

from cvs.lib.report.inference_payload import build_inference_report_payload
from cvs.lib.report.rundeck.config_adapter import resolve_report_config
from cvs.lib.report.rundeck.viewer_config import ViewerConfigBuilder, build_viewer_config
from cvs.lib.report.testing.fixtures import (
    generic_inference_report_config,
    generic_sweep_profile,
    generic_variant,
    two_cell_inf_res,
)
from cvs.lib.report.types import InferenceReportConfig


class TestViewerConfig(unittest.TestCase):
    def test_build_viewer_config_from_sweep_profile(self):
        profile = generic_sweep_profile()
        config = resolve_report_config(profile)
        vc = build_viewer_config(profile, config)

        self.assertEqual(vc["group_by"], ["isl", "osl"])
        self.assertGreaterEqual({f["field"] for f in vc["filters"]}, {"isl", "osl", "policy", "host"})
        self.assertIn("client.output_throughput", vc["metrics"])
        self.assertIn("client.output_throughput", vc["heatmap_metrics"])
        self.assertTrue(any(col.get("field") == "isl" for col in vc["table_columns"]))
        self.assertTrue(vc["interactivity"]["enabled"])
        self.assertTrue(vc["sweep_charts"]["enabled"])
        self.assertEqual(vc["interactivity"]["tpot_metric"], "client.mean_tpot_ms")
        self.assertEqual(vc["interactivity"]["output_throughput_metric"], "client.output_throughput")
        self.assertEqual(vc["interactivity"]["total_throughput_metric"], "client.total_token_throughput")

    def test_sglang_profile_interactivity_metrics(self):
        import json

        from cvs.lib.report.profile import profile_json_path

        profile = json.loads(profile_json_path("sglang").read_text(encoding="utf-8"))
        inter = profile["viewer"]["interactivity"]
        self.assertEqual(inter["tpot_metric"], "mean_tpot_ms")
        self.assertEqual(inter["output_throughput_metric"], "output_throughput_per_sec")

        config = resolve_report_config(generic_sweep_profile())
        config = InferenceReportConfig(
            **{
                **config.__dict__,
                "suite_id": "sglang",
                "headline_metric": "output_throughput_per_sec",
                "metric_prefix": "",
                "results_columns": (
                    ("Output tok/s", "output_throughput_per_sec"),
                    ("Mean TPOT (ms)", "mean_tpot_ms"),
                ),
            }
        )
        vc = ViewerConfigBuilder(profile, config).build()
        self.assertEqual(vc["interactivity"]["tpot_metric"], "mean_tpot_ms")
        self.assertIn("output_throughput_per_sec", vc["metrics"])
        self.assertIn("mean_tpot_ms", vc["metrics"])

    def test_megatron_profile_viewer_config(self):
        import json

        from cvs.lib.report.profile import profile_json_path

        profile = json.loads(profile_json_path("megatron").read_text(encoding="utf-8"))
        config = resolve_report_config(profile)
        vc = ViewerConfigBuilder(profile, config).build()
        self.assertEqual(vc["group_by"], ["mbs", "gbs"])
        self.assertEqual(
            {f["field"] for f in vc["filters"]},
            {"mbs", "gbs", "precision", "tp", "pp"},
        )
        self.assertEqual(vc["concurrency_field"], "precision")
        self.assertEqual(vc["heatmap_row_fields"], ["mbs", "gbs", "tp", "pp"])
        self.assertFalse(vc["interactivity"]["enabled"])
        self.assertFalse(vc["sweep_charts"]["enabled"])
        self.assertIn("training.throughput_per_gpu", vc["metrics"])
        self.assertTrue(any(col.get("field") == "mbs" for col in vc["table_columns"]))
        self.assertTrue(any(col.get("field") == "precision" for col in vc["table_columns"]))

    def test_sweep_charts_follow_interactivity_when_unset(self):
        profile = generic_sweep_profile()
        profile["viewer"] = {"interactivity": {"enabled": False}}
        config = resolve_report_config(profile)
        vc = ViewerConfigBuilder(profile, config).build()
        self.assertFalse(vc["sweep_charts"]["enabled"])

    def test_viewer_config_builder_class(self):
        profile = generic_sweep_profile()
        config = resolve_report_config(profile)
        vc = ViewerConfigBuilder(profile, config).build()
        self.assertEqual(vc["group_by"], ["isl", "osl"])

    def test_build_viewer_config_legacy_config_only(self):
        config = generic_inference_report_config()
        vc = build_viewer_config({}, config)

        self.assertEqual(vc["group_by"], ["isl", "osl"])
        self.assertEqual(vc["default_heatmap_metric"], config.headline_metric)
        self.assertGreaterEqual(len(vc["table_columns"]), 5)

    def test_inference_payload_includes_viewer_config(self):
        config = generic_inference_report_config()
        payload = build_inference_report_payload(
            config=config,
            variant_config=generic_variant(),
            inf_res_dict=two_cell_inf_res(),
            lifecycle_report={},
        )
        self.assertIn("viewer_config", payload)
        self.assertEqual(payload["viewer_config"]["group_by"], ["isl", "osl"])


if __name__ == "__main__":
    unittest.main()
