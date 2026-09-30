'''Unit tests for deck profile source resolution and inheritance.'''

import json
import unittest

from jsonschema import Draft202012Validator

from cvs.lib.report.profile import DEFAULT_SOURCES, load_json_profile, profile_json_path, sources_for_profile
from cvs.lib.report.rundeck.config_builder import make_inference_report_config


class TestProfile(unittest.TestCase):
    def test_rccl_profile_validates_against_schema(self):
        schema = json.loads(profile_json_path("schema").read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(load_json_profile("rccl"))

    def test_status_matrix_health_cards_validate_against_schema(self):
        schema = json.loads(profile_json_path("schema").read_text(encoding="utf-8"))
        profile = {
            "schema_version": 1,
            "profile_id": "health_example",
            "suite_id": "health_example",
            "report_basename": "health_example_run_deck",
            "title": "Health Run Deck",
            "dataset_builder": "status_matrix",
            "interactive_viewer": True,
            "sources": {"results": "cvs_results_dict"},
            "cards": [
                {
                    "type": "run_card",
                    "id": "run-card",
                    "title": "Run card",
                    "bind": "datasets.status_matrix.run_card_display",
                },
                {
                    "type": "status_overview",
                    "id": "overview",
                    "title": "Health overview",
                    "bind": "datasets.status_matrix.overview",
                },
                {
                    "type": "metric_charts",
                    "id": "metrics",
                    "title": "Metrics",
                    "bind": "datasets.status_matrix.metric_charts",
                    "when_empty": "hide",
                },
                {
                    "type": "status_matrix",
                    "id": "results",
                    "title": "Full results",
                    "bind": "datasets.status_matrix",
                    "hint": "Click a cell's items to expand that node Ã— group.",
                },
            ],
        }
        Draft202012Validator(schema).validate(profile)

    def test_transferbench_profile_validates_against_schema(self):
        schema = json.loads(profile_json_path("schema").read_text(encoding="utf-8"))
        profile = load_json_profile("transferbench_cvs")
        Draft202012Validator(schema).validate(profile)
        self.assertEqual(profile["dataset_builder"], "status_matrix")
        self.assertEqual(profile["sources"]["results"], "transferbench_res_dict")
        self.assertTrue(profile["interactive_viewer"])
        self.assertEqual(
            [card["type"] for card in profile["cards"]],
            ["run_card", "status_overview", "metric_charts", "status_matrix"],
        )
        self.assertEqual(profile["cards"][2]["when_empty"], "hide")
        self.assertTrue(profile["cards"][3]["hint"])

    def test_health_profiles_extend_shared_card_stack(self):
        schema = json.loads(profile_json_path("schema").read_text(encoding="utf-8"))
        base = load_json_profile("health_status_base")
        Draft202012Validator(schema).validate(base)
        self.assertEqual(
            [card["id"] for card in base["cards"]],
            ["run-card", "overview", "metrics", "results"],
        )

        rvs_raw = json.loads(profile_json_path("rvs_cvs").read_text(encoding="utf-8"))
        transferbench_raw = json.loads(profile_json_path("transferbench_cvs").read_text(encoding="utf-8"))
        self.assertEqual(rvs_raw["extends"], "health_status_base")
        self.assertEqual(rvs_raw["cards"], [{"id": "results", "hint": rvs_raw["cards"][0]["hint"]}])
        self.assertEqual(transferbench_raw["extends"], "health_status_base")
        self.assertEqual([card["id"] for card in transferbench_raw["cards"]], ["metrics", "results"])

        rvs = load_json_profile("rvs_cvs")
        transferbench = load_json_profile("transferbench_cvs")
        self.assertEqual([card["id"] for card in rvs["cards"]], ["run-card", "overview", "metrics", "results"])
        self.assertEqual(
            [card["id"] for card in transferbench["cards"]],
            ["run-card", "overview", "metrics", "results"],
        )
        self.assertEqual(transferbench["cards"][2]["title"], "Bandwidth highlights")
        self.assertEqual(rvs["cards"][2]["title"], "Measurements")

    def test_rvs_profile_validates_against_schema(self):
        schema = json.loads(profile_json_path("schema").read_text(encoding="utf-8"))
        profile = load_json_profile("rvs_cvs")
        Draft202012Validator(schema).validate(profile)
        self.assertEqual(profile["dataset_builder"], "status_matrix")
        self.assertEqual(profile["sources"]["results"], "rvs_res_dict")

    def test_default_sources_for_legacy_preset(self):
        cfg = make_inference_report_config(
            suite_id="demo",
            results_columns=(),
            metric_units={},
            tier_metric_specs=lambda _c, _t: {},
        )
        self.assertEqual(sources_for_profile(cfg), DEFAULT_SOURCES)

    def test_rccl_stems_share_one_profile(self):
        for stem in ("rccl", "rccl_perf", "rccl_regression", "rccl_pairwise"):
            profile = load_json_profile(stem)
            self.assertIsNotNone(profile, stem)
            self.assertEqual(profile["suite_id"], "rccl")
            self.assertEqual(profile["dataset_builder"], "series")
            self.assertEqual(
                profile["hooks"]["run_card_display"],
                "cvs.lib.report.profiles.hooks.rccl_run_card:rccl_run_card_display",
            )

    def test_sglang_stems_share_one_profile(self):
        for stem in ("sglang_single", "sglang_distributed", "sglang_disagg_distributed"):
            profile = load_json_profile(stem)
            self.assertIsNotNone(profile, stem)
            self.assertEqual(profile["suite_id"], "sglang")
            self.assertEqual(profile["report_basename"], "sglang_run_deck")
            self.assertEqual(
                profile["hooks"]["run_card_display"],
                "cvs.lib.report.profiles.hooks.sglang_run_card:sglang_run_card_display",
            )

    def test_megatron_stems_share_one_profile(self):
        for stem in ("megatron", "megatron_single", "megatron_distributed"):
            profile = load_json_profile(stem)
            self.assertIsNotNone(profile, stem)
            self.assertEqual(profile["suite_id"], "megatron")
            self.assertEqual(profile["dataset_builder"], "training_sweep")
            self.assertEqual(profile["sources"]["results"], "train_res_dict")
            self.assertIn("smoke", profile["lifecycle"]["session_labels"])
            self.assertEqual(
                profile["hooks"]["run_card_display"],
                "cvs.lib.report.profiles.hooks.megatron_run_card:megatron_run_card_display",
            )

    def test_vllm_hooks_point_at_canonical_metric_contract(self):
        profile = load_json_profile("vllm")
        self.assertEqual(
            profile["hooks"]["metric_units"],
            "cvs.lib.inference.utils.vllm_metrics:METRIC_UNITS",
        )
        self.assertEqual(
            profile["hooks"]["metric_verdict"],
            "cvs.lib.inference.utils.vllm_metrics:metric_verdict",
        )
        self.assertNotIn("run_card_display", profile.get("hooks", {}))

    def test_vllm_profile_cell_highlights_include_output_throughput(self):
        from cvs.lib.report.rundeck.config_adapter import build_inference_config_from_profile

        profile = load_json_profile("vllm")
        self.assertIsNotNone(profile)
        config = build_inference_config_from_profile(profile)
        shorts = [short for short, _label in config.cell_highlights]
        self.assertIn("output_throughput", shorts)
        self.assertEqual(config.full_metric("output_throughput"), "output_throughput")

    def test_sglang_profile_preserves_empty_metric_prefix(self):
        from cvs.lib.report.rundeck.config_adapter import build_inference_config_from_profile

        profile = load_json_profile("sglang")
        self.assertIsNotNone(profile)
        profile = dict(profile)
        hooks = dict(profile.get("hooks") or {})
        hooks.pop("run_card_display", None)
        profile["hooks"] = hooks
        config = build_inference_config_from_profile(profile)
        self.assertEqual(config.metric_prefix, "")
        self.assertEqual(config.headline_metric, "output_throughput_per_sec")
        self.assertEqual(config.full_metric("output_throughput_per_sec"), "output_throughput_per_sec")
        self.assertEqual(config.metric_tier_order, ("throughput", "latency", "health", "record"))

    def test_xdit_stems_share_one_profile(self):
        stems = (
            "xdit_flux_dev_single",
            "xdit_flux_dev_distributed",
            "xdit_wan22_14b_single",
            "xdit_wan22_14b_diffusers_single",
            "xdit_wan22_14b_diffusers_distributed",
        )
        for stem in stems:
            with self.subTest(stem=stem):
                profile = load_json_profile(stem)
                self.assertEqual(profile["suite_id"], "xdit")
                self.assertEqual(profile["report_basename"], "xdit_run_deck")

    def test_xdit_profile_resolves_latency_metrics(self):
        from cvs.lib.report.rundeck.config_adapter import build_inference_config_from_profile

        config = build_inference_config_from_profile(load_json_profile("xdit"))

        self.assertEqual(config.metric_prefix, "")
        self.assertEqual(config.metric_tier_order, ("latency", "record"))
        self.assertEqual(config.full_metric("avg_pipe_time_s"), "avg_pipe_time_s")


if __name__ == "__main__":
    unittest.main()
