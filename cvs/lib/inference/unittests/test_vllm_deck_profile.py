'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import unittest
from pathlib import Path

from cvs.lib.inference.vllm_topology import EffectiveVllmTopology
from cvs.lib.inference.utils.vllm_config_loader import load_variant
from cvs.lib.inference.utils.vllm_metrics import (
    CLIENT_METRICS,
    METRIC_CATEGORIES,
    METRIC_REGISTRY,
    VLLM_RESULTS_COLUMNS,
    tier_metric_specs,
)
from cvs.lib.report.auto_register import try_auto_register_suite_report
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.registry import get_resolved_profile, resolve_suite_report_config
from cvs.lib.report.rundeck.config_adapter import build_inference_config_from_profile
from cvs.lib.report.inference import build_inference_report_payload


class TestVllmDeckProfile(unittest.TestCase):
    def test_results_columns_fixed_positional_prefix(self):
        fixed = VLLM_RESULTS_COLUMNS[:7]
        self.assertEqual(
            fixed,
            (
                ("Model", None),
                ("GPU", None),
                ("ISL", None),
                ("OSL", None),
                ("Policy", None),
                ("Conc", None),
                ("Host", None),
            ),
        )

    def test_metric_tiers_subset_of_tier_order(self):
        profile = load_json_profile("vllm")
        config = build_inference_config_from_profile(profile)
        self.assertEqual(config.metric_tier_order, METRIC_CATEGORIES)
        self.assertNotIn("record", config.metric_tier_order)

    def test_gated_metrics_partitioned_exactly_once(self):
        tiered = [
            definition.name
            for category in METRIC_CATEGORIES
            for definition in METRIC_REGISTRY
            if definition.category == category
        ]
        self.assertEqual(tiered, [definition.name for definition in METRIC_REGISTRY])

    def test_gated_metrics_subset_of_client_metrics(self):
        client_short = {short for short, _unit in CLIENT_METRICS}
        registered_client = {definition.name for definition in METRIC_REGISTRY if definition.datasource == "client"}
        self.assertEqual(client_short, registered_client)

    def test_tier_metric_specs_throughput(self):
        cell = {
            "output_throughput": {"kind": "min", "value": 1},
            "mean_ttft_ms": {"kind": "max", "value": 2},
        }
        specs = tier_metric_specs(cell, "throughput")
        self.assertIn("output_throughput", specs)
        self.assertNotIn("mean_ttft_ms", specs)

    def test_tier_metric_specs_has_no_static_record_tier(self):
        cell = {"num_prompts": {"kind": "min", "value": 100}}
        self.assertEqual(tier_metric_specs(cell, "record"), {})
        self.assertEqual(tier_metric_specs(cell, "run_health"), cell)

    def test_vllm_json_profile_identity(self):
        profile = load_json_profile("vllm")
        self.assertIsNotNone(profile)
        cfg = build_inference_config_from_profile(profile)
        self.assertEqual(cfg.suite_id, "vllm")
        self.assertEqual(cfg.inference_test_substring, "test_vllm_inference")
        self.assertEqual(cfg.row_card_test_names, ("test_verify_cell_metrics",))
        self.assertEqual(cfg.metric_prefix, "")
        self.assertEqual(cfg.metric_contract, {"id": "vllm-bare", "version": 1})

    def test_vllm_profile_lifecycle_labels_match_what_suite_records(self):
        profile = load_json_profile("vllm")
        cfg = build_inference_config_from_profile(profile)
        suite_recorded = {
            "container_launch",
            "topology_discovery",
            "model_fetch",
            "openai_smoke",
            "server_ready",
            "teardown",
        }
        self.assertTrue(set(cfg.session_lifecycle_labels) <= suite_recorded)
        self.assertTrue(set(cfg.cell_lifecycle_labels) <= suite_recorded)

    def test_auto_register_resolves_split_suite_stems(self):
        class _FakeConfig:
            pass

        for stem in ("vllm_single", "vllm_distributed"):
            with self.subTest(stem=stem):
                cfg = _FakeConfig()
                cfg._suite_name = stem
                cfg._suite_report_config = None
                self.assertTrue(try_auto_register_suite_report(cfg))
                profile = get_resolved_profile(cfg)
                self.assertIsInstance(profile, dict)
                self.assertEqual(profile["suite_id"], "vllm")
                self.assertEqual(resolve_suite_report_config(cfg).suite_id, "vllm")

    def test_split_suite_stems_reuse_vllm_profile(self):
        for stem in ("vllm_single", "vllm_distributed"):
            with self.subTest(stem=stem):
                profile = load_json_profile(stem)
                self.assertIsNotNone(profile)
                self.assertEqual(profile["suite_id"], "vllm")

    def test_distributed_cell_uses_bound_threshold_key(self):
        root = Path(__file__).resolve().parents[3]
        variant = load_variant(
            root / "input/config_file/inference/vllm/mi3xx_vllm_llama33-70b_fp8_distributed.json",
            {"username": "test"},
        )
        variant.bind_effective_topology(EffectiveVllmTopology("distributed", ("node0", "node1"), 2))
        cell_id = variant.expected_cells()[0]
        variant.enforce_thresholds = True
        variant.thresholds = {cell_id: {"output_throughput": {"kind": "min", "value": 1000.0}}}
        run = variant.resolved_runs()[0]
        key = (variant.model_id, "", str(run.cell.isl), str(run.cell.osl), run.cell.key, run.cell.concurrency)
        profile = load_json_profile("vllm_distributed")
        payload = build_inference_report_payload(
            config=build_inference_config_from_profile(profile),
            variant_config=variant,
            inf_res_dict={key: {"node0": {"output_throughput": 500.0}}},
            lifecycle_report={},
        )
        self.assertEqual(payload["cells"][0]["cell_id"], cell_id)
        self.assertEqual(payload["cells"][0]["tiers"]["throughput"], "fail")
        self.assertEqual(payload["overall_status"], "fail")
        self.assertEqual(payload["metric_contract"], {"id": "vllm-bare", "version": 1})
        self.assertEqual(
            payload["viewer_config"]["metric_contract"],
            {"id": "vllm-bare", "version": 1},
        )


if __name__ == "__main__":
    unittest.main()
