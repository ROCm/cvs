'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import unittest

from cvs.lib.inference.utils.vllm_parsing import (
    CLIENT_METRICS,
    GATED_METRICS,
    METRIC_TIER_ORDER,
    METRIC_TIERS,
    VLLM_RESULTS_COLUMNS,
    tier_metric_specs,
)
from cvs.lib.report.presets.vllm import VLLM_REPORT_CONFIG


class TestVllmReportPreset(unittest.TestCase):
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
        self.assertTrue(set(METRIC_TIERS) <= set(METRIC_TIER_ORDER))

    def test_gated_metrics_partitioned_exactly_once(self):
        tiered = [m for names in METRIC_TIERS.values() for m in names]
        # No duplicates across tiers.
        self.assertEqual(len(tiered), len(set(tiered)))
        # Every gated metric lands in exactly one non-record tier.
        self.assertEqual(set(tiered), set(GATED_METRICS))

    def test_gated_metrics_subset_of_client_metrics(self):
        client_short = {short for short, _unit in CLIENT_METRICS}
        missing = GATED_METRICS - client_short
        self.assertEqual(missing, set(), f"GATED_METRICS not in CLIENT_METRICS: {missing}")

    def test_tier_metric_specs_throughput(self):
        cell = {
            "client.output_throughput": {"kind": "min_tok_s", "value": 1},
            "client.mean_ttft_ms": {"kind": "max_ms", "value": 2},
        }
        specs = tier_metric_specs(cell, "throughput")
        self.assertIn("client.output_throughput", specs)
        self.assertNotIn("client.mean_ttft_ms", specs)

    def test_tier_metric_specs_record_includes_non_tiered(self):
        cell = {
            "client.num_prompts": {"kind": "within", "value": 100},
            "client.output_throughput": {"kind": "min_tok_s", "value": 1},
        }
        specs = tier_metric_specs(cell, "record")
        self.assertIn("client.num_prompts", specs)
        self.assertNotIn("client.output_throughput", specs)

    def test_preset_config_identity(self):
        self.assertEqual(VLLM_REPORT_CONFIG.suite_id, "vllm")
        self.assertEqual(VLLM_REPORT_CONFIG.inference_test_substring, "test_vllm_inference")
        self.assertEqual(VLLM_REPORT_CONFIG.row_card_test_names, ("test_metric",))

    def test_preset_lifecycle_labels_match_what_suite_records(self):
        # Guard against drift: the vLLM suite (cvs/tests/inference/vllm/vllm.py)
        # records exactly these session-level stages via lifecycle.record(...).
        suite_recorded = {
            "container_launch",
            "topology_discovery",
            "model_fetch",
            "server_ready",
            "teardown",
        }
        self.assertTrue(set(VLLM_REPORT_CONFIG.session_lifecycle_labels) <= suite_recorded)
        self.assertTrue(set(VLLM_REPORT_CONFIG.cell_lifecycle_labels) <= suite_recorded)

    def test_auto_register_resolves_vllm_stem(self):
        from cvs.lib.report.auto_register import try_auto_register_inference_suite_report
        from cvs.lib.report.registry import get_suite_report_config

        class _FakeConfig:
            pass

        cfg = _FakeConfig()
        cfg._suite_name = "vllm"
        cfg._suite_report_config = None
        registered = try_auto_register_inference_suite_report(cfg)
        self.assertTrue(registered)
        self.assertIs(get_suite_report_config(cfg), VLLM_REPORT_CONFIG)


if __name__ == "__main__":
    unittest.main()
