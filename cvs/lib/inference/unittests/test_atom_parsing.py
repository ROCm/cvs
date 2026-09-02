'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import unittest

from cvs.lib.inference.atom.atom_parsing import (
    CLIENT_METRICS,
    ENFORCED_METRICS,
    GATED_METRICS,
    METRIC_TIERS,
    METRIC_UNITS,
    evaluate_specs_for_actuals,
    sglang_bench_jsonl_to_raw,
    tier_metric_specs,
)


class TestATOMAtomParsing(unittest.TestCase):
    def test_sglang_bench_jsonl_to_raw_maps_total_throughput(self):
        text = '{"completed": 10, "total_throughput": 99.5, "output_throughput": 50.0}\n'
        raw = sglang_bench_jsonl_to_raw(text)
        self.assertEqual(raw["total_token_throughput"], 99.5)

    def test_gated_metrics_include_w1_extras(self):
        for name in ("per_gpu_throughput", "output_tput_per_gpu", "p99_tpot_ms", "p99_ttft_ms"):
            self.assertIn(name, GATED_METRICS)

    def test_enforced_metrics_cover_all_tiers(self):
        tiered = {m for names in METRIC_TIERS.values() for m in names}
        self.assertEqual(ENFORCED_METRICS, frozenset(tiered))

    def test_metric_units_export_for_run_deck_profile(self):
        self.assertIn("output_throughput", METRIC_UNITS)
        self.assertEqual(METRIC_UNITS["efficiency_pct"], "%")

    def test_tier_metric_specs_throughput(self):
        cell = {
            "output_throughput": {"kind": "min_tok_s", "value": 1},
            "mean_ttft_ms": {"kind": "max_ms", "value": 2},
        }
        specs = tier_metric_specs(cell, "throughput")
        self.assertIn("output_throughput", specs)
        self.assertNotIn("mean_ttft_ms", specs)

    def test_tier_metric_specs_tpot_uses_p99_tail(self):
        cell = {
            "mean_tpot_ms": {"kind": "max_ms", "value": 46.8},
            "p99_tpot_ms": {"kind": "max_ms", "value": 51.36},
            "p95_tpot_ms": {"kind": "max_ms", "value": 53.76},
        }
        specs = tier_metric_specs(cell, "tpot")
        self.assertIn("p99_tpot_ms", specs)
        self.assertNotIn("p95_tpot_ms", specs)

    def test_tier_metric_specs_record_includes_non_tiered(self):
        cell = {
            "median_ttft_ms": {"kind": "max_ms", "value": 9},
            "output_throughput": {"kind": "min_tok_s", "value": 1},
        }
        specs = tier_metric_specs(cell, "record")
        self.assertIn("median_ttft_ms", specs)
        self.assertNotIn("output_throughput", specs)

    def test_tier_metric_specs_scaling(self):
        cell = {
            "scaling.efficiency_pct": {"kind": "min", "value": 50},
            "output_throughput": {"kind": "min_tok_s", "value": 1},
        }
        specs = tier_metric_specs(cell, "scaling")
        self.assertEqual(specs, {"scaling.efficiency_pct": {"kind": "min", "value": 50}})

    def test_evaluate_specs_for_actuals_maps_client_namespace(self):
        specs = {"output_throughput": {"kind": "min_tok_s", "value": 1}}
        actuals = {"client.output_throughput": 100.0, "client.mean_ttft_ms": 5.0}
        eval_actuals, eval_specs = evaluate_specs_for_actuals(specs, actuals, metric_tier="throughput")
        self.assertEqual(eval_actuals, {"output_throughput": 100.0})
        self.assertEqual(eval_specs, specs)

    def test_gated_metrics_subset_of_client_metrics(self):
        client_short = {short for short, _unit in CLIENT_METRICS}
        missing = GATED_METRICS - client_short
        self.assertEqual(missing, set(), f"GATED_METRICS not in CLIENT_METRICS: {missing}")

    def test_health_tier_metrics_in_enforced_set(self):
        for name in ("success_rate", "failed"):
            self.assertIn(name, ENFORCED_METRICS)
            self.assertIn(name, METRIC_TIERS["health"])


if __name__ == "__main__":
    unittest.main()
