'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import unittest

from cvs.lib.inference.utils.inferencex_atom_parsing import (
    CLIENT_METRICS,
    ENFORCED_METRICS,
    GATED_METRICS,
    METRIC_TIERS,
    tier_metric_specs,
)


class TestInferenceXAtomParsing(unittest.TestCase):
    def test_gated_metrics_include_w1_ix_extras(self):
        for name in ("per_gpu_throughput", "output_tput_per_gpu", "p99_tpot_ms", "p99_ttft_ms"):
            self.assertIn(name, GATED_METRICS)

    def test_enforced_metrics_cover_all_tiers(self):
        tiered = {m for names in METRIC_TIERS.values() for m in names}
        self.assertEqual(ENFORCED_METRICS, frozenset(tiered))

    def test_tier_metric_specs_throughput(self):
        cell = {
            "client.output_throughput": {"kind": "min_tok_s", "value": 1},
            "client.mean_ttft_ms": {"kind": "max_ms", "value": 2},
        }
        specs = tier_metric_specs(cell, "throughput")
        self.assertIn("client.output_throughput", specs)
        self.assertNotIn("client.mean_ttft_ms", specs)

    def test_tier_metric_specs_tpot_uses_p99_tail(self):
        cell = {
            "client.mean_tpot_ms": {"kind": "max_ms", "value": 46.8},
            "client.p99_tpot_ms": {"kind": "max_ms", "value": 51.36},
            "client.p95_tpot_ms": {"kind": "max_ms", "value": 53.76},
        }
        specs = tier_metric_specs(cell, "tpot")
        self.assertIn("client.p99_tpot_ms", specs)
        self.assertNotIn("client.p95_tpot_ms", specs)

    def test_tier_metric_specs_record_includes_non_tiered(self):
        cell = {
            "client.median_ttft_ms": {"kind": "max_ms", "value": 9},
            "client.output_throughput": {"kind": "min_tok_s", "value": 1},
        }
        specs = tier_metric_specs(cell, "record")
        self.assertIn("client.median_ttft_ms", specs)
        self.assertNotIn("client.output_throughput", specs)

    def test_tier_metric_specs_scaling(self):
        cell = {
            "scaling.efficiency_pct": {"kind": "min", "value": 50},
            "client.output_throughput": {"kind": "min_tok_s", "value": 1},
        }
        specs = tier_metric_specs(cell, "scaling")
        self.assertEqual(specs, {"scaling.efficiency_pct": {"kind": "min", "value": 50}})

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
