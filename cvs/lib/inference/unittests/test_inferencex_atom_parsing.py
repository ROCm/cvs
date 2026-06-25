'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import unittest

from cvs.lib.inference.utils.inferencex_atom_parsing import (
    ENFORCED_METRICS,
    GATED_METRICS,
    METRIC_TIERS,
    tier_metric_specs,
)


class TestInferenceXAtomParsing(unittest.TestCase):
    def test_gated_metrics_include_w1_ix_extras(self):
        for name in ("per_gpu_throughput", "output_tput_per_gpu", "p95_tpot_ms", "p99_ttft_ms"):
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

    def test_tier_metric_specs_record_includes_non_tiered(self):
        cell = {
            "client.median_ttft_ms": {"kind": "max_ms", "value": 9},
            "client.output_throughput": {"kind": "min_tok_s", "value": 1},
        }
        specs = tier_metric_specs(cell, "record")
        self.assertIn("client.median_ttft_ms", specs)
        self.assertNotIn("client.output_throughput", specs)


if __name__ == "__main__":
    unittest.main()
