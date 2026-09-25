'''Unit tests for Megatron Run Deck metric tiers.'''

import unittest

from cvs.lib.training.megatron.utils.megatron_metrics import METRIC_UNITS, tier_metric_specs


class TestMegatronMetrics(unittest.TestCase):
    def test_unknown_tier_empty(self):
        self.assertEqual(tier_metric_specs({"training.throughput_per_gpu": {"kind": "min", "value": 1}}, "nope"), {})

    def test_units_include_throughput(self):
        self.assertEqual(METRIC_UNITS["throughput_per_gpu"], "TFLOP/s/GPU")
        self.assertEqual(METRIC_UNITS["step_time_p95_ms"], "ms")


if __name__ == "__main__":
    unittest.main()
