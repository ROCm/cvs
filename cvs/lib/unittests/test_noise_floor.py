"""Unit tests for cvs/lib/noise_floor.py (CVS docker-mode P12)."""

import unittest
from unittest.mock import MagicMock

from cvs.lib import noise_floor


SAMPLE_TB_OUTPUT = """\
TransferBench v1.59
Test 1:
  Executor: GPU 00 -> GPU 00 |     64.00 MB | summary    1234.56 GB/s
"""

SAMPLE_TB_OUTPUT_NOMATCH = "Help: usage info, no GB/s value here\n"


class TestParseTransferBenchBw(unittest.TestCase):
    def test_parses_first_gbps(self):
        self.assertAlmostEqual(
            noise_floor.parse_transferbench_bw(SAMPLE_TB_OUTPUT), 1234.56
        )

    def test_no_match_returns_none(self):
        self.assertIsNone(noise_floor.parse_transferbench_bw(SAMPLE_TB_OUTPUT_NOMATCH))

    def test_empty_input_returns_none(self):
        self.assertIsNone(noise_floor.parse_transferbench_bw(""))

    def test_picks_first_when_multiple(self):
        text = "first 100.0 GB/s ... second 200.0 GB/s"
        self.assertAlmostEqual(noise_floor.parse_transferbench_bw(text), 100.0)


class TestComputeCV(unittest.TestCase):
    def test_low_variance_under_1pct(self):
        cv = noise_floor.compute_cv([100.0, 100.5, 99.8, 100.1, 100.2])
        self.assertIsNotNone(cv)
        self.assertLess(cv, 0.01)

    def test_high_variance_over_5pct(self):
        cv = noise_floor.compute_cv([50.0, 100.0, 150.0])
        self.assertGreater(cv, 0.05)

    def test_too_few_samples_returns_none(self):
        self.assertIsNone(noise_floor.compute_cv([42.0]))
        self.assertIsNone(noise_floor.compute_cv([]))

    def test_zero_mean_returns_none(self):
        self.assertIsNone(noise_floor.compute_cv([0.0, 0.0, 0.0]))

    def test_drops_none_samples(self):
        cv = noise_floor.compute_cv([100.0, None, 100.5, 99.8])
        self.assertIsNotNone(cv)


class TestEvaluateCV(unittest.TestCase):
    def test_pass_below_threshold(self):
        summary = {"node-01": {"cv": 0.005}}
        self.assertEqual(noise_floor.evaluate_cv(summary, 0.01), {"node-01": "pass"})

    def test_fail_above_threshold(self):
        summary = {"node-01": {"cv": 0.05}}
        self.assertEqual(noise_floor.evaluate_cv(summary, 0.01), {"node-01": "fail"})

    def test_inconclusive_when_cv_none(self):
        summary = {"node-01": {"cv": None}}
        self.assertEqual(
            noise_floor.evaluate_cv(summary, 0.01), {"node-01": "inconclusive"}
        )


class TestMeasureNoiseFloor(unittest.TestCase):
    def _phdl_with_outputs(self, per_iter_outputs):
        m = MagicMock()
        # First exec writes the probe config; remaining N execs run TransferBench.
        m.exec.side_effect = [{"node-01": ""}] + [
            {"node-01": o} for o in per_iter_outputs
        ]
        return m

    def test_clean_node_low_cv(self):
        outputs = [
            "100.0 GB/s",
            "100.5 GB/s",
            "99.8 GB/s",
            "100.1 GB/s",
            "100.2 GB/s",
        ]
        phdl = self._phdl_with_outputs(outputs)
        summary = noise_floor.measure_noise_floor(phdl, iterations=5)
        self.assertEqual(len(summary["node-01"]["samples_gbps"]), 5)
        self.assertLess(summary["node-01"]["cv"], 0.01)
        self.assertEqual(summary["node-01"]["missing"], 0)

    def test_noisy_node_high_cv(self):
        outputs = ["50.0 GB/s", "100.0 GB/s", "150.0 GB/s"]
        phdl = self._phdl_with_outputs(outputs)
        summary = noise_floor.measure_noise_floor(phdl, iterations=3)
        self.assertGreater(summary["node-01"]["cv"], 0.05)

    def test_missing_samples_counted(self):
        outputs = ["100.0 GB/s", "garbage no match", "100.2 GB/s"]
        phdl = self._phdl_with_outputs(outputs)
        summary = noise_floor.measure_noise_floor(phdl, iterations=3)
        self.assertEqual(summary["node-01"]["missing"], 1)
        self.assertEqual(len(summary["node-01"]["samples_gbps"]), 2)


if __name__ == "__main__":
    unittest.main()
