# cvs/lib/unittests/test_regression_lib.py
"""
Unit tests for the paired A/B regression detector (cvs.lib.regression_lib).

Includes deterministic correctness tests plus Monte-Carlo sweeps that verify the
detector's two most important properties for CI:
  * running a candidate against an identical-distribution reference produces an
    extremely low false-positive rate (trustworthy / no flaky failures), and
  * a genuine regression larger than the size-tier threshold is reliably caught.
"""

import math
import random
import unittest

import cvs.lib.regression_lib as reg

KiB = 1024
MiB = 1024 * 1024
GiB = 1024 * 1024 * 1024

# Message size sweep used by the simulations: 1 KiB .. 4 GiB, powers of two.
SWEEP_SIZES = [1 << e for e in range(10, 33)]  # 2^10 (1KiB) .. 2^32 (4GiB)


def _true_bw(size):
    """A plausible bandwidth curve: latency-bound small, plateau at large size."""
    plateau = 350.0  # GB/s
    half = 16 * MiB
    return plateau * size / (size + half)


def _cv_for_size(size):
    """Run-to-run coefficient of variation: noisy small, tight large."""
    if size <= 1 * MiB:
        return 0.12
    if size <= 64 * MiB:
        return 0.05
    return 0.025


def _make_run(rng, collective="AllReduce", dtype="float", in_place=1, bw_scale=1.0, sizes=None):
    """Build one simulated sweep (list of rccl-style rows) with Gaussian noise."""
    rows = []
    for size in (sizes or SWEEP_SIZES):
        cv = _cv_for_size(size)
        mean = _true_bw(size) * bw_scale
        val = mean * (1.0 + rng.gauss(0.0, cv))
        val = max(val, 0.01)
        rows.append({
            "name": collective,
            "size": size,
            "type": dtype,
            "inPlace": in_place,
            "busBw": val,
            "algBw": val * 0.5,
            "time": 1.0,
        })
    return rows


class TestHelpers(unittest.TestCase):
    def test_percentile_basic(self):
        data = [1, 2, 3, 4, 5]
        self.assertEqual(reg.percentile(data, 0), 1)
        self.assertEqual(reg.percentile(data, 100), 5)
        self.assertEqual(reg.percentile(data, 50), 3)
        self.assertEqual(reg.median(data), 3)

    def test_percentile_single(self):
        self.assertEqual(reg.percentile([42], 25), 42)

    def test_summarize(self):
        s = reg.summarize_samples([10, 20, 30])
        self.assertEqual(s["n"], 3)
        self.assertEqual(s["min"], 10)
        self.assertEqual(s["max"], 30)
        self.assertEqual(s["median"], 20)
        self.assertAlmostEqual(s["mean"], 20.0)

    def test_size_tier_and_threshold(self):
        self.assertEqual(reg.size_tier(1 * KiB), "small")
        self.assertEqual(reg.size_tier(1 * MiB), "small")
        self.assertEqual(reg.size_tier(2 * MiB), "mid")
        self.assertEqual(reg.size_tier(64 * MiB), "mid")
        self.assertEqual(reg.size_tier(128 * MiB), "large")
        self.assertEqual(reg.size_tier(4 * GiB), "large")
        self.assertEqual(reg.threshold_for(1 * KiB), 0.20)
        self.assertEqual(reg.threshold_for(2 * MiB), 0.10)
        self.assertEqual(reg.threshold_for(1 * GiB), 0.05)

    def test_merge_config_overrides(self):
        cfg = reg.merge_config({"thresholds": {"large": 0.08}, "adjacency_min_run": 3})
        self.assertEqual(cfg["thresholds"]["large"], 0.08)
        self.assertEqual(cfg["thresholds"]["small"], 0.20)  # untouched
        self.assertEqual(cfg["adjacency_min_run"], 3)


class TestCompareKey(unittest.TestCase):
    def test_clear_regression_is_candidate(self):
        a = [100.0, 101.0, 99.0, 100.5, 100.0]
        b = [80.0, 81.0, 79.0, 80.5, 80.0]  # ~20% lower, tight
        v = reg.compare_key(a, b, size_bytes=1 * GiB)  # large tier, 5% thr
        self.assertTrue(v["candidate"])
        self.assertGreater(v["rel_drop"], 0.05)

    def test_noise_only_not_candidate(self):
        a = [100.0, 102.0, 98.0, 101.0, 99.0]
        b = [101.0, 99.0, 100.0, 98.0, 102.0]  # same distribution
        v = reg.compare_key(a, b, size_bytes=1 * GiB)
        self.assertFalse(v["candidate"])

    def test_insufficient_repeats_inconclusive(self):
        v = reg.compare_key([100.0], [50.0], size_bytes=1 * GiB)
        self.assertEqual(v["verdict"], reg.INCONCLUSIVE)

    def test_below_floor_inconclusive(self):
        v = reg.compare_key([0.1, 0.1, 0.1], [0.01, 0.01, 0.01], size_bytes=1 * KiB,
                            config={"min_bandwidth_floor": 0.5})
        self.assertEqual(v["verdict"], reg.INCONCLUSIVE)

    def test_separation_gate_blocks_overlap(self):
        # Median drop (10%) exceeds the 5% large-tier threshold, but B is highly
        # variable and overlaps A, so the separation gate must veto the candidate.
        a = [100.0, 100.0, 100.0, 100.0, 100.0]
        b = [70.0, 85.0, 90.0, 110.0, 115.0]  # median 90, but p75(B)=110 >= p25(A)=100
        v = reg.compare_key(a, b, size_bytes=1 * GiB)
        self.assertGreater(v["rel_drop"], 0.05)  # threshold gate alone would pass
        self.assertFalse(v["candidate"])          # separation gate vetoes it


class TestDetectRegressions(unittest.TestCase):
    def test_identical_runs_no_regression(self):
        rng = random.Random(0)
        runs = [_make_run(rng) for _ in range(5)]
        # Compare the exact same runs A vs A.
        report = reg.detect_regressions(runs, runs)
        self.assertFalse(report["summary"]["has_regression"])
        self.assertEqual(report["summary"]["regressions"], 0)

    def test_real_regression_band_confirmed(self):
        rng = random.Random(1)
        a_runs = [_make_run(rng, bw_scale=1.0) for _ in range(5)]
        # B is 15% slower across ALL sizes -> large tier (5%/10%) easily flagged,
        # and the regression spans many adjacent sizes so adjacency confirms.
        b_runs = [_make_run(rng, bw_scale=0.85) for _ in range(5)]
        report = reg.detect_regressions(a_runs, b_runs)
        self.assertTrue(report["summary"]["has_regression"])
        # Large-tier sizes must be among the confirmed regressions.
        big = [r for r in report["regressions"] if r["key"]["size"] >= 128 * MiB]
        self.assertTrue(len(big) >= 2)

    def test_isolated_candidate_not_confirmed(self):
        # Construct A and B identical except one isolated large size dropped hard.
        sizes = [256 * MiB, 512 * MiB, 1 * GiB, 2 * GiB, 4 * GiB]
        a_runs = []
        b_runs = []
        for _ in range(5):
            a_rows = [{"name": "AllReduce", "size": s, "type": "float", "inPlace": 1,
                       "busBw": 300.0, "algBw": 150.0, "time": 1.0} for s in sizes]
            b_rows = []
            for s in sizes:
                bw = 300.0
                if s == 1 * GiB:  # single isolated size regressed 30%
                    bw = 210.0
                b_rows.append({"name": "AllReduce", "size": s, "type": "float", "inPlace": 1,
                               "busBw": bw, "algBw": bw / 2, "time": 1.0})
            a_runs.append(a_rows)
            b_runs.append(b_rows)
        report = reg.detect_regressions(a_runs, b_runs, config={"adjacency_min_run": 2})
        # The single isolated size is a candidate but must NOT be confirmed.
        self.assertFalse(report["summary"]["has_regression"])
        self.assertEqual(report["summary"]["candidates"], 1)

    def test_isolated_candidate_confirmed_when_adjacency_disabled(self):
        sizes = [1 * GiB, 2 * GiB]
        a_runs = [[{"name": "AllReduce", "size": 1 * GiB, "type": "float", "inPlace": 1,
                    "busBw": 300.0, "algBw": 150.0, "time": 1.0},
                   {"name": "AllReduce", "size": 2 * GiB, "type": "float", "inPlace": 1,
                    "busBw": 300.0, "algBw": 150.0, "time": 1.0}] for _ in range(5)]
        b_runs = [[{"name": "AllReduce", "size": 1 * GiB, "type": "float", "inPlace": 1,
                    "busBw": 210.0, "algBw": 105.0, "time": 1.0},
                   {"name": "AllReduce", "size": 2 * GiB, "type": "float", "inPlace": 1,
                    "busBw": 300.0, "algBw": 150.0, "time": 1.0}] for _ in range(5)]
        report = reg.detect_regressions(a_runs, b_runs, config={"adjacency_min_run": 1})
        self.assertTrue(report["summary"]["has_regression"])

    def test_small_message_regression_below_threshold_skipped(self):
        # A 12% drop at a small (noisy) size is below the 20% small-tier threshold.
        size = 4 * KiB
        a_runs = [[{"name": "AllReduce", "size": size, "type": "float", "inPlace": 1,
                    "busBw": 10.0, "algBw": 5.0, "time": 1.0}] for _ in range(5)]
        b_runs = [[{"name": "AllReduce", "size": size, "type": "float", "inPlace": 1,
                    "busBw": 8.8, "algBw": 4.4, "time": 1.0}] for _ in range(5)]
        report = reg.detect_regressions(a_runs, b_runs)
        self.assertFalse(report["summary"]["has_regression"])


class TestMonteCarlo(unittest.TestCase):
    """Empirical stability checks with realistic noise."""

    N_TRIALS = 200
    REPEATS = 5

    def test_false_positive_rate_is_tiny(self):
        """A vs A (same distribution): confirmed regressions should be ~never."""
        rng = random.Random(12345)
        false_positives = 0
        for _ in range(self.N_TRIALS):
            a_runs = [_make_run(rng, bw_scale=1.0) for _ in range(self.REPEATS)]
            b_runs = [_make_run(rng, bw_scale=1.0) for _ in range(self.REPEATS)]
            report = reg.detect_regressions(a_runs, b_runs)
            if report["summary"]["has_regression"]:
                false_positives += 1
        fp_rate = false_positives / self.N_TRIALS
        # Triple-gated detector should essentially never false-positive.
        self.assertLessEqual(fp_rate, 0.01, f"false-positive rate too high: {fp_rate:.3f}")

    def test_detection_rate_for_real_regression(self):
        """A 15% uniform slowdown should be caught nearly every time."""
        rng = random.Random(999)
        detected = 0
        for _ in range(self.N_TRIALS):
            a_runs = [_make_run(rng, bw_scale=1.0) for _ in range(self.REPEATS)]
            b_runs = [_make_run(rng, bw_scale=0.85) for _ in range(self.REPEATS)]
            report = reg.detect_regressions(a_runs, b_runs)
            if report["summary"]["has_regression"]:
                detected += 1
        detect_rate = detected / self.N_TRIALS
        self.assertGreaterEqual(detect_rate, 0.95, f"detection rate too low: {detect_rate:.3f}")


class TestThresholdDerivation(unittest.TestCase):
    def test_measure_noise_reflects_input_cv(self):
        """measured CV should be in the right ballpark for each tier's noise."""
        rng = random.Random(7)
        # Many repeats of the same build -> control dataset.
        control = [_make_run(rng, bw_scale=1.0) for _ in range(20)]
        noise = reg.measure_noise(control)
        # large tier noise (cv ~0.025) should be well below small tier (cv ~0.12)
        self.assertIsNotNone(noise["large"])
        self.assertIsNotNone(noise["mid"])
        self.assertLess(noise["large"]["cv_p95"], 0.08)
        self.assertGreater(noise["mid"]["cv_p95"], noise["large"]["cv_p95"])

    def test_derive_thresholds_above_noise(self):
        rng = random.Random(8)
        control = [_make_run(rng, bw_scale=1.0) for _ in range(20)]
        derived = reg.derive_thresholds(control, safety_factor=2.0)
        th = derived["thresholds"]
        # Each derived threshold must sit above the measured p95 noise for the tier.
        for tier in ("mid", "large"):
            if derived["noise"][tier]:
                self.assertGreaterEqual(th[tier], derived["noise"][tier]["cv_p95"])
        # Thresholds respect the configured minimums.
        self.assertGreaterEqual(th["large"], 0.03)

    def test_derived_thresholds_give_zero_false_positives(self):
        """End-to-end: thresholds derived from control data => no A/B false positives."""
        rng = random.Random(101)
        control = [_make_run(rng, bw_scale=1.0) for _ in range(15)]
        derived = reg.derive_thresholds(control, safety_factor=2.0)
        cfg = {"thresholds": derived["thresholds"]}
        fp = 0
        for _ in range(150):
            a = [_make_run(rng, bw_scale=1.0) for _ in range(7)]
            b = [_make_run(rng, bw_scale=1.0) for _ in range(7)]
            if reg.detect_regressions(a, b, config=cfg)["summary"]["has_regression"]:
                fp += 1
        self.assertEqual(fp, 0, f"derived thresholds produced {fp} false positives")


if __name__ == "__main__":
    unittest.main()
