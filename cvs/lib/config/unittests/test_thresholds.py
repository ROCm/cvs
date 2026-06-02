"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

# Unit tests for cvs/lib/config/thresholds.py: the threshold verdict evaluators
# and their discriminated-union parsing.
#
# Pinned invariants:
#   - Each evaluator's verdict semantics: rate/percentile/goodput on scalars and
#     samples, convergence/monotonicity/stability over trajectories, and a
#     fail-with-actual=None verdict on missing data.
#   - The discriminated union round-trips through parse_config with extra="forbid"
#     on each member; an unknown type or misspelled field raises ConfigError.
#   - Operator-authored numeric fields (percentile, by_step) are range-checked at
#     load, not indexed/sliced blindly at evaluate time.

import unittest

from pydantic import ValidationError

from cvs.lib.config import parse_config
from cvs.lib.config.loader import ConfigError
from cvs.lib.config.thresholds import (
    ConvergenceThreshold,
    GoodputThreshold,
    MonotonicityThreshold,
    PercentileThreshold,
    RateThreshold,
    ResultView,
    StabilityThreshold,
)

from ._fixtures import iter_bases


class TestThresholds(unittest.TestCase):
    def test_percentile_explicit_op(self):
        view = ResultView(samples=[{"ttft_ms": x} for x in [10, 20, 30, 40, 200]])
        ok = PercentileThreshold(metric="ttft_ms", percentile=50, op="<=", value=100).evaluate(view)
        self.assertTrue(ok.passed)
        bad = PercentileThreshold(metric="ttft_ms", percentile=99, op="<=", value=50).evaluate(view)
        self.assertFalse(bad.passed)

    def test_rate(self):
        view = ResultView(scalars={"total_throughput": 1500})
        self.assertTrue(RateThreshold(metric="total_throughput", op=">=", value=1200).evaluate(view).passed)

    def test_goodput_filtered(self):
        samples = [{"ttft_ms": 100, "tpot_ms": 10} for _ in range(8)] + [{"ttft_ms": 999, "tpot_ms": 99}]
        view = ResultView(scalars={"elapsed_s": 1.0}, samples=samples)
        verdict = GoodputThreshold(op=">=", value=8, where={"ttft_ms": "<=300", "tpot_ms": "<=40"}).evaluate(view)
        self.assertEqual(verdict.actual, 8.0)
        self.assertTrue(verdict.passed)

    def test_convergence_full_vs_by_step(self):
        traj = [{"step": i, "metric": "loss", "value": v} for i, v in enumerate([5.0, 1.0, 0.05])]
        view = ResultView(trajectory=traj)
        self.assertTrue(ConvergenceThreshold(metric="loss", target=0.0, epsilon=0.1).evaluate(view).passed)
        # by_step=2 only inspects [5.0, 1.0] -> never within epsilon -> not converged.
        early = ConvergenceThreshold(metric="loss", target=0.0, epsilon=0.1, by_step=2).evaluate(view)
        self.assertFalse(early.passed)

    def test_stability_variance_and_thin_data(self):
        steady = ResultView(samples=[{"ttft_ms": 10.0} for _ in range(5)])
        self.assertTrue(StabilityThreshold(metric="ttft_ms", max_variance=1.0).evaluate(steady).passed)
        noisy = ResultView(samples=[{"ttft_ms": v} for v in [1.0, 100.0, 1.0, 100.0]])
        self.assertFalse(StabilityThreshold(metric="ttft_ms", max_variance=1.0).evaluate(noisy).passed)
        thin = StabilityThreshold(metric="ttft_ms", max_variance=1.0).evaluate(ResultView(samples=[{"ttft_ms": 10.0}]))
        self.assertFalse(thin.passed)
        self.assertIn("not enough", thin.detail)

    def test_missing_data_verdicts(self):
        empty = ResultView()
        for verdict in (
            RateThreshold(metric="total_throughput", op=">=", value=1).evaluate(empty),
            PercentileThreshold(metric="ttft_ms", op="<=", value=1).evaluate(empty),
            GoodputThreshold(op=">=", value=1, where={"ttft_ms": "<=1"}).evaluate(empty),
        ):
            self.assertFalse(verdict.passed)
            self.assertIsNone(verdict.actual)


THRESHOLD_TYPES = [
    {"type": "rate", "metric": "total_throughput", "op": ">=", "value": 1000},
    {"type": "percentile", "metric": "ttft_ms", "percentile": 99, "op": "<=", "value": 50},
    {"type": "goodput", "op": ">=", "value": 8, "where": {"ttft_ms": "<=300"}},
    {"type": "monotonicity", "metric": "loss", "direction": "non_increasing"},
    {"type": "convergence", "metric": "loss", "target": 0.0, "epsilon": 0.1},
    {"type": "stability", "metric": "ttft_ms", "max_variance": 5.0},
]


class TestThresholdUnionParsing(unittest.TestCase):
    """The discriminated union is the core G2 parsing contract; exercise it via
    parse_config (not just direct construction) so the discriminator and each
    member's extra="forbid" are actually covered. Thresholds live on the base
    config, so this is parametrized over every framework fixture."""

    def test_all_threshold_types_round_trip(self):
        for framework, base in iter_bases():
            with self.subTest(framework=framework):
                cfg = parse_config({**base, "thresholds": THRESHOLD_TYPES})
                self.assertEqual(
                    [t.type for t in cfg.thresholds],
                    ["rate", "percentile", "goodput", "monotonicity", "convergence", "stability"],
                )

    def test_unknown_threshold_type_rejected(self):
        for framework, base in iter_bases():
            with self.subTest(framework=framework):
                with self.assertRaises(ConfigError):
                    parse_config({**base, "thresholds": [{"type": "bogus", "metric": "x"}]})

    def test_misspelled_threshold_field_rejected(self):
        bad = {"type": "rate", "metric": "x", "op": ">=", "value": 1, "bogus": 2}
        for framework, base in iter_bases():
            with self.subTest(framework=framework):
                with self.assertRaises(ConfigError):
                    parse_config({**base, "thresholds": [bad]})

    def test_b2_malformed_where_rejected_via_config(self):
        bad = {"type": "goodput", "op": ">=", "value": 1, "where": {"ttft_ms": "garbage"}}
        for framework, base in iter_bases():
            with self.subTest(framework=framework):
                with self.assertRaises(ConfigError):
                    parse_config({**base, "thresholds": [bad]})


class TestThresholdBakeIns(unittest.TestCase):
    def test_b1_monotonicity_guards_windowed_tail(self):
        # series len 3, window 0.25 -> tail = last 1 point. Pre-fix this would
        # vacuously pass (empty pairwise loop); post-fix it is not-enough-data.
        traj = [{"step": i, "metric": "loss", "value": v} for i, v in enumerate([3.0, 2.0, 1.0])]
        verdict = MonotonicityThreshold(metric="loss", window=0.25).evaluate(ResultView(trajectory=traj))
        self.assertFalse(verdict.passed)
        self.assertIn("not enough", verdict.detail)

    def test_b1_monotonicity_still_evaluates_full_window(self):
        traj = [{"step": i, "metric": "loss", "value": v} for i, v in enumerate([5.0, 4.0, 3.0, 2.0, 1.0])]
        verdict = MonotonicityThreshold(metric="loss", window=1.0).evaluate(ResultView(trajectory=traj))
        self.assertTrue(verdict.passed)

    def test_b2_goodput_rejects_malformed_where_at_load(self):
        with self.assertRaises(ValidationError):
            GoodputThreshold(op=">=", value=8, where={"ttft_ms": "garbage"})

    def test_b2_goodput_accepts_wellformed_where(self):
        GoodputThreshold(op=">=", value=8, where={"ttft_ms": "<=300", "tpot_ms": ">= 1.5"})


class TestNumericFieldBounds(unittest.TestCase):
    """Operator-authored numeric fields must be range-checked at load, not
    indexed/sliced blindly at evaluate time (the percentile IndexError class)."""

    def test_percentile_out_of_range_rejected_directly(self):
        for bad in (150, -5, 100.1):
            with self.assertRaises(ValidationError):
                PercentileThreshold(metric="x", percentile=bad, op="<=", value=1)

    def test_percentile_out_of_range_rejected_via_config(self):
        for framework, base in iter_bases():
            for bad in (150, -5):
                with self.subTest(framework=framework, percentile=bad):
                    thr = {"type": "percentile", "metric": "x", "op": "<=", "value": 1, "percentile": bad}
                    with self.assertRaises(ConfigError):
                        parse_config({**base, "thresholds": [thr]})

    def test_percentile_boundaries_accepted(self):
        for ok in (0, 99, 100):
            PercentileThreshold(metric="x", percentile=ok, op="<=", value=1)

    def test_convergence_by_step_must_be_positive(self):
        for bad in (0, -1):
            with self.assertRaises(ValidationError):
                ConvergenceThreshold(metric="loss", target=0.0, epsilon=0.1, by_step=bad)
        ConvergenceThreshold(metric="loss", target=0.0, epsilon=0.1, by_step=1)


if __name__ == "__main__":
    unittest.main()
