'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs/lib/report/verdict.py threshold evaluation.
'''

import unittest

from cvs.lib.report.verdict import ThresholdViolation, _check_one, evaluate_all


class CheckOneTests(unittest.TestCase):
    def test_min_pass_and_fail(self):
        self.assertIsNone(_check_one("m", 10, {"kind": "min", "value": 5}))
        self.assertIsNotNone(_check_one("m", 3, {"kind": "min", "value": 5}))

    def test_max_pass_and_fail(self):
        self.assertIsNone(_check_one("m", 3, {"kind": "max", "value": 5}))
        self.assertIsNotNone(_check_one("m", 9, {"kind": "max", "value": 5}))

    def test_info_never_gates(self):
        # Record-only: always passes regardless of value (mirrors utils/verdict).
        self.assertIsNone(_check_one("m", 999999, {"kind": "info", "value": 0}))
        self.assertIsNone(_check_one("m", 0, {"kind": "info", "value": 100}))

    def test_unknown_kind_still_flagged(self):
        self.assertIsNotNone(_check_one("m", 1, {"kind": "bogus", "value": 1}))


class EvaluateAllTests(unittest.TestCase):
    def test_info_metric_does_not_raise(self):
        # A present info metric must not be a violation even under enforcement.
        evaluate_all({"m": 12345.0}, {"m": {"kind": "info", "value": 1.0}})  # no raise

    def test_gating_metric_violation_raises(self):
        with self.assertRaises(ThresholdViolation):
            evaluate_all({"m": 1.0}, {"m": {"kind": "min", "value": 5.0}})

    def test_none_actual_raises_even_for_info(self):
        # None handling is unchanged: a missing value is reported before kind logic.
        with self.assertRaises(ThresholdViolation):
            evaluate_all({"m": None}, {"m": {"kind": "info", "value": 1.0}})


if __name__ == "__main__":
    unittest.main()
