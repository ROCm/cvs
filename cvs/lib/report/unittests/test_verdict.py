'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.report.verdict.
'''

import unittest

from cvs.lib.report.verdict import ThresholdViolation, evaluate_all


class TestEvaluateAll(unittest.TestCase):
    def test_info_spec_never_fails(self):
        self.assertIsNone(evaluate_all({"failed": 99.0}, {"failed": {"kind": "info", "value": 0}}))
        with self.assertRaises(ThresholdViolation) as caught:
            evaluate_all({"failed": 99.0}, {"failed": {"kind": "max", "value": 0}})
        self.assertEqual(caught.exception.violations, ["failed: actual 99.0 > max 0.0"])

    def test_unknown_kind_still_fails(self):
        with self.assertRaises(ThresholdViolation) as caught:
            evaluate_all({"failed": 1.0}, {"failed": {"kind": "bogus", "value": 0}})
        self.assertEqual(caught.exception.violations, ["failed: unknown threshold kind 'bogus'"])


if __name__ == "__main__":
    unittest.main()
