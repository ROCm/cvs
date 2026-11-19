'''Unit tests for the Run Deck threshold check in cvs/lib/report/verdict.py.'''

import unittest

from cvs.lib.report.verdict import _check_one


class TestCheckOne(unittest.TestCase):
    def test_an_info_spec_records_without_a_violation(self):
        self.assertIsNone(_check_one("output_throughput", 1.0, {"kind": "info", "value": 0.0}))

    def test_a_gating_spec_still_reports_its_violation(self):
        self.assertEqual(
            _check_one("output_throughput", 1.0, {"kind": "min", "value": 2.0}),
            "output_throughput: actual 1.0 < min 2.0",
        )


if __name__ == "__main__":
    unittest.main()
