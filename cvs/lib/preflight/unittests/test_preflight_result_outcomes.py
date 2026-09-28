"""Unit tests for mapping recorded preflight results onto pytest row outcomes."""

import os
import sys
import unittest

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.tests.preflight import preflight_checks


class TestPreflightUpdateTestResult(unittest.TestCase):
    """A recorded skip or failure must surface on the row, not report as passed."""

    def setUp(self):
        preflight_checks.globals.error_list.clear()

    def _skip_reason(self, result):
        with self.assertRaises(pytest.skip.Exception) as ctx:
            preflight_checks.preflight_update_test_result(result)
        return str(ctx.exception)

    def _fail_reason(self, result):
        with self.assertRaises(pytest.fail.Exception) as ctx:
            preflight_checks.preflight_update_test_result(result)
        return str(ctx.exception)

    def test_no_result_keeps_row_passing(self):
        self.assertIsNone(preflight_checks.preflight_update_test_result())
        self.assertIsNone(preflight_checks.preflight_update_test_result({'status': 'PASS'}))

    def test_status_skipped_skips_row(self):
        reason = self._skip_reason({'status': 'SKIPPED', 'skipped': True, 'message': 'disabled by configuration'})
        self.assertIn('disabled by configuration', reason)

    def test_skipped_flag_without_status_skips_row(self):
        reason = self._skip_reason({'mode': 'skip', 'skipped': True, 'message': 'L2 ping disabled'})
        self.assertIn('L2 ping disabled', reason)

    def test_blocked_result_skips_row(self):
        reason = self._skip_reason(preflight_checks._blocked_by_node_health('RDMA connectivity'))
        self.assertIn('mandatory node-health admission failed', reason)

    def test_status_fail_fails_row(self):
        reason = self._fail_reason({'status': 'FAIL', 'message': 'gate failed; see report'})
        self.assertIn('gate failed; see report', reason)

    def test_per_node_failure_fails_row_and_names_nodes(self):
        reason = self._fail_reason({'nodeB': {'status': 'FAIL'}, 'nodeA': {'status': 'FAIL'}, 'nodeC': {}})
        self.assertIn('nodeA', reason)
        self.assertIn('nodeB', reason)

    def test_per_node_all_passing_keeps_row_passing(self):
        self.assertIsNone(
            preflight_checks.preflight_update_test_result({'nodeA': {'status': 'PASS'}, 'nodeB': {'status': 'PASS'}})
        )

    def test_warning_status_keeps_row_passing(self):
        self.assertIsNone(preflight_checks.preflight_update_test_result({'status': 'WARNING'}))

    def test_accumulated_errors_are_cleared_before_reporting(self):
        preflight_checks.globals.error_list.extend(['boom'])
        preflight_checks.preflight_update_test_result({'status': 'PASS'})
        self.assertEqual(preflight_checks.globals.error_list, [])


if __name__ == "__main__":
    unittest.main()
