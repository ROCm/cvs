"""Unit tests for mapping recorded preflight results onto pytest row outcomes."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.report.health_lifecycle import HealthLifecycle
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

    def test_summary_sections_are_not_reported_as_nodes(self):
        reason = self._fail_reason(
            {
                'status': 'FAIL',
                'failed_nodes': ['nodeA'],
                'node_results': {'nodeA': {'status': 'FAIL'}, 'nodeB': {'status': 'PASS'}},
                'vpod_membership': {'status': 'FAIL', 'errors': ['mismatch']},
                'setup_results': {'status': 'FAIL', 'node_results': {}},
                'pod_membership': {'status': 'FAIL'},
            }
        )
        self.assertIn('nodeA', reason)
        self.assertNotIn('vpod_membership', reason)
        self.assertNotIn('setup_results', reason)
        self.assertNotIn('pod_membership', reason)
        self.assertNotIn('node_results', reason)

    def test_node_results_names_hosts_when_failed_nodes_is_absent(self):
        reason = self._fail_reason(
            {
                'status': 'FAIL',
                'node_results': {'nodeB': {'status': 'FAIL'}, 'nodeA': {'status': 'PASS'}},
                'pod_membership': {'status': 'FAIL'},
            }
        )
        self.assertIn('nodeB', reason)
        self.assertNotIn('nodeA', reason)
        self.assertNotIn('pod_membership', reason)

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


class TestTimedPreflightChecksSurfaceOutcomes(unittest.TestCase):
    def setUp(self):
        preflight_checks.globals.error_list.clear()
        self.previous_results = dict(preflight_checks.preflight_results)
        preflight_checks.preflight_results.clear()
        self.assertIsNone(preflight_checks._rundeck_slot['target'])

    def tearDown(self):
        preflight_checks.preflight_results.clear()
        preflight_checks.preflight_results.update(self.previous_results)

    def test_decorated_check_fail_reaches_pytest_without_lifecycle(self):
        @preflight_checks._timed_stage(preflight_checks.GID_CONSISTENCY)
        def check(lifecycle=None):
            preflight_checks.preflight_update_test_result({'status': 'FAIL', 'message': 'GID index invalid'})

        with self.assertRaises(pytest.fail.Exception) as ctx:
            check()
        self.assertIn('GID index invalid', str(ctx.exception))

    def test_decorated_check_skip_reaches_pytest_without_lifecycle(self):
        @preflight_checks._timed_stage(preflight_checks.GID_CONSISTENCY)
        def check(lifecycle=None):
            preflight_checks.preflight_update_test_result({'status': 'SKIPPED', 'skipped': True, 'message': 'disabled'})

        with self.assertRaises(pytest.skip.Exception) as ctx:
            check()
        self.assertIn('disabled', str(ctx.exception))

    def test_decorated_check_records_stage_when_failing(self):
        @preflight_checks._timed_stage(preflight_checks.GID_CONSISTENCY)
        def check(lifecycle=None):
            preflight_checks.preflight_update_test_result({'status': 'FAIL', 'message': 'GID index invalid'})

        lifecycle = HealthLifecycle()
        with self.assertRaises(pytest.fail.Exception):
            check(lifecycle=lifecycle)
        self.assertEqual(lifecycle.report['health'][0][0], preflight_checks.GID_CONSISTENCY)

    def test_interface_check_skip_mode_skips_row(self):
        config = {'connectivity_check': {'rdma': {'connectivity_mode': 'skip'}}}
        with self.assertRaises(pytest.skip.Exception):
            preflight_checks.test_interface_name_consistency(MagicMock(), config)

    def test_gid_check_node_failure_fails_row(self):
        orch = MagicMock()
        orch.all.reachable_hosts = ['nodeA']
        config = {
            'connectivity_check': {'rdma': {'connectivity_mode': 'basic', 'gid_index': '3', 'interfaces': ['eth0']}}
        }
        with patch.object(preflight_checks, 'GidConsistencyCheck') as gid_checker:
            gid_checker.return_value.run.return_value = {
                'nodeA': {'status': 'FAIL', 'errors': ['GID index 3 missing'], 'interfaces': {}}
            }
            with self.assertRaises(pytest.fail.Exception) as ctx:
                preflight_checks.test_gid_consistency(orch, config)

        self.assertIn('nodeA', str(ctx.exception))


class TestPreflightReportOverallVerdict(unittest.TestCase):
    """The report row must fail when the summary overall status is FAIL."""

    def setUp(self):
        preflight_checks.globals.error_list.clear()
        self.previous_results = dict(preflight_checks.preflight_results)
        preflight_checks.preflight_results.clear()
        self.request = SimpleNamespace(config=SimpleNamespace())

    def tearDown(self):
        preflight_checks.preflight_results.clear()
        preflight_checks.preflight_results.update(self.previous_results)

    def test_passing_summary_keeps_report_row_passing(self):
        config = {'node_check': {'enabled': False}, 'reporting': {'generate_html_report': False}}
        self.assertIsNone(preflight_checks.test_generate_preflight_report(MagicMock(), config, self.request))
        self.assertEqual(preflight_checks.preflight_results['summary']['overall_status'], 'PASS')

    def test_failing_summary_fails_report_row(self):
        config = {'node_check': {'enabled': False}, 'reporting': {'generate_html_report': False}}
        preflight_checks.preflight_results['rdma_connectivity'] = {
            'mode': 'basic',
            'total_pairs': 2,
            'successful_pairs': 1,
            'failed_pairs': 1,
        }
        with self.assertRaises(pytest.fail.Exception) as ctx:
            preflight_checks.test_generate_preflight_report(MagicMock(), config, self.request)
        message = str(ctx.exception)
        self.assertIn('overall status is FAIL', message)
        self.assertIn('Rdma Connectivity', message)
        self.assertEqual(preflight_checks.preflight_results['summary']['overall_status'], 'FAIL')

    def test_node_health_gate_still_fails_before_the_overall_message(self):
        config = {'node_check': {'enabled': True}, 'reporting': {'generate_html_report': False}}
        preflight_checks.preflight_results['node_health'] = {
            'status': 'FAIL',
            'failed_nodes': ['nodeA'],
            'node_results': {'nodeA': {'status': 'FAIL'}},
        }
        with self.assertRaises(pytest.fail.Exception) as ctx:
            preflight_checks.test_generate_preflight_report(MagicMock(), config, self.request)
        self.assertIn('Mandatory node-health admission gate failed', str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
