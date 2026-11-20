"""Unit tests for crashed preflight checks in the report summary."""

import unittest
from unittest.mock import MagicMock

from cvs.lib.preflight.report import CRASHED_CHECKS_KEY, PreflightReportGenerator, _apply_crashed_checks
from cvs.tests.preflight import preflight_checks


class TestCrashedCheckSummary(unittest.TestCase):
    def setUp(self):
        preflight_checks.globals.error_list.clear()
        self.previous_results = dict(preflight_checks.preflight_results)
        preflight_checks.preflight_results.clear()

    def tearDown(self):
        preflight_checks.preflight_results.clear()
        preflight_checks.preflight_results.update(self.previous_results)

    def test_missing_payload_crash_is_error(self):
        results = {CRASHED_CHECKS_KEY: {'rdma_connectivity': 'RuntimeError: boom'}}
        summary = PreflightReportGenerator(MagicMock(), results, {})._generate_preflight_summary()
        self.assertEqual(summary['checks']['rdma_connectivity']['status'], 'ERROR')
        self.assertEqual(summary['overall_status'], 'FAIL')
        self.assertFalse(any('ready for performance testing' in line for line in summary['recommendations']))

    def test_partial_passing_payload_crash_is_error(self):
        results = {
            'rdma_connectivity': {'mode': 'basic', 'total_pairs': 1, 'successful_pairs': 1, 'failed_pairs': 0},
            CRASHED_CHECKS_KEY: {'rdma_connectivity': 'RuntimeError: boom'},
        }
        summary = PreflightReportGenerator(MagicMock(), results, {})._generate_preflight_summary()
        self.assertEqual(summary['checks']['rdma_connectivity']['status'], 'ERROR')
        self.assertEqual(summary['checks']['rdma_connectivity']['total_pairs'], 1)

    def test_skipped_payload_crash_is_error(self):
        results = {
            'rdma_connectivity': {'status': 'SKIPPED', 'skipped': True, 'message': 'x'},
            CRASHED_CHECKS_KEY: {'rdma_connectivity': 'RuntimeError: boom'},
        }
        summary = PreflightReportGenerator(MagicMock(), results, {})._generate_preflight_summary()
        self.assertEqual(summary['checks']['rdma_connectivity']['status'], 'ERROR')

    def test_recorded_failure_keeps_status_and_text(self):
        original = {'status': 'FAIL', 'summary': '1/2'}
        checks = {'gid_consistency': original}
        self.assertEqual(_apply_crashed_checks(checks, {'gid_consistency': 'E: x'}), [])
        self.assertEqual(checks['gid_consistency'], original)

    def test_reachability_label_maps_to_ssh_reachability(self):
        results = {CRASHED_CHECKS_KEY: {'node_reachability': 'RuntimeError: boom'}}
        summary = PreflightReportGenerator(MagicMock(), results, {})._generate_preflight_summary()
        self.assertEqual(summary['checks']['ssh_reachability']['status'], 'ERROR')
        self.assertEqual(summary['overall_status'], 'FAIL')

    def test_node_smoke_tier2_crash_is_error(self):
        results = {
            'node_smoke_tier1': {
                'node_results': {'n0': {'status': 'PASS'}},
                'failed_nodes': [],
                'unknown_nodes': [],
            },
            CRASHED_CHECKS_KEY: {'node_smoke_tier2': 'TypeError: x'},
        }
        summary = PreflightReportGenerator(MagicMock(), results, {})._generate_preflight_summary()
        self.assertEqual(summary['checks']['node_smoke_tier2']['status'], 'ERROR')
        self.assertNotEqual(summary['checks']['node_smoke_tier1']['status'], 'ERROR')

    def test_no_crashes_is_noop(self):
        checks = {'gid_consistency': {'status': 'PASS', 'summary': 'ok'}}
        self.assertEqual(_apply_crashed_checks(checks, None), [])
        self.assertEqual(_apply_crashed_checks(checks, {}), [])
        self.assertEqual(checks['gid_consistency']['status'], 'PASS')

    def test_html_renders_crashed_check_and_escapes_message(self):
        results = {CRASHED_CHECKS_KEY: {'rdma_connectivity': 'ValueError: <bad> & co'}}
        gen = PreflightReportGenerator(MagicMock(), results, {})
        gen.results['summary'] = gen._generate_preflight_summary()
        html_text = gen._generate_html_content()
        self.assertIn('Check did not finish', html_text)
        self.assertIn('&lt;bad&gt; &amp; co', html_text)
        self.assertNotIn('<bad>', html_text)


if __name__ == '__main__':
    unittest.main()
