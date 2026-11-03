"""An empty node set and a status-only skip must not score as a pass."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.report import PreflightReportGenerator
from cvs.tests.preflight import preflight_checks

BACKFILL = {'status': 'SKIPPED', 'message': 'Check was skipped due to earlier failures'}


class TestEmptyNodeSetIsNotAPass(unittest.TestCase):
    def setUp(self):
        self.generator = PreflightReportGenerator(MagicMock(), {}, {})

    def test_status_skipped_without_flag_is_skipped(self):
        ifoe = self.generator._summarize_ifoe_l2_results(BACKFILL)
        tier1 = self.generator._summarize_node_smoke_tier1_results(BACKFILL)
        tier3 = self.generator._summarize_node_smoke_tier3_results(BACKFILL)
        transferbench = self.generator._summarize_transferbench_smoke_results(BACKFILL)
        self.assertEqual(ifoe['status'], 'SKIPPED')
        self.assertEqual(tier1['status'], 'SKIPPED')
        self.assertEqual(tier3['status'], 'SKIPPED')
        self.assertEqual(self.generator._summarize_node_smoke_tier2_results(BACKFILL)['status'], 'SKIPPED')
        self.assertEqual(transferbench['status'], 'SKIPPED')
        self.assertEqual(self.generator._summarize_ssh_connectivity_results(BACKFILL)['status'], 'SKIPPED')
        self.assertIn('earlier failures', ifoe['summary'])
        self.assertIn('earlier failures', tier1['summary'])
        self.assertNotIn('0/0', ifoe['summary'])
        self.assertNotIn('0/0', tier1['summary'])
        self.assertNotIn('0/0', transferbench['summary'])

    def test_ran_with_no_nodes_is_error(self):
        empty_nodes = {'node_results': {}}
        self.assertEqual(self.generator._summarize_ifoe_l2_results(empty_nodes)['status'], 'ERROR')
        self.assertEqual(self.generator._summarize_node_smoke_tier1_results(empty_nodes)['status'], 'ERROR')
        self.assertEqual(self.generator._summarize_node_smoke_tier3_results(empty_nodes)['status'], 'ERROR')
        tier2 = {'tier2_perf': True, 'node_results': {}}
        self.assertEqual(self.generator._summarize_node_smoke_tier2_results(tier2)['status'], 'ERROR')
        health = {'status': 'PASS', 'node_results': {}}
        self.assertEqual(self.generator._summarize_node_health_results(health)['status'], 'ERROR')
        reachability = {'total_nodes': 0, 'reachable_nodes': 0, 'unreachable_nodes': []}
        self.assertEqual(self.generator._summarize_reachability_results(reachability)['status'], 'ERROR')
        self.assertNotIn('0/0', self.generator._summarize_reachability_results(reachability)['summary'])
        pairs = {'total_pairs': 0, 'successful_pairs': 0, 'failed_pairs': 0, 'mode': 'basic'}
        self.assertEqual(self.generator._summarize_connectivity_results(pairs)['status'], 'ERROR')
        self.assertEqual(self.generator._summarize_ssh_connectivity_results(pairs)['status'], 'ERROR')

    def test_measured_results_still_pass(self):
        tier1 = {'node_results': {'node0': {'status': 'PASS'}}, 'failed_nodes': [], 'unknown_nodes': []}
        tier1_summary = self.generator._summarize_node_smoke_tier1_results(tier1)
        self.assertEqual(tier1_summary['total_nodes'], 1)
        self.assertNotEqual(tier1_summary['status'], 'ERROR')
        reachability = {'total_nodes': 2, 'reachable_nodes': 2, 'unreachable_nodes': []}
        self.assertEqual(self.generator._summarize_reachability_results(reachability)['status'], 'PASS')
        pairs = {'total_pairs': 1, 'successful_pairs': 1, 'failed_pairs': 0, 'mode': 'basic'}
        self.assertEqual(self.generator._summarize_connectivity_results(pairs)['status'], 'PASS')

    def test_too_few_rdma_hosts_stay_skipped(self):
        skipped = {
            'skipped': True,
            'message': 'Too few nodes for RDMA after reachability, interface, and GID pruning',
            'total_pairs': 0,
            'successful_pairs': 0,
            'failed_pairs': 0,
        }
        self.assertEqual(self.generator._summarize_connectivity_results(skipped)['status'], 'SKIPPED')

    def test_missing_reachability_stays_unknown(self):
        self.assertEqual(self.generator._summarize_reachability_results(None)['status'], 'UNKNOWN')
        self.assertEqual(self.generator._summarize_reachability_results({})['status'], 'UNKNOWN')

    def test_backfill_summary_passes_until_reachability_is_empty(self):
        results = {
            'gid_consistency': BACKFILL,
            'rdma_mtu': BACKFILL,
            'rdma_connectivity': BACKFILL,
            'rocm_versions': BACKFILL,
            'interface_names': BACKFILL,
            'node_health': BACKFILL,
            'ifoe_l2_connectivity': BACKFILL,
            'transferbench_smoke': BACKFILL,
            'node_smoke_tier1': BACKFILL,
            'node_smoke_tier3': BACKFILL,
        }
        summary = PreflightReportGenerator(MagicMock(), results, {})._generate_preflight_summary()
        self.assertEqual(summary['checks']['ifoe_l2_connectivity']['status'], 'SKIPPED')
        self.assertEqual(summary['checks']['node_smoke_tier1']['status'], 'SKIPPED')
        self.assertEqual(summary['checks']['node_smoke_tier3']['status'], 'SKIPPED')
        self.assertEqual(summary['checks']['ssh_reachability']['status'], 'UNKNOWN')
        self.assertEqual(summary['overall_status'], 'PASS')

        results['node_reachability'] = {'total_nodes': 0, 'reachable_nodes': 0, 'unreachable_nodes': []}
        summary = PreflightReportGenerator(MagicMock(), results, {})._generate_preflight_summary()
        self.assertEqual(summary['checks']['ssh_reachability']['status'], 'ERROR')
        self.assertEqual(summary['overall_status'], 'FAIL')

    def test_ifoe_skip_html_uses_status(self):
        html_report = self.generator._generate_ifoe_l2_html(BACKFILL)
        self.assertIn('earlier failures', html_report)
        self.assertNotIn('0/0', html_report)


class TestEmptyReachabilityFailsReportRow(unittest.TestCase):
    def setUp(self):
        preflight_checks.globals.error_list.clear()
        self.previous_results = dict(preflight_checks.preflight_results)
        preflight_checks.preflight_results.clear()
        self.request = SimpleNamespace(config=SimpleNamespace())

    def tearDown(self):
        preflight_checks.preflight_results.clear()
        preflight_checks.preflight_results.update(self.previous_results)

    def test_empty_reachability_fails_the_report_row(self):
        config = {'node_check': {'enabled': False}, 'reporting': {'generate_html_report': False}}
        preflight_checks.preflight_results['node_reachability'] = {
            'total_nodes': 0,
            'reachable_nodes': 0,
            'unreachable_nodes': [],
        }
        with (
            patch.object(preflight_checks.log, 'error') as error,
            patch.object(preflight_checks.log, 'info') as info,
            self.assertRaises(pytest.fail.Exception) as ctx,
        ):
            preflight_checks.test_generate_preflight_report(MagicMock(), config, self.request)
        self.assertIn('overall status is FAIL', str(ctx.exception))
        self.assertIn('Ssh Reachability', str(ctx.exception))
        error_lines = [str(call.args[0]) for call in error.call_args_list if call.args]
        info_lines = [str(call.args[0]) for call in info.call_args_list if call.args]
        self.assertTrue(any('Ssh Reachability: ERROR' in line for line in error_lines))
        self.assertFalse(any('Ssh Reachability: ERROR' in line for line in info_lines))
        summary = preflight_checks.preflight_results['summary']
        self.assertEqual(summary['checks']['ssh_reachability']['status'], 'ERROR')
        self.assertEqual(summary['checks']['ifoe_l2_connectivity']['status'], 'SKIPPED')
        self.assertEqual(summary['checks']['node_smoke_tier1']['status'], 'SKIPPED')
        self.assertEqual(summary['checks']['node_smoke_tier3']['status'], 'SKIPPED')


if __name__ == '__main__':
    unittest.main()
