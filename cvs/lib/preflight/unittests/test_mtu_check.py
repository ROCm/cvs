"""Unit tests for RDMA MTU preflight validation."""

import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from cvs.lib.preflight.mtu_check import RdmaMtuCheck
from cvs.lib.preflight.report import PreflightReportGenerator
from cvs.parsers.schemas import PreflightConfigFile
from cvs.tests.preflight import preflight_checks


def _probe(dev, netdev='eth0', netdev_mtu='9000', active='4096', max_mtu='4096'):
    return '\n'.join(
        (
            f'DEVICE:{dev}',
            f'NETDEV:{netdev}',
            f'NETDEV_MTU:{netdev_mtu}',
            f'ACTIVE_MTU:{active}',
            f'MAX_MTU:{max_mtu}',
        )
    )


class TestBuildProbeCommand(unittest.TestCase):
    def test_command_probes_quoted_interfaces_and_both_mtus(self):
        checker = RdmaMtuCheck(MagicMock(), ['mlx5_0', 'bad name'], gid_index='7')
        command = checker._build_mtu_probe_command()
        self.assertIn("mlx5_0 'bad name'", command)
        self.assertIn('ibv_devinfo -d "$dev" -i 1', command)
        self.assertIn('/sys/class/net/$ndev/mtu', command)
        self.assertIn('rdma link show "$dev/1"', command)
        self.assertIn('gid_attrs/ndevs/7', command)
        self.assertIn('{print $2; exit}', command)

        quoted = RdmaMtuCheck(MagicMock(), ['rdma0'], gid_index='$(unsafe)')._build_mtu_probe_command()
        self.assertIn("gid_attrs/ndevs/\"'$(unsafe)'", quoted)


class TestEvaluateNode(unittest.TestCase):
    def setUp(self):
        self.checker = RdmaMtuCheck(MagicMock(), ['rdma0'])

    def test_good_mtus_pass_and_record_measurements(self):
        result = self.checker._evaluate_node('n1', _probe('rdma0'))
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['interfaces']['rdma0']['status'], 'OK')
        self.assertEqual(result['interfaces']['rdma0']['netdev_mtu'], 9000)
        self.assertEqual(result['interfaces']['rdma0']['active_mtu'], 4096)
        self.assertEqual(result['interfaces']['rdma0']['max_mtu'], 4096)
        self.assertEqual(result['min_netdev_mtu'], 4200)
        self.assertEqual(result['min_active_mtu'], 4096)

    def test_low_netdev_and_active_mtu_fail(self):
        result = self.checker._evaluate_node('n1', _probe('rdma0', netdev_mtu='1500', active='1024'))
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(len(result['errors']), 2)
        self.assertIn('jumbo', result['errors'][0])
        self.assertIn('eth0', result['errors'][0])
        self.assertIn('1024', result['errors'][1])

    def test_only_low_active_mtu_fails(self):
        result = self.checker._evaluate_node('n1', _probe('rdma0', active='1024'))
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(len(result['errors']), 1)
        self.assertIn('active_mtu', result['errors'][0])
        self.assertIn('netdev eth0 MTU 9000', result['errors'][0])

    def test_threshold_boundary_passes(self):
        result = self.checker._evaluate_node('n1', _probe('rdma0', netdev_mtu='4200'))
        self.assertEqual(result['status'], 'PASS')

    def test_missing_device_fails(self):
        result = self.checker._evaluate_node('n1', 'DEVICE:rdma0\nDEVICE_MISSING:Interface not found')
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('not found', result['errors'][0])

    def test_empty_netdev_mtu_fails(self):
        result = self.checker._evaluate_node('n1', _probe('rdma0', netdev_mtu=''))
        self.assertIn('Could not determine netdev MTU', result['errors'][0])

    def test_empty_active_mtu_fails(self):
        result = self.checker._evaluate_node('n1', _probe('rdma0', active=''))
        self.assertEqual(len(result['errors']), 1)
        self.assertIn('ibv_devinfo', result['errors'][0])

    def test_nonnumeric_netdev_mtu_fails_without_exception(self):
        result = self.checker._evaluate_node('n1', _probe('rdma0', netdev_mtu='abc'))
        self.assertEqual(result['status'], 'FAIL')
        self.assertIsNone(result['interfaces']['rdma0']['netdev_mtu'])

    def test_zero_thresholds_disable_individual_comparisons(self):
        active_disabled = RdmaMtuCheck(MagicMock(), ['rdma0'], min_active_mtu=0)
        netdev_disabled = RdmaMtuCheck(MagicMock(), ['rdma0'], min_netdev_mtu=0)
        self.assertEqual(active_disabled._evaluate_node('n1', _probe('rdma0', active='1024'))['status'], 'PASS')
        self.assertEqual(netdev_disabled._evaluate_node('n1', _probe('rdma0', netdev_mtu='1500'))['status'], 'PASS')

    def test_absent_expected_interface_fails(self):
        result = self.checker._evaluate_node('n1', _probe('other'))
        self.assertIn('No MTU data returned for rdma0', result['errors'])

    def test_unreachable_output_fails_with_context(self):
        result = self.checker._evaluate_node('n1', 'ABORT: Host Unreachable Error')
        self.assertEqual(result['status'], 'FAIL')
        self.assertTrue(any('Host Unreachable Error' in error for error in result['errors']))


class TestRun(unittest.TestCase):
    def test_one_parallel_probe_returns_node_results(self):
        orch = MagicMock()
        orch.all.exec.return_value = {
            'n1': _probe('rdma0'),
            'n2': _probe('rdma0', netdev_mtu='1500', active='1024'),
        }
        checker = RdmaMtuCheck(orch, ['rdma0'])
        results = checker.run()
        orch.all.exec.assert_called_once()
        self.assertEqual(orch.all.exec.call_args.kwargs['timeout'], 60)
        self.assertEqual(set(results), {'n1', 'n2'})
        self.assertEqual(results['n1']['status'], 'PASS')
        self.assertEqual(results['n2']['status'], 'FAIL')


class TestRdmaMtuSchema(unittest.TestCase):
    def test_defaults(self):
        config = PreflightConfigFile.model_validate(
            {'connectivity_check': {'rdma': {'connectivity_mode': 'basic', 'interfaces': ['x']}}}
        )
        self.assertTrue(config.connectivity_check.rdma.mtu_check)
        self.assertEqual(config.connectivity_check.rdma.min_netdev_mtu, 4200)
        self.assertEqual(config.connectivity_check.rdma.min_active_mtu, 4096)

    def test_invalid_thresholds(self):
        for value in (3000, -1):
            key = 'min_active_mtu' if value == 3000 else 'min_netdev_mtu'
            with self.subTest(key=key), self.assertRaises(ValidationError):
                PreflightConfigFile.model_validate({'connectivity_check': {'rdma': {key: value}}})


class TestRdmaMtuReport(unittest.TestCase):
    def setUp(self):
        self.good = RdmaMtuCheck(MagicMock(), ['rdma0'])._evaluate_node('n1', _probe('rdma0'))
        self.bad = RdmaMtuCheck(MagicMock(), ['rdma0'])._evaluate_node(
            'n<2', _probe('rdma0', netdev_mtu='1500', active='1024')
        )
        self.report = PreflightReportGenerator(MagicMock(), {}, {})

    def test_pass_fail_and_skipped_summaries(self):
        passing = self.report._summarize_rdma_mtu_results({'n1': self.good})
        failing = self.report._summarize_rdma_mtu_results({'n1': self.good, 'n<2': self.bad})
        skipped = self.report._summarize_rdma_mtu_results({'status': 'SKIPPED', 'skipped': True})
        self.assertEqual(passing['status'], 'PASS')
        self.assertEqual(failing['status'], 'FAIL')
        self.assertEqual(failing['failed_nodes'], ['n<2'])
        self.assertTrue(failing['summary'].startswith('1/2'))
        self.assertEqual(failing['netdev_mtus'], [1500, 9000])
        self.assertEqual(skipped['status'], 'SKIPPED')

    def test_failure_html_escapes_values(self):
        self.assertEqual(self.report._generate_rdma_mtu_html({'n1': self.good}), '')
        self.assertEqual(self.report._generate_rdma_mtu_html({'status': 'SKIPPED', 'skipped': True}), '')
        section = self.report._generate_rdma_mtu_html({'n<2': self.bad})
        self.assertIn('RDMA MTU (Jumbo Frames) Issues', section)
        self.assertIn('n&lt;2', section)
        self.assertIn('1500', section)
        self.assertNotIn('n<2', section)

    def test_overall_summary_and_recommendation(self):
        self.report.results = {'rdma_mtu': {'n<2': self.bad}}
        summary = self.report._generate_preflight_summary()
        self.assertEqual(summary['checks']['rdma_mtu']['status'], 'FAIL')
        self.assertEqual(summary['overall_status'], 'FAIL')
        self.assertTrue(any('jumbo' in recommendation for recommendation in summary['recommendations']))


class TestRdmaMtuPreflightStep(unittest.TestCase):
    def setUp(self):
        self.previous_results = dict(preflight_checks.preflight_results)
        preflight_checks.preflight_results.clear()

    def tearDown(self):
        preflight_checks.preflight_results.clear()
        preflight_checks.preflight_results.update(self.previous_results)

    def test_skip_mode_does_not_run_checker(self):
        with (
            patch.object(preflight_checks, 'RdmaMtuCheck') as checker,
            patch.object(preflight_checks, 'preflight_update_test_result'),
        ):
            preflight_checks.test_rdma_mtu(MagicMock(), {'connectivity_check': {'rdma': {'connectivity_mode': 'skip'}}})
        self.assertEqual(preflight_checks.preflight_results['rdma_mtu']['status'], 'SKIPPED')
        checker.assert_not_called()

    def test_disabled_flag_does_not_run_checker(self):
        for flag in (False, 'false'):
            with (
                self.subTest(flag=flag),
                patch.object(preflight_checks, 'RdmaMtuCheck') as checker,
                patch.object(preflight_checks, 'preflight_update_test_result'),
            ):
                config = {'connectivity_check': {'rdma': {'connectivity_mode': 'basic', 'mtu_check': flag}}}
                preflight_checks.test_rdma_mtu(MagicMock(), config)
                self.assertEqual(preflight_checks.preflight_results['rdma_mtu']['status'], 'SKIPPED')
                checker.assert_not_called()

    def test_failure_is_stored_without_pruning(self):
        orch = MagicMock()
        config = {
            'connectivity_check': {
                'rdma': {
                    'connectivity_mode': 'basic',
                    'gid_index': '7',
                    'interfaces': ['enp4s0np0'],
                    'min_netdev_mtu': 9000,
                }
            }
        }
        failing = {'n1': {'status': 'FAIL', 'errors': ['low MTU'], 'interfaces': {'enp4s0np0': {'status': 'FAIL'}}}}
        with (
            patch.object(preflight_checks, 'RdmaMtuCheck') as checker,
            patch.object(preflight_checks, 'preflight_update_test_result') as update,
            patch.object(preflight_checks, '_prune_nodes_from_phdl') as prune,
        ):
            checker.return_value.run.return_value = failing
            preflight_checks.test_rdma_mtu(orch, config)
        checker.assert_called_once_with(orch, ['enp4s0np0'], 9000, 4096, '7', config)
        self.assertIs(preflight_checks.preflight_results['rdma_mtu'], failing)
        update.assert_called_once_with(failing)
        prune.assert_not_called()


if __name__ == '__main__':
    unittest.main()
