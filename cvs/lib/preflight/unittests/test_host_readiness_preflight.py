"""Unit tests for the host readiness preflight steps, report summary and Run Deck records."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from cvs.lib.preflight.report import HOST_READINESS_CHECKS, PreflightReportGenerator
from cvs.lib.preflight.rundeck import HOST_KERNEL_PARAMS, build_preflight_deck
from cvs.parsers.schemas import PreflightConfigFile
from cvs.tests.preflight import preflight_checks

PASS_NODE = {'status': 'PASS', 'errors': [], 'warnings': [], 'details': {}}
WARN_NODE = {'status': 'PASS', 'errors': [], 'warnings': ['dynamic value'], 'details': {}}
FAIL_NODE = {'status': 'FAIL', 'errors': ['Kernel parameter iommu=pt is not set <x>'], 'warnings': [], 'details': {}}


class _PreflightResultsTest(unittest.TestCase):
    def setUp(self):
        self.previous_results = dict(preflight_checks.preflight_results)
        preflight_checks.preflight_results.clear()

    def tearDown(self):
        preflight_checks.preflight_results.clear()
        preflight_checks.preflight_results.update(self.previous_results)


class TestHostReadinessSteps(_PreflightResultsTest):
    def test_absent_section_is_skipped_without_running(self):
        with (
            patch.object(preflight_checks, 'KernelParamsCheck') as checker,
            patch.object(preflight_checks, 'preflight_update_test_result') as update,
        ):
            preflight_checks.test_host_kernel_params(MagicMock(), {})
        checker.assert_not_called()
        stored = preflight_checks.preflight_results['host_kernel_params']
        self.assertEqual(stored['status'], 'SKIPPED')
        self.assertIn('host_readiness.kernel_params is not configured', stored['message'])
        update.assert_called_once_with(stored)

    def test_disabled_section_is_skipped(self):
        config = {'host_readiness': {'os_tuning': {'enabled': False}}}
        with (
            patch.object(preflight_checks, 'OsTuningCheck') as checker,
            patch.object(preflight_checks, 'preflight_update_test_result'),
        ):
            preflight_checks.test_host_os_tuning(MagicMock(), config)
        checker.assert_not_called()
        self.assertIn('is disabled', preflight_checks.preflight_results['host_os_tuning']['message'])

    def test_invalid_section_fails_with_message(self):
        config = {'host_readiness': {'kernel_params': {'on_mismatch': 'ignore'}}}
        with (
            patch.object(preflight_checks, 'KernelParamsCheck') as checker,
            patch.object(preflight_checks, 'preflight_update_test_result') as update,
        ):
            preflight_checks.test_host_kernel_params(MagicMock(), config)
        checker.assert_not_called()
        stored = preflight_checks.preflight_results['host_kernel_params']
        self.assertEqual(stored['status'], 'FAIL')
        self.assertIn('Invalid host_readiness.kernel_params', stored['message'])
        update.assert_called_once_with(stored)

    def test_kernel_params_runs_with_schema_defaults_and_metadata_stripped(self):
        orch = MagicMock()
        config = {'host_readiness': {'kernel_params': {'_comment': 'doc', 'on_mismatch': 'warn'}}}
        results = {'n1': FAIL_NODE}
        with (
            patch.object(preflight_checks, 'KernelParamsCheck') as checker,
            patch.object(preflight_checks, 'preflight_update_test_result') as update,
        ):
            checker.return_value.run.return_value = results
            preflight_checks.test_host_kernel_params(orch, config)
        args, kwargs = checker.call_args
        self.assertIs(args[0], orch)
        self.assertIn('iommu=pt', args[1])
        self.assertEqual(kwargs['required_intel'], ['intel_iommu=on'])
        self.assertEqual(kwargs['on_mismatch'], 'warn')
        self.assertIs(preflight_checks.preflight_results['host_kernel_params'], results)
        update.assert_called_once_with(results)

    def test_env_check_follows_container_orchestrator(self):
        orch = MagicMock()
        orch.orchestrator_type = 'container'
        with (
            patch.object(preflight_checks, 'PersistentEnvCheck') as checker,
            patch.object(preflight_checks, 'preflight_update_test_result'),
        ):
            checker.return_value.run.return_value = {'n1': PASS_NODE}
            preflight_checks.test_host_persistent_env(orch, {'host_readiness': {'env': {}}})
        self.assertTrue(checker.call_args.kwargs['check_container'])
        self.assertEqual(checker.call_args.args[1]['HIP_FORCE_DEV_KERNARG'], '1')

    def test_idle_state_passes_thresholds(self):
        config = {'host_readiness': {'idle_state': {'samples': 5, 'temp_max_c': {'hotspot': 90}}}}
        with (
            patch.object(preflight_checks, 'GpuIdleStateCheck') as checker,
            patch.object(preflight_checks, 'preflight_update_test_result'),
        ):
            checker.return_value.run.return_value = {'n1': PASS_NODE}
            preflight_checks.test_gpu_idle_state(MagicMock(), config)
        self.assertEqual(checker.call_args.kwargs['samples'], 5)
        self.assertEqual(checker.call_args.kwargs['temp_max_c'], {'hotspot': 90})

    def test_platform_extras_passes_ldd_settings_only_when_enabled(self):
        for enabled in (False, True):
            config = {'host_readiness': {'platform_extras': {'rocm_lib_deps': {'enabled': enabled}}}}
            with (
                self.subTest(enabled=enabled),
                patch.object(preflight_checks, 'PlatformExtrasCheck') as checker,
                patch.object(preflight_checks, 'preflight_update_test_result'),
            ):
                checker.return_value.run.return_value = {'n1': PASS_NODE}
                preflight_checks.test_host_platform_extras(MagicMock(), config)
                deps = checker.call_args.kwargs['rocm_lib_deps']
                self.assertEqual(deps is not None, enabled)

    def test_inventory_exports_json_and_uses_backend_interfaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                'connectivity_check': {'rdma': {'interfaces': ['rocep28s0']}},
                'reporting': {'artifacts_root_dir': tmp},
                'host_readiness': {'inventory': {'sku_profiles': [{'name': 'gpu', 'cpu_sockets': 2}]}},
            }
            with (
                patch.object(preflight_checks, 'HostInventoryCheck') as checker,
                patch.object(preflight_checks, 'preflight_update_test_result'),
            ):
                checker.return_value.run.return_value = {'n1': PASS_NODE}
                preflight_checks.test_host_inventory(MagicMock(), config)
            kwargs = checker.call_args.kwargs
            self.assertEqual(kwargs['backend_rdma_devices'], ['rocep28s0'])
            self.assertEqual(kwargs['sku_profiles'][0]['cpu_sockets'], 2)
            self.assertIsNone(kwargs['sku_profiles'][0]['dimm_count'])
            path = preflight_checks.preflight_results['host_inventory_export']
            self.assertTrue(path.startswith(tmp))
            self.assertTrue(os.path.isfile(path))

    def test_inventory_export_can_be_disabled(self):
        config = {'host_readiness': {'inventory': {'export_json': False}}}
        with (
            patch.object(preflight_checks, 'HostInventoryCheck') as checker,
            patch.object(preflight_checks, 'export_inventory') as export,
            patch.object(preflight_checks, 'preflight_update_test_result'),
        ):
            checker.return_value.run.return_value = {'n1': PASS_NODE}
            preflight_checks.test_host_inventory(MagicMock(), config)
        export.assert_not_called()
        self.assertNotIn('host_inventory_export', preflight_checks.preflight_results)


class TestHostReadinessSchema(unittest.TestCase):
    def test_sections_default_to_absent(self):
        self.assertIsNone(PreflightConfigFile.model_validate({}).host_readiness)

    def test_defaults_and_validation(self):
        config = PreflightConfigFile.model_validate({'host_readiness': {'kernel_params': {}, 'env': {}}})
        self.assertIn('pci=bfsort', config.host_readiness.kernel_params.required)
        self.assertEqual(config.host_readiness.env.sources, ['/etc/environment'])
        self.assertIsNone(config.host_readiness.os_tuning)
        for bad in (
            {'kernel_params': {'on_mismatch': 'ignore'}},
            {'kernel_params': {'requird': []}},
            {'env': {'required': {'1BAD': 'x'}}},
            {'inventory': {'sku_profiles': [{'name': 'a', 'cpu_sockets': 0}]}},
            {'idle_state': {'samples': 0}},
            {'unknown_section': {}},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                PreflightConfigFile.model_validate({'host_readiness': bad})

    def test_metadata_keys_are_ignored(self):
        config = PreflightConfigFile.model_validate(
            {'host_readiness': {'_comment': 'x', 'inventory': {'_example_sku_profiles': [{'bogus': 1}]}}}
        )
        self.assertEqual(config.host_readiness.inventory.sku_profiles, [])


class TestHostReadinessReport(unittest.TestCase):
    def setUp(self):
        self.report = PreflightReportGenerator(MagicMock(), {}, {})

    def test_summary_statuses(self):
        summarize = self.report._summarize_host_readiness_results
        self.assertEqual(summarize('host_kernel_params', None)['status'], 'SKIPPED')
        self.assertEqual(summarize('host_kernel_params', {'status': 'SKIPPED', 'skipped': True})['status'], 'SKIPPED')
        passing = summarize('host_kernel_params', {'n1': PASS_NODE, 'n2': WARN_NODE})
        self.assertEqual(passing['status'], 'PASS')
        self.assertEqual(passing['summary'], '2/2 nodes pass (1 with warnings)')
        failing = summarize('host_kernel_params', {'n1': PASS_NODE, 'n2': FAIL_NODE})
        self.assertEqual((failing['status'], failing['failed_nodes']), ('FAIL', ['n2']))
        invalid = summarize('host_kernel_params', {'status': 'FAIL', 'message': 'Invalid host_readiness.kernel_params'})
        self.assertEqual((invalid['status'], invalid['summary']), ('ERROR', 'Invalid host_readiness.kernel_params'))

    def test_overall_status_and_recommendation(self):
        self.report.results = {'host_kernel_params': {'n1': FAIL_NODE}}
        summary = self.report._generate_preflight_summary()
        for check in HOST_READINESS_CHECKS:
            self.assertIn(check, summary['checks'])
        self.assertEqual(summary['overall_status'], 'FAIL')
        self.assertTrue(any('GRUB' in r for r in summary['recommendations']))

    def test_html_lists_failed_and_warned_nodes_escaped(self):
        self.assertEqual(self.report._generate_host_readiness_html('host_kernel_params', {'n1': PASS_NODE}), '')
        self.assertEqual(self.report._generate_host_readiness_html('host_kernel_params', {'status': 'SKIPPED'}), '')
        section = self.report._generate_host_readiness_html(
            'host_kernel_params', {'n<1>': FAIL_NODE, 'n2': WARN_NODE, 'n3': PASS_NODE}
        )
        self.assertIn('Host Kernel Parameters', section)
        self.assertIn('n&lt;1&gt;', section)
        self.assertIn('is not set &lt;x&gt;', section)
        self.assertIn('warning: dynamic value', section)
        self.assertNotIn('<td>n3</td>', section)
        self.assertIn('/tmp/a&amp;b', self.report._generate_host_inventory_export_html('/tmp/a&b'))
        self.assertEqual(self.report._generate_host_inventory_export_html(None), '')


class TestHostReadinessRundeck(unittest.TestCase):
    def test_records_per_node_and_skipped(self):
        deck = build_preflight_deck(
            {HOST_KERNEL_PARAMS: {'n1': FAIL_NODE, 'n2': WARN_NODE}, 'host_os_tuning': {'status': 'SKIPPED'}},
            {'node_dict': {'n1': {}, 'n2': {}}},
        )
        nodes = deck['groups'][HOST_KERNEL_PARAMS]['nodes']
        self.assertEqual(nodes['n1']['status'], 'fail')
        self.assertEqual(nodes['n1']['items'][0]['name'], HOST_KERNEL_PARAMS)
        self.assertEqual(nodes['n2']['status'], 'pass')
        self.assertIn('pass with warning', nodes['n2']['items_summary'])
        self.assertEqual(deck['groups']['host_os_tuning']['nodes']['n1']['status'], 'na')


if __name__ == '__main__':
    unittest.main()
