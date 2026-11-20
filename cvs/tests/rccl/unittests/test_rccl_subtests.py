'''Unit tests for RCCL suite case-report wiring.'''

import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cvs.lib import globals
from cvs.lib import utils_lib
from cvs.lib.report import benchmark_metric_registry as registry
from cvs.lib.report.rundeck.dataset_builders.series import overall_status_from_run
from cvs.tests.rccl import rccl_pairwise, rccl_perf, rccl_regression


class _CapturingSubtests:
    def __init__(self):
        self.calls = []
        self.failures = []

    @contextmanager
    def test(self, **kwargs):
        self.calls.append(kwargs)
        try:
            yield
        except AssertionError as exc:
            self.failures.append(str(exc))


class TestRcclSubtests(unittest.TestCase):
    def setUp(self):
        errors = patch.object(globals, 'error_list', [])
        errors.start()
        self.addCleanup(errors.stop)
        registry._ROWS_BY_NODEID.clear()
        registry._COLUMNS_BY_NODEID.clear()
        registry._SUBTEST_SUMMARY_COUNTED.clear()
        registry._SUBTEST_SUMMARY.update({'failed': 0, 'passed': 0, 'skipped': 0, 'recorded': 0})
        self.raw = [{'size': 2048, 'type': 'float', 'inPlace': 1, 'busBw': 100.0}]
        self.verdicts = [
            {
                'check': 'bus_bw',
                'dtype': 'float',
                'size': 1024,
                'actual': 100.0,
                'threshold': 80.0,
                'unit': 'GB/s',
                'status': 'pass',
                'message': '',
            },
            {
                'check': 'bus_bw',
                'dtype': 'float',
                'size': 2048,
                'actual': 10.0,
                'threshold': 80.0,
                'unit': 'GB/s',
                'status': 'fail',
                'message': 'low bandwidth',
            },
        ]

    def _request(self, name):
        return SimpleNamespace(
            node=SimpleNamespace(nodeid=f'cvs/tests/rccl/{name}.py::test_rccl_perf[all_reduce_perf]', stash={})
        )

    def _run_perf(
        self, module, regression=False, error=False, sudo=False, cvs_params=None, verdict_store=None, expect_fail=False
    ):
        request = self._request('rccl_regression' if regression else 'rccl_perf')
        fake = _CapturingSubtests()
        orch = MagicMock()
        if verdict_store is None:
            verdict_store = []
        with (
            patch.object(module, 'get_passwordless_sudo_status', return_value={'n0': sudo}),
            patch.object(module, 'rccl_res_dict', {}),
            patch.object(module, 'rccl_verdicts', verdict_store),
            patch.object(module.rccl_lib.RcclJob, 'from_config') as from_config,
        ):
            from_config.return_value.verdicts = self.verdicts
            from_config.return_value.run_perf.return_value = self.raw
            from_config.return_value.run_regression.return_value = self.raw
            if error:
                from_config.return_value.run_perf.side_effect = lambda: (
                    globals.error_list.append('low bandwidth') or self.raw
                )
            args = (orch, ['n0'], ['v0'], {'cvs_params': cvs_params or {}}, 'all_reduce_perf')
            if regression:
                args += ({'NCCL_ALGO': 'Ring'},)
            if error or expect_fail:
                with self.assertRaises(pytest.fail.Exception):
                    module.test_rccl_perf(*args, request, fake)
            else:
                module.test_rccl_perf(*args, request, fake)
        return request, fake, from_config, verdict_store

    def test_perf_reports_job_verdicts(self):
        request, fake, _, verdict_store = self._run_perf(rccl_perf)
        self.assertEqual(len(fake.calls), 2)
        self.assertTrue(all(call['collective'] == 'all_reduce_perf' for call in fake.calls))
        self.assertEqual(fake.failures, ['low bandwidth'])
        self.assertEqual(len(registry.benchmark_metric_rows_for_nodeid(request.node.nodeid)), 2)
        self.assertEqual(verdict_store, self.verdicts)

    def test_perf_parent_still_fails_on_recorded_errors(self):
        _, fake, _, _ = self._run_perf(rccl_perf, error=True)
        self.assertEqual(fake.failures, ['low bandwidth'])

    def test_regression_reports_job_verdicts(self):
        request, fake, from_config, verdict_store = self._run_perf(rccl_regression, regression=True)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.failures, ['low bandwidth'])
        self.assertEqual(len(registry.benchmark_metric_rows_for_nodeid(request.node.nodeid)), 2)
        self.assertEqual(from_config.call_args.kwargs['env_overrides'], {'NCCL_ALGO': 'Ring'})
        self.assertEqual(verdict_store, self.verdicts)

    def test_perf_verdicts_accumulate_across_jobs(self):
        verdict_store = []
        self._run_perf(rccl_perf, verdict_store=verdict_store)
        self._run_perf(rccl_perf, verdict_store=verdict_store)
        self.assertEqual(verdict_store, self.verdicts * 2)

    def test_sudo_runs_bracket_the_job_with_a_dmesg_window(self):
        for module, regression, label in (
            (rccl_perf, False, 'all_reduce_perf'),
            (rccl_regression, True, 'all_reduce_perf NCCL_ALGO=Ring'),
        ):
            for flag, use_markers in (('True', True), ('False', False)):
                with self.subTest(module=module.__name__, flag=flag):
                    order = []

                    def record(step, module=module):
                        job = module.rccl_lib.RcclJob.from_config.return_value
                        ran = job.run_perf.called or job.run_regression.called
                        order.append(f'{step} {"after" if ran else "before"} job')

                    with patch.object(module, 'DmesgWindow') as window_class:
                        window_class.return_value.open.side_effect = lambda: record('open')
                        window_class.return_value.close_and_scan.side_effect = lambda: record('scan')
                        self._run_perf(module, regression=regression, sudo=True, cvs_params={'dmesg_during_test': flag})
                    self.assertEqual(window_class.call_args.args[1:], (label, use_markers))
                    self.assertEqual(order, ['open before job', 'scan after job'])

    def test_runs_without_sudo_skip_the_dmesg_window(self):
        for module, regression in ((rccl_perf, False), (rccl_regression, True)):
            with self.subTest(module=module.__name__):
                with patch.object(module, 'DmesgWindow') as window_class:
                    self._run_perf(module, regression=regression, cvs_params={'dmesg_during_test': 'True'})
                window_class.assert_not_called()

    def test_failures_after_passing_checks_fail_pytest_and_deck(self):
        for module, regression in ((rccl_perf, False), (rccl_regression, True)):
            for source in ('dmesg', 'ecc_blocks', 'snapshot'):
                with self.subTest(module=module.__name__, source=source), ExitStack() as stack:
                    self.verdicts = [
                        dict(self.verdicts[0]),
                        {**self.verdicts[1], 'check': 'bw_dip', 'status': 'pass', 'message': ''},
                    ]
                    sudo = source != 'ecc_blocks'
                    cvs_params = {}
                    if source == 'dmesg':
                        window_class = stack.enter_context(patch.object(module, 'DmesgWindow'))
                        window_class.return_value.close_and_scan.side_effect = lambda: utils_lib.fail_test(
                            'dmesg: GPU reset begin on n0'
                        )
                    elif source == 'ecc_blocks':
                        cvs_params = {'report_ecc_delta': 'True', 'report_ecc_blocks': ['BOGUS']}
                        if regression:
                            stack.enter_context(
                                patch.dict(
                                    rccl_regression._ecc_after_carry, {'blocks': None, 'snapshot': None}, clear=True
                                )
                            )
                    else:
                        cvs_params = {'cluster_snapshot_debug': 'True'}
                        stack.enter_context(patch.object(module, 'DmesgWindow'))
                        snapshots = stack.enter_context(patch.object(module, 'create_cluster_metrics_snapshot'))
                        stack.enter_context(patch.object(module, 'compare_cluster_metrics_snapshots'))

                        def capture_snapshot(_handle):
                            if snapshots.call_count == 2:
                                utils_lib.fail_test('amd-smi RAS JSON parse failed')
                            return {}

                        snapshots.side_effect = capture_snapshot
                    verdict_store = []
                    self._run_perf(
                        module,
                        regression=regression,
                        sudo=sudo,
                        cvs_params=cvs_params,
                        verdict_store=verdict_store,
                        expect_fail=True,
                    )
                    self.assertTrue(all(verdict['status'] == 'pass' for verdict in verdict_store))
                    self.assertEqual(len(verdict_store), 2)
                    self.assertEqual(overall_status_from_run(verdict_store, False), 'pass')
                    self.assertEqual(overall_status_from_run(verdict_store, True), 'fail')

    def test_failure_without_thresholds_fails_pytest_and_deck(self):
        self.verdicts = []
        verdict_store = []
        with patch.object(rccl_perf, 'DmesgWindow') as window_class:
            window_class.return_value.close_and_scan.side_effect = lambda: utils_lib.fail_test(
                'dmesg: GPU reset begin on n0'
            )
            self._run_perf(rccl_perf, sudo=True, verdict_store=verdict_store, expect_fail=True)
        self.assertEqual(verdict_store, [])
        self.assertEqual(overall_status_from_run(verdict_store, True), 'fail')

    def test_pairwise_reports_each_phase_and_node(self):
        request = SimpleNamespace(
            node=SimpleNamespace(nodeid='cvs/tests/rccl/rccl_pairwise.py::test_rccl_pairwise', stash={})
        )
        fake = _CapturingSubtests()

        def run_pairwise(_orch, node_pair_mgmt, **_kwargs):
            return (None, False) if node_pair_mgmt[-1] == 'n2' else (self.raw, True)

        with (
            patch.object(rccl_pairwise, 'is_managed_compute', return_value=False),
            patch.object(rccl_pairwise, '_persist_pairwise_artifact'),
            patch.object(rccl_pairwise, '_phase1_survivors', None),
            patch.object(rccl_pairwise, 'run_pairwise_rccl', side_effect=run_pairwise),
        ):
            with self.assertRaises(pytest.fail.Exception):
                rccl_pairwise.test_rccl_pairwise(None, ['n0', 'n1', 'n2'], {}, ['v0', 'v1', 'v2'], request, fake)
            self.assertEqual(
                fake.calls, [{'phase': '0', 'node': 'n0'}, {'phase': '1', 'node': 'n1'}, {'phase': '1', 'node': 'n2'}]
            )
            self.assertEqual(len(fake.failures), 1)
            globals.error_list = []
            second = _CapturingSubtests()
            rccl_pairwise.test_rccl_incremental(None, ['n0', 'n1', 'n2'], {}, ['v0', 'v1', 'v2'], request, second)
            self.assertEqual(second.calls, [{'phase': '2', 'node': 'n1'}])

    def test_pairwise_phase0_failure_reports_one_case_and_aborts(self):
        request = SimpleNamespace(
            node=SimpleNamespace(nodeid='cvs/tests/rccl/rccl_pairwise.py::test_rccl_pairwise', stash={})
        )
        fake = _CapturingSubtests()
        with (
            patch.object(rccl_pairwise, 'is_managed_compute', return_value=False),
            patch.object(rccl_pairwise, 'run_pairwise_rccl', return_value=(None, False)),
        ):
            with self.assertRaises(pytest.fail.Exception):
                rccl_pairwise.test_rccl_pairwise(None, ['n0'], {}, ['v0'], request, fake)
        self.assertEqual(fake.calls, [{'phase': '0', 'node': 'n0'}])
        self.assertEqual(len(fake.failures), 1)
        self.assertEqual(registry.benchmark_metric_rows_for_nodeid(request.node.nodeid)[0]['status'], 'fail')
