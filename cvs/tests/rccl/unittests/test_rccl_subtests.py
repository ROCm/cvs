'''Unit tests for RCCL suite case-report wiring.'''

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cvs.lib import globals
from cvs.lib.report import benchmark_metric_registry as registry
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

    def _run_perf(self, module, regression=False, error=False):
        request = self._request('rccl_regression' if regression else 'rccl_perf')
        fake = _CapturingSubtests()
        orch = MagicMock()
        with (
            patch.object(module, 'get_passwordless_sudo_status', return_value={'n0': False}),
            patch.object(module, 'rccl_res_dict', {}),
            patch.object(module.rccl_lib.RcclJob, 'from_config') as from_config,
        ):
            from_config.return_value.verdicts = self.verdicts
            from_config.return_value.run_perf.return_value = self.raw
            from_config.return_value.run_regression.return_value = self.raw
            if error:
                from_config.return_value.run_perf.side_effect = (
                    lambda: globals.error_list.append('low bandwidth') or self.raw
                )
            args = (orch, ['n0'], ['v0'], {'cvs_params': {}}, 'all_reduce_perf')
            if regression:
                args += ({'NCCL_ALGO': 'Ring'},)
            if error:
                with self.assertRaises(pytest.fail.Exception):
                    module.test_rccl_perf(*args, request, fake)
            else:
                module.test_rccl_perf(*args, request, fake)
        return request, fake, from_config

    def test_perf_reports_job_verdicts(self):
        request, fake, _ = self._run_perf(rccl_perf)
        self.assertEqual(len(fake.calls), 2)
        self.assertTrue(all(call['collective'] == 'all_reduce_perf' for call in fake.calls))
        self.assertEqual(fake.failures, ['low bandwidth'])
        self.assertEqual(len(registry.benchmark_metric_rows_for_nodeid(request.node.nodeid)), 2)

    def test_perf_parent_still_fails_on_recorded_errors(self):
        _, fake, _ = self._run_perf(rccl_perf, error=True)
        self.assertEqual(fake.failures, ['low bandwidth'])

    def test_regression_reports_job_verdicts(self):
        request, fake, from_config = self._run_perf(rccl_regression, regression=True)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.failures, ['low bandwidth'])
        self.assertEqual(len(registry.benchmark_metric_rows_for_nodeid(request.node.nodeid)), 2)
        self.assertEqual(from_config.call_args.kwargs['env_overrides'], {'NCCL_ALGO': 'Ring'})

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
