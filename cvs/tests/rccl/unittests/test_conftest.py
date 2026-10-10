'''Unit tests for RCCL report hooks.'''

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

from cvs.lib.report import benchmark_metric_registry as registry
from cvs.lib.report.render.perf_metric_table import is_benchmark_metrics_extra
from cvs.tests.rccl import conftest
from cvs.tests.rccl._case_report import RCCL_CASE_TESTS


class _Outcome:
    def __init__(self, report):
        self.report = report

    def get_result(self):
        return self.report


class TestRcclConftest(unittest.TestCase):
    def setUp(self):
        registry._ROWS_BY_NODEID.clear()
        registry._COLUMNS_BY_NODEID.clear()
        registry._SUBTEST_SUMMARY_COUNTED.clear()
        registry._SUBTEST_SUMMARY.update({'failed': 0, 'passed': 0, 'skipped': 0, 'recorded': 0})
        self.nodeid = 'cvs/tests/rccl/rccl_perf.py::test_rccl_perf[all_reduce_perf]'
        self.item = SimpleNamespace(nodeid=self.nodeid, stash={})
        self.rows = [{'node': '', 'metric': '0:bus_bw', 'label': 'all_reduce_perf bus_bw', 'status': 'pass'}]
        registry.record_benchmark_metric_rows(self.item, self.rows)

    def _report(self, nodeid=None, extras=None):
        return SimpleNamespace(
            nodeid=nodeid or self.nodeid,
            when='call',
            extras=[] if extras is None else extras,
            user_properties=[],
        )

    def _run_makereport(self, report):
        hook = conftest.pytest_runtest_makereport(self.item, None)
        next(hook)
        with self.assertRaises(StopIteration):
            hook.send(_Outcome(report))

    def test_makereport_attaches_single_panel_and_full_log(self):
        full_log = {'format_type': 'url', 'name': 'Full Log', 'content': 'test.html'}
        report = self._report(extras=[full_log, dict(full_log)])
        self._run_makereport(report)
        self.assertEqual(sum(is_benchmark_metrics_extra(extra) for extra in report.extras), 1)
        self.assertEqual(sum(extra.get('name') == 'Full Log' for extra in report.extras), 1)
        self.assertEqual(registry.benchmark_metric_rows_from_report(report), self.rows)

    def test_makereport_skips_subtest_context_and_other_tests(self):
        report = self._report(extras=[{'name': 'marker'}])
        with patch.object(conftest, 'called_from_subtest_context', return_value=True):
            self._run_makereport(report)
        self.assertEqual(report.extras, [{'name': 'marker'}])
        other = self._report(nodeid='cvs/tests/rccl/rccl_perf.py::test_collect_hostinfo')
        self._run_makereport(other)
        self.assertEqual(other.extras, [])

    def test_row_and_html_hooks_only_touch_case_rows(self):
        pairwise = self._report(nodeid='cvs/tests/rccl/rccl_pairwise.py::test_rccl_pairwise')
        registry.stamp_benchmark_metric_rows_on_report(pairwise, self.rows)
        cells = ['<td class="col-result">Passed</td>']
        data = ['captured log']
        conftest.pytest_html_results_table_row(pairwise, cells)
        conftest.pytest_html_results_table_html(pairwise, data)
        self.assertIn('cvs-benchmark-collapsible', cells[0])
        self.assertEqual(data, [])
        other = self._report(nodeid='cvs/tests/rccl/rccl_pairwise.py::test_gen_graph')
        registry.stamp_benchmark_metric_rows_on_report(other, self.rows)
        cells = ['<td class="col-result">Passed</td>']
        data = ['captured log']
        conftest.pytest_html_results_table_row(other, cells)
        conftest.pytest_html_results_table_html(other, data)
        self.assertEqual(cells, ['<td class="col-result">Passed</td>'])
        self.assertEqual(data, ['captured log'])

    def test_sessionfinish_patches_each_case_test(self):
        session = SimpleNamespace(config=SimpleNamespace(option=SimpleNamespace(htmlpath='/tmp/rccl.html')))
        with patch.object(conftest, 'patch_benchmark_metrics_into_html') as patch_html:
            hook = conftest.pytest_sessionfinish(session, 0)
            next(hook)
            with self.assertRaises(StopIteration):
                next(hook)
        self.assertEqual(
            patch_html.call_args_list,
            [call(Path('/tmp/rccl.html'), benchmark_test_name=name) for name in RCCL_CASE_TESTS],
        )
        session.config.option.htmlpath = None
        with patch.object(conftest, 'patch_benchmark_metrics_into_html') as patch_html:
            hook = conftest.pytest_sessionfinish(session, 0)
            next(hook)
            with self.assertRaises(StopIteration):
                next(hook)
            patch_html.assert_not_called()
