'''Unit tests for vLLM pytest-html metric hooks.'''

import unittest
from types import SimpleNamespace

from cvs.lib.report import benchmark_metric_registry as registry
from cvs.lib.report.render.perf_metric_table import is_benchmark_metrics_extra
from cvs.tests.inference.vllm import conftest as vllm_conftest


class _FakeStash(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _Outcome:
    def __init__(self, report):
        self.report = report

    def get_result(self):
        return self.report


class TestVllmMetricReportHook(unittest.TestCase):
    def setUp(self):
        registry._ROWS_BY_NODEID.clear()
        registry._COLUMNS_BY_NODEID.clear()

    def test_makereport_attaches_and_stamps_metric_panel(self):
        nodeid = 'cvs/tests/inference/vllm/vllm_single.py::test_verify_cell_metrics[cell]'
        item = SimpleNamespace(nodeid=nodeid, stash=_FakeStash())
        rows = [
            {
                'node': 'head',
                'metric': 'output_throughput',
                'status': 'record',
                'actual': 99,
                'spec': {'kind': 'min', 'value': 100},
                'enforced': False,
            }
        ]
        registry.record_benchmark_metric_rows(item, rows)
        full_log = {'format_type': 'url', 'name': 'Full Log', 'content': 'test_html/full.html'}
        report = SimpleNamespace(
            nodeid=nodeid,
            when='call',
            extras=[full_log, dict(full_log)],
            user_properties=[],
        )

        hook = vllm_conftest.pytest_runtest_makereport(item, None)
        next(hook)
        with self.assertRaises(StopIteration):
            hook.send(_Outcome(report))

        self.assertEqual(sum(is_benchmark_metrics_extra(extra) for extra in report.extras), 1)
        self.assertEqual(sum(extra.get('name') == 'Full Log' for extra in report.extras), 1)
        self.assertEqual(
            dict(report.user_properties)[registry.BENCHMARK_METRIC_ROWS_USER_PROPERTY],
            rows,
        )


if __name__ == '__main__':
    unittest.main()
