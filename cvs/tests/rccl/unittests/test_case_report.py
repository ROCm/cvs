'''Unit tests for RCCL case reporting.'''

import unittest
from contextlib import contextmanager
from types import SimpleNamespace

from cvs.lib.report import benchmark_metric_registry as registry
from cvs.tests.rccl._case_report import RcclCaseReporter


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


class TestRcclCaseReporter(unittest.TestCase):
    def setUp(self):
        registry._ROWS_BY_NODEID.clear()
        registry._COLUMNS_BY_NODEID.clear()
        registry._SUBTEST_SUMMARY_COUNTED.clear()
        registry._SUBTEST_SUMMARY.update({'failed': 0, 'passed': 0, 'skipped': 0, 'recorded': 0})
        self.nodeid = 'cvs/tests/rccl/rccl_perf.py::test_rccl_perf[all_reduce_perf]'
        self.fake = _CapturingSubtests()
        request = SimpleNamespace(node=SimpleNamespace(nodeid=self.nodeid, stash={}))
        self.reporter = RcclCaseReporter(request, self.fake)

    def rows(self):
        return registry.benchmark_metric_rows_for_nodeid(self.nodeid)

    def test_report_records_rows_and_subtests(self):
        self.reporter.report('a', 'first', True, 'unused', actual=10, threshold=5, unit='GB/s', node='n0')
        self.reporter.report('b', 'second', False, 'low bandwidth', node='n1')
        self.assertEqual(self.fake.calls, [{'node': 'n0'}, {'node': 'n1'}])
        self.assertEqual(self.fake.failures, ['low bandwidth'])
        self.assertEqual([row['status'] for row in self.rows()], ['pass', 'fail'])
        self.assertEqual([row['label'] for row in self.rows()], ['first', 'second'])
        self.assertEqual(self.rows()[0]['spec'], {'kind': '>=', 'value': 5})
        self.assertEqual(self.rows()[0]['reason'], '')

    def test_report_verdicts_labels(self):
        verdicts = [
            {
                'check': 'bus_bw',
                'dtype': 'float',
                'size': 2048,
                'actual': 10.0,
                'threshold': 76.0,
                'unit': 'GB/s',
                'status': 'fail',
                'message': 'low',
            },
            {
                'check': 'bus_bw',
                'dtype': None,
                'size': None,
                'actual': None,
                'threshold': None,
                'unit': None,
                'status': 'fail',
                'message': 'missing',
            },
        ]
        self.reporter.report_verdicts(verdicts, 'all_reduce_perf')
        self.assertEqual(
            self.rows()[0]['label'], 'all_reduce_perf bus_bw float size=2048: 10.00 GB/s (threshold >= 76.00 GB/s)'
        )
        self.assertEqual(self.rows()[1]['label'], 'all_reduce_perf bus_bw')
        self.assertEqual(
            self.fake.calls[0], {'collective': 'all_reduce_perf', 'check': 'bus_bw', 'dtype': 'float', 'size': 2048}
        )
        self.assertEqual(self.fake.calls[1], {'collective': 'all_reduce_perf', 'check': 'bus_bw'})

    def test_duplicate_verdicts_keep_distinct_rows(self):
        verdict = {
            'check': 'bus_bw',
            'dtype': 'float',
            'size': 2048,
            'actual': 10.0,
            'threshold': 5.0,
            'unit': 'GB/s',
            'status': 'pass',
            'message': '',
        }
        self.reporter.report_verdicts([verdict, verdict], 'all_reduce_perf')
        self.assertEqual(len(self.rows()), 2)

    def test_report_phase_labels(self):
        self.reporter.report_phase(
            '2', 'n1', 'Phase2 2-node cluster (adding n1)', True, '', best_bw=350.0, min_bw=300.0
        )
        self.reporter.report_phase('0', 'n0', 'Phase0 sanity n0', False, 'failed')
        self.assertEqual(
            self.rows()[0]['label'],
            'phase 2 n1 (Phase2 2-node cluster (adding n1)): 350.00 GB/s (threshold >= 300.00 GB/s)',
        )
        self.assertEqual(self.fake.calls[0], {'phase': '2', 'node': 'n1'})
        self.assertEqual(self.rows()[1]['label'], 'phase 0 n0 (Phase0 sanity n0)')
        self.assertIsNone(self.rows()[1]['unit'])
