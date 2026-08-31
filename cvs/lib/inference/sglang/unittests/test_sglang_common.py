'''Unit tests for cvs/lib/inference/sglang/sglang_common.py.'''

import unittest
from unittest import mock

from cvs.lib.inference.sglang import sglang_common


_SAMPLE_BENCH_LOG = """
Successful requests: 100
Benchmark duration (s): 120.5
Total input tokens: 819200
Total generated tokens: 51200
Request throughput (req/s): 0.833333
Output token throughput (tok/s): 426.666667
Mean TTFT (ms): 45.5
Median TTFT (ms): 40.0
P99 TTFT (ms): 120.0
Mean TPOT (ms): 12.5
Median TPOT (ms): 12.9
P99 TPOT (ms): 18.0
Mean ITL (ms): 11.0
Median ITL (ms): 10.5
P99 ITL (ms): 16.0
Mean E2E Latency (ms): 250.0
Median E2E Latency (ms): 240.0
P90 E2E Latency (ms): 300.0
P95 E2E Latency (ms): 320.0
P99 E2E Latency (ms): 350.0
Serving Benchmark Result
"""


class _FakeSubtests:
    def __init__(self):
        self.failures = []

    class _Ctx:
        def __init__(self, outer):
            self._outer = outer

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is not None:
                self._outer.failures.append(exc)
                return True
            return False

    def test(self, **kwargs):
        return self._Ctx(self)


class TestSglangCommonHelpers(unittest.TestCase):
    def test_first_output(self):
        self.assertEqual(sglang_common.first_output({'a': 'x'}), 'x')
        self.assertEqual(sglang_common.first_output({}), '')

    def test_normalize_hosts(self):
        self.assertEqual(sglang_common.normalize_hosts(None), [])
        self.assertEqual(sglang_common.normalize_hosts('host1'), ['host1'])

    def test_thresholds_from_expected_latency(self):
        specs = sglang_common.thresholds_from_expected({'mean_ttft_ms': 100.0})
        self.assertEqual(specs['mean_ttft_ms']['kind'], 'max_ms')

    def test_perf_enforce_thresholds_defaults_true(self):
        self.assertTrue(sglang_common.perf_enforce_thresholds({}))

    def test_perf_enforce_thresholds_reads_bench_serv_random(self):
        bp = {
            'inference_tests': {
                'bench_serv_random': {'enforce_thresholds': False},
            }
        }
        self.assertFalse(sglang_common.perf_enforce_thresholds(bp))

    def test_node_threshold_actuals_filters_metrics(self):
        inference = {'node1': {'mean_ttft_ms': '50.0', 'other': '1'}}
        thresholds = {'mean_ttft_ms': {'kind': 'max_ms', 'value': 100.0}}
        actuals = sglang_common.node_threshold_actuals(inference, 'node1', thresholds)
        self.assertEqual(actuals, {'mean_ttft_ms': 50.0})

    def test_metric_threshold_violation_missing_metric(self):
        violation = sglang_common.metric_threshold_violation(
            'mean_ttft_ms',
            {},
            {'kind': 'max_ms', 'value': 100.0},
        )
        self.assertIn('missing from actuals', violation)

    def test_metric_threshold_violation_passes_within_threshold(self):
        violation = sglang_common.metric_threshold_violation(
            'mean_ttft_ms',
            {'mean_ttft_ms': 50.0},
            {'kind': 'max_ms', 'value': 100.0},
        )
        self.assertIsNone(violation)

    def test_parse_inference_bench_results_basic(self):
        parsed = sglang_common.parse_inference_bench_results(
            {'node1': _SAMPLE_BENCH_LOG},
            bench_num_prompts=100,
        )
        node = parsed['node1']
        self.assertEqual(node['successful_requests'], '100')
        self.assertEqual(node['benchmark_duration'], '120.5')
        self.assertEqual(node['median_tpot_ms'], '12.9')
        self.assertEqual(node['goodput'], '1.000000')

    def test_parse_inference_bench_results_disagg_extras(self):
        parsed = sglang_common.parse_inference_bench_results(
            {'node1': _SAMPLE_BENCH_LOG},
            num_gpus_for_per_gpu_throughput=16,
            include_itl=True,
            include_extended_e2e_percentiles=True,
        )
        node = parsed['node1']
        self.assertEqual(node['mean_itl_ms'], '11.0')
        self.assertEqual(node['p90_e2e_latency_ms'], '300.0')
        self.assertEqual(node['output_throughput_per_gpu_per_sec'], '26.666667')

    def test_finalize_inference_verification(self):
        host_exec = mock.Mock(return_value={'head': 'Mon Aug 12 18:00'})
        with mock.patch.object(sglang_common.time, 'sleep'):
            end = sglang_common.finalize_inference_verification(host_exec)
        host_exec.assert_called_once()
        self.assertEqual(end, {'head': 'Mon Aug 12 18:00'})

    def test_verify_inference_results_passes(self):
        host_exec = mock.Mock(return_value={'head': 'time'})
        with mock.patch.object(sglang_common.time, 'sleep'):
            end = sglang_common.verify_inference_results(
                {'node1': {'mean_ttft_ms': '50.0'}},
                {'mean_ttft_ms': 100.0},
                host_exec,
            )
        self.assertEqual(end, {'head': 'time'})

    def test_verify_inference_results_subtests(self):
        host_exec = mock.Mock(return_value={'head': 'time'})
        lifecycle = mock.Mock()
        lifecycle.perf_metric_rows = {}
        with mock.patch.object(sglang_common.time, 'sleep'):
            passed, end = sglang_common.verify_inference_results_subtests(
                {'node1': {'mean_ttft_ms': '50.0'}},
                {'mean_ttft_ms': 100.0},
                host_exec,
                _FakeSubtests(),
                'bench_serv',
                lifecycle=lifecycle,
                report_nodeid='test::node',
            )
        self.assertTrue(passed)
        self.assertEqual(end, {'head': 'time'})
        self.assertEqual(lifecycle.perf_metric_rows['test::node'][0]['status'], 'pass')

    def test_verify_inference_results_subtests_gates_violation(self):
        host_exec = mock.Mock(return_value={'head': 'time'})
        lifecycle = mock.Mock()
        lifecycle.perf_metric_rows = {}
        subtests = _FakeSubtests()
        with mock.patch.object(sglang_common.time, 'sleep'):
            passed, _end = sglang_common.verify_inference_results_subtests(
                {'node1': {'mean_ttft_ms': '500.0', 'p99_ttft_ms': '10.0'}},
                {'mean_ttft_ms': 100.0, 'p99_ttft_ms': 100.0},
                host_exec,
                subtests,
                'bench_serv',
                lifecycle=lifecycle,
                report_nodeid='nid',
            )
        self.assertFalse(passed)
        rows = {r['metric']: r['status'] for r in lifecycle.perf_metric_rows['nid']}
        self.assertEqual(rows, {'mean_ttft_ms': 'fail', 'p99_ttft_ms': 'pass'})
        self.assertEqual(len(subtests.failures), 1)

    def test_verify_inference_results_subtests_record_only_passes_on_violation(self):
        host_exec = mock.Mock(return_value={'head': 'time'})
        lifecycle = mock.Mock()
        lifecycle.perf_metric_rows = {}
        subtests = _FakeSubtests()
        with mock.patch.object(sglang_common.time, 'sleep'):
            passed, _end = sglang_common.verify_inference_results_subtests(
                {'node1': {'mean_ttft_ms': '500.0', 'p99_ttft_ms': '10.0'}},
                {'mean_ttft_ms': 100.0, 'p99_ttft_ms': 100.0},
                host_exec,
                subtests,
                'bench_serv',
                lifecycle=lifecycle,
                report_nodeid='nid',
                enforce_thresholds=False,
            )
        self.assertTrue(passed)
        rows = {r['metric']: r['status'] for r in lifecycle.perf_metric_rows['nid']}
        self.assertEqual(rows, {'mean_ttft_ms': 'pass', 'p99_ttft_ms': 'pass'})
        self.assertEqual(subtests.failures, [])

    def test_verify_inference_results_subtests_record_only_fails_without_results(self):
        host_exec = mock.Mock(return_value={'head': 'time'})
        lifecycle = mock.Mock()
        lifecycle.perf_metric_rows = {}
        subtests = _FakeSubtests()
        with mock.patch.object(sglang_common.time, 'sleep'):
            passed, _end = sglang_common.verify_inference_results_subtests(
                {},
                {'mean_ttft_ms': 100.0},
                host_exec,
                subtests,
                'bench_serv',
                lifecycle=lifecycle,
                report_nodeid='nid',
                enforce_thresholds=False,
            )
        self.assertFalse(passed)
        self.assertEqual(subtests.failures, [])

    def test_poll_for_inference_completion_success(self):
        log_text = {'bench': _SAMPLE_BENCH_LOG}

        def fetch_log_tail():
            return log_text

        with mock.patch.object(sglang_common.time, 'sleep'):
            result = sglang_common.poll_for_inference_completion(
                fetch_log_tail,
                sglang_common.parse_inference_bench_results,
                iterations=2,
                waittime_between_iters=0,
            )
        self.assertEqual(result['status'], 'success')
        self.assertIn('bench', result['results'])


if __name__ == '__main__':
    unittest.main()
