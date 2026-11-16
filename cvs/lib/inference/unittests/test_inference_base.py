import unittest
from unittest.mock import MagicMock, patch

from cvs.lib.inference.base import InferenceBaseJob


VLLM_SERVING_OUTPUT = """============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  123.45
Total input tokens:                      1024000
Total generated tokens:                  512000.5
Request throughput (req/s):              8.10
Output token throughput (tok/s):         8294.71
Total Token throughput (tok/s):          16589.42
---------------Time to First Token----------------
Mean TTFT (ms):                          123.45
Median TTFT (ms):                        110.00
P99 TTFT (ms):                           400.12
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          15.03
Median TPOT (ms):                        15.00
P99 TPOT (ms):                           16.00
---------------Inter-token Latency----------------
Mean ITL (ms):                           15.01
Median ITL (ms):                         14.62
P99 ITL (ms):                            27.51
----------------End-to-end Latency----------------
Mean E2EL (ms):                          19668.87
Median E2EL (ms):                        19835.18
P99 E2EL (ms):                           34034.53
==================================================
"""

EXPECTED_METRICS = {
    'successful_requests': '1000',
    'benchmark_duration': '123',
    'total_input_tokens': '1024000',
    'total_generated_tokens': '512000.5',
    'request_throughput_per_sec': '8.10',
    'output_throughput_per_sec': '8294.71',
    'total_throughput_per_sec': '16589.42',
    'mean_ttft_ms': '123.45',
    'median_ttft_ms': '110.00',
    'p99_ttft_ms': '400.12',
    'mean_tpot_ms': '15.03',
    'median_tpot_ms': '15.00',
    'p99_tpot_ms': '16.00',
    'mean_itl_ms': '15.01',
    'median_itl_ms': '14.62',
    'p99_itl_ms': '27.51',
    'mean_e2el_ms': '19668.87',
    'median_e2el_ms': '19835.18',
    'p99_e2el_ms': '34034.53',
}


def _job(**attrs):
    job = InferenceBaseJob.__new__(InferenceBaseJob)
    for key, value in attrs.items():
        setattr(job, key, value)
    return job


class TestGetInferenceResultsDict(unittest.TestCase):
    def test_parses_every_metric_from_vllm_serving_output(self):
        job = _job()
        result = job.get_inference_results_dict({'node1': VLLM_SERVING_OUTPUT})

        self.assertEqual(result, {'node1': EXPECTED_METRICS})
        self.assertEqual(job.inference_results_dict, {'node1': EXPECTED_METRICS})

    def test_case_insensitive_match(self):
        result = _job().get_inference_results_dict({'node1': VLLM_SERVING_OUTPUT.lower()})

        self.assertEqual(result, {'node1': EXPECTED_METRICS})

    def test_missing_metrics_are_omitted(self):
        result = _job().get_inference_results_dict({'node1': 'no benchmark here'})

        self.assertEqual(result, {'node1': {}})

    def test_literal_parentheses_required(self):
        result = _job().get_inference_results_dict({'node1': 'Request throughput req/s:   8.10'})

        self.assertNotIn('request_throughput_per_sec', result['node1'])


class TestExecNicSetupScripts(unittest.TestCase):
    def setUp(self):
        self.s_phdl = MagicMock()
        self.job = _job(
            distributed_inference=True,
            nic_type='thor2',
            container_name='c',
            nccl_ib_gid_index='0',
            s_phdl=self.s_phdl,
        )

    @patch('cvs.lib.inference.base.fail_test')
    def test_bnxt_hca_present_passes(self, fail_test):
        self.s_phdl.exec.return_value = {'n1': 'hca_id:\tbnxt_re0\n\ttransport: InfiniBand'}

        self.job.exec_nic_setup_scripts()

        fail_test.assert_not_called()
        self.assertEqual(self.job.nccl_ib_gid_index, 3)

    @patch('cvs.lib.inference.base.fail_test')
    def test_non_bnxt_hca_fails(self, fail_test):
        self.s_phdl.exec.return_value = {'n1': 'hca_id:\tmlx5_0'}

        self.job.exec_nic_setup_scripts()

        fail_test.assert_called_once()
        self.assertIn('n1', fail_test.call_args.args[0])

    @patch('cvs.lib.inference.base.fail_test')
    def test_hca_id_without_whitespace_fails(self, fail_test):
        self.s_phdl.exec.return_value = {'n1': 'hca_id:bnxt_re0'}

        self.job.exec_nic_setup_scripts()

        fail_test.assert_called_once()

    def test_non_distributed_is_noop(self):
        self.job.distributed_inference = False

        self.job.exec_nic_setup_scripts()

        self.s_phdl.exec.assert_not_called()
