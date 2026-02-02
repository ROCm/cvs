# cvs/lib/inference/unittests/test_base.py
import unittest
from unittest.mock import MagicMock, patch
from cvs.lib.inference.base import InferenceBaseJob


class TestVerifyInferenceResults(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.mock_c_phdl = MagicMock()
        self.mock_s_phdl = MagicMock()

        # Create minimal config for InferenceBaseJob
        inference_config = {
            'container_name': 'test_container',
            'log_dir': '/test/logs',
        }

        benchmark_params = {
            'gpt-oss-120b': {
                'server_script': 'test_server.sh',
                'bench_serv_script': 'test_bench.py',
                'input_sequence_length': '1024',
                'output_sequence_length': '8192',
                'tensor_parallelism': '1',
                'max_concurrency': '32',
                'result_dict': {
                    'ISL=1024,OSL=8192,TP=1,CONC=32': {
                        'total_throughput_per_sec': '4000',
                        'mean_ttft_ms': '250',
                        'mean_tpot_ms': '15',
                    }
                },
            }
        }

        self.job = InferenceBaseJob(
            c_phdl=self.mock_c_phdl,
            s_phdl=self.mock_s_phdl,
            model_name='gpt-oss-120b',
            inference_config_dict=inference_config,
            benchmark_params_dict=benchmark_params,
            hf_token='test_token',
            gpu_type='mi355x',
        )

        # Set up test results that would come from get_inference_results_dict
        self.job.inference_results_dict = {
            'node1': {
                'total_throughput_per_sec': '4500',
                'mean_ttft_ms': '200',
                'mean_tpot_ms': '12',
            }
        }

    @patch('cvs.lib.inference.base.verify_dmesg_for_errors')
    def test_verify_with_valid_results_all_pass(self, mock_verify_dmesg):
        """Test verification passes when all metrics meet thresholds."""
        # verify_inference_results should complete without raising exception
        self.job.verify_inference_results()
        mock_verify_dmesg.assert_called_once()

    @patch('cvs.lib.inference.base.verify_dmesg_for_errors')
    @patch('cvs.lib.inference.base.fail_test')
    def test_verify_with_throughput_below_threshold(self, mock_fail_test, mock_verify_dmesg):
        """Test verification fails when throughput is below threshold."""
        # Set throughput below expected
        self.job.inference_results_dict['node1']['total_throughput_per_sec'] = '3000'

        # Configure fail_test to raise an exception when called
        mock_fail_test.side_effect = Exception("Throughput below threshold")

        with self.assertRaises(Exception):
            self.job.verify_inference_results()
        mock_fail_test.assert_called_once()

    @patch('cvs.lib.inference.base.verify_dmesg_for_errors')
    @patch('cvs.lib.inference.base.fail_test')
    def test_verify_with_latency_above_threshold(self, mock_fail_test, mock_verify_dmesg):
        """Test verification fails when latency exceeds threshold."""
        # Set latency above expected
        self.job.inference_results_dict['node1']['mean_ttft_ms'] = '300'

        # Configure fail_test to raise an exception when called
        mock_fail_test.side_effect = Exception("Latency above threshold")

        with self.assertRaises(Exception):
            self.job.verify_inference_results()
        mock_fail_test.assert_called_once()

    def test_verify_with_missing_result_dict(self):
        """Test verification skips when result_dict is missing from config."""
        # Remove result_dict from benchmark params
        del self.job.bp_dict['result_dict']

        with patch('builtins.print') as mock_print:
            self.job.verify_inference_results()
            # Should print warning and return early
            warning_calls = [call for call in mock_print.call_args_list if 'WARNING: No result_dict' in str(call)]
            self.assertTrue(len(warning_calls) > 0)

    def test_verify_with_missing_config_key(self):
        """Test verification skips when specific config key is not in result_dict."""
        # Change concurrency so config_key won't match
        self.job.bp_dict['max_concurrency'] = '64'

        with patch('builtins.print') as mock_print:
            self.job.verify_inference_results()
            # Should print warning and return early
            warning_calls = [
                call for call in mock_print.call_args_list if 'WARNING: No expected results for config' in str(call)
            ]
            self.assertTrue(len(warning_calls) > 0)

    @patch('cvs.lib.inference.base.verify_dmesg_for_errors')
    def test_verify_with_empty_inference_results_dict(self, mock_verify_dmesg):
        """Test verification when inference_results_dict is empty."""
        # Simulate empty results dict
        self.job.inference_results_dict = {}

        # Should complete without exception because validation loop never executes
        self.job.verify_inference_results()
        # Verification should pass because loop over empty dict means validation_passed stays True

    @patch('cvs.lib.inference.base.verify_dmesg_for_errors')
    def test_verify_with_missing_metric_in_results(self, mock_verify_dmesg):
        """Test verification skips missing metrics with warning."""
        # Remove one metric from results
        del self.job.inference_results_dict['node1']['mean_tpot_ms']

        with patch('builtins.print') as mock_print:
            self.job.verify_inference_results()
            # Should print warning about missing metric
            warning_calls = [
                call
                for call in mock_print.call_args_list
                if 'WARNING: Metric' in str(call) and 'not found' in str(call)
            ]
            self.assertTrue(len(warning_calls) > 0)

    @patch('cvs.lib.inference.base.verify_dmesg_for_errors')
    @patch('cvs.lib.inference.base.fail_test')
    def test_verify_stores_failed_status_on_exception(self, mock_fail_test, mock_verify_dmesg):
        """Test that validation raises exception on invalid number format."""
        # Force an exception during validation by providing invalid float string
        self.job.inference_results_dict['node1']['mean_ttft_ms'] = 'invalid_number'

        # Should raise exception when trying to convert 'invalid_number' to float
        with self.assertRaises(ValueError):
            self.job.verify_inference_results()


class TestGetInferenceResultsDict(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.mock_c_phdl = MagicMock()
        self.mock_s_phdl = MagicMock()

        inference_config = {
            'container_name': 'test_container',
            'log_dir': '/test/logs',
        }

        benchmark_params = {
            'gpt-oss-120b': {
                'server_script': 'test_server.sh',
                'bench_serv_script': 'test_bench.py',
            }
        }

        self.job = InferenceBaseJob(
            c_phdl=self.mock_c_phdl,
            s_phdl=self.mock_s_phdl,
            model_name='gpt-oss-120b',
            inference_config_dict=inference_config,
            benchmark_params_dict=benchmark_params,
            hf_token='test_token',
            gpu_type='mi355x',
        )

    def test_get_inference_results_dict_parses_all_metrics(self):
        """Test that all metrics are parsed from benchmark output."""
        benchmark_output = """
Successful requests:                     800
Benchmark duration (s):                  400.07
Total input tokens:                      5916180
Total generated tokens:                  736700
Request throughput (req/s):              2.00
Output token throughput (tok/s):         1841.44
Total Token throughput (tok/s):          16629.36
---------------Time to First Token----------------
Mean TTFT (ms):                          210.29
Median TTFT (ms):                        172.69
P99 TTFT (ms):                           1118.68
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.43
Median TPOT (ms):                        8.45
P99 TPOT (ms):                           9.05
---------------Inter-token Latency----------------
Mean ITL (ms):                           8.44
Median ITL (ms):                         6.68
P99 ITL (ms):                            115.43
----------------End-to-end Latency----------------
Mean E2EL (ms):                          7966.19
Median E2EL (ms):                        7984.96
P99 E2EL (ms):                           9095.12
"""
        out_dict = {'node1': benchmark_output}

        result = self.job.get_inference_results_dict(out_dict)

        # Verify it creates inference_results_dict (with 's')
        self.assertTrue(hasattr(self.job, 'inference_results_dict'))
        self.assertIn('node1', result)
        self.assertEqual(result['node1']['successful_requests'], '800')
        self.assertEqual(result['node1']['total_throughput_per_sec'], '16629.36')
        self.assertEqual(result['node1']['mean_ttft_ms'], '210.29')
        self.assertEqual(result['node1']['mean_tpot_ms'], '8.43')
        self.assertEqual(result['node1']['p99_itl_ms'], '115.43')


if __name__ == '__main__':
    unittest.main()
