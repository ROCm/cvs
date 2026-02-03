# cvs/lib/inference/unittests/test_vllm.py
import unittest
from unittest.mock import MagicMock, patch
from cvs.lib.inference.base import InferenceBaseJob
from cvs.lib.inference.vllm import VllmJob


class TestCollectTestResult(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.mock_c_phdl = MagicMock()
        self.mock_s_phdl = MagicMock()

        # Create minimal config for VllmJob
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
                'sequence_combinations': [
                    {'isl': '1024', 'osl': '1024', 'name': 'balanced'},
                    {'isl': '1024', 'osl': '8192', 'name': 'long_generation'},
                    {'isl': '8192', 'osl': '1024', 'name': 'long_context'},
                ],
            }
        }

        self.job = VllmJob(
            c_phdl=self.mock_c_phdl,
            s_phdl=self.mock_s_phdl,
            model_name='gpt-oss-120b',
            inference_config_dict=inference_config,
            benchmark_params_dict=benchmark_params,
            hf_token='test_token',
            gpu_type='mi355x',
        )

        # Clear class variable for test isolation
        InferenceBaseJob.all_test_results = {}

    def tearDown(self):
        """Clean up after tests."""
        InferenceBaseJob.all_test_results = {}

    def test_collect_test_result_with_success_status(self):
        """Test collecting successful test results."""
        # Set up inference results
        self.job.inference_results_dict = {
            'node1': {
                'total_throughput_per_sec': '4500',
                'mean_ttft_ms': '200',
                'mean_tpot_ms': '12',
            }
        }

        self.job.collect_test_result("success")

        # Check that results were stored in class variable
        expected_key = ('gpt-oss-120b', 'mi355x', '1024', '8192', 'long_generation', 32)
        self.assertIn(expected_key, InferenceBaseJob.all_test_results)
        self.assertEqual(InferenceBaseJob.all_test_results[expected_key]['status'], 'success')
        self.assertEqual(InferenceBaseJob.all_test_results[expected_key]['results'], self.job.inference_results_dict)

    def test_collect_test_result_with_failed_status(self):
        """Test collecting failed test results."""
        self.job.inference_results_dict = {
            'node1': {
                'total_throughput_per_sec': '3000',
                'mean_ttft_ms': '300',
                'mean_tpot_ms': '20',
            }
        }

        self.job.collect_test_result("failed")

        expected_key = ('gpt-oss-120b', 'mi355x', '1024', '8192', 'long_generation', 32)
        self.assertIn(expected_key, InferenceBaseJob.all_test_results)
        self.assertEqual(InferenceBaseJob.all_test_results[expected_key]['status'], 'failed')

    def test_store_test_result_finds_correct_sequence_name(self):
        """Test that store_test_result correctly identifies sequence combination name."""
        # Set to long_context combination
        self.job.bp_dict['input_sequence_length'] = '8192'
        self.job.bp_dict['output_sequence_length'] = '1024'
        self.job.bp_dict['max_concurrency'] = '16'

        self.job.inference_results_dict = {'node1': {'total_throughput_per_sec': '16509'}}

        self.job.collect_test_result("success")

        expected_key = ('gpt-oss-120b', 'mi355x', '8192', '1024', 'long_context', 16)
        self.assertIn(expected_key, VllmJob.all_test_results)

    def test_store_test_result_with_unknown_sequence(self):
        """Test handling of unknown sequence combination."""
        # Set to a combination not in sequence_combinations
        self.job.bp_dict['input_sequence_length'] = '2048'
        self.job.bp_dict['output_sequence_length'] = '4096'

        self.job.inference_results_dict = {'node1': {'total_throughput_per_sec': '5000'}}

        self.job.collect_test_result("success")

        # Should use "unknown" as sequence name
        expected_key = ('gpt-oss-120b', 'mi355x', '2048', '4096', 'unknown', 32)
        self.assertIn(expected_key, VllmJob.all_test_results)

    def test_collect_test_result_with_empty_inference_results_dict(self):
        """Test that collect_test_result handles empty inference_results_dict."""
        # Don't set inference_results_dict or set it to empty
        self.job.inference_results_dict = {}

        # Should not store anything if inference_results_dict is empty
        self.job.collect_test_result("success")
        self.assertEqual(len(InferenceBaseJob.all_test_results), 0)

    def test_collect_test_result_accumulates_multiple_tests(self):
        """Test that multiple test results accumulate in class variable."""
        # Test 1: long_generation conc32
        self.job.inference_results_dict = {'node1': {'total_throughput_per_sec': '4000'}}
        self.job.collect_test_result("success")

        # Test 2: long_context conc16
        self.job.bp_dict['input_sequence_length'] = '8192'
        self.job.bp_dict['output_sequence_length'] = '1024'
        self.job.bp_dict['max_concurrency'] = '16'
        self.job.inference_results_dict = {'node1': {'total_throughput_per_sec': '16500'}}
        self.job.collect_test_result("success")

        # Should have both results
        self.assertEqual(len(InferenceBaseJob.all_test_results), 2)

    def test_collect_test_result_overwrites_same_config(self):
        """Test that collecting results for same config overwrites previous."""
        self.job.inference_results_dict = {'node1': {'total_throughput_per_sec': '4000'}}
        self.job.collect_test_result("success")

        # Store again with different results
        self.job.inference_results_dict = {'node1': {'total_throughput_per_sec': '4500'}}
        self.job.collect_test_result("success")

        # Should have only 1 entry (overwritten)
        self.assertEqual(len(InferenceBaseJob.all_test_results), 1)
        expected_key = ('gpt-oss-120b', 'mi355x', '1024', '8192', 'long_generation', 32)
        self.assertEqual(
            InferenceBaseJob.all_test_results[expected_key]['results']['node1']['total_throughput_per_sec'], '4500'
        )


class TestPrintAllResults(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        InferenceBaseJob.all_test_results = {}

    def tearDown(self):
        """Clean up after tests."""
        InferenceBaseJob.all_test_results = {}

    @patch('cvs.lib.inference.vllm.update_test_result')
    @patch('builtins.print')
    def test_print_all_results_with_empty_results(self, mock_print, mock_update):
        """Test printing when no results are stored."""
        VllmJob.print_all_results()

        # Should still call update_test_result
        mock_update.assert_called_once()

    @patch('cvs.lib.inference.vllm.update_test_result')
    @patch('builtins.print')
    def test_print_all_results_with_single_result(self, mock_print, mock_update):
        """Test printing with a single test result."""
        InferenceBaseJob.all_test_results = {
            ('gpt-oss-120b', 'mi355x', '1024', '8192', 'long_generation', 32): {
                'node1': {
                    'successful_requests': '640',
                    'total_throughput_per_sec': '4038',
                    'mean_ttft_ms': '230',
                    'mean_tpot_ms': '13',
                    'p99_itl_ms': '150',
                }
            }
        }

        VllmJob.print_all_results()

        # Check that table was printed
        table_printed = any('gpt-oss-120b' in str(call) for call in mock_print.call_args_list)
        self.assertTrue(table_printed or len(mock_print.call_args_list) > 0)

    @patch('cvs.lib.inference.vllm.update_test_result')
    @patch('builtins.print')
    def test_print_all_results_with_multiple_results(self, mock_print, mock_update):
        """Test printing with multiple test results."""
        InferenceBaseJob.all_test_results = {
            ('gpt-oss-120b', 'mi355x', '1024', '8192', 'long_generation', 32): {
                'node1': {
                    'successful_requests': '640',
                    'total_throughput_per_sec': '4038',
                    'mean_ttft_ms': '230',
                    'mean_tpot_ms': '13',
                    'p99_itl_ms': '150',
                }
            },
            ('gpt-oss-120b', 'mi355x', '8192', '1024', 'long_context', 16): {
                'node1': {
                    'successful_requests': '800',
                    'total_throughput_per_sec': '16509',
                    'mean_ttft_ms': '350',
                    'mean_tpot_ms': '20',
                    'p99_itl_ms': '200',
                }
            },
        }

        VllmJob.print_all_results()

        # Should have printed both results
        self.assertTrue(len(mock_print.call_args_list) > 0)


class TestClearAllResults(unittest.TestCase):
    def test_clear_all_results(self):
        """Test that clear_all_results empties the class variable."""
        InferenceBaseJob.all_test_results = {('test', 'gpu', '1024', '1024', 'balanced', 16): {}}

        VllmJob.clear_all_results()

        self.assertEqual(InferenceBaseJob.all_test_results, {})


if __name__ == '__main__':
    unittest.main()
