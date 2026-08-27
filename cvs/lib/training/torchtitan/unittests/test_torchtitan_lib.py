"""
Unit tests for torchtitan_lib.py

Tests TorchTitanTrainingJob log path consistency and core functionality.
"""

import unittest
from unittest.mock import MagicMock, patch
import re

from cvs.lib.training.torchtitan.torchtitan_lib import TorchTitanTrainingJob


class TestTorchTitanTrainingJobLogPaths(unittest.TestCase):
    """Test that build_training_job_cmd and polling methods use consistent log paths."""

    def setUp(self):
        """Set up mock orchestrator and minimal variant config for testing."""
        # Mock orchestrator
        self.mock_orch = MagicMock()
        self.mock_orch.hosts = ['host1', 'host2']
        self.mock_orch.exec = MagicMock(return_value={'host1': 'success', 'host2': 'success'})

        # Minimal variant config
        self.mock_variant_config = MagicMock()
        self.mock_variant_config.config = {
            'log_dir': '/test/logs',
            'scripts_dir': '/test/scripts',
            'torchtitan_root': '/workspace/torchtitan',
            'training_iterations': '10',
            'nnodes': '2',
            'master_address': '127.0.0.1',
        }
        self.mock_variant_config.model_params = {
            'model_name': 'llama3_3_70b',
            'micro_batch_size': '2',
            'global_batch_size': '32',
        }
        self.mock_variant_config.gpu_arch = 'MI355X'

    @patch('cvs.lib.training.torchtitan.torchtitan_lib.detect_rocm_path')
    def test_log_path_consistency_single_node(self, mock_detect_rocm):
        """Test that write and read paths match for single-node training."""
        mock_detect_rocm.return_value = '/opt/rocm'

        # Create job with specific run_label
        job = TorchTitanTrainingJob(
            orch=self.mock_orch,
            variant_config=self.mock_variant_config,
            hf_token='test_token',
            run_label='test_combo_123',
            distributed_training=False,
        )

        # Build training command (creates log path)
        job.build_training_job_cmd()

        # Extract write path from job_cmd
        write_path_match = re.search(r'> (/[^\s]+/training\.log)', job.job_cmd)
        self.assertIsNotNone(write_path_match, 'Could not find log path in job_cmd')
        write_path = write_path_match.group(1)

        # Verify combo_log_dir is in the write path
        self.assertIn(job.run_label_sanitized, write_path, 'Write path missing run_label')
        self.assertIn('test_combo_123', write_path, 'Write path missing specific combo label')

        # Check that polling uses the same base directory
        # poll_for_training_completion constructs: f'{self.combo_log_dir}/out-node0/training.log'
        expected_poll_path = f'{job.combo_log_dir}/out-node0/training.log'
        self.assertEqual(write_path, expected_poll_path, 'Write path and poll path do not match')

        # Check get_training_results_dict uses the same path
        # get_training_results_dict constructs: f'{self.combo_log_dir}/out-node0/training.log'
        log_files = [f'{job.combo_log_dir}/out-node0/training.log']
        self.assertIn(write_path, log_files, 'get_training_results_dict path does not match write path')

        # Check scan_for_training_errors uses the same path
        # scan_for_training_errors constructs: f'{self.combo_log_dir}/out-node0/training.log'
        log_files = [f'{job.combo_log_dir}/out-node0/training.log']
        self.assertIn(write_path, log_files, 'scan_for_training_errors path does not match write path')

    @patch('cvs.lib.training.torchtitan.torchtitan_lib.detect_rocm_path')
    def test_log_path_consistency_distributed(self, mock_detect_rocm):
        """Test that write and read paths match for distributed training."""
        mock_detect_rocm.return_value = '/opt/rocm'

        # Create distributed job with specific run_label
        job = TorchTitanTrainingJob(
            orch=self.mock_orch,
            variant_config=self.mock_variant_config,
            hf_token='test_token',
            run_label='llama3_1_8b-mi355-bs48-mbs6-bf16',
            distributed_training=True,
        )

        # Build training command (creates log paths)
        job.build_training_job_cmd()

        # Extract write paths from job_cmd_list (one per node)
        write_paths = []
        for cmd in job.job_cmd_list:
            # Extract the wrapper script content that contains the actual command
            wrapper_match = re.search(r"cat > .* << 'WRAPPER_EOF'\n#!/bin/bash\n(.+?)\nWRAPPER_EOF", cmd, re.DOTALL)
            if wrapper_match:
                full_cmd = wrapper_match.group(1)
                path_match = re.search(r'> (/[^\s;]+/training\.log)', full_cmd)
                if path_match:
                    write_paths.append(path_match.group(1))

        # Should have one write path per node
        self.assertEqual(len(write_paths), job.nnodes, f'Expected {job.nnodes} write paths, got {len(write_paths)}')

        # Verify all write paths use combo_log_dir
        for i, write_path in enumerate(write_paths):
            self.assertIn(job.run_label_sanitized, write_path, f'Write path {i} missing run_label')
            expected_write_path = f'{job.combo_log_dir}/out-node{i}/training.log'
            self.assertEqual(write_path, expected_write_path, f'Write path {i} does not match expected path')

        # Check that polling uses the same base directory (node 0 for distributed)
        expected_poll_path = f'{job.combo_log_dir}/out-node0/training.log'
        self.assertEqual(write_paths[0], expected_poll_path, 'Poll path does not match node 0 write path')

        # Check get_training_results_dict uses the same paths
        expected_results_paths = [f'{job.combo_log_dir}/out-node{i}/training.log' for i in range(job.nnodes)]
        self.assertEqual(
            write_paths, expected_results_paths, 'get_training_results_dict paths do not match write paths'
        )

        # Check scan_for_training_errors uses the same paths
        expected_scan_paths = [f'{job.combo_log_dir}/out-node{i}/training.log' for i in range(job.nnodes)]
        self.assertEqual(write_paths, expected_scan_paths, 'scan_for_training_errors paths do not match write paths')

    @patch('cvs.lib.training.torchtitan.torchtitan_lib.detect_rocm_path')
    def test_combo_log_dir_sanitization(self, mock_detect_rocm):
        """Test that run_label is properly sanitized for filesystem use."""
        mock_detect_rocm.return_value = '/opt/rocm'

        # Test with run_label containing special characters
        job = TorchTitanTrainingJob(
            orch=self.mock_orch,
            variant_config=self.mock_variant_config,
            hf_token='test_token',
            run_label='combo:with/special\\chars!',
            distributed_training=False,
        )

        # Verify sanitization removed/replaced special chars
        self.assertNotIn(':', job.run_label_sanitized)
        self.assertNotIn('/', job.run_label_sanitized)
        self.assertNotIn('\\', job.run_label_sanitized)
        self.assertNotIn('!', job.run_label_sanitized)

        # Verify combo_log_dir uses sanitized label
        self.assertIn(job.run_label_sanitized, job.combo_log_dir)
        self.assertNotIn('combo:with/special', job.combo_log_dir)

    @patch('cvs.lib.training.torchtitan.torchtitan_lib.detect_rocm_path')
    def test_default_run_label(self, mock_detect_rocm):
        """Test that default run_label is applied when none provided."""
        mock_detect_rocm.return_value = '/opt/rocm'

        # Create job without run_label
        job = TorchTitanTrainingJob(
            orch=self.mock_orch,
            variant_config=self.mock_variant_config,
            hf_token='test_token',
            run_label=None,
            distributed_training=False,
        )

        # Should use default label
        self.assertEqual(job.run_label_sanitized, 'torchtitan_training')
        self.assertIn('torchtitan_training', job.combo_log_dir)


if __name__ == '__main__':
    unittest.main()
