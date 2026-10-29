"""
Unit tests for torchtitan_lib.py

Tests TorchTitanTrainingJob log path consistency and core functionality.
"""

import unittest
from unittest.mock import MagicMock, patch
import re

from cvs.lib.training.torchtitan.torchtitan_lib import (
    TorchTitanTrainingJob,
    _hf_token_export_line,
    _redact_logged_text,
    _require_hf_token_file,
)

FAKE_CRED = 'fake-cred'
FAKE_CRED_FILE = '/test/.hf_token'


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
            'data_cache_dir': '/test/cache',
            'hf_token_file': '/test/.hf_token',
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
        self.mock_variant_config.train_params = self.mock_variant_config.model_params
        self.mock_variant_config.job_config_dict.return_value = self.mock_variant_config.config
        self.mock_variant_config.gpu_name = 'MI355X'
        self.mock_variant_config.gpu_arch = 'MI355X'

    @patch('cvs.lib.training.torchtitan.torchtitan_lib.detect_rocm_path')
    def test_log_path_consistency_single_node(self, mock_detect_rocm):
        """Test that write and read paths match for single-node training."""
        mock_detect_rocm.return_value = '/opt/rocm'

        # Create job with specific run_label
        job = TorchTitanTrainingJob(
            orch=self.mock_orch,
            variant_config=self.mock_variant_config,
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
            run_label=None,
            distributed_training=False,
        )

        # Should use default label
        self.assertEqual(job.run_label_sanitized, 'torchtitan_training')
        self.assertIn('torchtitan_training', job.combo_log_dir)


if __name__ == '__main__':
    unittest.main()


class TestDistributedConfigStructure(unittest.TestCase):
    """Test that distributed configs have the required flattened structure."""

    def test_distributed_configs_have_model_name(self):
        """Test that all distributed configs have train_params.model_name at top level."""
        import json
        import glob
        import os

        # Find all mi355x distributed config files
        config_dir = os.path.join(os.path.dirname(__file__), "../../../../input/config_file/training/torchtitan")
        pattern = os.path.join(config_dir, "mi355x_torchtitan_*_distributed.json")
        config_files = glob.glob(pattern)

        self.assertGreater(len(config_files), 0, "No distributed config files found")

        for config_file in config_files:
            with self.subTest(config=os.path.basename(config_file)):
                with open(config_file, "r") as f:
                    config = json.load(f)

                # Assert train_params exists
                self.assertIn("train_params", config, f"{config_file}: Missing train_params")

                train_params = config["train_params"]

                # Assert model_name is at top level (not nested)
                self.assertIn("model_name", train_params, f"{config_file}: Missing train_params.model_name")

                # Assert model_name is a valid TORCHTITAN_MODELS key
                model_name = train_params["model_name"]
                valid_models = [
                    "llama3_1_8b",
                    "llama3_1_70b",
                    "llama3_1_405b",
                    "llama3_3_70b",
                    "qwen3_32b",
                    "deepseek_v2_lite",
                ]
                self.assertIn(
                    model_name,
                    valid_models,
                    f"{config_file}: model_name {model_name} is not a valid TORCHTITAN_MODELS key",
                )

                # Assert tokenizer_model exists at top level
                self.assertIn(
                    "tokenizer_model",
                    train_params,
                    f"{config_file}: Missing train_params.tokenizer_model (required by test_download_tokenizer)",
                )

                # Assert train_params is NOT nested (no multi_node key)
                self.assertNotIn(
                    "multi_node",
                    train_params,
                    f"{config_file}: train_params should not contain nested multi_node structure",
                )


class TestHfTokenFile(unittest.TestCase):
    """The token stays on the node. Launch commands read the file, they do not embed it."""

    def test_export_line_reads_the_file_inside_the_container(self):
        line = _hf_token_export_line('/home/user/.hf_token')
        self.assertIn('[ -s /home/user/.hf_token ]', line)
        self.assertIn("export HF_TOKEN=\"$(tr -d '[:space:]' < /home/user/.hf_token)\"", line)
        self.assertIn('exit 1', line)

    def test_export_line_fails_when_the_path_is_unset(self):
        line = _hf_token_export_line('  ')
        self.assertIn('paths.hf_token_file is not configured', line)
        self.assertIn('exit 1', line)

    def test_require_raises_when_the_path_is_unset(self):
        with patch('cvs.lib.training.torchtitan.torchtitan_lib.fail_test') as fail:
            with self.assertRaises(FileNotFoundError):
                _require_hf_token_file(MagicMock(), '')
        fail.assert_called_once()

    def test_require_raises_when_the_file_is_missing_on_a_host(self):
        orch = MagicMock()
        orch.hosts = ['n0', 'n1']
        orch.exec.return_value = {'n0': {'exit_code': 0}, 'n1': {'exit_code': 1}}
        with patch('cvs.lib.training.torchtitan.torchtitan_lib.fail_test') as fail:
            with self.assertRaises(FileNotFoundError):
                _require_hf_token_file(orch, '/home/user/.hf_token')
        fail.assert_called_once()
        command = orch.exec.call_args.args[0]
        self.assertIn('test -s', command)
        self.assertIn('/home/user/.hf_token', command)
        self.assertTrue(orch.exec.call_args.kwargs.get('detailed'))

    def test_redact_logged_text_masks_a_bare_hub_token(self):
        bare = 'hf_' + ('c' * 34)
        redacted = _redact_logged_text(f'Invalid credentials for token {bare} and .hf_token')
        self.assertIn('token <redacted>', redacted)
        self.assertIn('.hf_token', redacted)
        self.assertNotIn(bare, redacted)

    @patch('cvs.lib.training.torchtitan.torchtitan_lib.detect_rocm_path')
    @patch('cvs.lib.training.torchtitan.torchtitan_lib.log')
    def test_download_and_launch_hide_token_values(self, mock_log, mock_detect):
        mock_detect.return_value = '/opt/rocm'
        orch = MagicMock()
        orch.hosts = ['n0']
        command_secret = 'hf_' + ('a' * 34)
        launch_secret = 'hf_' + ('e' * 34)
        download_secret = 'hf_' + ('f' * 34)

        def _exec(cmd, **kwargs):
            if str(cmd).startswith('test -s'):
                return {'n0': {'exit_code': 0}}
            if 'download_hf_assets.py' in str(cmd):
                return {'n0': f'error: Invalid credentials for token {download_secret}'}
            if str(cmd).startswith('bash -c'):
                return {'n0': f'launch failed for token {launch_secret}'}
            return {'n0': ''}

        orch.exec.side_effect = _exec
        variant = MagicMock()
        variant.job_config_dict.return_value = {
            'log_dir': '/test/logs',
            'scripts_dir': '/test/scripts',
            'data_cache_dir': '/test/cache',
            'hf_token_file': FAKE_CRED_FILE,
            'torchtitan_root': '/workspace/torchtitan',
            'training_iterations': '10',
        }
        variant.train_params = {'model_name': 'llama3_1_8b'}
        variant.gpu_name = 'MI355X'
        job = TorchTitanTrainingJob(
            orch=orch,
            variant_config=variant,
            run_label='token',
            distributed_training=False,
        )
        job.job_cmd = _hf_token_export_line(FAKE_CRED_FILE) + f'echo token {command_secret}'
        job.download_hf_assets()
        job.start_training_job()
        logged = ' '.join(
            str(arg) for call in mock_log.info.call_args_list + mock_log.warning.call_args_list for arg in call.args
        )
        preview = next(
            str(arg) for call in mock_log.info.call_args_list for arg in call.args if 'echo token' in str(arg)
        )
        self.assertIn(FAKE_CRED_FILE, preview)
        self.assertIn('HF_TOKEN=<redacted>', preview)
        self.assertNotIn(command_secret, preview)
        launch_call = next(call for call in orch.exec.call_args_list if str(call.args[0]).startswith('bash -c'))
        self.assertIn(command_secret, launch_call.args[0])
        self.assertFalse(launch_call.kwargs.get('print_console', True))
        download_call = next(call for call in orch.exec.call_args_list if 'download_hf_assets.py' in str(call.args[0]))
        self.assertNotIn(download_secret, download_call.args[0])
        self.assertNotIn(command_secret, logged)
        self.assertNotIn(launch_secret, logged)
        self.assertNotIn(download_secret, logged)

    @patch('cvs.lib.training.torchtitan.torchtitan_lib.detect_rocm_path')
    @patch('cvs.lib.training.torchtitan.torchtitan_lib.log')
    def test_distributed_launch_hides_token_values(self, mock_log, mock_detect):
        """The distributed 'bash <wrapper script>' launch must not print its raw stdout/stderr."""
        mock_detect.return_value = '/opt/rocm'
        orch = MagicMock()
        orch.hosts = ['n0', 'n1']
        launch_secret = 'hf_' + ('d' * 34)

        def _exec(cmd, **kwargs):
            if str(cmd).startswith('test -s'):
                return {'n0': {'exit_code': 0}, 'n1': {'exit_code': 0}}
            if str(cmd).startswith('bash ') and 'distributed_wrapper_script' in str(cmd):
                return {'n0': f'launch failed for token {launch_secret}'}
            return {'n0': '', 'n1': ''}

        orch.exec.side_effect = _exec
        variant = MagicMock()
        variant.job_config_dict.return_value = {
            'log_dir': '/test/logs',
            'scripts_dir': '/test/scripts',
            'data_cache_dir': '/test/cache',
            'hf_token_file': FAKE_CRED_FILE,
            'torchtitan_root': '/workspace/torchtitan',
            'training_iterations': '10',
            'nnodes': '2',
            'master_address': '127.0.0.1',
        }
        variant.train_params = {'model_name': 'llama3_1_8b'}
        variant.gpu_name = 'MI355X'
        job = TorchTitanTrainingJob(
            orch=orch,
            variant_config=variant,
            run_label='token-dist',
            distributed_training=True,
        )
        job.build_training_job_cmd()
        job.start_training_job()

        launch_calls = [
            call
            for call in orch.exec.call_args_list
            if str(call.args[0]).startswith('bash ') and 'distributed_wrapper_script' in str(call.args[0])
        ]
        self.assertTrue(launch_calls)
        for call in launch_calls:
            self.assertFalse(call.kwargs.get('print_console', True))
        logged = ' '.join(
            str(arg) for call in mock_log.info.call_args_list + mock_log.warning.call_args_list for arg in call.args
        )
        self.assertNotIn(launch_secret, logged)

    @patch('cvs.lib.training.torchtitan.torchtitan_lib.detect_rocm_path')
    def test_read_last_node_log_redacts_bare_token(self, mock_detect):
        """Callers store this text in reports; a failed node can leave a bare hub token on it."""
        mock_detect.return_value = '/opt/rocm'
        orch = MagicMock()
        orch.hosts = ['n0', 'n1']
        bare = 'hf_' + ('b' * 34)
        orch.exec.return_value = {'n1': f'step: 1 | loss: 0.5\nInvalid credentials for token {bare}'}
        variant = MagicMock()
        variant.job_config_dict.return_value = {
            'log_dir': '/test/logs',
            'scripts_dir': '/test/scripts',
            'data_cache_dir': '/test/cache',
            'hf_token_file': FAKE_CRED_FILE,
            'torchtitan_root': '/workspace/torchtitan',
            'training_iterations': '10',
        }
        variant.train_params = {'model_name': 'llama3_1_8b'}
        variant.gpu_name = 'MI355X'
        job = TorchTitanTrainingJob(orch=orch, variant_config=variant, run_label='tail', distributed_training=False)

        text = job._read_last_node_log()

        self.assertNotIn(bare, text)
        self.assertIn('step: 1 | loss: 0.5', text)
        self.assertFalse(orch.exec.call_args.kwargs.get('print_console', True))

    @patch('cvs.lib.training.torchtitan.torchtitan_lib.detect_rocm_path')
    def test_log_reads_do_not_print_raw_output(self, mock_detect):
        """get_training_results_dict, scan_for_training_errors, and poll_for_training_completion
        all cat/tail the training log, which can contain a bare hub token on failure."""
        mock_detect.return_value = '/opt/rocm'
        orch = MagicMock()
        orch.hosts = ['n0']
        orch.exec.return_value = {'n0': 'step: 1 | tps: 100.0 | loss: 0.5'}
        variant = MagicMock()
        variant.job_config_dict.return_value = {
            'log_dir': '/test/logs',
            'scripts_dir': '/test/scripts',
            'data_cache_dir': '/test/cache',
            'hf_token_file': FAKE_CRED_FILE,
            'torchtitan_root': '/workspace/torchtitan',
            'training_iterations': '1',
        }
        variant.train_params = {'model_name': 'llama3_1_8b'}
        variant.gpu_name = 'MI355X'
        job = TorchTitanTrainingJob(orch=orch, variant_config=variant, run_label='reads', distributed_training=False)

        job.get_training_results_dict()
        job.scan_for_training_errors()
        job.poll_for_training_completion(time_between_iters=0)

        cat_calls = [c for c in orch.exec.call_args_list if str(c.args[0]).startswith('cat ')]
        tail_calls = [c for c in orch.exec.call_args_list if str(c.args[0]).startswith('tail ')]
        self.assertTrue(cat_calls)
        self.assertTrue(tail_calls)
        for call in cat_calls + tail_calls:
            self.assertFalse(call.kwargs.get('print_console', True))
