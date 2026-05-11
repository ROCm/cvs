import unittest
from unittest.mock import patch, MagicMock
from cvs.lib.parallel.pssh_sharder import PsshSharder
from cvs.lib.parallel.config import ParallelConfig


class TestPsshSharder(unittest.TestCase):
    def setUp(self):
        self.config = ParallelConfig(hosts_per_shard=2, max_workers_per_cpu=4)
        self.sharder = PsshSharder(self.config)

    def test_chunk_hosts_even_division(self):
        """Test chunking hosts with even division."""
        hosts = ["host1", "host2", "host3", "host4"]
        chunks = list(self.sharder.chunk_hosts(hosts))
        expected = [["host1", "host2"], ["host3", "host4"]]
        self.assertEqual(chunks, expected)

    def test_chunk_hosts_uneven_division(self):
        """Test chunking hosts with uneven division."""
        hosts = ["host1", "host2", "host3", "host4", "host5"]
        chunks = list(self.sharder.chunk_hosts(hosts))
        expected = [["host1", "host2"], ["host3", "host4"], ["host5"]]
        self.assertEqual(chunks, expected)

    def test_chunk_hosts_single_chunk(self):
        """Test chunking where chunk size is larger than host list."""
        hosts = ["host1", "host2"]
        chunks = list(self.sharder.chunk_hosts(hosts))
        expected = [["host1", "host2"]]
        self.assertEqual(chunks, expected)

    def test_chunk_hosts_empty_list(self):
        """Test chunking empty host list."""
        hosts = []
        chunks = list(self.sharder.chunk_hosts(hosts))
        self.assertEqual(chunks, [])

    def test_chunk_hosts_single_host(self):
        """Test chunking single host."""
        hosts = ["host1"]
        chunks = list(self.sharder.chunk_hosts(hosts))
        expected = [["host1"]]
        self.assertEqual(chunks, expected)


class TestPsshSharderMethods(unittest.TestCase):
    def setUp(self):
        self.config = ParallelConfig(hosts_per_shard=2, max_workers_per_cpu=4)
        self.sharder = PsshSharder(self.config)

    def test_create_payloads_exec_mode(self):
        """Test creating payloads for exec mode."""
        host_chunks = [["host1", "host2"], ["host3", "host4"]]
        shard_init_kwargs = {'user': 'testuser', 'password': 'testpass'}

        payloads = self.sharder.create_payloads('exec', host_chunks, shard_init_kwargs, cmd='echo hello', timeout=30)

        expected = [
            {
                'operation': 'exec',
                'init': {'user': 'testuser', 'password': 'testpass', 'host_list': ['host1', 'host2']},
                'cmd': 'echo hello',
                'timeout': 30,
            },
            {
                'operation': 'exec',
                'init': {'user': 'testuser', 'password': 'testpass', 'host_list': ['host3', 'host4']},
                'cmd': 'echo hello',
                'timeout': 30,
            },
        ]
        self.assertEqual(payloads, expected)

    def test_create_payloads_cmd_list_mode(self):
        """Test creating payloads for cmd_list mode."""
        host_chunks = [["host1", "host2"]]
        shard_init_kwargs = {'user': 'testuser'}

        payloads = self.sharder.create_payloads(
            'exec_cmd_list', host_chunks, shard_init_kwargs, cmd_list=['echo 1', 'echo 2']
        )

        expected = [
            {
                'operation': 'exec_cmd_list',
                'init': {'user': 'testuser', 'host_list': ['host1', 'host2']},
                'cmd_list': ['echo 1', 'echo 2'],
            }
        ]
        self.assertEqual(payloads, expected)

    def test_create_payloads_empty_chunks(self):
        """Test creating payloads with empty chunks."""
        payloads = self.sharder.create_payloads('exec', [], {}, cmd='test')
        self.assertEqual(payloads, [])

    @patch('cvs.lib.parallel.pssh_sharder.ProcessPoolExecutor')
    def test_execute_sharded_empty_payloads(self, mock_executor):
        """Test executing with empty payloads."""
        result = self.sharder.execute_sharded([])
        self.assertEqual(result, [])
        mock_executor.assert_not_called()

    @patch('cvs.lib.parallel.pssh_sharder.ProcessPoolExecutor')
    def test_execute_sharded_with_payloads(self, mock_executor):
        """Test executing sharded operations."""
        # Mock executor and futures
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        mock_future1 = MagicMock()
        mock_future2 = MagicMock()
        mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]

        # Mock as_completed and results
        with patch('cvs.lib.parallel.pssh_sharder.as_completed') as mock_as_completed:
            mock_result1 = {'host1': 'result1', 'host2': 'result2'}
            mock_result2 = {'host3': 'result3'}

            mock_future1.result.return_value = mock_result1
            mock_future2.result.return_value = mock_result2
            mock_as_completed.return_value = [mock_future1, mock_future2]

            payloads = [
                {'operation': 'exec', 'init': {}, 'cmd': 'test1'},
                {'operation': 'exec', 'init': {}, 'cmd': 'test2'},
            ]

            result = self.sharder.execute_sharded(payloads)

            expected = [mock_result1, mock_result2]
            self.assertEqual(result, expected)

            # Verify executor was used correctly (max_workers = min(payloads, config.max_workers))
            # Verify executor was used correctly
            # Note: We can't easily test the exact ProcessPoolExecutor call due to mp_context object
            self.assertEqual(mock_executor_instance.submit.call_count, 2)

    def test_merge_results_complete_coverage(self):
        """Test merging results with complete host coverage."""
        shard_returns = [
            {'result': {'host1': 'result1', 'host2': 'result2'}},
            {'result': {'host3': 'result3', 'host4': 'result4'}},
        ]
        original_hosts = ['host1', 'host2', 'host3', 'host4']

        merged = self.sharder.merge_results(shard_returns, original_hosts)

        expected = {'host1': 'result1', 'host2': 'result2', 'host3': 'result3', 'host4': 'result4'}
        self.assertEqual(merged, expected)

    def test_merge_results_missing_hosts(self):
        """Test merging results with missing hosts."""
        shard_returns = [{'result': {'host1': 'result1'}}, {'result': {'host3': 'result3'}}]
        original_hosts = ['host1', 'host2', 'host3', 'host4']

        merged = self.sharder.merge_results(shard_returns, original_hosts)

        # Only hosts found in shard returns are included
        expected = {'host1': 'result1', 'host3': 'result3'}
        self.assertEqual(merged, expected)

    def test_merge_results_empty_returns(self):
        """Test merging with empty shard returns."""
        merged = self.sharder.merge_results([], ['host1', 'host2'])

        # With no shard returns, no hosts are found
        expected = {}
        self.assertEqual(merged, expected)

    def test_merge_results_with_none_results(self):
        """Test merging results when some shards return None (scp/reboot operations)."""
        shard_returns = [
            {'result': {'host1': 'exec_result', 'host2': 'exec_result'}},
            {'result': None},  # This happens for scp/reboot operations
        ]
        original_hosts = ['host1', 'host2', 'host3', 'host4']

        # This should not raise TypeError
        merged = self.sharder.merge_results(shard_returns, original_hosts)

        # Only hosts with actual results should be included
        expected = {'host1': 'exec_result', 'host2': 'exec_result'}
        self.assertEqual(merged, expected)

    def test_merge_results_all_none_results(self):
        """Test merging results when all shards return None (all scp/reboot operations)."""
        shard_returns = [
            {'result': None},  # scp operation
            {'result': None},  # reboot operation
        ]
        original_hosts = ['host1', 'host2', 'host3', 'host4']

        # This should not raise TypeError and should return empty dict
        merged = self.sharder.merge_results(shard_returns, original_hosts)

        expected = {}
        self.assertEqual(merged, expected)


class TestPsshShardWorker(unittest.TestCase):
    @patch('cvs.lib.parallel.pssh_sharder.Pssh')
    def test_pssh_shard_worker_exec_mode(self, mock_pssh_class):
        """Test shard worker in exec mode with direct operation calls."""
        mock_shard = MagicMock()
        mock_pssh_class.return_value = mock_shard
        mock_shard.exec.return_value = {'host1': 'output1', 'host2': 'output2'}
        mock_shard.reachable_hosts = ['host1', 'host2']
        mock_shard.unreachable_hosts = []

        payload = {
            'operation': 'exec',
            'init': {'log': None, 'host_list': ['host1', 'host2'], 'user': 'test'},
            'cmd': 'echo hello',
            'timeout': 30,
            'print_console': False,
        }

        result = PsshSharder.run_shard(payload)

        expected = {
            'result': {'host1': 'output1', 'host2': 'output2'},
            'reachable_hosts': ['host1', 'host2'],
            'unreachable_hosts': [],
        }
        self.assertEqual(result, expected)

        # Verify Pssh was created and exec was called directly
        mock_pssh_class.assert_called_once_with(
            log=None, host_list=['host1', 'host2'], user='test', process_output=False
        )
        mock_shard.exec.assert_called_once_with('echo hello', timeout=30, print_console=False)
        mock_shard.destroy_clients.assert_called_once()

    @patch('cvs.lib.parallel.pssh_sharder.Pssh')
    def test_pssh_shard_worker_cmd_list_mode(self, mock_pssh_class):
        """Test shard worker in cmd_list mode with direct operation calls."""
        mock_shard = MagicMock()
        mock_pssh_class.return_value = mock_shard
        mock_shard.exec_cmd_list.return_value = {'host1': 'cmd1_output'}
        mock_shard.reachable_hosts = ['host1']
        mock_shard.unreachable_hosts = []

        payload = {
            'operation': 'exec_cmd_list',
            'init': {'log': None, 'host_list': ['host1'], 'user': 'test'},
            'cmd_list': ['echo 1', 'echo 2'],
        }

        result = PsshSharder.run_shard(payload)

        expected = {'result': {'host1': 'cmd1_output'}, 'reachable_hosts': ['host1'], 'unreachable_hosts': []}
        self.assertEqual(result, expected)

        # Verify direct operation call
        mock_pssh_class.assert_called_once_with(log=None, host_list=['host1'], user='test', process_output=False)
        mock_shard.exec_cmd_list.assert_called_once_with(['echo 1', 'echo 2'], timeout=None, print_console=False)
        mock_shard.destroy_clients.assert_called_once()

    @patch('cvs.lib.parallel.pssh_sharder.Pssh')
    def test_pssh_shard_worker_scp_mode(self, mock_pssh_class):
        """Test shard worker in scp mode with direct operation calls."""
        mock_shard = MagicMock()
        mock_pssh_class.return_value = mock_shard
        mock_shard.reachable_hosts = ['host1']
        mock_shard.unreachable_hosts = []

        payload = {
            'operation': 'scp_file',
            'init': {'log': None, 'host_list': ['host1'], 'user': 'test'},
            'local_file': 'test.txt',
            'remote_file': '/tmp/test.txt',
            'recurse': False,
        }

        result = PsshSharder.run_shard(payload)

        expected = {'result': None, 'reachable_hosts': ['host1'], 'unreachable_hosts': []}
        self.assertEqual(result, expected)

        # Verify direct operation call
        mock_shard.scp_file.assert_called_once_with('test.txt', '/tmp/test.txt', recurse=False)

    @patch('cvs.lib.parallel.pssh_sharder.Pssh')
    def test_pssh_shard_worker_reboot_mode(self, mock_pssh_class):
        """Test shard worker in reboot mode with direct operation calls."""
        mock_shard = MagicMock()
        mock_pssh_class.return_value = mock_shard
        mock_shard.reachable_hosts = ['host1']
        mock_shard.unreachable_hosts = []

        payload = {
            'operation': 'reboot_connections',
            'init': {'log': None, 'host_list': ['host1'], 'user': 'test'},
        }

        result = PsshSharder.run_shard(payload)

        expected = {'result': None, 'reachable_hosts': ['host1'], 'unreachable_hosts': []}
        self.assertEqual(result, expected)

        # Verify direct operation call
        mock_shard.reboot_connections.assert_called_once_with()

    @patch('cvs.lib.parallel.pssh_sharder.Pssh')
    def test_pssh_shard_worker_unknown_operation(self, mock_pssh_class):
        """Test shard worker with unknown operation."""
        mock_shard = MagicMock()
        mock_pssh_class.return_value = mock_shard

        payload = {
            'operation': 'unknown_operation',
            'init': {'log': None, 'host_list': ['host1'], 'user': 'test'},
        }

        with self.assertRaises(ValueError) as cm:
            PsshSharder.run_shard(payload)

        self.assertIn('Unknown operation: unknown_operation', str(cm.exception))

    @patch('cvs.lib.parallel.pssh_sharder.Pssh')
    def test_pssh_shard_worker_registry_handler_mismatch(self, mock_pssh_class):
        """Test shard worker with operation in registry but missing handler (implementation error)."""
        mock_shard = MagicMock()
        mock_pssh_class.return_value = mock_shard

        # Temporarily add an operation to registry without corresponding handler
        from cvs.lib.parallel.pssh_sharder import SUPPORTED_OPERATIONS

        original_operations = SUPPORTED_OPERATIONS.copy()

        try:
            # Add a fake operation to registry
            SUPPORTED_OPERATIONS['fake_operation'] = {
                'required_params': [],
                'optional_params': [],
                'returns_result': False,
            }

            payload = {
                'operation': 'fake_operation',
                'init': {'log': None, 'host_list': ['host1'], 'user': 'test'},
            }

            with self.assertRaises(RuntimeError) as cm:
                PsshSharder.run_shard(payload)

            # Verify it's identified as a implementation error
            self.assertIn('Implementation bug', str(cm.exception))
            self.assertIn('fake_operation', str(cm.exception))
            self.assertIn('supported (in registry) but handler is missing', str(cm.exception))

        finally:
            # Restore original registry
            SUPPORTED_OPERATIONS.clear()
            SUPPORTED_OPERATIONS.update(original_operations)


class TestPsshSharderApiCompliance(unittest.TestCase):
    """
    Test suite to ensure PsshSharder supports all operations and parameters
    required by MultiProcessPssh, preventing API drift issues.
    """

    def test_sharder_supports_all_multiprocess_operations(self):
        """
        Verify that pssh_sharder supports all operations used by MultiProcessPssh.

        Uses the operation registry as the single source of truth instead of
        fragile regex parsing.
        """
        import re
        import inspect
        from cvs.lib.parallel.pssh_sharder import SUPPORTED_OPERATIONS

        # Get MultiProcessPssh source code to extract sharder operations
        from cvs.lib.parallel.multiprocess_pssh import MultiProcessPssh

        multiprocess_source = inspect.getsource(MultiProcessPssh)

        # Extract operation names from create_payloads calls
        operation_pattern = r"create_payloads\s*\(\s*['\"](\w+)['\"]"
        used_operations = set(re.findall(operation_pattern, multiprocess_source))

        # Get supported operations from registry (single source of truth)
        supported_operations = set(SUPPORTED_OPERATIONS.keys())

        # Find missing operations
        missing_operations = used_operations - supported_operations

        if missing_operations:
            error_msg = (
                f"PsshSharder missing support for {len(missing_operations)} operation(s) "
                f"used by MultiProcessPssh:\n"
                f"Missing operations: {sorted(missing_operations)}\n"
                f"Used by MultiProcessPssh: {sorted(used_operations)}\n"
                f"Supported by PsshSharder: {sorted(supported_operations)}\n\n"
                f"Add missing operations to SUPPORTED_OPERATIONS registry in pssh_sharder.py"
            )
            self.fail(error_msg)

        print(f"✓ Sharder Compatibility Check Passed: All {len(used_operations)} operations supported")

    def test_sharder_operations_can_execute_without_errors(self):
        """
        Test that all sharder operations can execute without parameter/method errors.

        Uses the operation registry to dynamically generate test payloads,
        ensuring all operations are tested consistently.
        """
        from cvs.lib.parallel.pssh_sharder import SUPPORTED_OPERATIONS

        config = ParallelConfig(hosts_per_shard=1, max_workers_per_cpu=1)
        sharder = PsshSharder(config)

        # Dynamically generate test payloads from operation registry
        test_payloads = {}

        for operation_name, op_config in SUPPORTED_OPERATIONS.items():
            payload = {
                'operation': operation_name,
                'init': {'log': None, 'host_list': ['testhost'], 'user': 'test'},
            }

            # Add required parameters with test values
            param_values = {
                'cmd': 'echo test',
                'cmd_list': ['echo test'],
                'timeout': 30,
                'detailed': False,
                'local_file': '/tmp/test.txt',
                'remote_file': '/remote/test.txt',
                'recurse': False,
                'suffix_separator': '_',
            }

            for param in op_config['required_params']:
                if param in param_values:
                    payload[param] = param_values[param]

            # Add some optional parameters for testing
            for param in op_config['optional_params']:
                if param in param_values:
                    payload[param] = param_values[param]

            test_payloads[operation_name] = payload

        # Mock all Pssh methods to avoid actual SSH calls
        with patch('cvs.lib.parallel.pssh_sharder.Pssh') as mock_pssh_class:
            mock_shard = MagicMock()
            mock_pssh_class.return_value = mock_shard

            # Configure return values
            mock_shard.exec.return_value = {'testhost': 'output'}
            mock_shard.exec_cmd_list.return_value = {'testhost': 'output'}
            mock_shard.download_file.return_value = {'testhost': '/tmp/test.txt_testhost'}
            mock_shard.reachable_hosts = ['testhost']
            mock_shard.unreachable_hosts = []

            failed_operations = []

            for operation_name, payload in test_payloads.items():
                try:
                    result = sharder.run_shard(payload)

                    # Verify result structure
                    self.assertIn('result', result)
                    self.assertIn('reachable_hosts', result)
                    self.assertIn('unreachable_hosts', result)

                    # Verify correct method was called on mock shard
                    if operation_name == 'exec':
                        mock_shard.exec.assert_called()
                    elif operation_name == 'cmd_list':
                        mock_shard.exec_cmd_list.assert_called()
                    elif operation_name == 'scp':
                        mock_shard.scp_file.assert_called()
                    elif operation_name == 'upload':
                        mock_shard.upload_file.assert_called()
                    elif operation_name == 'download':
                        mock_shard.download_file.assert_called()
                    elif operation_name == 'reboot':
                        mock_shard.reboot_connections.assert_called()

                except Exception as e:
                    failed_operations.append(
                        {'operation': operation_name, 'error': str(e), 'error_type': type(e).__name__}
                    )
                finally:
                    # Reset mocks for next iteration
                    mock_shard.reset_mock()

            if failed_operations:
                error_msg = "Sharder operations failed with runtime errors:\n"
                for failure in failed_operations:
                    error_msg += f"  {failure['operation']}: {failure['error_type']}: {failure['error']}\n"
                error_msg += "\nThis suggests missing parameters, wrong method names, or incorrect implementation."
                self.fail(error_msg)

            print(f"✓ Runtime Compatibility Check Passed: All {len(test_payloads)} operations execute correctly")

    def test_operation_registry_completeness(self):
        """
        Test that the operation registry is complete and consistent.

        Validates that all operations have proper configuration and that
        the referenced Pssh methods actually exist.
        """
        from cvs.lib.parallel.pssh_sharder import SUPPORTED_OPERATIONS
        from cvs.lib.parallel.pssh import Pssh
        import inspect

        errors = []

        for operation_name, op_config in SUPPORTED_OPERATIONS.items():
            # Check required fields
            required_fields = ['required_params', 'optional_params', 'returns_result']
            for field in required_fields:
                if field not in op_config:
                    errors.append(f"Operation '{operation_name}' missing required field '{field}'")

            # Check that operation name corresponds to an actual method in Pssh class
            if not hasattr(Pssh, operation_name):
                errors.append(f"Operation '{operation_name}' does not correspond to a method in Pssh class")

            # Check method signature compatibility
            if hasattr(Pssh, operation_name):
                try:
                    method = getattr(Pssh, operation_name)
                    sig = inspect.signature(method)
                    method_params = set(sig.parameters.keys()) - {'self'}  # Exclude 'self'

                    config_params = set(op_config['required_params'] + op_config['optional_params'])

                    # Check if all config parameters exist in method signature
                    missing_in_method = config_params - method_params
                    if missing_in_method:
                        errors.append(
                            f"Operation '{operation_name}': method missing parameters {sorted(missing_in_method)}"
                        )

                except Exception as e:
                    errors.append(f"Operation '{operation_name}': Could not validate method signature: {e}")

        if errors:
            error_msg = "Operation registry validation failed:\n"
            for error in errors:
                error_msg += f"  - {error}\n"
            error_msg += "\nFix the SUPPORTED_OPERATIONS registry in pssh_sharder.py"
            self.fail(error_msg)

        print(f"✓ Operation Registry Validation Passed: All {len(SUPPORTED_OPERATIONS)} operations properly configured")

    def test_registry_handler_consistency(self):
        """
        Test that all operations in registry have corresponding handlers in run_shard.

        This prevents the implementation error scenario where operations are added to
        the registry but corresponding handler code is forgotten.
        """
        from cvs.lib.parallel.pssh_sharder import SUPPORTED_OPERATIONS
        import inspect
        import re

        # Get the run_shard method source to extract handled operations
        from cvs.lib.parallel.pssh_sharder import PsshSharder

        run_shard_source = inspect.getsource(PsshSharder.run_shard)

        # Extract operation names from if/elif operation == 'name' and operation in ['name1', 'name2'] patterns
        # Pattern 1: operation == 'name'
        equals_pattern = r"operation == ['\"](\w+)['\"]"
        equals_operations = set(re.findall(equals_pattern, run_shard_source))

        # Pattern 2: operation in ['name1', 'name2', ...]
        in_pattern = r"operation in \[(.*?)\]"
        in_matches = re.findall(in_pattern, run_shard_source)
        in_operations = set()
        for match in in_matches:
            # Extract individual operation names from the list
            names = re.findall(r"['\"](\w+)['\"]", match)
            in_operations.update(names)

        handled_operations = equals_operations | in_operations

        # Operations in registry but not in handlers (implementation error)
        registry_operations = set(SUPPORTED_OPERATIONS.keys())
        missing_handlers = registry_operations - handled_operations

        # Operations in handlers but not in registry (less critical, but good to know)
        extra_handlers = handled_operations - registry_operations

        errors = []
        if missing_handlers:
            errors.append(f"Operations in registry but missing handlers: {sorted(missing_handlers)}")

        if extra_handlers:
            errors.append(f"Operations with handlers but not in registry: {sorted(extra_handlers)}")

        if errors:
            error_msg = "Registry-Handler consistency check failed:\n"
            for error in errors:
                error_msg += f"  - {error}\n"
            error_msg += "\nEnsure registry and handler code are synchronized."
            self.fail(error_msg)

        print(f"✓ Registry-Handler Consistency Check Passed: All {len(registry_operations)} operations have handlers")


if __name__ == "__main__":
    unittest.main()
