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
            'cmd_list', host_chunks, shard_init_kwargs, cmd_list=['echo 1', 'echo 2']
        )

        expected = [
            {
                'operation': 'cmd_list',
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
            'operation': 'cmd_list',
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
            'operation': 'scp',
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
            'operation': 'reboot',
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


if __name__ == "__main__":
    unittest.main()
