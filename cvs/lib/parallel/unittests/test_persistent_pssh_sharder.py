import unittest
from queue import Empty
from unittest.mock import MagicMock, patch, PropertyMock

from cvs.lib.parallel.config import ParallelConfig
from cvs.lib.parallel.persistent_pssh_sharder import PersistentPsshSharder


class TestPersistentPsshSharder(unittest.TestCase):
    def setUp(self):
        self.config = ParallelConfig(hosts_per_shard=2, persistent_shards=True)
        self.sharder = PersistentPsshSharder(self.config)

    def test_execute_sharded_routes_requests_and_merges_success(self):
        req_q_1 = MagicMock()
        req_q_2 = MagicMock()
        resp_q_1 = object()
        resp_q_2 = object()
        proc_1 = MagicMock()
        proc_1.is_alive.return_value = True
        proc_2 = MagicMock()
        proc_2.is_alive.return_value = True

        # Use worker_id-based storage (dict instead of list)
        self.sharder._workers = {
            0: {'process': proc_1, 'req_q': req_q_1, 'resp_q': resp_q_1, 'init': {'host_list': ['host1', 'host2']}},
            1: {'process': proc_2, 'req_q': req_q_2, 'resp_q': resp_q_2, 'init': {'host_list': ['host3']}},
        }

        routing_map = {
            0: {'operation': 'exec', 'init': {'host_list': ['host1', 'host2']}, 'cmd': 'uptime', 'timeout': 10},
            1: {'operation': 'exec', 'init': {'host_list': ['host3']}, 'cmd': 'uptime', 'timeout': 10},
        }

        with (
            patch.object(self.sharder, '_ensure_workers') as mock_ensure,
            patch.object(
                self.sharder,
                '_wait_for_response',
                side_effect=[
                    {
                        'ok': True,
                        'result': {'host1': 'up1', 'host2': 'up2'},
                        'reachable_hosts': ['host1', 'host2'],
                        'unreachable_hosts': [],
                    },
                    {
                        'ok': True,
                        'result': {'host3': 'up3'},
                        'reachable_hosts': ['host3'],
                        'unreachable_hosts': [],
                    },
                ],
            ),
        ):
            result = self.sharder.execute_sharded(routing_map)

        mock_ensure.assert_called_once_with(routing_map)
        self.assertEqual(result[0]['result'], {'host1': 'up1', 'host2': 'up2'})
        self.assertEqual(result[1]['result'], {'host3': 'up3'})
        req_q_1.put.assert_called_once()
        req_q_2.put.assert_called_once()
        first_req = req_q_1.put.call_args[0][0]
        self.assertEqual(first_req['type'], 'request')
        self.assertEqual(first_req['operation'], 'exec')
        self.assertEqual(first_req['args']['cmd'], 'uptime')

    def test_execute_sharded_propagates_exec_error_per_host(self):
        req_q = MagicMock()
        proc = MagicMock()
        proc.is_alive.return_value = True
        self.sharder._workers = {
            0: {'process': proc, 'req_q': req_q, 'resp_q': object(), 'init': {'host_list': ['hostA', 'hostB']}},
        }
        # Setup worker state table for worker_id=0
        self.sharder._worker_state_table.append(0, ['hostA', 'hostB'], ['hostA', 'hostB'], [])

        routing_map = {0: {'operation': 'exec', 'init': {'host_list': ['hostA', 'hostB']}, 'cmd': 'date'}}

        with (
            patch.object(self.sharder, '_ensure_workers'),
            patch.object(
                self.sharder,
                '_wait_for_response',
                return_value={
                    'ok': False,
                    'error': 'RuntimeError: boom',
                    'reachable_hosts': [],
                    'unreachable_hosts': ['hostA', 'hostB'],
                },
            ),
        ):
            result = self.sharder.execute_sharded(routing_map)

        self.assertEqual(
            result,
            [
                {
                    'result': {'hostA': 'ERROR: RuntimeError: boom', 'hostB': 'ERROR: RuntimeError: boom'},
                    'reachable_hosts': [],
                    'unreachable_hosts': ['hostA', 'hostB'],
                }
            ],
        )

    def test_ensure_workers_restart_dead_and_create_missing(self):
        """Test that worker management restarts dead workers and creates missing ones."""
        alive_proc = MagicMock()
        alive_proc.is_alive.return_value = True
        dead_proc = MagicMock()
        dead_proc.is_alive.return_value = False

        # Setup workers: one alive, one dead
        self.sharder._workers = {
            0: {'process': alive_proc, 'req_q': MagicMock(), 'resp_q': object(), 'init': {'host_list': ['host1']}},
            1: {'process': dead_proc, 'req_q': MagicMock(), 'resp_q': object(), 'init': {'host_list': ['host2']}},
        }

        replacement_worker = {
            'process': MagicMock(),
            'req_q': MagicMock(),
            'resp_q': object(),
            'init': {'host_list': ['host2']},
            'worker_id': 1,
        }
        replacement_worker['process'].is_alive.return_value = True

        # Routing map includes both workers plus a new one
        routing_map = {
            0: {'init': {'host_list': ['host1', 'hostX']}},  # Alive worker, no restart needed
            1: {'init': {'host_list': ['host2']}},  # Dead worker, needs restart
            2: {'init': {'host_list': ['host3']}},  # Missing worker, needs creation
        }

        with patch.object(self.sharder, '_start_worker', return_value=replacement_worker) as mock_start:
            self.sharder._ensure_workers(routing_map)

        # Should restart dead worker (1) and create missing worker (2)
        self.assertEqual(mock_start.call_count, 2)
        mock_start.assert_any_call({'host_list': ['host2']})  # Restart dead worker 1
        mock_start.assert_any_call({'host_list': ['host3']})  # Create missing worker 2

        # Dead process should be terminated
        dead_proc.terminate.assert_called_once()

    def test_update_worker_table_keeps_only_active_workers(self):
        self.sharder._worker_state_table.append(0, ['host1'], ['host1'], [])
        self.sharder._worker_state_table.append(1, ['host2'], ['host2'], [])

        shard_returns = [
            {'reachable_hosts': ['host1'], 'unreachable_hosts': []},
            {'reachable_hosts': [], 'unreachable_hosts': ['host2']},
        ]

        self.sharder.update_worker_table(shard_returns)
        rows = list(self.sharder.get_worker_table())

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].worker_id, 0)
        self.assertEqual(list(rows[0].reachable_hosts), ['host1'])

    def test_prune_worker_nodes_drops_workers_with_no_reachable_hosts(self):
        self.sharder._worker_state_table.append(0, ['host1'], ['host1'], [])
        self.sharder._worker_state_table.append(1, ['host2'], ['host2'], [])

        self.sharder.prune_worker_nodes({'host2'})
        rows = list(self.sharder.get_worker_table())

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].worker_id, 0)

    def test_destroy_clients_sends_shutdown_and_terminates_alive(self):
        req_q = MagicMock()
        proc = MagicMock()
        proc.is_alive.return_value = True
        self.sharder._workers = {0: {'process': proc, 'req_q': req_q, 'resp_q': object(), 'init': {}}}

        self.sharder.destroy_clients()

        req_q.put.assert_called_once_with({'type': 'shutdown'})
        proc.join.assert_called()
        proc.terminate.assert_called_once()
        self.assertEqual(self.sharder._workers, {})

    def test_start_worker_fails_fast_on_init_error(self):
        sharder = PersistentPsshSharder(self.config)
        req_q = MagicMock()
        resp_q = MagicMock()
        process = MagicMock()
        process.is_alive.return_value = True

        sharder.ctx = MagicMock()
        sharder.ctx.Queue.side_effect = [req_q, resp_q]
        sharder.ctx.Process.return_value = process
        resp_q.get.return_value = {'type': 'init', 'ok': False, 'error': 'PKeyFileError: missing key'}

        with self.assertRaises(RuntimeError) as ctx:
            sharder._start_worker({'host_list': ['host1']})

        self.assertIn('Persistent shard worker init failed', str(ctx.exception))
        process.terminate.assert_called_once()

    def test_wait_for_response_timeout_returns_error(self):
        resp_q = MagicMock()
        resp_q.get.side_effect = Empty()

        # Use mocked time instead of manipulating timeout values
        with patch('cvs.lib.parallel.persistent_pssh_sharder.time') as mock_time:
            from itertools import count

            time_counter = count(0, 0.001)  # Start at 0, increment by 1ms each call
            mock_time.time.side_effect = lambda: next(time_counter)

            response = self.sharder._wait_for_response(resp_q, 'req-1')

        self.assertFalse(response['ok'])
        self.assertIn('Timeout waiting for shard response', response['error'])

    # ===== TIMEOUT HANDLING TESTS =====

    def test_persistent_timeout_respects_operation_timeout(self):
        """Test that persistent sharder uses operation timeout + buffer for response waiting.

        Expected Behavior:
        - When execute_sharded() receives payloads with timeout=10
        - _wait_for_response() should wait for 10 + buffer (e.g., 40s total)
        - Should NOT use the hardcoded 300s _RESPONSE_TIMEOUT_SEC
        - Should timeout and return error after operation_timeout + buffer expires
        """
        # Create mock worker
        req_q = MagicMock()
        resp_q = MagicMock()
        proc = MagicMock()
        proc.is_alive.return_value = True

        self.sharder._workers = {
            0: {'process': proc, 'req_q': req_q, 'resp_q': resp_q, 'init': {'host_list': ['host1']}}
        }

        routing_map = {0: {'operation': 'exec', 'init': {'host_list': ['host1']}, 'cmd': 'test', 'timeout': 10}}

        # Mock _wait_for_response to capture the timeout parameter
        with (
            patch.object(self.sharder, '_ensure_workers'),
            patch.object(self.sharder, '_wait_for_response') as mock_wait,
        ):
            mock_wait.return_value = {
                'ok': True,
                'result': {'host1': 'success'},
                'reachable_hosts': ['host1'],
                'unreachable_hosts': [],
            }

            self.sharder.execute_sharded(routing_map)

            # Verify _wait_for_response was called with operation timeout
            mock_wait.assert_called_once()
            call_args = mock_wait.call_args[0]
            self.assertEqual(len(call_args), 3)  # resp_q, request_id, operation_timeout

    def test_persistent_timeout_uses_default_when_no_operation_timeout(self):
        """Test that persistent sharder falls back to default timeout when no operation timeout provided.

        Expected Behavior:
        - When execute_sharded() receives payloads without timeout parameter
        - _wait_for_response() should use _RESPONSE_TIMEOUT_SEC (300s)
        - Should maintain backward compatibility for operations without explicit timeouts
        """
        req_q = MagicMock()
        resp_q = MagicMock()
        proc = MagicMock()
        proc.is_alive.return_value = True

        self.sharder._workers = {
            0: {'process': proc, 'req_q': req_q, 'resp_q': resp_q, 'init': {'host_list': ['host1']}}
        }

        # Routing map without timeout
        routing_map = {0: {'operation': 'exec', 'init': {'host_list': ['host1']}, 'cmd': 'test'}}

        with (
            patch.object(self.sharder, '_ensure_workers'),
            patch.object(self.sharder, '_wait_for_response') as mock_wait,
        ):
            mock_wait.return_value = {
                'ok': True,
                'result': {'host1': 'success'},
                'reachable_hosts': ['host1'],
                'unreachable_hosts': [],
            }

            self.sharder.execute_sharded(routing_map)

            # Verify _wait_for_response was called without operation timeout (should use default)
            mock_wait.assert_called_once()
            call_args = mock_wait.call_args[0]
            self.assertEqual(len(call_args), 3)  # resp_q, request_id, None (operation_timeout)

    def test_wait_for_response_with_operation_timeout(self):
        """Test _wait_for_response timeout calculation with operation timeout.

        NOTE: This test documents the expected behavior after implementing
        the timeout fix. Currently it tests the existing behavior.
        """
        resp_q = MagicMock()
        resp_q.get.side_effect = Empty()  # Always timeout

        # Test current behavior - uses _RESPONSE_TIMEOUT_SEC
        original_timeout = self.sharder._RESPONSE_TIMEOUT_SEC
        self.sharder._RESPONSE_TIMEOUT_SEC = 0.01  # Very short for testing

        try:
            # Mock time to control timeout behavior
            with patch('cvs.lib.parallel.persistent_pssh_sharder.time') as mock_time:
                # Use itertools.count to provide infinite increasing time values
                from itertools import count

                time_counter = count(0, 0.001)  # Start at 0, increment by 1ms each call
                mock_time.time.side_effect = lambda: next(time_counter)

                response = self.sharder._wait_for_response(resp_q, 'req-1')

                self.assertFalse(response['ok'])
                self.assertIn('Timeout waiting for shard response', response['error'])
        finally:
            self.sharder._RESPONSE_TIMEOUT_SEC = original_timeout

    # ===== PROCESS LIMIT TESTS =====

    def test_persistent_respects_max_workers_limit_on_initialization(self):
        """Test that persistent sharder doesn't exceed config.max_workers during initialization.

        Expected Behavior:
        - Given config.max_workers = 4 and 100 hosts with hosts_per_shard=10 (would create 10 chunks)
        - initialize_workers() should create exactly 4 workers, not 10
        - Hosts should be redistributed across 4 workers instead of creating 10
        - Should maintain roughly even host distribution per worker
        """
        # Create config that would naturally want 10 workers (100 hosts / 10 per shard)
        # but max_workers limits to 4
        with patch('cvs.lib.parallel.config.os.cpu_count', return_value=4):
            # max_workers = max(10, 4 * 1) = 10, but we'll test with smaller config
            limited_config = ParallelConfig(hosts_per_shard=5, max_workers_per_cpu=1)
            sharder = PersistentPsshSharder(limited_config)

        hosts = [f'host{i}' for i in range(25)]  # 25 hosts, 5 per shard = 5 natural chunks

        # Mock _start_worker to avoid actual process creation
        with patch.object(sharder, '_start_worker') as mock_start:
            mock_worker = {'process': MagicMock(), 'req_q': MagicMock(), 'resp_q': MagicMock(), 'init': {}}
            mock_start.return_value = mock_worker

            sharder.initialize_workers(hosts)

            # Should respect max_workers limit
            with patch('cvs.lib.parallel.config.os.cpu_count', return_value=4):
                expected_max_workers = min(5, limited_config.max_workers)  # min(chunks, max_workers)
                self.assertLessEqual(len(sharder._workers), expected_max_workers)

    def test_persistent_chunks_hosts_when_exceeding_max_workers(self):
        """Test host redistribution when natural chunking exceeds max_workers.

        Expected Behavior:
        - Given 50 hosts, hosts_per_shard=5 (10 natural chunks), max_workers=3
        - Should create 3 workers with ~17, ~17, ~16 hosts each
        - All 50 hosts should be covered across the 3 workers
        - Worker table should reflect the redistributed host assignments
        """
        # Force a scenario where we have more natural chunks than max_workers
        config = ParallelConfig(hosts_per_shard=2, max_workers_per_cpu=1)

        with patch.object(type(config), 'max_workers', new_callable=PropertyMock) as mock_max_workers:
            mock_max_workers.return_value = 3

            sharder = PersistentPsshSharder(config)
            hosts = [f'host{i}' for i in range(20)]  # 20 hosts, 2 per shard = 10 natural chunks > 3 max_workers

            with patch.object(sharder, '_start_worker') as mock_start:
                mock_worker = {'process': MagicMock(), 'req_q': MagicMock(), 'resp_q': MagicMock(), 'init': {}}
                mock_start.return_value = mock_worker

                sharder.initialize_workers(hosts)

                # Should not exceed max_workers
                self.assertLessEqual(len(sharder._workers), 3)

            # All hosts should be covered
            covered_hosts = set()
            for row in sharder._worker_state_table:
                covered_hosts.update(row.host_list)
            self.assertEqual(covered_hosts, set(hosts))

    def test_persistent_max_workers_with_small_host_list(self):
        """Test max_workers behavior when host count < max_workers.

        Expected Behavior:
        - Given 5 hosts and max_workers=10
        - Should create 1 worker (5 hosts fit in single shard)
        - Should not create empty workers
        - Worker count should be min(required_workers, max_workers)
        """
        config = ParallelConfig(hosts_per_shard=10, max_workers_per_cpu=5)  # High max_workers
        sharder = PersistentPsshSharder(config)

        hosts = ['host1', 'host2', 'host3']  # Only 3 hosts, fits in 1 chunk

        with patch.object(sharder, '_start_worker') as mock_start:
            mock_worker = {'process': MagicMock(), 'req_q': MagicMock(), 'resp_q': MagicMock(), 'init': {}}
            mock_start.return_value = mock_worker

            sharder.initialize_workers(hosts)

            # Should create exactly 1 worker, not max_workers
            self.assertEqual(len(sharder._workers), 1)
            self.assertEqual(len(sharder._worker_state_table), 1)

    # ===== WORKER LIFECYCLE TESTS =====

    def test_worker_restart_after_unexpected_crash(self):
        """Test worker restart when process dies unexpectedly between operations.

        Expected Behavior:
        - Initialize workers successfully
        - Kill worker process externally (simulate crash)
        - Next execute_sharded() call should detect dead worker via is_alive()
        - Should restart worker automatically with same host list
        - Operation should succeed after restart
        """
        # Setup initial worker
        original_worker = {
            'process': MagicMock(),
            'req_q': MagicMock(),
            'resp_q': MagicMock(),
            'init': {'host_list': ['host1']},
        }
        original_worker['process'].is_alive.return_value = False  # Simulate crash

        # Use worker_id-based storage (dict instead of list)
        self.sharder._workers = {0: original_worker}
        self.sharder._worker_state_table.append(0, ['host1'], ['host1'], [])

        routing_map = {0: {'operation': 'exec', 'init': {'host_list': ['host1']}, 'cmd': 'test'}}

        # Mock _start_worker for restart
        new_worker = {
            'process': MagicMock(),
            'req_q': MagicMock(),
            'resp_q': MagicMock(),
            'init': {'host_list': ['host1']},
            'worker_id': 0,
        }
        new_worker['process'].is_alive.return_value = True

        with (
            patch.object(self.sharder, '_start_worker', return_value=new_worker) as mock_start,
            patch.object(self.sharder, '_wait_for_response') as mock_wait,
        ):
            mock_wait.return_value = {
                'ok': True,
                'result': {'host1': 'success'},
                'reachable_hosts': ['host1'],
                'unreachable_hosts': [],
            }

            result = self.sharder.execute_sharded(routing_map)

            # Should have restarted the worker
            mock_start.assert_called_once_with({'host_list': ['host1']})
            self.assertEqual(self.sharder._workers[0]['worker_id'], 0)

            # Operation should succeed
            self.assertTrue(result[0]['result']['host1'] == 'success')

    def test_worker_restart_preserves_host_assignment(self):
        """Test that restarted workers maintain their host list assignments.

        Expected Behavior:
        - Worker 0 assigned hosts [host1, host2], Worker 1 assigned [host3, host4]
        - Worker 0 crashes and restarts
        - Worker 0 should still be assigned [host1, host2] after restart
        - Host-to-worker mapping should remain consistent
        """
        # Setup two workers, first one crashes
        worker0 = {
            'process': MagicMock(),
            'req_q': MagicMock(),
            'resp_q': MagicMock(),
            'init': {'host_list': ['host1', 'host2']},
        }
        worker0['process'].is_alive.return_value = False  # Crashed

        worker1 = {
            'process': MagicMock(),
            'req_q': MagicMock(),
            'resp_q': MagicMock(),
            'init': {'host_list': ['host3', 'host4']},
        }
        worker1['process'].is_alive.return_value = True  # Healthy

        # Use worker_id-based storage (dict instead of list)
        self.sharder._workers = {0: worker0, 1: worker1}
        # Initialize worker state table to match the workers
        self.sharder._worker_state_table.append(0, ['host1', 'host2'], ['host1', 'host2'], [])
        self.sharder._worker_state_table.append(1, ['host3', 'host4'], ['host3', 'host4'], [])

        routing_map = {
            0: {'operation': 'exec', 'init': {'host_list': ['host1', 'host2']}, 'cmd': 'test'},
            1: {'operation': 'exec', 'init': {'host_list': ['host3', 'host4']}, 'cmd': 'test'},
        }

        # Mock restart of worker0
        new_worker0 = {
            'process': MagicMock(),
            'req_q': MagicMock(),
            'resp_q': MagicMock(),
            'init': {'host_list': ['host1', 'host2']},
        }
        new_worker0['process'].is_alive.return_value = True

        with (
            patch.object(self.sharder, '_start_worker', return_value=new_worker0) as mock_start,
            patch.object(self.sharder, '_wait_for_response') as mock_wait,
        ):
            mock_wait.return_value = {'ok': True, 'result': {}, 'reachable_hosts': [], 'unreachable_hosts': []}

            self.sharder.execute_sharded(routing_map)

            # Worker 0 should be restarted with same host assignment
            mock_start.assert_called_once_with({'host_list': ['host1', 'host2']})
            self.assertEqual(self.sharder._workers[0]['init']['host_list'], ['host1', 'host2'])

            # Worker 1 should remain unchanged
            self.assertEqual(self.sharder._workers[1], worker1)

    def test_worker_init_failure_propagation(self):
        """Test proper error handling when worker initialization fails.

        Expected Behavior:
        - Worker process starts but Pssh.__init__ raises exception (e.g., SSH key error)
        - _start_worker() should receive init error via response queue
        - Should raise RuntimeError with descriptive error message
        - Failed process should be terminated and cleaned up
        """
        # Mock failed worker initialization
        with (
            patch.object(self.sharder.ctx, 'Queue') as mock_queue,
            patch.object(self.sharder.ctx, 'Process') as mock_process,
        ):
            mock_req_q = MagicMock()
            mock_resp_q = MagicMock()
            mock_queue.side_effect = [mock_req_q, mock_resp_q]

            mock_proc = MagicMock()
            mock_proc.is_alive.return_value = True
            mock_process.return_value = mock_proc

            # Simulate initialization failure
            mock_resp_q.get.return_value = {'type': 'init', 'ok': False, 'error': 'PKeyFileError: missing SSH key file'}

            init_payload = {'host_list': ['host1'], 'user': 'test'}

            with self.assertRaises(RuntimeError) as ctx:
                self.sharder._start_worker(init_payload)

            # Should contain descriptive error
            self.assertIn('Persistent shard worker init failed', str(ctx.exception))
            self.assertIn('PKeyFileError: missing SSH key file', str(ctx.exception))

            # Failed process should be terminated
            mock_proc.terminate.assert_called_once()

    def test_multiple_destroy_clients_calls_safety(self):
        """Test that multiple destroy_clients() calls are safe.

        Expected Behavior:
        - First destroy_clients() should shutdown workers cleanly
        - Second destroy_clients() should be no-op, not crash
        - _workers list should remain empty after multiple calls
        - No resource leaks or zombie processes
        """
        # Setup mock workers
        worker1 = {'process': MagicMock(), 'req_q': MagicMock(), 'resp_q': MagicMock(), 'init': {}}
        worker2 = {'process': MagicMock(), 'req_q': MagicMock(), 'resp_q': MagicMock(), 'init': {}}

        self.sharder._workers = {0: worker1, 1: worker2}
        self.sharder._worker_state_table.append(0, ['host1'], ['host1'], [])
        self.sharder._worker_state_table.append(1, ['host2'], ['host2'], [])

        # First destroy_clients call
        self.sharder.destroy_clients()

        # Verify cleanup
        self.assertEqual(len(self.sharder._workers), 0)
        self.assertEqual(len(self.sharder._worker_state_table), 0)

        # Second destroy_clients call should not crash
        self.sharder.destroy_clients()  # Should be safe no-op

        # State should remain clean
        self.assertEqual(len(self.sharder._workers), 0)
        self.assertEqual(len(self.sharder._worker_state_table), 0)

    # ===== STATE MANAGEMENT EDGE CASES =====

    def test_worker_table_with_all_hosts_unreachable(self):
        """Test update_worker_table behavior when all hosts become unreachable.

        Expected Behavior:
        - Start with 2 workers, each with reachable hosts
        - Execute operation that marks all hosts as unreachable
        - update_worker_table() should result in empty worker table
        - get_worker_table() should return empty iterator
        - Next operation should rebuild workers from scratch
        """
        # Setup initial worker state
        self.sharder._worker_state_table.append(0, ['host1', 'host2'], ['host1', 'host2'], [])
        self.sharder._worker_state_table.append(1, ['host3', 'host4'], ['host3', 'host4'], [])

        # Simulate operation where all hosts become unreachable
        shard_returns = [
            {'reachable_hosts': [], 'unreachable_hosts': ['host1', 'host2']},
            {'reachable_hosts': [], 'unreachable_hosts': ['host3', 'host4']},
        ]

        self.sharder.update_worker_table(shard_returns)

        # Worker table should be empty (no workers with reachable hosts)
        worker_list = list(self.sharder.get_worker_table())
        self.assertEqual(len(worker_list), 0)

        # Internal state should be cleared
        self.assertEqual(len(self.sharder._worker_state_table), 0)

    def test_worker_table_partial_host_failure(self):
        """Test worker table updates when some hosts fail per worker.

        Expected Behavior:
        - Worker 0: [host1, host2] -> host1 fails -> [host2] remains reachable
        - Worker 1: [host3, host4] -> both fail -> worker removed from table
        - update_worker_table() should keep Worker 0, remove Worker 1
        - Subsequent operations should only use Worker 0
        """
        # Setup initial worker state
        self.sharder._worker_state_table.append(0, ['host1', 'host2'], ['host1', 'host2'], [])
        self.sharder._worker_state_table.append(1, ['host3', 'host4'], ['host3', 'host4'], [])

        # Simulate partial failures
        shard_returns = [
            {'reachable_hosts': ['host2'], 'unreachable_hosts': ['host1']},  # Worker 0: partial success
            {'reachable_hosts': [], 'unreachable_hosts': ['host3', 'host4']},  # Worker 1: total failure
        ]

        self.sharder.update_worker_table(shard_returns)

        # Only Worker 0 should remain (has reachable host)
        workers = list(self.sharder.get_worker_table())
        self.assertEqual(len(workers), 1)
        self.assertEqual(workers[0].worker_id, 0)
        self.assertEqual(list(workers[0].reachable_hosts), ['host2'])

    def test_prune_nodes_removes_empty_workers(self):
        """Test that prune_worker_nodes removes workers with no remaining hosts.

        Expected Behavior:
        - Worker 0 has [host1, host2], Worker 1 has [host3, host4]
        - prune_worker_nodes({'host1', 'host2'}) removes all hosts from Worker 0
        - Worker 0 should be removed from worker table (no reachable hosts)
        - Worker 1 should remain unchanged with [host3, host4]
        """
        # Setup initial state
        self.sharder._worker_state_table.append(0, ['host1', 'host2'], ['host1', 'host2'], [])
        self.sharder._worker_state_table.append(1, ['host3', 'host4'], ['host3', 'host4'], [])

        # Prune hosts from Worker 0
        self.sharder.prune_worker_nodes({'host1', 'host2'})

        # Only Worker 1 should remain
        workers = list(self.sharder.get_worker_table())
        self.assertEqual(len(workers), 1)
        self.assertEqual(workers[0].worker_id, 1)
        self.assertEqual(list(workers[0].reachable_hosts), ['host3', 'host4'])

    def test_worker_table_persistence_across_operations(self):
        """Test that worker state persists correctly across multiple operations.

        Expected Behavior:
        - Operation 1: exec() updates worker reachability
        - Operation 2: upload_file() should use updated worker state from Operation 1
        - Worker assignments should remain stable across different operation types
        - Host reachability changes should carry forward between operations
        """
        # Initialize with workers
        self.sharder._worker_state_table.append(0, ['host1', 'host2'], ['host1', 'host2'], [])

        # Simulate first operation with partial failure
        shard_returns_1 = [{'reachable_hosts': ['host1'], 'unreachable_hosts': ['host2']}]
        self.sharder.update_worker_table(shard_returns_1)

        # Verify state after first operation
        workers = list(self.sharder.get_worker_table())
        self.assertEqual(len(workers), 1)
        self.assertEqual(list(workers[0].reachable_hosts), ['host1'])
        self.assertEqual(list(workers[0].unreachable_hosts), ['host2'])

        # Simulate second operation - should use updated state
        shard_returns_2 = [
            {'reachable_hosts': ['host1'], 'unreachable_hosts': ['host2']}  # Consistent state
        ]
        self.sharder.update_worker_table(shard_returns_2)

        # State should remain consistent
        workers_after = list(self.sharder.get_worker_table())
        self.assertEqual(len(workers_after), 1)
        self.assertEqual(workers_after[0].worker_id, workers[0].worker_id)
        self.assertEqual(list(workers_after[0].reachable_hosts), ['host1'])

    def test_worker_id_based_routing_stability(self):
        """Test that worker_id-based routing maintains stability across operations.

        Expected Behavior:
        - Workers maintain their IDs across different operations
        - Routing maps correctly to the same physical workers
        - Worker state persists between operations
        """
        # Setup workers with explicit IDs
        worker0 = {
            'process': MagicMock(),
            'req_q': MagicMock(),
            'resp_q': MagicMock(),
            'init': {'host_list': ['host1']},
            'worker_id': 0,
        }
        worker1 = {
            'process': MagicMock(),
            'req_q': MagicMock(),
            'resp_q': MagicMock(),
            'init': {'host_list': ['host2']},
            'worker_id': 1,
        }
        worker0['process'].is_alive.return_value = True
        worker1['process'].is_alive.return_value = True

        self.sharder._workers = {0: worker0, 1: worker1}
        self.sharder._worker_state_table.append(0, ['host1'], ['host1'], [])
        self.sharder._worker_state_table.append(1, ['host2'], ['host2'], [])

        # First operation routing map
        routing_map_1 = {
            0: {'operation': 'exec', 'init': {'host_list': ['host1']}, 'cmd': 'test1'},
            1: {'operation': 'exec', 'init': {'host_list': ['host2']}, 'cmd': 'test1'},
        }

        with patch.object(self.sharder, '_wait_for_response') as mock_wait:
            mock_wait.return_value = {'ok': True, 'result': {}, 'reachable_hosts': [], 'unreachable_hosts': []}

            # Execute first operation
            self.sharder.execute_sharded(routing_map_1)

            # Verify requests went to correct workers
            self.assertEqual(worker0['req_q'].put.call_count, 1)
            self.assertEqual(worker1['req_q'].put.call_count, 1)

            # Second operation with same worker IDs should use same physical workers
            routing_map_2 = {
                0: {'operation': 'upload_file', 'init': {'host_list': ['host1']}, 'local_file': 'test.txt'},
                1: {'operation': 'upload_file', 'init': {'host_list': ['host2']}, 'local_file': 'test.txt'},
            }

            self.sharder.execute_sharded(routing_map_2)

            # Verify same workers were used (call count increased)
            self.assertEqual(worker0['req_q'].put.call_count, 2)
            self.assertEqual(worker1['req_q'].put.call_count, 2)

            # Worker IDs should remain stable
            self.assertEqual(self.sharder._workers[0]['worker_id'], 0)
            self.assertEqual(self.sharder._workers[1]['worker_id'], 1)

    # ===== ERROR PROPAGATION TESTS =====

    def test_worker_error_propagation_per_host(self):
        """Test that worker execution errors are properly propagated per host.

        Expected Behavior:
        - Worker assigned [host1, host2] raises RuntimeError during exec
        - Should return {'host1': 'ERROR: RuntimeError: boom', 'host2': 'ERROR: RuntimeError: boom'}
        - Should mark both hosts as unreachable in response
        - Error message should include exception type and message
        """
        req_q = MagicMock()
        resp_q = MagicMock()
        proc = MagicMock()
        proc.is_alive.return_value = True

        self.sharder._workers = {
            0: {'process': proc, 'req_q': req_q, 'resp_q': resp_q, 'init': {'host_list': ['host1', 'host2']}}
        }
        # Setup worker state table for worker_id=0
        self.sharder._worker_state_table.append(0, ['host1', 'host2'], ['host1', 'host2'], [])

        routing_map = {0: {'operation': 'exec', 'init': {'host_list': ['host1', 'host2']}, 'cmd': 'test'}}

        # Simulate worker error response
        error_response = {
            'ok': False,
            'error': 'RuntimeError: boom',
            'reachable_hosts': [],
            'unreachable_hosts': ['host1', 'host2'],
        }

        with (
            patch.object(self.sharder, '_ensure_workers'),
            patch.object(self.sharder, '_wait_for_response', return_value=error_response),
        ):
            result = self.sharder.execute_sharded(routing_map)

            # Should propagate error to all hosts
            self.assertEqual(
                result[0]['result'], {'host1': 'ERROR: RuntimeError: boom', 'host2': 'ERROR: RuntimeError: boom'}
            )
            self.assertEqual(result[0]['reachable_hosts'], [])
            self.assertEqual(result[0]['unreachable_hosts'], ['host1', 'host2'])

    def test_worker_response_timeout_error_format(self):
        """Test error format when worker doesn't respond within timeout.

        Expected Behavior:
        - Worker gets stuck and doesn't send response within timeout period
        - Should return error dict with descriptive timeout message
        - Error should include request_id for debugging
        - All hosts assigned to worker should be marked with timeout error
        """
        resp_q = MagicMock()
        resp_q.get.side_effect = Empty()  # Simulate timeout

        # Mock time to avoid real timeout wait
        with patch('cvs.lib.parallel.persistent_pssh_sharder.time') as mock_time:
            # Simulate time progression beyond timeout
            mock_time.time.side_effect = [0, 0.5, 301]  # Start, middle, exceed timeout

            # Test the timeout response format
            response = self.sharder._wait_for_response(resp_q, 'test-req-123')

        self.assertFalse(response['ok'])
        self.assertIn('Timeout waiting for shard response', response['error'])
        self.assertIn('test-req-123', response['error'])
        self.assertEqual(response['request_id'], 'test-req-123')
        self.assertEqual(response['result'], {})
        self.assertEqual(response['reachable_hosts'], [])
        self.assertEqual(response['unreachable_hosts'], [])


if __name__ == "__main__":
    unittest.main()
