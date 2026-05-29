import unittest
import time
from unittest.mock import MagicMock, patch

from cvs.lib.parallel.config import ParallelConfig
from cvs.lib.parallel.persistent_pssh_sharder import PersistentPsshSharder


class TestPersistentPerformanceIntegration(unittest.TestCase):
    """Integration and performance tests for persistent sharder implementation."""

    def setUp(self):
        self.config = ParallelConfig(hosts_per_shard=2, persistent_shards=True)
        self.mock_log = MagicMock()
        self.host_list = ["host1", "host2", "host3", "host4"]

    def test_persistent_worker_reuse_performance(self):
        """Test that persistent workers actually reuse SSH connections.

        Expected Behavior:
        - Execute multiple operations on same hosts
        - Should not see SSH connection establishment overhead after first operation
        - Worker processes should remain alive between operations
        - Connection reuse should improve operation latency
        """
        sharder = PersistentPsshSharder(self.config)

        # Mock worker processes that stay alive
        mock_workers = []
        for i in range(2):
            worker = {
                'process': MagicMock(),
                'req_q': MagicMock(),
                'resp_q': MagicMock(),
                'init': {'host_list': self.host_list[i * 2 : (i + 1) * 2]},
            }
            worker['process'].is_alive.return_value = True
            mock_workers.append(worker)

        # Convert to dict format for new implementation
        sharder._workers = {i: worker for i, worker in enumerate(mock_workers)}
        sharder._worker_state_table.append(0, ['host1', 'host2'], ['host1', 'host2'], [])
        sharder._worker_state_table.append(1, ['host3', 'host4'], ['host3', 'host4'], [])

        routing_map = {
            0: {'operation': 'exec', 'init': {'host_list': ['host1', 'host2']}, 'cmd': 'uptime'},
            1: {'operation': 'exec', 'init': {'host_list': ['host3', 'host4']}, 'cmd': 'uptime'},
        }

        with patch.object(sharder, '_wait_for_response') as mock_wait:
            mock_wait.return_value = {'ok': True, 'result': {}, 'reachable_hosts': [], 'unreachable_hosts': []}

            # Execute first operation
            sharder.execute_sharded(routing_map)

            # Execute second operation - should reuse workers
            sharder.execute_sharded(routing_map)

            # Workers should remain alive and not be restarted
            for worker in sharder._workers.values():
                worker['process'].is_alive.assert_called()
                worker['process'].terminate.assert_not_called()

    def test_concurrent_destroy_and_execute_safety(self):
        """Test thread safety between destroy_clients() and execute_sharded().

        Expected Behavior:
        - If destroy_clients() called while execute_sharded() is running
        - Should not crash or leave zombie processes
        - execute_sharded() should fail gracefully with clear error
        - All worker processes should be properly terminated
        """
        sharder = PersistentPsshSharder(self.config)

        # Setup mock workers
        worker = {'process': MagicMock(), 'req_q': MagicMock(), 'resp_q': MagicMock(), 'init': {'host_list': ['host1']}}
        worker['process'].is_alive.return_value = True
        sharder._workers = {0: worker}

        routing_map = {0: {'operation': 'exec', 'init': {'host_list': ['host1']}, 'cmd': 'test'}}

        # Simulate destroy_clients being called during execute_sharded
        def side_effect(*args):
            # Call destroy_clients during execution
            sharder.destroy_clients()
            return {
                'ok': False,
                'result': {},
                'reachable_hosts': [],
                'unreachable_hosts': ['host1'],
                'error': 'Worker terminated during execution',
            }

        with (
            patch.object(sharder, '_ensure_workers'),
            patch.object(sharder, '_wait_for_response', side_effect=side_effect),
        ):
            # Should not crash, should handle gracefully
            sharder.execute_sharded(routing_map)

            # Workers should be cleaned up
            self.assertEqual(len(sharder._workers), 0)
            worker['process'].terminate.assert_called()

    def test_worker_process_count_limits_enforced(self):
        """Test that persistent sharder respects max_workers configuration limits.

        Expected Behavior:
        - Configuration with max_workers=2 should never create more than 2 processes
        - Even with large host lists that would naturally create more chunks
        - Host distribution should be rebalanced across available workers
        """
        # Create config inside patch context to ensure max_workers=2
        with patch('cvs.lib.parallel.config.os.cpu_count', return_value=2):  # Force max_workers = 2
            limited_config = ParallelConfig(hosts_per_shard=1, max_workers_per_cpu=1)
            sharder = PersistentPsshSharder(limited_config)

            # Large host list that would naturally create 10 chunks
            large_host_list = [f'host{i}' for i in range(10)]

            with patch.object(sharder, '_start_worker') as mock_start:
                mock_worker = {'process': MagicMock(), 'req_q': MagicMock(), 'resp_q': MagicMock(), 'init': {}}
                mock_start.return_value = mock_worker

                sharder.initialize_workers(large_host_list)

                # Should respect max_workers limit (2), not create 10 workers
                self.assertLessEqual(len(sharder._workers), 2)
                self.assertLessEqual(mock_start.call_count, 2)

                # All hosts should still be covered
                covered_hosts = set()
                for row in sharder._worker_state_table:
                    covered_hosts.update(row.host_list)
                self.assertEqual(covered_hosts, set(large_host_list))

    def test_timeout_buffer_configuration(self):
        """Test configurable timeout buffer for persistent operations.

        Expected Behavior:
        - Config with response_timeout_buffer=45 should add 45s to operation timeouts
        - exec(timeout=10) should wait 10+45=55 seconds for response
        - Buffer should account for IPC overhead and network latency
        - Should be tunable for different environments (local vs remote clusters)
        """
        # This test assumes we implement configurable timeout buffer
        # For now, test the current hardcoded behavior and document expected enhancement

        sharder = PersistentPsshSharder(self.config)

        # Mock response queue that times out
        from queue import Empty

        resp_q = MagicMock()
        resp_q.get.side_effect = Empty()  # Simulate timeout

        import cvs.lib.parallel.persistent_pssh_sharder as pssh_module

        # Test current behavior - should use _RESPONSE_TIMEOUT_SEC
        original_timeout = sharder._RESPONSE_TIMEOUT_SEC
        test_timeout = 0.01  # Very short for testing
        sharder._RESPONSE_TIMEOUT_SEC = test_timeout

        try:
            with patch.object(pssh_module, 'time') as mock_time:
                # Use itertools.count to provide infinite increasing time values
                from itertools import count

                time_counter = count(0, 0.001)  # Start at 0, increment by 1ms each call
                mock_time.time.side_effect = lambda: next(time_counter)

                response = sharder._wait_for_response(resp_q, 'test-req')

                self.assertFalse(response['ok'])
                self.assertIn('Timeout waiting for shard response', response['error'])
        finally:
            sharder._RESPONSE_TIMEOUT_SEC = original_timeout

    def test_large_scale_host_distribution(self):
        """Test worker distribution with very large host lists.

        Expected Behavior:
        - 1000+ hosts should be distributed evenly across workers
        - No worker should be significantly over/under loaded
        - All hosts should be assigned to exactly one worker
        - Performance should remain reasonable with large host counts
        """
        config = ParallelConfig(hosts_per_shard=50, max_workers_per_cpu=4)

        with patch('cvs.lib.parallel.config.os.cpu_count', return_value=8):  # max_workers = 32
            sharder = PersistentPsshSharder(config)

        # Create large host list
        large_host_list = [f'host{i:04d}' for i in range(1000)]

        with patch.object(sharder, '_start_worker') as mock_start:
            mock_worker = {'process': MagicMock(), 'req_q': MagicMock(), 'resp_q': MagicMock(), 'init': {}}
            mock_start.return_value = mock_worker

            start_time = time.time()
            sharder.initialize_workers(large_host_list)
            initialization_time = time.time() - start_time

            # Should complete initialization reasonably quickly (< 1 second)
            self.assertLess(initialization_time, 1.0)

            # Verify all hosts are covered
            covered_hosts = set()
            host_counts = []
            for row in sharder._worker_state_table:
                covered_hosts.update(row.host_list)
                host_counts.append(len(row.host_list))

            self.assertEqual(len(covered_hosts), 1000)
            self.assertEqual(covered_hosts, set(large_host_list))

            # Verify reasonable load distribution (no worker should have > 2x average)
            if host_counts:
                avg_hosts = sum(host_counts) / len(host_counts)
                max_hosts = max(host_counts)
                self.assertLess(max_hosts, avg_hosts * 2.0)

    def test_error_recovery_and_resilience(self):
        """Test system resilience to various error conditions.

        Expected Behavior:
        - Worker crashes should not affect other workers
        - Network errors should be isolated to affected hosts
        - System should gracefully degrade and recover
        - Error states should be clearly reported
        """
        sharder = PersistentPsshSharder(self.config)

        # Setup mixed worker states: healthy, crashed, and network-failed
        healthy_worker = {
            'process': MagicMock(),
            'req_q': MagicMock(),
            'resp_q': MagicMock(),
            'init': {'host_list': ['host1']},
        }
        healthy_worker['process'].is_alive.return_value = True

        crashed_worker = {
            'process': MagicMock(),
            'req_q': MagicMock(),
            'resp_q': MagicMock(),
            'init': {'host_list': ['host2']},
        }
        crashed_worker['process'].is_alive.return_value = False  # Crashed

        sharder._workers = {0: healthy_worker, 1: crashed_worker}

        routing_map = {
            0: {'operation': 'exec', 'init': {'host_list': ['host1']}, 'cmd': 'test'},
            1: {'operation': 'exec', 'init': {'host_list': ['host2']}, 'cmd': 'test'},
        }

        # Mock successful restart of crashed worker
        new_worker = {
            'process': MagicMock(),
            'req_q': MagicMock(),
            'resp_q': MagicMock(),
            'init': {'host_list': ['host2']},
        }
        new_worker['process'].is_alive.return_value = True

        with (
            patch.object(sharder, '_start_worker', return_value=new_worker) as mock_start,
            patch.object(sharder, '_wait_for_response') as mock_wait,
        ):
            # Mock responses: healthy worker succeeds, restarted worker succeeds
            mock_wait.side_effect = [
                {'ok': True, 'result': {'host1': 'success'}, 'reachable_hosts': ['host1'], 'unreachable_hosts': []},
                {'ok': True, 'result': {'host2': 'success'}, 'reachable_hosts': ['host2'], 'unreachable_hosts': []},
            ]

            result = sharder.execute_sharded(routing_map)

            # Crashed worker should have been restarted
            mock_start.assert_called_once_with({'host_list': ['host2']})

            # Both operations should succeed after recovery
            self.assertEqual(len(result), 2)
            self.assertTrue(result[0]['result']['host1'] == 'success')
            self.assertTrue(result[1]['result']['host2'] == 'success')

    def test_memory_usage_and_cleanup(self):
        """Test that persistent workers don't accumulate memory leaks over time.

        Expected Behavior:
        - Multiple operations should not increase memory usage significantly
        - Worker state tables should be cleaned up properly
        - No references to old results should be retained
        - Destroy operations should fully clean up resources
        """
        sharder = PersistentPsshSharder(self.config)

        # Setup workers
        workers = []
        for i in range(3):
            worker = {
                'process': MagicMock(),
                'req_q': MagicMock(),
                'resp_q': MagicMock(),
                'init': {'host_list': [f'host{i + 1}']},
            }
            worker['process'].is_alive.return_value = True
            workers.append(worker)

        # Convert to dict format
        sharder._workers = {i: worker for i, worker in enumerate(workers)}

        # Initialize worker table
        for i, worker in enumerate(workers):
            sharder._worker_state_table.append(i, [f'host{i + 1}'], [f'host{i + 1}'], [])

        initial_worker_count = len(sharder._workers)
        initial_table_size = len(sharder._worker_state_table)

        # Simulate multiple operations
        for _ in range(10):
            routing_map = {
                i: {'operation': 'exec', 'init': {'host_list': [f'host{i + 1}']}, 'cmd': f'test{i}'} for i in range(3)
            }

            with patch.object(sharder, '_ensure_workers'), patch.object(sharder, '_wait_for_response') as mock_wait:
                mock_wait.return_value = {'ok': True, 'result': {}, 'reachable_hosts': [], 'unreachable_hosts': []}

                sharder.execute_sharded(routing_map)

        # Resource counts should remain stable
        self.assertEqual(len(sharder._workers), initial_worker_count)
        self.assertEqual(len(sharder._worker_state_table), initial_table_size)

        # Cleanup should remove all resources
        sharder.destroy_clients()

        self.assertEqual(len(sharder._workers), 0)
        self.assertEqual(len(sharder._worker_state_table), 0)

        # All worker processes should be terminated
        for worker in workers:
            worker['process'].terminate.assert_called()


if __name__ == "__main__":
    unittest.main()
