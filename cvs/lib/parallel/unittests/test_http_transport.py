import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pssh.exceptions import Timeout

from cvs.core.agent.http_client import HostOutput, HTTPConnectionError
from cvs.lib.parallel.http_transport import HttpTransport
from cvs.lib.parallel.transport import BaseTransport


def _ok(host, line='ok'):
    return HostOutput(host=host, stdout=[line], stderr=[], exit_code=0, exception=None)


class TestHttpTransport(unittest.TestCase):
    def setUp(self):
        self.http_patcher = patch('cvs.lib.parallel.http_transport.ParallelHTTPClient')
        self.mock_http_cls = self.http_patcher.start()
        self.addCleanup(self.http_patcher.stop)
        self.mock_http = MagicMock()
        self.mock_http.run_command = AsyncMock(return_value=[_ok('h1')])
        self.mock_http.health = AsyncMock(return_value={'h1': True})
        self.mock_http.destroy = AsyncMock()
        self.mock_http.shutdown = AsyncMock(return_value={'h1': True})
        self.mock_http.rebuild = MagicMock()
        self.mock_http_cls.return_value = self.mock_http

    def _make(self, hosts=None, urls=None, token='tok'):
        hosts = list(hosts or ['h1'])
        urls = urls if urls is not None else {host: f'http://{host}:9' for host in hosts}
        transport = HttpTransport(hosts, agent_urls=urls, token=token)
        self.addCleanup(transport.destroy)
        return transport

    def test_is_base_transport(self):
        self.assertTrue(issubclass(HttpTransport, BaseTransport))

    def test_missing_token_raises(self):
        with self.assertRaisesRegex(ValueError, "non-empty token"):
            HttpTransport(['h1'], agent_urls={'h1': 'http://h1:9'}, token='')

    def test_missing_agent_urls_raises(self):
        with self.assertRaisesRegex(ValueError, "agent_urls"):
            HttpTransport(['h1'], agent_urls={}, token='tok')

    def test_unknown_host_raises(self):
        with self.assertRaisesRegex(ValueError, "No agent URL"):
            self._make(hosts=['h1', 'missing'], urls={'h1': 'http://h1:9'})

    def test_run_command_is_synchronous(self):
        transport = self._make()
        outputs = transport.client.run_command('hostname', stop_on_errors=False, read_timeout=3)
        self.assertEqual(outputs[0].host, 'h1')
        self.mock_http.run_command.assert_awaited_once()
        kwargs = self.mock_http.run_command.await_args.kwargs
        self.assertEqual(kwargs['stop_on_errors'], False)
        self.assertEqual(kwargs['read_timeout'], 3)

    def test_run_command_forwards_host_args_and_inactivity_timeout(self):
        transport = self._make(hosts=['h1', 'h2'])
        self.mock_http.run_command = AsyncMock(return_value=[_ok('h1'), _ok('h2')])
        transport.client.run_command('%s', host_args=['echo a', 'echo b'], inactivity_timeout=7)
        kwargs = self.mock_http.run_command.await_args.kwargs
        self.assertEqual(kwargs['host_args'], ['echo a', 'echo b'])
        self.assertEqual(kwargs['inactivity_timeout'], 7)

    def test_declares_capabilities_parallel_handle_branches_on(self):
        self.assertEqual(HttpTransport.prune_exception_types, (HTTPConnectionError,))
        self.assertTrue(HttpTransport.remote_inactivity_timeout)

    def test_agent_timeout_surfaces_as_a_timeout_exception(self):
        """A killed-on-timeout command must not read as an ordinary nonzero exit."""
        transport = self._make()
        self.mock_http.run_command = AsyncMock(
            return_value=[HostOutput(host='h1', stdout=[], stderr=[], exit_code=-15, exception=None, timed_out=True)]
        )
        outputs = transport.client.run_command('sleep 100', read_timeout=1)
        self.assertIsInstance(outputs[0].exception, Timeout)
        self.assertIn('h1', str(outputs[0].exception))

    def test_transport_level_exception_is_not_overwritten_by_timeout(self):
        transport = self._make()
        original = HTTPConnectionError('refused')
        self.mock_http.run_command = AsyncMock(
            return_value=[
                HostOutput(host='h1', stdout=[], stderr=[], exit_code=None, exception=original, timed_out=True)
            ]
        )
        outputs = transport.client.run_command('hostname')
        self.assertIs(outputs[0].exception, original)

    def test_truncated_output_is_flagged_in_stderr(self):
        transport = self._make()
        self.mock_http.run_command = AsyncMock(
            return_value=[HostOutput(host='h1', stdout=['a'], stderr=[], exit_code=0, exception=None, truncated=True)]
        )
        outputs = transport.client.run_command('cat big')
        self.assertIn('ABORT: Output Truncated by agent on Host: h1', outputs[0].stderr)

    def test_clean_result_is_left_untouched(self):
        transport = self._make()
        outputs = transport.client.run_command('hostname')
        self.assertIsNone(outputs[0].exception)
        self.assertEqual(outputs[0].stderr, [])

    def test_check_connectivity_returns_unhealthy_hosts(self):
        transport = self._make(hosts=['h1', 'h2'])
        self.mock_http.health = AsyncMock(return_value={'h1': True, 'h2': False})
        self.assertEqual(transport.check_connectivity(['h1', 'h2']), ['h2'])

    def test_check_connectivity_probes_only_the_suspect_hosts(self):
        """One flaky node must not cause a cluster-wide health fan-out on every exec."""
        transport = self._make(hosts=['h1', 'h2'])
        self.mock_http.health = AsyncMock(return_value={'h2': False})
        self.mock_http_cls.reset_mock()

        self.assertEqual(transport.check_connectivity(['h2']), ['h2'])

        self.mock_http_cls.assert_called_once_with({'h2': 'http://h2:9'}, 'tok', connect_timeout=None)
        # The throwaway probe pool is released rather than left open.
        self.assertEqual(self.mock_http.destroy.await_count, 1)

    def test_check_connectivity_treats_unregistered_host_as_unreachable(self):
        transport = self._make(hosts=['h1'])
        self.mock_http_cls.reset_mock()
        self.assertEqual(transport.check_connectivity(['nobody']), ['nobody'])
        self.mock_http_cls.assert_not_called()

    def test_check_connectivity_empty_hosts(self):
        transport = self._make()
        self.assertEqual(transport.check_connectivity([]), [])
        self.mock_http.health.assert_not_called()

    def test_rebuild_updates_url_map_without_new_client(self):
        transport = self._make(hosts=['h1', 'h2'])
        transport.rebuild(['h1'])
        self.mock_http.rebuild.assert_called_once_with({'h1': 'http://h1:9'})
        self.assertEqual(self.mock_http_cls.call_count, 1)
        self.assertEqual(transport.client._hosts, ['h1'])

    def test_rebuild_after_destroy_recreates_the_client(self):
        """SshTransport.rebuild recovers from a destroy(); this must not diverge."""
        transport = self._make(hosts=['h1', 'h2'])
        transport.destroy()
        transport.rebuild(['h1'])
        self.assertEqual(transport.client._hosts, ['h1'])
        self.assertEqual(transport.client.run_command('hostname')[0].host, 'h1')

    def test_transports_share_one_event_loop_thread(self):
        first = self._make(hosts=['h1'])
        second = first.client_for_hosts(['h1'])
        self.addCleanup(second.destroy)
        self.assertIs(first._loop, second._loop)

    def test_client_for_hosts_is_independent_transport(self):
        transport = self._make(hosts=['h1', 'h2'])
        subset = transport.client_for_hosts(['h2'])
        self.addCleanup(subset.destroy)
        self.assertIsInstance(subset, HttpTransport)
        self.assertEqual(subset.hosts, ['h2'])
        self.assertEqual(self.mock_http_cls.call_count, 2)
        self.mock_http_cls.assert_called_with(
            {'h2': 'http://h2:9'},
            'tok',
            connect_timeout=None,
        )

    def test_destroy_closes_client_and_does_not_shutdown_agents(self):
        transport = self._make()
        transport.destroy()
        self.mock_http.destroy.assert_awaited()
        self.mock_http.shutdown.assert_not_called()
        with self.assertRaises(AttributeError):
            _ = transport.client

    def test_shutdown_agents_posts_shutdown(self):
        transport = self._make()
        result = transport.shutdown_agents(stop_on_errors=True)
        self.assertEqual(result, {'h1': True})
        self.mock_http.shutdown.assert_awaited_once_with(stop_on_errors=True)

    def test_wait_until_healthy_returns_when_all_ok(self):
        transport = self._make(hosts=['h1'])
        self.mock_http.health = AsyncMock(side_effect=[{'h1': False}, {'h1': True}])
        with patch('cvs.lib.parallel.http_transport.time.sleep'):
            results = transport.wait_until_healthy(timeout=1)
        self.assertEqual(results, {'h1': True})
        self.assertEqual(self.mock_http.health.await_count, 2)

    def test_wait_until_healthy_times_out(self):
        transport = self._make(hosts=['h1'])
        self.mock_http.health = AsyncMock(return_value={'h1': False})
        with self.assertRaisesRegex(TimeoutError, "not healthy"):
            transport.wait_until_healthy(timeout=0)


class TestHttpTransportSharedFsCopy(unittest.TestCase):
    def setUp(self):
        self.http_patcher = patch('cvs.lib.parallel.http_transport.ParallelHTTPClient')
        mock_http_cls = self.http_patcher.start()
        self.addCleanup(self.http_patcher.stop)
        mock_http = MagicMock()
        mock_http.destroy = AsyncMock()
        mock_http.run_command = AsyncMock(return_value=[])
        mock_http.health = AsyncMock(return_value={})
        mock_http.shutdown = AsyncMock(return_value={})
        mock_http_cls.return_value = mock_http
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.transport = HttpTransport(
            ['h1', 'h2'],
            agent_urls={'h1': 'http://h1:9', 'h2': 'http://h2:9'},
            token='tok',
        )
        self.addCleanup(self.transport.destroy)

    def test_copy_file_writes_once_and_returns_per_host_stubs(self):
        src = self.root / 'src.txt'
        dst = self.root / 'dst.txt'
        src.write_text('hello')
        cmds = self.transport.client.copy_file(str(src), str(dst))
        self.transport.client.pool.join()
        self.assertEqual(len(cmds), 2)
        for cmd in cmds:
            self.assertIsNone(cmd.get())
        self.assertEqual(dst.read_text(), 'hello')

    def test_copy_file_reports_the_same_outcome_to_every_host(self):
        cmds = self.transport.client.copy_file('/no/such/src', str(self.root / 'dst.txt'))
        self.assertEqual(len(cmds), 2)
        for cmd in cmds:
            with self.assertRaises(OSError):
                cmd.get()

    def test_copy_file_same_path_is_success(self):
        src = self.root / 'same.txt'
        src.write_text('x')
        cmds = self.transport.client.copy_file(str(src), str(src))
        cmds[0].get()

    def test_copy_file_failure_surfaces_on_get(self):
        cmds = self.transport.client.copy_file('/no/such/src', str(self.root / 'dst.txt'))
        with self.assertRaises(OSError):
            cmds[0].get()

    def test_copy_file_copy_args(self):
        a = self.root / 'a.txt'
        b = self.root / 'b.txt'
        a.write_text('A')
        b.write_text('B')
        dst_a = self.root / 'out_a.txt'
        dst_b = self.root / 'out_b.txt'
        cmds = self.transport.client.copy_file(
            '%(local_file)s',
            '%(remote_file)s',
            copy_args=[
                {'local_file': str(a), 'remote_file': str(dst_a)},
                {'local_file': str(b), 'remote_file': str(dst_b)},
            ],
        )
        for cmd in cmds:
            cmd.get()
        self.assertEqual(dst_a.read_text(), 'A')
        self.assertEqual(dst_b.read_text(), 'B')

    def test_copy_remote_file_suffixes_per_host(self):
        remote = self.root / 'remote.txt'
        remote.write_text('payload')
        local = str(self.root / 'local.txt')
        cmds = self.transport.client.copy_remote_file(str(remote), local)
        for cmd in cmds:
            cmd.get()
        self.assertEqual((self.root / 'local.txt_h1').read_text(), 'payload')
        self.assertEqual((self.root / 'local.txt_h2').read_text(), 'payload')

    def test_copy_file_recurse_copies_directory(self):
        src = self.root / 'tree'
        src.mkdir()
        (src / 'f.txt').write_text('nested')
        dst = self.root / 'tree-out'
        cmds = self.transport.client.copy_file(str(src), str(dst), recurse=True)
        cmds[0].get()
        self.assertEqual((dst / 'f.txt').read_text(), 'nested')


if __name__ == '__main__':
    unittest.main()
