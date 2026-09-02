import unittest
from unittest.mock import patch, MagicMock, call

from cvs.lib.parallel.ssh_transport import SshTransport


class TestSshTransportInit(unittest.TestCase):
    @patch('cvs.lib.parallel.phandle.ParallelSSHClient')
    def test_init_with_password(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        transport = SshTransport(['h1', 'h2'], user='user', password='pass', num_retries=1)

        mock_client_cls.assert_called_once_with(
            ['h1', 'h2'],
            user='user',
            password='pass',
            keepalive_seconds=30,
            num_retries=1,
        )
        self.assertIs(transport.client, mock_client_cls.return_value)

    @patch('cvs.lib.parallel.phandle.ParallelSSHClient')
    def test_init_with_pkey(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        transport = SshTransport(['h1'], user='user', pkey='id_ed25519')

        mock_client_cls.assert_called_once_with(
            ['h1'],
            user='user',
            pkey='id_ed25519',
            keepalive_seconds=30,
        )
        self.assertEqual(transport.hosts, ['h1'])


class TestSshTransportRebuild(unittest.TestCase):
    @patch('cvs.lib.parallel.phandle.ParallelSSHClient')
    def test_rebuild_recreates_client_for_new_hosts(self, mock_client_cls):
        first_client = MagicMock()
        second_client = MagicMock()
        mock_client_cls.side_effect = [first_client, second_client]

        transport = SshTransport(['h1'], user='user', password='pass')
        transport.rebuild(['h2', 'h3'])

        self.assertEqual(transport.hosts, ['h2', 'h3'])
        self.assertIs(transport.client, second_client)
        self.assertEqual(
            mock_client_cls.call_args_list,
            [
                call(['h1'], user='user', password='pass', keepalive_seconds=30),
                call(['h2', 'h3'], user='user', password='pass', keepalive_seconds=30),
            ],
        )


class TestSshTransportCheckConnectivity(unittest.TestCase):
    @patch('cvs.lib.parallel.phandle.ParallelSSHClient')
    def test_check_connectivity_empty_hosts(self, mock_client_cls):
        transport = SshTransport(['h1'], user='user', password='pass')
        self.assertEqual(transport.check_connectivity([]), [])
        mock_client_cls.assert_called_once()

    @patch('cvs.lib.parallel.phandle.ParallelSSHClient')
    def test_check_connectivity_returns_unreachable_hosts(self, mock_client_cls):
        main_client = MagicMock()
        probe_client = MagicMock()
        mock_client_cls.side_effect = [main_client, probe_client]

        ok = MagicMock(host='h1', exception=None)
        bad = MagicMock(host='h2', exception=ConnectionError('down'))
        probe_client.run_command.return_value = [ok, bad]

        transport = SshTransport(['h1', 'h2'], user='user', password='pass')
        unreachable = transport.check_connectivity(['h1', 'h2'])

        self.assertEqual(unreachable, ['h2'])
        probe_client.run_command.assert_called_once_with('echo 1', stop_on_errors=False, read_timeout=2)
        self.assertEqual(
            mock_client_cls.call_args_list[1],
            call(
                ['h1', 'h2'],
                user='user',
                password='pass',
                keepalive_seconds=30,
                timeout=2,
                num_retries=0,
            ),
        )


class TestSshTransportClientForHosts(unittest.TestCase):
    @patch('cvs.lib.parallel.phandle.ParallelSSHClient')
    def test_client_for_hosts_builds_subset_transport(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        transport = SshTransport(['h1', 'h2'], user='user', password='pass', allow_agent=False)

        subset = transport.client_for_hosts(['h2'])

        self.assertIsInstance(subset, SshTransport)
        self.assertEqual(subset.hosts, ['h2'])
        self.assertEqual(subset.user, 'user')
        self.assertEqual(subset.password, 'pass')
        self.assertEqual(subset.ssh_client_kwargs, {'allow_agent': False})
        self.assertEqual(
            mock_client_cls.call_args_list[-1],
            call(['h2'], user='user', password='pass', keepalive_seconds=30, allow_agent=False),
        )


class TestSshTransportDestroy(unittest.TestCase):
    """destroy must tear the SSH client down explicitly.

    A timed-out exec leaves the per-host greenlet in client.cmds unfinished.
    That greenlet's callable is a bound method of the ParallelSSHClient, so the
    client stays reachable unless pending greenlets are killed and host clients
    disconnected explicitly.
    """

    @patch('cvs.lib.parallel.phandle.ParallelSSHClient')
    def _make_transport(self, mock_client_cls, host_clients=None, cmds=None):
        mock_client = MagicMock()
        mock_client.cmds = cmds
        mock_client._host_clients = host_clients if host_clients is not None else {}
        mock_client_cls.return_value = mock_client
        transport = SshTransport(['host1'], user='user', password='pass')
        return transport, mock_client

    def test_destroy_disconnects_each_host_client(self):
        host_client = MagicMock()
        transport, client = self._make_transport(host_clients={(0, 'host1'): host_client})

        transport.destroy()

        host_client._disconnect.assert_called_once_with()
        with self.assertRaises(AttributeError):
            _ = transport.client

    def test_destroy_kills_pending_command_greenlets(self):
        greenlet = MagicMock()
        transport, client = self._make_transport(cmds=[greenlet])

        with patch('cvs.lib.parallel.ssh_transport.killall') as mock_killall:
            transport.destroy()

        mock_killall.assert_called_once_with([greenlet], block=True, timeout=5)
        self.assertIsNone(client.cmds)

    def test_destroy_survives_disconnect_errors(self):
        dead = MagicMock()
        dead._disconnect.side_effect = OSError('connection already gone')
        alive = MagicMock()
        transport, client = self._make_transport(host_clients={(0, 'h1'): dead, (1, 'h2'): alive})

        transport.destroy()

        alive._disconnect.assert_called_once_with()

    def test_destroy_logs_disconnect_errors(self):
        dead = MagicMock()
        dead._disconnect.side_effect = OSError('connection already gone')
        transport, client = self._make_transport(host_clients={(0, 'h1'): dead})

        with patch('cvs.lib.parallel.ssh_transport.log') as mock_log:
            transport.destroy()

        mock_log.debug.assert_called_once()
        self.assertIn('Error disconnecting SSH client', mock_log.debug.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
