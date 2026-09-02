import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

from cvs.core.agent.mesh import AgentMesh, _addresses, _ptr_names


class TestAgentMesh(unittest.TestCase):
    def setUp(self):
        AgentMesh.reset()
        self.addCleanup(AgentMesh.reset)

    def test_install_builds_host_urls(self):
        snapshot = {
            0: SimpleNamespace(hostname='n0', port=9000),
            1: SimpleNamespace(hostname='n1', port=9001),
        }
        mesh = AgentMesh.install(snapshot, 'secret-token')
        self.assertEqual(
            mesh.urls_by_host,
            {'n0': 'http://n0:9000', 'n1': 'http://n1:9001'},
        )
        self.assertEqual(mesh.token, 'secret-token')
        self.assertIs(AgentMesh.get(), mesh)

    def test_duplicate_hostname_raises(self):
        snapshot = {
            0: SimpleNamespace(hostname='n0', port=9000),
            1: SimpleNamespace(hostname='n0', port=9001),
        }
        with self.assertRaisesRegex(ValueError, "duplicate agent hostname"):
            AgentMesh.install(snapshot, 'tok')

    def test_empty_token_raises(self):
        with self.assertRaisesRegex(ValueError, "non-empty auth token"):
            AgentMesh.install({}, '')

    def test_get_before_install_raises(self):
        with self.assertRaisesRegex(RuntimeError, "not installed"):
            AgentMesh.get()

    def test_install_from_agent_dir_reads_secret(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        Path(tmp.name, 'secret').write_text('from-file\n')
        snapshot = {0: SimpleNamespace(hostname='n0', port=1)}
        mesh = AgentMesh.install_from_agent_dir(snapshot, tmp.name)
        self.assertEqual(mesh.token, 'from-file')
        self.assertEqual(mesh.urls_by_host, {'n0': 'http://n0:1'})


class TestAgentMeshResolve(unittest.TestCase):
    """resolve() bridges cluster-file host names to the names agents registered under."""

    def setUp(self):
        AgentMesh.reset()
        self.addCleanup(AgentMesh.reset)
        self.mesh = AgentMesh.install(
            {
                0: SimpleNamespace(hostname='node01', port=9000),
                1: SimpleNamespace(hostname='node02', port=9001),
            },
            'tok',
        )

    def test_exact_names_resolve(self):
        self.assertEqual(
            self.mesh.resolve(['node01', 'node02']),
            {'node01': 'http://node01:9000', 'node02': 'http://node02:9001'},
        )

    def test_fqdn_in_cluster_file_matches_short_registration(self):
        with patch('cvs.core.agent.mesh._addresses', return_value=[]):
            resolved = self.mesh.resolve(['node01.cluster.local'])
        self.assertEqual(resolved, {'node01.cluster.local': 'http://node01:9000'})

    def test_management_ips_in_cluster_file_match_by_address(self):
        mapping = {
            'node01': ['10.0.0.1'],
            'node02': ['10.0.0.2'],
            '10.0.0.2': ['10.0.0.2'],
        }
        with patch('cvs.core.agent.mesh._addresses', side_effect=lambda n: mapping.get(n, [])):
            resolved = self.mesh.resolve(['10.0.0.2'])
        self.assertEqual(resolved, {'10.0.0.2': 'http://node02:9001'})

    def test_second_a_record_matches_cluster_ip(self):
        mapping = {
            'node01': ['10.1.0.1', '10.0.0.1'],
            'node02': ['10.0.0.2'],
            '10.0.0.1': ['10.0.0.1'],
        }
        with patch('cvs.core.agent.mesh._addresses', side_effect=lambda n: mapping.get(n, [])):
            resolved = self.mesh.resolve(['10.0.0.1'])
        self.assertEqual(resolved, {'10.0.0.1': 'http://node01:9000'})

    def test_reverse_dns_matches_registered_short_name(self):
        mapping = {
            'node01': ['10.1.0.1'],
            '10.0.0.1': ['10.0.0.1'],
        }
        with (
            patch('cvs.core.agent.mesh._addresses', side_effect=lambda n: mapping.get(n, [])),
            patch(
                'cvs.core.agent.mesh._ptr_names',
                side_effect=lambda ip: ['node01.cluster.local'] if ip == '10.0.0.1' else [],
            ),
        ):
            resolved = self.mesh.resolve(['10.0.0.1'])
        self.assertEqual(resolved, {'10.0.0.1': 'http://node01:9000'})

    def test_unresolvable_host_raises_naming_both_sides(self):
        with patch('cvs.core.agent.mesh._addresses', return_value=[]):
            with self.assertRaises(ValueError) as ctx:
                self.mesh.resolve(['node01', 'login-node'])
        message = str(ctx.exception)
        self.assertIn('login-node', message)
        self.assertIn('node01', message)

    def test_resolution_preserves_requested_order(self):
        self.assertEqual(list(self.mesh.resolve(['node02', 'node01'])), ['node02', 'node01'])

    def test_addresses_resolved_once_per_host(self):
        """Both address-based rungs share one lookup; on the failure path each can block
        until DNS times out, and paying it twice turns a clear error into a slow one."""
        with patch('cvs.core.agent.mesh._addresses', return_value=[]) as mock_addresses:
            with self.assertRaises(ValueError):
                self.mesh.resolve(['login-node'])
        self.assertEqual(mock_addresses.call_args_list.count(call('login-node')), 1)

    def test_agent_indexes_are_built_once_across_resolves(self):
        with patch('cvs.core.agent.mesh._addresses', return_value=[]) as mock_addresses:
            self.mesh.resolve(['node01'])
            first = mock_addresses.call_count
            self.mesh.resolve(['node02'])
            self.assertEqual(mock_addresses.call_count, first)


class TestAgentMeshAmbiguity(unittest.TestCase):
    """A cluster name that two agents claim must fail, not silently pick the first."""

    def setUp(self):
        AgentMesh.reset()
        self.addCleanup(AgentMesh.reset)

    def test_colliding_short_names_are_refused(self):
        mesh = AgentMesh.install(
            {
                0: SimpleNamespace(hostname='node01.dc1', port=9000),
                1: SimpleNamespace(hostname='node01.dc2', port=9001),
            },
            'tok',
        )
        with patch('cvs.core.agent.mesh._addresses', return_value=[]):
            with self.assertRaisesRegex(ValueError, "match more than one registered agent"):
                mesh.resolve(['node01'])

    def test_exact_name_still_wins_over_an_ambiguous_short_name(self):
        mesh = AgentMesh.install(
            {
                0: SimpleNamespace(hostname='node01.dc1', port=9000),
                1: SimpleNamespace(hostname='node01.dc2', port=9001),
            },
            'tok',
        )
        with patch('cvs.core.agent.mesh._addresses', return_value=[]):
            self.assertEqual(mesh.resolve(['node01.dc2']), {'node01.dc2': 'http://node01.dc2:9001'})

    def test_shared_address_is_refused(self):
        mesh = AgentMesh.install(
            {
                0: SimpleNamespace(hostname='alpha', port=9000),
                1: SimpleNamespace(hostname='beta', port=9001),
            },
            'tok',
        )
        # Both agent names publish the same A record, so an IP-keyed cluster file cannot
        # tell them apart.
        with patch('cvs.core.agent.mesh._addresses', return_value=['10.0.0.1']):
            with self.assertRaisesRegex(ValueError, "match more than one registered agent"):
                mesh.resolve(['10.0.0.1'])


class TestMeshDnsHelpers(unittest.TestCase):
    def test_addresses_skips_loopback(self):
        infos = [
            (socket.AF_INET, 0, 0, '', ('127.0.0.1', 0)),
            (socket.AF_INET, 0, 0, '', ('10.0.0.1', 0)),
        ]
        with patch('cvs.core.agent.mesh.socket.getaddrinfo', return_value=infos):
            self.assertEqual(_addresses('node01'), ['10.0.0.1'])

    def test_addresses_normalizes_ipv4_mapped_loopback(self):
        infos = [(socket.AF_INET6, 0, 0, '', ('::ffff:127.0.0.1', 0, 0, 0))]
        with patch('cvs.core.agent.mesh.socket.getaddrinfo', return_value=infos):
            self.assertEqual(_addresses('node01'), [])

    def test_addresses_keeps_every_non_loopback_record(self):
        infos = [
            (socket.AF_INET, 0, 0, '', ('10.1.0.1', 0)),
            (socket.AF_INET, 0, 0, '', ('10.0.0.1', 0)),
        ]
        with patch('cvs.core.agent.mesh.socket.getaddrinfo', return_value=infos):
            self.assertEqual(_addresses('node01'), ['10.1.0.1', '10.0.0.1'])

    def test_addresses_oserror_returns_empty(self):
        with patch('cvs.core.agent.mesh.socket.getaddrinfo', side_effect=OSError):
            self.assertEqual(_addresses('nope'), [])

    def test_ptr_names_returns_primary_and_aliases(self):
        with patch(
            'cvs.core.agent.mesh.socket.gethostbyaddr',
            return_value=('node01.cluster.local', ['node01'], ['10.0.0.1']),
        ):
            self.assertEqual(_ptr_names('10.0.0.1'), ['node01.cluster.local', 'node01'])

    def test_ptr_names_oserror_returns_empty(self):
        with patch('cvs.core.agent.mesh.socket.gethostbyaddr', side_effect=OSError):
            self.assertEqual(_ptr_names('10.0.0.1'), [])


if __name__ == '__main__':
    unittest.main()
