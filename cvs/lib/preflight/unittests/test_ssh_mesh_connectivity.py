"""Unit tests for the SSH mesh connectivity preflight check."""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.ssh_mesh_connectivity import SshMeshConnectivityCheck


class TestSshMeshConnectivityCheck(unittest.TestCase):
    def _make_checker(self, phdl, peer_map, ssh_timeout_sec=10):
        return SshMeshConnectivityCheck(phdl, peer_map, ssh_timeout_sec=ssh_timeout_sec)

    def test_all_peers_reachable_pass(self):
        phdl = MagicMock()
        phdl.reachable_hosts = ['node1', 'node2', 'node3']
        peer_map = {'node1': '10.0.0.1', 'node2': '10.0.0.2', 'node3': '10.0.0.3'}
        phdl.exec_cmd_list.return_value = {
            'node1': "SSH_MESH_TOTAL:2\nSSH_MESH_PASS:2\nSSH_MESH_FAIL:0\nSSH_MESH_FAILED_PEERS:",
            'node2': "SSH_MESH_TOTAL:2\nSSH_MESH_PASS:2\nSSH_MESH_FAIL:0\nSSH_MESH_FAILED_PEERS:",
            'node3': "SSH_MESH_TOTAL:2\nSSH_MESH_PASS:2\nSSH_MESH_FAIL:0\nSSH_MESH_FAILED_PEERS:",
        }
        checker = self._make_checker(phdl, peer_map)
        results = checker.run()

        for node, result in results.items():
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(result['failed_peers'], [])
            self.assertEqual(result['errors'], [])

    def test_some_peers_unreachable_warning(self):
        phdl = MagicMock()
        phdl.reachable_hosts = ['node1', 'node2']
        peer_map = {'node1': '10.0.0.1', 'node2': '10.0.0.2'}
        phdl.exec_cmd_list.return_value = {
            'node1': "SSH_MESH_TOTAL:1\nSSH_MESH_PASS:0\nSSH_MESH_FAIL:1\nSSH_MESH_FAILED_PEERS:10.0.0.2",
            'node2': "SSH_MESH_TOTAL:1\nSSH_MESH_PASS:1\nSSH_MESH_FAIL:0\nSSH_MESH_FAILED_PEERS:",
        }
        checker = self._make_checker(phdl, peer_map)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertEqual(results['node1']['failed_peers'], ['10.0.0.2'])
        self.assertIn('SSH mesh failed to 1 peer', results['node1']['errors'][0])
        self.assertEqual(results['node2']['status'], 'PASS')

    def test_malformed_output_treated_as_warning(self):
        phdl = MagicMock()
        phdl.reachable_hosts = ['node1', 'node2']
        peer_map = {'node1': '10.0.0.1', 'node2': '10.0.0.2'}
        phdl.exec_cmd_list.return_value = {
            'node1': "garbage output with no recognizable markers",
            'node2': "",
        }
        checker = self._make_checker(phdl, peer_map)
        results = checker.run()

        # total==0 with no failed_peers parsed still can't be trusted as PASS.
        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertEqual(results['node2']['status'], 'WARNING')

    def test_single_node_cluster_has_no_peers_and_passes(self):
        phdl = MagicMock()
        phdl.reachable_hosts = ['node1']
        peer_map = {'node1': '10.0.0.1'}
        checker = self._make_checker(phdl, peer_map)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['total_peers'], 0)
        phdl.exec_cmd_list.assert_called_once()

    def test_no_reachable_hosts_returns_empty(self):
        phdl = MagicMock()
        phdl.reachable_hosts = []
        checker = self._make_checker(phdl, {})
        results = checker.run()
        self.assertEqual(results, {})
        phdl.exec_cmd_list.assert_not_called()

    def test_exec_cmd_list_called_exactly_once_with_positional_list(self):
        phdl = MagicMock()
        phdl.reachable_hosts = ['node1', 'node2']
        peer_map = {'node1': '10.0.0.1', 'node2': '10.0.0.2'}
        phdl.exec_cmd_list.return_value = {
            'node1': "SSH_MESH_TOTAL:1\nSSH_MESH_PASS:1\nSSH_MESH_FAIL:0\nSSH_MESH_FAILED_PEERS:",
            'node2': "SSH_MESH_TOTAL:1\nSSH_MESH_PASS:1\nSSH_MESH_FAIL:0\nSSH_MESH_FAILED_PEERS:",
        }
        checker = self._make_checker(phdl, peer_map)
        checker.run()

        phdl.exec_cmd_list.assert_called_once()
        (cmd_list,), _ = phdl.exec_cmd_list.call_args
        self.assertIsInstance(cmd_list, list)
        self.assertEqual(len(cmd_list), 2)


if __name__ == '__main__':
    unittest.main()
