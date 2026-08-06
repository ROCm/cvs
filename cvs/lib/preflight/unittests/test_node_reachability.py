"""Unit tests for the ping reachability and uptime preflight checks."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.node_reachability import PingReachabilityCheck, UptimeCheck


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stdout=b""):
        self.returncode = returncode
        self.stdout = stdout


class TestPingReachabilityCheck(unittest.TestCase):
    def test_all_nodes_reachable_pass(self):
        phdl = MagicMock()
        node_ip_map = {'node1': '10.0.0.1', 'node2': '10.0.0.2'}
        checker = PingReachabilityCheck(phdl, node_ip_map, count=4, timeout_sec=1)

        with patch(
            'subprocess.run',
            return_value=_FakeCompletedProcess(returncode=0, stdout=b"4 packets transmitted, 4 received"),
        ):
            results = checker.run()

        self.assertEqual(set(results.keys()), {'node1', 'node2'})
        for node, result in results.items():
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(result['errors'], [])

    def test_unreachable_node_fails(self):
        phdl = MagicMock()
        node_ip_map = {'node1': '10.0.0.1'}
        checker = PingReachabilityCheck(phdl, node_ip_map, count=4, timeout_sec=1)

        with patch('subprocess.run', return_value=_FakeCompletedProcess(returncode=1, stdout=b"100% packet loss")):
            results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('ping to 10.0.0.1 failed', results['node1']['errors'][0])

    def test_ping_subprocess_timeout_treated_as_unreachable(self):
        import subprocess

        phdl = MagicMock()
        node_ip_map = {'node1': '10.0.0.1'}
        checker = PingReachabilityCheck(phdl, node_ip_map, count=4, timeout_sec=1)

        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired(cmd='ping', timeout=5)):
            results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('timed out', results['node1']['errors'][0])

    def test_ping_subprocess_missing_binary_treated_as_unreachable(self):
        phdl = MagicMock()
        node_ip_map = {'node1': '10.0.0.1'}
        checker = PingReachabilityCheck(phdl, node_ip_map, count=4, timeout_sec=1)

        with patch('subprocess.run', side_effect=OSError("no such file")):
            results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('failed to start', results['node1']['errors'][0])

    def test_empty_node_ip_map_returns_empty_results(self):
        phdl = MagicMock()
        checker = PingReachabilityCheck(phdl, {})
        results = checker.run()
        self.assertEqual(results, {})


class TestUptimeCheck(unittest.TestCase):
    def test_uptime_collected_pass(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': ' 10:00:00 up 5 days,  2:00,  1 user,  load average: 0.10, 0.05, 0.01',
            'node2': ' 10:00:00 up 1 day,   1:00,  1 user,  load average: 0.00, 0.00, 0.00',
        }
        checker = UptimeCheck(phdl)
        results = checker.run()

        self.assertEqual(set(results.keys()), {'node1', 'node2'})
        for node, result in results.items():
            self.assertEqual(result['status'], 'PASS')
            self.assertTrue(result['uptime'])
            self.assertEqual(result['errors'], [])

    def test_empty_output_fails(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': ''}
        checker = UptimeCheck(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('no output', results['node1']['errors'][0])

    def test_none_output_fails(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': None}
        checker = UptimeCheck(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
