"""Unit tests for the /etc/hosts consistency preflight check."""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.etc_hosts_consistency import EtcHostsConsistencyCheck


class TestEtcHostsConsistencyCheck(unittest.TestCase):
    def test_all_expected_ips_present_pass(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': "127.0.0.1 localhost\n10.0.0.1 node1\n10.0.0.2 node2\n",
        }
        checker = EtcHostsConsistencyCheck(phdl, expected_ips=['10.0.0.1', '10.0.0.2'])
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['missing_ips'], [])
        self.assertEqual(results['node1']['errors'], [])

    def test_missing_expected_ip_warning(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': "10.0.0.1 node1\n",
        }
        checker = EtcHostsConsistencyCheck(phdl, expected_ips=['10.0.0.1', '10.0.0.2'])
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertEqual(results['node1']['missing_ips'], ['10.0.0.2'])
        self.assertIn('missing entries', results['node1']['errors'][0])

    def test_extra_entries_mismatch_warning(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': "10.0.0.1 node1\n10.0.0.5 wrong-host\n",
        }
        checker = EtcHostsConsistencyCheck(
            phdl,
            expected_ips=['10.0.0.1'],
            extra_entries=[{'hostname': 'gpu-node-01', 'ip': '10.0.0.5'}],
        )
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertEqual(results['node1']['missing_extra_entries'], ['gpu-node-01=10.0.0.5'])

    def test_extra_entries_match_pass(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': "10.0.0.1 node1\n10.0.0.5 gpu-node-01\n",
        }
        checker = EtcHostsConsistencyCheck(
            phdl,
            expected_ips=['10.0.0.1'],
            extra_entries=[{'hostname': 'gpu-node-01', 'ip': '10.0.0.5'}],
        )
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['missing_extra_entries'], [])

    def test_empty_hosts_file_all_missing(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': ''}
        checker = EtcHostsConsistencyCheck(phdl, expected_ips=['10.0.0.1'])
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertEqual(results['node1']['missing_ips'], ['10.0.0.1'])

    def test_comments_and_blank_lines_ignored(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': "# comment line\n\n10.0.0.1 node1 # inline comment\n",
        }
        checker = EtcHostsConsistencyCheck(phdl, expected_ips=['10.0.0.1'])
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')

    def test_hostname_expected_address_present_pass(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': "10.0.212.208 smci300x-ccs-aus-e04-19\n10.0.212.203 smci300x-ccs-aus-e07-03\n",
        }
        checker = EtcHostsConsistencyCheck(
            phdl,
            expected_ips=['smci300x-ccs-aus-e04-19', 'smci300x-ccs-aus-e07-03'],
        )
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['missing_ips'], [])

    def test_hostname_expected_address_missing_warning(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': "10.0.212.208 smci300x-ccs-aus-e04-19\n",
        }
        checker = EtcHostsConsistencyCheck(
            phdl,
            expected_ips=['smci300x-ccs-aus-e04-19', 'smci300x-ccs-aus-e07-03'],
        )
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertEqual(results['node1']['missing_ips'], ['smci300x-ccs-aus-e07-03'])


if __name__ == '__main__':
    unittest.main()
