"""Unit tests for the ROCm version check preflight module."""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.version_check import RocmVersionCheck


def _make_amd_smi_json(version):
    return json.dumps([{'rocm_version': version}])


class TestRocmVersionCheck(unittest.TestCase):
    def _make_checker(self, phdl, expected_version='7.2.0'):
        return RocmVersionCheck(phdl, expected_version)

    def test_all_nodes_pass(self):
        phdl = MagicMock()
        phdl.all.exec.return_value = {
            'node1': _make_amd_smi_json('7.2.0'),
            'node2': _make_amd_smi_json('7.2.0'),
        }
        checker = self._make_checker(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node2']['status'], 'PASS')
        self.assertEqual(results['node1']['errors'], [])

    def test_version_mismatch_fail(self):
        phdl = MagicMock()
        phdl.all.exec.return_value = {
            'node1': _make_amd_smi_json('7.2.0'),
            'node2': _make_amd_smi_json('6.3.0'),
        }
        checker = self._make_checker(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node2']['status'], 'FAIL')
        self.assertIn('mismatch', results['node2']['errors'][0])

    def test_not_found_output_fail(self):
        phdl = MagicMock()
        phdl.all.exec.return_value = {'node1': 'NOT_FOUND'}
        checker = self._make_checker(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertEqual(results['node1']['detected_version'], 'NOT_FOUND')

    def test_pruned_node_abort_output_skipped(self):
        """Nodes that were pruned by test_node_reachability emit ABORT output;
        they must be reported SKIPPED rather than FAIL."""
        phdl = MagicMock()
        phdl.all.exec.return_value = {
            'node1': _make_amd_smi_json('7.2.0'),
            'node2': '\nABORT: Host Unreachable Error',
        }
        checker = self._make_checker(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node2']['status'], 'SKIPPED')
        self.assertEqual(results['node2']['errors'], [])

    def test_pruned_node_does_not_count_as_fail(self):
        """ABORT/unreachable nodes must not inflate the FAIL count."""
        phdl = MagicMock()
        phdl.all.exec.return_value = {
            'node1': _make_amd_smi_json('7.2.0'),
            'node2': _make_amd_smi_json('7.2.0'),
            'node3': 'ABORT: Host Unreachable Error',
        }
        checker = self._make_checker(phdl)
        results = checker.run()

        failed = [n for n, r in results.items() if r['status'] == 'FAIL']
        self.assertEqual(failed, [])
        self.assertEqual(results['node3']['status'], 'SKIPPED')

    def test_malformed_json_output_fail(self):
        phdl = MagicMock()
        phdl.all.exec.return_value = {'node1': 'not valid json'}
        checker = self._make_checker(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertEqual(results['node1']['detected_version'], 'NOT_FOUND')

    def test_empty_output_fail(self):
        phdl = MagicMock()
        phdl.all.exec.return_value = {'node1': ''}
        checker = self._make_checker(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')

    def test_no_nodes_returns_empty(self):
        phdl = MagicMock()
        phdl.all.exec.return_value = {}
        checker = self._make_checker(phdl)
        results = checker.run()

        self.assertEqual(results, {})

    def test_ssh_motd_banner_prefix_is_stripped(self):
        """SSH MOTD/auth banners prepended before JSON must not break parsing."""
        banner = "WARNING: Conductor Auth\nThis is a restricted system.\n"
        json_payload = _make_amd_smi_json('7.2.0')
        phdl = MagicMock()
        phdl.all.exec.return_value = {'node1': banner + json_payload}
        checker = self._make_checker(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['detected_version'], '7.2.0')
        self.assertEqual(results['node1']['errors'], [])

    def test_banner_only_output_returns_not_found(self):
        """Output that contains a banner but no JSON must return NOT_FOUND gracefully."""
        phdl = MagicMock()
        phdl.all.exec.return_value = {'node1': 'WARNING: Conductor Auth\nNo JSON here'}
        checker = self._make_checker(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertEqual(results['node1']['detected_version'], 'NOT_FOUND')


if __name__ == '__main__':
    unittest.main()
