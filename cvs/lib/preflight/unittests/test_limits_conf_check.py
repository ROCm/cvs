"""Unit tests for the /etc/security/limits.conf preflight check."""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.limits_conf_check import LimitsConfCheck


REQUIRED_LINES = [
    "* soft memlock unlimited",
    "* hard memlock unlimited",
]


class TestLimitsConfCheck(unittest.TestCase):
    def test_all_required_lines_present_pass(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': "* soft memlock unlimited\n* hard memlock unlimited\n",
        }
        checker = LimitsConfCheck(phdl, REQUIRED_LINES)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['missing_lines'], [])
        self.assertEqual(results['node1']['errors'], [])

    def test_whitespace_insensitive_match_pass(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': "*   soft   memlock   unlimited\n*\thard\tmemlock\tunlimited\n",
        }
        checker = LimitsConfCheck(phdl, REQUIRED_LINES)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')

    def test_missing_line_fails(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': "* soft memlock unlimited\n",
        }
        checker = LimitsConfCheck(phdl, REQUIRED_LINES)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertEqual(results['node1']['missing_lines'], ["* hard memlock unlimited"])
        self.assertIn('missing 1 required line', results['node1']['errors'][0])

    def test_empty_file_all_missing_fails(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': ''}
        checker = LimitsConfCheck(phdl, REQUIRED_LINES)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertEqual(len(results['node1']['missing_lines']), 2)

    def test_empty_required_lines_always_pass(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': ''}
        checker = LimitsConfCheck(phdl, [])
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')


if __name__ == '__main__':
    unittest.main()
