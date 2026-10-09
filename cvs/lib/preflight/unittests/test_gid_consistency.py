"""Tests for preflight GID consistency checks."""

import unittest
from unittest.mock import MagicMock

from cvs.lib.preflight.gid_consistency import GidConsistencyCheck


class TestGidConsistency(unittest.TestCase):
    def setUp(self):
        self.orch = MagicMock()
        self.devices = ['nic0', 'nic1']
        self.good = (
            'DEVICE:nic0\nLINK_LAYER:Ethernet\nGID:0000:0000:0000:0000:0000:ffff:c000:0201\nGID_TYPE:RoCE v2\n'
            'DEVICE:nic1\nLINK_LAYER:Ethernet\nGID:0000:0000:0000:0000:0000:ffff:c000:0202\nGID_TYPE:RoCE v2\n'
        )

    def run_check(self, output, expected_type='RoCE v2'):
        self.orch.all.exec.return_value = {'nodeA': output}
        check = GidConsistencyCheck(self.orch, '3', self.devices, expected_gid_type=expected_type)
        return check.run()['nodeA']

    def test_all_interfaces_valid_and_command_reads_type(self):
        result = self.run_check(self.good)
        self.assertEqual(result['status'], 'PASS')
        for interface in result['interfaces'].values():
            self.assertEqual(interface['status'], 'OK')
            self.assertEqual(interface['gid_type'], 'RoCE v2')
            self.assertEqual(interface['gid_index'], '3')
        self.assertIn('gid_attrs/types/3', self.orch.all.exec.call_args.args[0])

    def test_wrong_type_records_error(self):
        result = self.run_check(self.good.replace('GID_TYPE:RoCE v2\n', 'GID_TYPE:IB/RoCE v1\n', 1))
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['interfaces']['nic0']['status'], 'WRONG_TYPE')
        for fragment in ('nic0', 'index 3', 'IB/RoCE v1'):
            self.assertIn(fragment, result['interfaces']['nic0']['error'])
        self.assertEqual(len(result['errors']), 1)

    def test_absent_interface_and_unreported_interface(self):
        missing = self.run_check(
            self.good.replace(
                'DEVICE:nic1\nLINK_LAYER:Ethernet\nGID:0000:0000:0000:0000:0000:ffff:c000:0202\nGID_TYPE:RoCE v2\n',
                'DEVICE:nic1\nDEVICE_MISSING\n',
            )
        )
        self.assertEqual(missing['interfaces']['nic1']['status'], 'DEVICE_MISSING')
        self.assertIn('error', missing['interfaces']['nic1'])
        unreported = self.run_check(self.good.split('DEVICE:nic1')[0])
        self.assertEqual(unreported['status'], 'FAIL')
        self.assertEqual(unreported['interfaces']['nic1']['status'], 'NOT_REPORTED')

    def test_zero_gid(self):
        result = self.run_check(self.good.replace('0000:ffff:c000:0201', '0000:0000:0000:0000'))
        self.assertEqual(result['interfaces']['nic0']['status'], 'EMPTY')
        self.assertTrue(result['interfaces']['nic0']['error'].startswith('GID index 3 is empty on'))

    def test_link_local_warns(self):
        result = self.run_check(
            self.good.replace('0000:0000:0000:0000:0000:ffff:c000:0201', 'fe80:0000:0000:0000:0000:0000:0000:0001')
        )
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(len(result['warnings']), 1)

    def test_any_type_accepts_v1(self):
        result = self.run_check(self.good.replace('GID_TYPE:RoCE v2', 'GID_TYPE:IB/RoCE v1'), 'any')
        self.assertEqual(result['status'], 'PASS')
