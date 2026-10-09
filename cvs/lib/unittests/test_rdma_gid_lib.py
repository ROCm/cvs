"""Tests for shared RDMA GID probing and validation."""

import unittest

from cvs.lib import rdma_gid_lib as gid


IPV4_GID = '0000:0000:0000:0000:0000:ffff:c000:0201'
LINK_GID = 'fe80:0000:0000:0000:0000:0000:0000:0001'


class TestGidProbe(unittest.TestCase):
    def test_index_modes_and_type_matching(self):
        self.assertEqual(gid.normalize_gid_index(' 3 '), '3')
        for value in (None, '', ' AUTO '):
            self.assertTrue(gid.is_auto_gid_index(value))
        self.assertFalse(gid.is_auto_gid_index('3'))
        self.assertTrue(gid.gid_type_matches('RoCE v2', ' roce  V2 '))
        self.assertTrue(gid.gid_type_matches('IB/RoCE v1', 'any'))
        self.assertFalse(gid.gid_type_matches('IB/RoCE v1', 'RoCE v2'))

    def test_probe_command_targets_devices_and_types(self):
        command = gid.build_gid_probe_cmd(3, ['rocep28s0'])
        for fragment in ('/sys/class/infiniband/rocep28s0', 'gids/3', 'gid_attrs/types/3', 'link_layer'):
            self.assertIn(fragment, command)
        self.assertIn('/sys/class/infiniband/*', gid.build_gid_probe_cmd('3'))

    def test_probe_rejects_invalid_indices(self):
        for index in ('auto', '3; rm -rf /', -1):
            with self.subTest(index=index), self.assertRaises(ValueError):
                gid.build_gid_probe_cmd(index)

    def test_parse_probe_output(self):
        output = (
            'noise\nDEVICE:rocep0\nLINK_LAYER:Ethernet\nGID:' + IPV4_GID + '\nGID_TYPE:RoCE v2\n'
            'DEVICE:rocep1\nGID_MISSING\nDEVICE:rocep2\nDEVICE_MISSING\n'
        )
        entries = gid.parse_gid_probe_output(output)
        self.assertEqual(entries['rocep0']['gid'], IPV4_GID)
        self.assertEqual(entries['rocep0']['gid_type'], 'RoCE v2')
        self.assertIsNone(entries['rocep1']['gid'])
        self.assertFalse(entries['rocep2']['present'])
        self.assertEqual(gid.parse_gid_probe_output(''), {})
        self.assertEqual(gid.parse_gid_probe_output(None), {})

    def test_check_entry_statuses(self):
        entry = {'present': True, 'link_layer': 'Ethernet', 'gid': IPV4_GID, 'gid_type': 'RoCE v2'}
        self.assertEqual(gid.check_gid_entry('rocep0', entry, 3), (gid.GID_OK, ''))
        wrong = dict(entry, gid_type='IB/RoCE v1')
        status, message = gid.check_gid_entry('rocep0', wrong, 3)
        self.assertEqual(status, gid.GID_WRONG_TYPE)
        for fragment in ('rocep0', 'index 3', 'IB/RoCE v1', 'RoCE v2'):
            self.assertIn(fragment, message)
        self.assertIn('unknown', gid.check_gid_entry('rocep0', dict(entry, gid_type=''), 3)[1])
        self.assertEqual(gid.check_gid_entry('rocep0', dict(entry, gid=gid.ZERO_GID), 3)[0], gid.GID_EMPTY)
        self.assertEqual(gid.check_gid_entry('rocep0', dict(entry, gid=None), 3)[0], gid.GID_MISSING)
        self.assertEqual(gid.check_gid_entry('rocep0', dict(entry, present=False), 3)[0], gid.GID_DEVICE_MISSING)
        self.assertEqual(gid.check_gid_entry('rocep0', None, 3)[0], gid.GID_NOT_REPORTED)
        self.assertEqual(gid.check_gid_entry('ib0', dict(wrong, link_layer='InfiniBand'), 3)[0], gid.GID_OK)
        self.assertEqual(gid.check_gid_entry('rocep0', wrong, 3, 'any')[0], gid.GID_OK)
        self.assertEqual(gid.check_gid_entry('rocep0', entry, 3, ' roce  V2 ')[0], gid.GID_OK)

    def test_gid_scope(self):
        self.assertEqual(gid.gid_scope(IPV4_GID), 'ipv4-mapped')
        self.assertEqual(gid.gid_scope(LINK_GID), 'link-local')
        self.assertEqual(gid.gid_scope('2001:db8::1'), 'global')

    def test_parse_table_output(self):
        output = f'GIDENT|rocep0|3|{IPV4_GID}|RoCE v2\nGIDENT|rocep0|bad|x|RoCE v2\nGIDENT|broken\n'
        self.assertEqual(gid.parse_gid_table_output(output), {'rocep0': {3: {'gid': IPV4_GID, 'gid_type': 'RoCE v2'}}})
        self.assertIn('GIDENT|', gid.build_gid_table_cmd(['rocep0']))


class TestCommonIndex(unittest.TestCase):
    def setUp(self):
        self.nodes = {'A': ['nic0', 'nic1'], 'B': ['nic0', 'nic1']}
        self.table = {
            0: {'gid': IPV4_GID, 'gid_type': 'IB/RoCE v1'},
            1: {'gid': LINK_GID, 'gid_type': 'RoCE v2'},
            3: {'gid': IPV4_GID, 'gid_type': 'RoCE v2'},
        }
        self.tables = {node: {device: dict(self.table) for device in devices} for node, devices in self.nodes.items()}

    def test_selects_common_ipv4_v2(self):
        self.assertEqual(gid.select_common_gid_index(self.tables, self.nodes), ('3', []))

    def test_selects_lowest_common(self):
        for devices in self.tables.values():
            for table in devices.values():
                table[5] = {'gid': IPV4_GID, 'gid_type': 'RoCE v2'}
        self.assertEqual(gid.select_common_gid_index(self.tables, self.nodes), ('3', []))

    def test_reports_nic_without_candidates(self):
        self.tables['B']['nic1'].pop(3)
        index, errors = gid.select_common_gid_index(self.tables, self.nodes)
        self.assertIsNone(index)
        self.assertIn('Node B NIC nic1', errors[0])

    def test_reports_disjoint_indices(self):
        self.tables['B']['nic0'].pop(3)
        self.tables['B']['nic1'].pop(3)
        for table in self.tables['B'].values():
            table[5] = {'gid': IPV4_GID, 'gid_type': 'RoCE v2'}
        index, errors = gid.select_common_gid_index(self.tables, self.nodes)
        self.assertIsNone(index)
        self.assertIn('on every NIC', errors[0])
