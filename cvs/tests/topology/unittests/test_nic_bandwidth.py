"""Unit tests for the NIC bandwidth suite's collection helpers.

Node names and PCI addresses are synthetic; no orchestrator or cluster is used.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.tests.topology import nic_bandwidth

MAPPING = {
    'nodeA': {
        'card0': {'gpu_bdf': '0001:01:00.0', 'rdma_dev': 'nic0'},
        'card10': {'gpu_bdf': '0002:01:00.0', 'rdma_dev': 'nic1'},
        'card2': {'gpu_bdf': '0003:01:00.0'},
    }
}

PROBE = '\n'.join(
    [
        'CVSNIC|nic0|0001:44:00.0|1|4: ACTIVE|400 Gb/sec (4X NDR)',
        'CVSNIC|nic1|0002:44:00.0|1|4: ACTIVE|400 Gb/sec (4X NDR)',
        'CVSNIC|nic9|0000:04:00.0|1|4: ACTIVE|400 Gb/sec (4X NDR)',
    ]
)


def _orch(exec_result):
    orch = MagicMock()
    orch.exec.return_value = {host: {'output': output, 'exit_code': 0} for host, output in exec_result.items()}
    return orch


class TestGpuBdfsByNode(unittest.TestCase):
    def test_gpu_order_follows_the_card_number_not_the_string(self):
        with patch.object(nic_bandwidth, 'get_gpu_nic_mapping_dict', return_value=MAPPING):
            bdfs = nic_bandwidth._gpu_bdfs_by_node(MagicMock())
        self.assertEqual(bdfs['nodeA'], ['0001:01:00.0', '0003:01:00.0', '0002:01:00.0'])

    def test_unmapped_card_without_a_bdf_is_skipped(self):
        mapping = {'nodeA': {'card0': {'rdma_dev': 'nic0'}}}
        with patch.object(nic_bandwidth, 'get_gpu_nic_mapping_dict', return_value=mapping):
            self.assertEqual(nic_bandwidth._gpu_bdfs_by_node(MagicMock())['nodeA'], [])

    def test_no_mapping_is_not_an_error(self):
        with patch.object(nic_bandwidth, 'get_gpu_nic_mapping_dict', return_value=None):
            self.assertEqual(nic_bandwidth._gpu_bdfs_by_node(MagicMock()), {})


SETTINGS = {'tb_binary': 'TransferBench', 'use_sudo': False, 'nic_filter': None, 'transferbench_env': {}}

# TransferBench's own numbering, deliberately not the device-name order.
TB_TABLE = {
    'nodeA': [{'nic': 'nic1', 'index': 0, 'closest_gpus': [7]}, {'nic': 'nic0', 'index': 1, 'closest_gpus': [4]}]
}


class TestCollectInventory(unittest.TestCase):
    def _collect(self, mapping, table=None, failures=None):
        with (
            patch.object(nic_bandwidth, 'get_gpu_nic_mapping_dict', return_value=mapping),
            patch.object(nic_bandwidth.nic_table, 'query_nic_table', return_value=(table or TB_TABLE, failures or {})),
        ):
            return nic_bandwidth._collect_inventory(_orch({'nodeA': PROBE}), SETTINGS)[:2]

    def test_selects_backend_nics_and_records_why_others_were_left_out(self):
        mapping = {'nodeA': {'card0': {'gpu_bdf': '0001:01:00.0'}, 'card1': {'gpu_bdf': '0002:01:00.0'}}}
        inventory, excluded = self._collect(mapping)

        self.assertEqual(sorted(nic['nic'] for nic in inventory['nodeA']), ['nic0', 'nic1'])
        self.assertIn('nic9', [nic['nic'] for nic in excluded['nodeA']])

    def test_addressing_indices_come_from_transferbench_not_from_pci_order(self):
        # PCI affinity decides which NICs are backend NICs; TransferBench's
        # table decides how the transfer file addresses them. nic0 is first by
        # name but index 1 to TransferBench, and its GPU is 4, not 0.
        mapping = {'nodeA': {'card0': {'gpu_bdf': '0001:01:00.0'}, 'card1': {'gpu_bdf': '0002:01:00.0'}}}
        inventory, _ = self._collect(mapping)

        got = {nic['nic']: (nic['index'], nic['gpu']) for nic in inventory['nodeA']}
        self.assertEqual(got, {'nic1': (0, 7), 'nic0': (1, 4)})

    def test_a_nic_transferbench_cannot_address_is_excluded_with_a_reason(self):
        mapping = {'nodeA': {'card0': {'gpu_bdf': '0001:01:00.0'}, 'card1': {'gpu_bdf': '0002:01:00.0'}}}
        table = {'nodeA': [{'nic': 'nic0', 'index': 0, 'closest_gpus': [0]}]}
        inventory, excluded = self._collect(mapping, table=table)

        self.assertEqual([nic['nic'] for nic in inventory['nodeA']], ['nic0'])
        reasons = {nic['nic']: nic.get('reason') for nic in excluded['nodeA']}
        self.assertEqual(reasons.get('nic1'), nic_bandwidth.nic_inventory.EXCLUDED_NOT_IN_TB_TABLE)

    def test_node_with_no_usable_nics_is_absent_from_the_inventory(self):
        inventory, excluded = self._collect({'nodeA': {}})

        self.assertNotIn('nodeA', inventory)
        self.assertEqual(len(excluded['nodeA']), 3)

    def test_an_unavailable_nic_table_is_reported_separately_from_having_no_nics(self):
        # A node with eight healthy NICs whose TransferBench will not run must
        # not read as a node with nothing to measure: the first is a config
        # error to fix, the second is a cluster with no backend fabric.
        mapping = {'nodeA': {'card0': {'gpu_bdf': '0001:01:00.0'}, 'card1': {'gpu_bdf': '0002:01:00.0'}}}
        with (
            patch.object(nic_bandwidth, 'get_gpu_nic_mapping_dict', return_value=mapping),
            patch.object(
                nic_bandwidth.nic_table,
                'query_nic_table',
                return_value=({}, {'nodeA': 'TransferBench exited 127'}),
            ),
        ):
            inventory, _, table_failures = nic_bandwidth._collect_inventory(_orch({'nodeA': PROBE}), SETTINGS)

        self.assertEqual(inventory, {})
        self.assertEqual(table_failures, {'nodeA': 'TransferBench exited 127'})


if __name__ == '__main__':
    unittest.main()
