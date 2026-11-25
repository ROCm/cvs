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
        'CVSNIC|nic0|0001:44:00.0|4: ACTIVE|400 Gb/sec (4X NDR)',
        'CVSNIC|nic1|0002:44:00.0|4: ACTIVE|400 Gb/sec (4X NDR)',
        'CVSNIC|nic9|0000:04:00.0|4: ACTIVE|400 Gb/sec (4X NDR)',
    ]
)


def _orch(exec_result):
    phdl = MagicMock()
    phdl.exec.return_value = exec_result
    orch = MagicMock()
    orch.all = phdl
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


class TestCollectInventory(unittest.TestCase):
    def test_selects_backend_nics_and_records_why_others_were_left_out(self):
        mapping = {'nodeA': {'card0': {'gpu_bdf': '0001:01:00.0'}, 'card1': {'gpu_bdf': '0002:01:00.0'}}}
        with patch.object(nic_bandwidth, 'get_gpu_nic_mapping_dict', return_value=mapping):
            inventory, excluded = nic_bandwidth._collect_inventory(_orch({'nodeA': PROBE}))

        self.assertEqual([nic['nic'] for nic in inventory['nodeA']], ['nic0', 'nic1'])
        self.assertEqual([nic['nic'] for nic in excluded['nodeA']], ['nic9'])

    def test_node_with_no_usable_nics_is_absent_from_the_inventory(self):
        with patch.object(nic_bandwidth, 'get_gpu_nic_mapping_dict', return_value={'nodeA': {}}):
            inventory, excluded = nic_bandwidth._collect_inventory(_orch({'nodeA': PROBE}))

        self.assertNotIn('nodeA', inventory)
        self.assertEqual(len(excluded['nodeA']), 3)


if __name__ == '__main__':
    unittest.main()
