"""Unit tests for the per-node RDMA NIC inventory.

The NIC/GPU layout used here mirrors a real 4-GPU, 8-NIC accelerator node:
each GPU sits alone in its own PCI domain and the backend NICs share that
domain, while one management NIC sits in a domain with no GPU. Node names are
synthetic.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.topology import nic_inventory

GPU_BDFS = ['0001:01:00.0', '0002:01:00.0', '0003:01:00.0', '0004:01:00.0']

# nic name -> bdf, in the layout the hardware reports.
NIC_LAYOUT = [
    ('ionic_0', '0000:04:00.0'),
    ('ionic_1', '0001:44:00.0'),
    ('ionic_2', '0001:48:00.0'),
    ('ionic_3', '0002:44:00.0'),
    ('ionic_4', '0003:44:00.0'),
    ('ionic_5', '0003:48:00.0'),
    ('ionic_6', '0004:44:00.0'),
    ('ionic_7', '0004:48:00.0'),
]


def _probe_output(active_nics=None, banner=True):
    active = NIC_LAYOUT if active_nics is None else [nic for nic in NIC_LAYOUT if nic[0] in active_nics]
    active_names = {name for name, _ in active}
    lines = []
    if banner:
        # A login shell prints a banner before the command's own output.
        lines += ['', '  ###  WELCOME BANNER  ###', 'Last login: never']
    for name, bdf in NIC_LAYOUT:
        if name in active_names:
            lines.append(f'{nic_inventory.PROBE_MARKER}|{name}|{bdf}|4: ACTIVE|400 Gb/sec (4X NDR)')
        else:
            lines.append(f'{nic_inventory.PROBE_MARKER}|{name}|{bdf}|1: DOWN|2.5 Gb/sec (1X SDR)')
    return '\n'.join(lines)


class TestParseNicProbe(unittest.TestCase):
    def test_banner_and_other_output_is_ignored(self):
        nics = nic_inventory.parse_nic_probe(_probe_output())
        self.assertEqual([nic['nic'] for nic in nics], [name for name, _ in NIC_LAYOUT])

    def test_link_state_is_interpreted(self):
        nics = nic_inventory.parse_nic_probe(_probe_output(active_nics={'ionic_0'}))
        by_name = {nic['nic']: nic for nic in nics}
        self.assertTrue(by_name['ionic_0']['active'])
        self.assertFalse(by_name['ionic_1']['active'])

    def test_malformed_and_empty_lines_are_skipped(self):
        text = '\n'.join(
            [
                f'{nic_inventory.PROBE_MARKER}|ionic_0',
                f'{nic_inventory.PROBE_MARKER}||0000:04:00.0|4: ACTIVE|400',
                f'{nic_inventory.PROBE_MARKER}|ionic_1|0001:44:00.0|4: ACTIVE|400',
                'unrelated|line|with|pipes|here',
            ]
        )
        nics = nic_inventory.parse_nic_probe(text)
        self.assertEqual([nic['nic'] for nic in nics], ['ionic_1'])

    def test_non_string_output_yields_nothing(self):
        self.assertEqual(nic_inventory.parse_nic_probe(None), [])


class TestSelectBackendNics(unittest.TestCase):
    def test_each_nic_maps_to_the_gpu_in_its_own_pci_domain(self):
        nics = nic_inventory.parse_nic_probe(_probe_output())
        selected, _ = nic_inventory.select_backend_nics(nics, GPU_BDFS)

        self.assertEqual(
            {nic['nic']: nic['gpu'] for nic in selected},
            {'ionic_1': 0, 'ionic_2': 0, 'ionic_3': 1, 'ionic_4': 2, 'ionic_5': 2, 'ionic_6': 3, 'ionic_7': 3},
        )

    def test_management_nic_without_a_local_gpu_is_excluded_with_a_reason(self):
        nics = nic_inventory.parse_nic_probe(_probe_output())
        selected, excluded = nic_inventory.select_backend_nics(nics, GPU_BDFS)

        self.assertNotIn('ionic_0', [nic['nic'] for nic in selected])
        reasons = {nic['nic']: nic['reason'] for nic in excluded}
        self.assertEqual(reasons['ionic_0'], nic_inventory.EXCLUDED_NO_LOCAL_GPU)

    def test_down_link_is_excluded_and_reported_not_dropped(self):
        nics = nic_inventory.parse_nic_probe(_probe_output(active_nics={'ionic_1', 'ionic_2'}))
        selected, excluded = nic_inventory.select_backend_nics(nics, GPU_BDFS)

        self.assertEqual([nic['nic'] for nic in selected], ['ionic_1', 'ionic_2'])
        down = {nic['nic'] for nic in excluded if nic['reason'] == nic_inventory.EXCLUDED_LINK_DOWN}
        self.assertEqual(down, {'ionic_3', 'ionic_4', 'ionic_5', 'ionic_6', 'ionic_7'})

    def test_node_without_gpus_selects_no_backend_nics(self):
        nics = nic_inventory.parse_nic_probe(_probe_output())
        selected, excluded = nic_inventory.select_backend_nics(nics, [])
        self.assertEqual(selected, [])
        self.assertEqual(len(excluded), len(NIC_LAYOUT))


class TestCollectNicInventory(unittest.TestCase):
    def _orch(self, exec_result):
        phdl = MagicMock()
        phdl.exec.return_value = exec_result
        orch = MagicMock()
        orch.all = phdl
        return orch

    def test_collects_per_node_and_reports_a_node_with_no_devices(self):
        orch = self._orch({'nodeA': _probe_output(), 'nodeB': 'no infiniband here'})
        nics_by_node, failures = nic_inventory.collect_nic_inventory(orch)

        self.assertEqual(len(nics_by_node['nodeA']), len(NIC_LAYOUT))
        self.assertNotIn('nodeB', nics_by_node)
        self.assertIn('nodeB', failures)

    def test_probe_lines_are_marked_so_banners_cannot_be_mistaken_for_devices(self):
        orch = self._orch({'nodeA': _probe_output()})
        nic_inventory.collect_nic_inventory(orch)
        cmd = orch.all.exec.call_args[0][0]
        self.assertIn(nic_inventory.PROBE_MARKER, cmd)


if __name__ == '__main__':
    unittest.main()
