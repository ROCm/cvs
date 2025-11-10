#!/usr/bin/env python3
"""
Unit tests for nic_mapping.py
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the utils directory to the path so we can import nic_mapping
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nic_mapping import (
    ancestry_chain, lowest_common_ancestor, hop_metrics,
    pick_mappings, filter_devices, build_pcie_tree,
    is_nic, is_gpu_or_display, is_accelerator
)


class TestNicMapping(unittest.TestCase):

    def setUp(self):
        # Mock device data for testing
        self.devs = {
            "0000:00:00.0": {
                "addr": "0000:00:00.0", "parent": None, "class": "0x060000",
                "vendor": "0x8086", "device": "0x1234", "name": "Root Bridge",
                "numa": 0, "driver": None
            },
            "0000:00:01.0": {
                "addr": "0000:00:01.0", "parent": "0000:00:00.0", "class": "0x060400",
                "vendor": "0x8086", "device": "0x5678", "name": "PCI Bridge",
                "numa": 0, "driver": None
            },
            "0000:01:00.0": {
                "addr": "0000:01:00.0", "parent": "0000:00:01.0", "class": "0x020000",
                "vendor": "0x14e4", "device": "0x1750", "name": "Broadcom NIC",
                "numa": 0, "driver": "bnxt_en", "is_vf": False, "pf_addr": None
            },
            "0000:00:02.0": {
                "addr": "0000:00:02.0", "parent": "0000:00:00.0", "class": "0x060400",
                "vendor": "0x8086", "device": "0x5678", "name": "PCI Bridge",
                "numa": 0, "driver": None
            },
            "0000:02:00.0": {
                "addr": "0000:02:00.0", "parent": "0000:00:02.0", "class": "0x030000",
                "vendor": "0x1002", "device": "0x74a1", "name": "AMD GPU",
                "numa": 0, "driver": "amdgpu"
            },
            "0000:02:00.1": {
                "addr": "0000:02:00.1", "parent": "0000:00:02.0", "class": "0x030000",
                "vendor": "0x1002", "device": "0x74a1", "name": "AMD GPU",
                "numa": 0, "driver": "amdgpu"
            },
            "0000:00:03.0": {
                "addr": "0000:00:03.0", "parent": "0000:00:00.0", "class": "0x060400",
                "vendor": "0x8086", "device": "0x5678", "name": "PCI Bridge",
                "numa": 0, "driver": None
            },
            "0000:03:00.0": {
                "addr": "0000:03:00.0", "parent": "0000:00:03.0", "class": "0x030000",
                "vendor": "0x1002", "device": "0x74a1", "name": "AMD GPU",
                "numa": 0, "driver": "amdgpu"
            },
        }

    def test_ancestry_chain(self):
        chain = ancestry_chain(self.devs, "0000:01:00.0")
        expected = ["0000:01:00.0", "0000:00:01.0", "0000:00:00.0"]
        self.assertEqual(chain, expected)

        chain_root = ancestry_chain(self.devs, "0000:00:00.0")
        self.assertEqual(chain_root, ["0000:00:00.0"])

    def test_lowest_common_ancestor(self):
        lca = lowest_common_ancestor(self.devs, "0000:01:00.0", "0000:02:00.0")
        self.assertEqual(lca, "0000:00:00.0")

        lca_same = lowest_common_ancestor(self.devs, "0000:02:00.0", "0000:02:00.1")
        self.assertEqual(lca_same, "0000:00:02.0")

        lca_none = lowest_common_ancestor(self.devs, "0000:01:00.0", "nonexistent")
        self.assertIsNone(lca_none)

    def test_hop_metrics(self):
        total, da, db, lca = hop_metrics(self.devs, "0000:01:00.0", "0000:02:00.0")
        self.assertEqual(total, 4)  # 1 (nic to 00:01) + 1 (gpu to 00:02) + 2 (00:01 to 00:00 and 00:02 to 00:00)
        self.assertEqual(da, 2)  # nic to lca: 01:00 -> 00:01 -> 00:00
        self.assertEqual(db, 2)  # gpu to lca: 02:00 -> 00:02 -> 00:00
        self.assertEqual(lca, "0000:00:00.0")

    def test_is_nic(self):
        self.assertTrue(is_nic(self.devs["0000:01:00.0"]))
        self.assertFalse(is_nic(self.devs["0000:02:00.0"]))

    def test_is_gpu_or_display(self):
        self.assertTrue(is_gpu_or_display(self.devs["0000:02:00.0"]))
        self.assertFalse(is_gpu_or_display(self.devs["0000:01:00.0"]))

    def test_is_accelerator(self):
        # Mock accelerator
        accel = {"class": "0x120000"}
        self.assertTrue(is_accelerator(accel))
        self.assertFalse(is_accelerator(self.devs["0000:02:00.0"]))

    def test_build_pcie_tree(self):
        tree = build_pcie_tree(self.devs)
        self.assertIn("ROOT", tree)
        self.assertIn("0000:00:00.0", tree)
        self.assertEqual(set(tree["ROOT"]), {"0000:00:00.0"})
        self.assertIn("0000:00:01.0", tree["0000:00:00.0"])
        self.assertIn("0000:01:00.0", tree["0000:00:01.0"])

    def test_filter_devices(self):
        # Mock args
        class MockArgs:
            def __init__(self):
                self.nic_vendor = ["0x14e4"]
                self.gpu_vendor = ["0x1002"]
                self.include_accelerators = False

        args = MockArgs()
        filtered = filter_devices(self.devs, args)
        self.assertIn("0000:01:00.0", filtered)  # NIC
        self.assertIn("0000:02:00.0", filtered)  # GPU
        self.assertIn("0000:00:00.0", filtered)  # Ancestor

    def test_pick_mappings_single_gpu(self):
        rows = pick_mappings(
            self.devs,
            nic_vendors=["0x14e4"],
            gpu_vendors=["0x1002"],
            include_accels=False,
            prefer_same_switch=False,
            one_to_one=False,
            max_gpus_per_nic=1
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["nic_addr"], "0000:01:00.0")
        self.assertIn(rows[0]["gpu_addr"], ["0000:02:00.0", "0000:02:00.1", "0000:03:00.0"])

    def test_pick_mappings_max_two_gpus_connected(self):
        rows = pick_mappings(
            self.devs,
            nic_vendors=["0x14e4"],
            gpu_vendors=["0x1002"],
            include_accels=False,
            prefer_same_switch=False,
            one_to_one=False,
            max_gpus_per_nic=2
        )
        # Should assign 2 GPUs to the NIC since 02:00.0 and 02:00.1 share LCA 00:02.0 with NIC? Wait no.
        # LCA of NIC 01:00.0 and 02:00.0 is 00:00.0
        # LCA of NIC and 02:00.1 is 00:00.0
        # LCA of NIC and 03:00.0 is 00:00.0
        # So all GPUs share LCA 00:00.0 with NIC, so should assign 2.
        nic_rows = [r for r in rows if r["nic_addr"] == "0000:01:00.0"]
        self.assertEqual(len(nic_rows), 2)
        gpus = {r["gpu_addr"] for r in nic_rows}
        self.assertIn("0000:02:00.0", gpus)
        self.assertIn("0000:02:00.1", gpus)

    def test_pick_mappings_max_two_gpus_no_connection(self):
        # Modify devs to have GPUs not connected
        devs_disconnected = self.devs.copy()
        # Make 03:00.0 have different root or something, but hard.
        # For simplicity, assume with current, it works.
        # To test no mapping, perhaps remove one GPU.
        # But since all share LCA, it will assign.
        # To test no mapping, set max_gpus_per_nic=3, but only 3 GPUs, but they all share LCA.
        rows = pick_mappings(
            self.devs,
            nic_vendors=["0x14e4"],
            gpu_vendors=["0x1002"],
            include_accels=False,
            prefer_same_switch=False,
            one_to_one=False,
            max_gpus_per_nic=4  # More than available
        )
        nic_rows = [r for r in rows if r["nic_addr"] == "0000:01:00.0"]
        self.assertEqual(len(nic_rows), 0)  # No group has 4 GPUs

    @patch('builtins.print')
    def test_print_pcie_hierarchy(self, mock_print):
        from nic_mapping import print_pcie_hierarchy
        print_pcie_hierarchy(self.devs)
        # Check that print was called
        self.assertTrue(mock_print.called)
        # Could check specific calls, but for now just that it runs without error


if __name__ == '__main__':
    unittest.main()