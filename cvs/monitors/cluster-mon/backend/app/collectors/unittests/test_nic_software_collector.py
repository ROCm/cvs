"""
Contract tests for NICSoftwareCollector.
"""

import json
import unittest

from app.unittests.testing import FakeSshManager
from app.collectors.nic_software_collector import NICSoftwareCollector


class TestNICSoftwareCollector(unittest.IsolatedAsyncioTestCase):
    async def test_collect_rdma_statistics_detailed_parses_counters(self):
        rdma_json = json.dumps([{"ifname": "rdma0", "port": 1, "ifindex": 5, "rx_pkts": 123, "tx_pkts": 45}])
        ssh = FakeSshManager(["node1"], command_map={"rdma statistic show --json": {"node1": rdma_json}})

        result = await NICSoftwareCollector().collect_rdma_statistics_detailed(ssh)

        # ifname/port/ifindex are metadata and must be excluded; only int stats kept.
        self.assertEqual(result["node1"]["rdma0/1"], {"rx_pkts": 123, "tx_pkts": 45})

    async def test_collect_rdma_statistics_empty_list(self):
        ssh = FakeSshManager(["node1"], command_map={"rdma statistic show --json": {"node1": "[]"}})
        result = await NICSoftwareCollector().collect_rdma_statistics_detailed(ssh)
        self.assertEqual(result["node1"], {})

    async def test_collect_pci_device_info_parses_devices(self):
        lspci = "01:00.0 Ethernet controller [0200]: Mellanox Technologies MT2910 Family\n"
        ssh = FakeSshManager(["node1"], command_map={"lspci -nn": {"node1": lspci}})

        result = await NICSoftwareCollector().collect_pci_device_info(ssh)

        devices = result["node1"]["devices"]
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]["pci_address"], "01:00.0")
        self.assertIn("Ethernet controller", devices[0]["description"])

    async def test_pci_device_info_error_marker(self):
        ssh = FakeSshManager(["node1"], command_map={"lspci -nn": {"node1": "ABORT: Host Unreachable Error"}})
        result = await NICSoftwareCollector().collect_pci_device_info(ssh)
        self.assertEqual(result["node1"], {"error": "ABORT: Host Unreachable Error"})


if __name__ == "__main__":
    unittest.main()
