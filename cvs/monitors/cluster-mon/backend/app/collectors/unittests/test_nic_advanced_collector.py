"""
Contract tests for NICAdvancedCollector.
"""

import json
import unittest

from app.unittests.testing import FakeSshManager
from app.collectors.nic_advanced_collector import NICAdvancedCollector


class TestNICAdvancedCollector(unittest.IsolatedAsyncioTestCase):
    async def test_collect_congestion_info_keeps_only_congestion_stats(self):
        rdma_json = json.dumps(
            [
                {
                    "ifname": "rdma0",
                    "port": 1,
                    "rx_pkts": 100,  # not congestion-related -> dropped
                    "req_rx_pkt_seq_err": 5,  # matches 'err'
                    "tx_rdma_ack_timeout": 2,  # matches 'timeout'
                    "rx_rdma_ecn_pkts": 7,  # matches 'ecn'
                }
            ]
        )
        ssh = FakeSshManager(["node1"], command_map={"rdma statistic show --json": {"node1": rdma_json}})

        result = await NICAdvancedCollector().collect_congestion_info(ssh)

        stats = result["node1"]["rdma0/1"]
        self.assertEqual(stats["req_rx_pkt_seq_err"], 5)
        self.assertEqual(stats["tx_rdma_ack_timeout"], 2)
        self.assertEqual(stats["rx_rdma_ecn_pkts"], 7)
        self.assertNotIn("rx_pkts", stats)

    async def test_collect_nic_pcie_info_parses_link_and_gen(self):
        lspci = "\n".join(
            [
                "01:00.0 Ethernet controller: Broadcom Inc. BCM57608",
                "    LnkCap: Port #0, Speed 32GT/s, Width x16",
                "    LnkSta: Speed 32GT/s, Width x16",
            ]
        )
        ssh = FakeSshManager(["node1"], command_map={"lspci -vvv": {"node1": lspci}})

        result = await NICAdvancedCollector().collect_nic_pcie_info(ssh)

        dev = result["node1"]["01:00.0"]
        self.assertEqual(dev["pcie_gen"], "Gen5")
        self.assertEqual(dev["link_speed_cap"], "32GT/s")
        self.assertEqual(dev["link_width_cap"], "x16")
        self.assertEqual(dev["link_speed_current"], "32GT/s")


if __name__ == "__main__":
    unittest.main()
