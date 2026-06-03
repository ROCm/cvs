"""
Contract tests for NICMetricsCollector (synchronous ssh_manager.exec path).
"""

import json
import unittest

from app.unittests.testing import FakeSshManager
from app.collectors.nic_collector import NICMetricsCollector


class TestNICMetricsCollector(unittest.IsolatedAsyncioTestCase):
    async def test_collect_rdma_links_parses_lines(self):
        rdma_link = "link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev ens1\n"
        ssh = FakeSshManager(["node1"], command_map={"rdma link": {"node1": rdma_link}})

        result = await NICMetricsCollector().collect_rdma_links(ssh)

        self.assertEqual(
            result["node1"]["mlx5_0/1"],
            {"state": "ACTIVE", "physical_state": "LINK_UP", "netdev": "ens1"},
        )

    async def test_collect_rdma_links_error_marker(self):
        ssh = FakeSshManager(["node1"], command_map={"rdma link": {"node1": "ERROR: boom"}})
        result = await NICMetricsCollector().collect_rdma_links(ssh)
        self.assertEqual(result["node1"], {"error": "ERROR: boom"})

    async def test_collect_rdma_stats_parses_counters(self):
        stats = json.dumps([{"ifname": "rdma0", "port": 1, "ifindex": 9, "rx_pkts": 10, "tx_pkts": 20}])
        ssh = FakeSshManager(["node1"], command_map={"rdma statistic show --json": {"node1": stats}})

        result = await NICMetricsCollector().collect_rdma_stats(ssh)

        self.assertEqual(result["node1"]["rdma0/1"], {"rx_pkts": 10, "tx_pkts": 20})

    async def test_collect_rdma_resources_parses(self):
        res = "0: bnxt_re0: pd 1 cq 2 qp 3 mr 0\n"
        ssh = FakeSshManager(["node1"], command_map={"rdma res": {"node1": res}})

        result = await NICMetricsCollector().collect_rdma_resources(ssh)

        self.assertEqual(result["node1"]["bnxt_re0"], {"pd": 1, "cq": 2, "qp": 3, "mr": 0})


if __name__ == "__main__":
    unittest.main()
