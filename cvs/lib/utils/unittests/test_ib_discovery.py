'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import unittest

from cvs.lib.utils.ib_discovery import discover_socket_netdev_name, resolve_multinode_fabric


class _NetdevOrch:
    def __init__(self, hosts, responses):
        self.hosts = list(hosts)
        self._responses = dict(responses)

    def exec(self, cmd, hosts=None, **kwargs):
        target_hosts = list(hosts) if hosts is not None else self.hosts
        return {h: self._responses.get((h, cmd), "") for h in target_hosts}


class TestDiscoverSocketNetdev(unittest.TestCase):
    def test_resolves_common_netdev_from_cluster_ips(self):
        h0, h1 = "10.32.80.112", "10.32.80.113"
        orch = _NetdevOrch(
            [h0, h1],
            {
                (h0, _cmd_for_ip(h0)): "ens51f1np1\n",
                (h1, _cmd_for_ip(h1)): "ens51f1np1\n",
            },
        )
        self.assertEqual(discover_socket_netdev_name(orch, master_addr=h0), "ens51f1np1")

    def test_raises_on_asymmetric_netdev_names(self):
        h0, h1 = "10.32.80.112", "10.32.80.113"
        orch = _NetdevOrch(
            [h0, h1],
            {
                (h0, _cmd_for_ip(h0)): "ens51f1np1\n",
                (h1, _cmd_for_ip(h1)): "ens51f1np2\n",
            },
        )
        with self.assertRaisesRegex(RuntimeError, "asymmetric netdev"):
            discover_socket_netdev_name(orch, master_addr=h0)

    def test_rejects_mlx5_hca_name(self):
        h0 = "10.32.80.112"
        orch = _NetdevOrch([h0], {(h0, _cmd_for_ip(h0)): "mlx5_0\n"})
        with self.assertRaisesRegex(RuntimeError, "IB HCA name"):
            discover_socket_netdev_name(orch, master_addr=h0)


class TestResolveMultinodeFabric(unittest.TestCase):
    def test_resolves_hcas_and_netdev_from_auto_config(self):
        from cvs.lib.utils.ib_discovery import _IBVDEVINFO_CMD, _SYSFS_CMD

        h0, h1 = "10.32.80.112", "10.32.80.113"
        orch = _NetdevOrch(
            [h0, h1],
            {
                (h0, _IBVDEVINFO_CMD): "",
                (h1, _IBVDEVINFO_CMD): "",
                (h0, _SYSFS_CMD): "mlx5_0 mlx5_1 ",
                (h1, _SYSFS_CMD): "mlx5_0 mlx5_1 ",
                (h0, _cmd_for_ip(h0)): "ens51f1np1\n",
                (h1, _cmd_for_ip(h1)): "ens51f1np1\n",
            },
        )
        hcas, netdev = resolve_multinode_fabric(
            orch,
            ib_hca_devices="auto",
            ib_netdev="auto",
            master_addr=h0,
        )
        self.assertEqual(hcas, ["mlx5_0", "mlx5_1"])
        self.assertEqual(netdev, "ens51f1np1")


def _cmd_for_ip(ip: str) -> str:
    from cvs.lib.utils.ib_discovery import _netdev_for_ip_cmd

    return _netdev_for_ip_cmd(ip)


if __name__ == "__main__":
    unittest.main()
