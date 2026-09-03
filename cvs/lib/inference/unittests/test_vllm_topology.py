import unittest
from types import SimpleNamespace

from cvs.cli_plugins.list_plugin import ListPlugin
from cvs.lib.inference.vllm_topology import build_vllm_targets, resolve_vllm_topology, scope_vllm_cluster


def _variant(pp="1", ray=False, ib_netdev=None):
    return SimpleNamespace(
        server_params=SimpleNamespace(
            pipeline_parallel_size=int(pp), distributed_executor_backend="ray" if ray else "mp"
        ),
        ib_netdev=ib_netdev,
    )


class TestVllmTopology(unittest.TestCase):
    def test_single_uses_scoped_first_host(self):
        targets, pp = build_vllm_targets("single", _variant(), ["node0"])
        self.assertEqual(targets, (("node0",),))
        self.assertEqual(pp, 1)

    def test_single_rejects_unscoped_orchestrator(self):
        with self.assertRaisesRegex(ValueError, "first host"):
            build_vllm_targets("single", _variant(), ["node0", "node1"])

    def test_single_cluster_scope_keeps_only_first_node(self):
        cluster = {
            "node_dict": {"node0": {"vpc_ip": "10.0.0.1"}, "node1": {"vpc_ip": "10.0.0.2"}},
            "head_node_dict": {"mgmt_ip": "node1"},
        }
        scoped = scope_vllm_cluster("single", cluster)
        self.assertEqual(list(scoped["node_dict"]), ["node0"])
        self.assertEqual(scoped["head_node_dict"]["mgmt_ip"], "node0")
        self.assertEqual(list(cluster["node_dict"]), ["node0", "node1"])

    def test_distributed_uses_all_hosts(self):
        targets, pp = build_vllm_targets("distributed", _variant(pp="2", ib_netdev="eth0"), ["node0", "node1"])
        self.assertEqual(targets, (("node0", "node1"),))
        self.assertEqual(pp, 2)

    def test_distributed_rejects_more_than_two_hosts(self):
        with self.assertRaisesRegex(ValueError, "exactly two"):
            resolve_vllm_topology("distributed", _variant(pp="2", ib_netdev="eth0"), ["node0", "node1", "node2"])

    def test_distributed_one_host_uses_singleton_fallback(self):
        targets, pp = build_vllm_targets("distributed", _variant(pp="2"), ["node0"])
        self.assertEqual(targets, (("node0",),))
        self.assertEqual(pp, 1)

    def test_single_rejects_pipeline_parallel_recipe(self):
        with self.assertRaisesRegex(ValueError, "pipeline_parallel_size"):
            build_vllm_targets("single", _variant(pp="2"), ["node0"])

    def test_multi_host_mp_requires_pipeline_parallelism(self):
        with self.assertRaisesRegex(ValueError, "pipeline_parallel_size"):
            build_vllm_targets("distributed", _variant(), ["node0", "node1"])

    def test_multi_host_ray_requires_network_interface(self):
        with self.assertRaisesRegex(ValueError, "ib_netdev"):
            build_vllm_targets("distributed", _variant(ray=True), ["node0", "node1"])

    def test_cli_discovers_split_suites_only(self):
        tests = ListPlugin.discover_tests()["cvs"]
        self.assertIn("vllm_single", tests)
        self.assertIn("vllm_distributed", tests)
        self.assertNotIn("vllm", tests)
