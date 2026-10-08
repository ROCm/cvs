import unittest
from types import SimpleNamespace

from cvs.cli_plugins.list_plugin import ListPlugin
from cvs.lib.inference.vllm_topology import (
    EffectiveVllmTopology,
    build_vllm_targets,
    resolve_vllm_topology,
    scope_vllm_cluster,
)


def _variant(pp="1", ray=False, ib_netdev=None, env=None):
    return SimpleNamespace(
        server_params=SimpleNamespace(
            pipeline_parallel_size=int(pp), distributed_executor_backend="ray" if ray else "mp"
        ),
        ib_netdev=ib_netdev,
        container=SimpleNamespace(env=dict(env or {})),
    )


class TestVllmTopology(unittest.TestCase):
    def test_single_uses_scoped_first_host(self):
        targets, pp = build_vllm_targets("single", _variant(), ["node0"])
        self.assertEqual(targets, (("node0",),))
        self.assertEqual(pp, 1)

    def test_single_rejects_unscoped_orchestrator(self):
        with self.assertRaisesRegex(ValueError, "first host"):
            build_vllm_targets("single", _variant(), ["node0", "node1"])

    def test_single_scope_uses_insertion_order_and_overwrites_configured_head(self):
        cluster = {
            "node_dict": {"node0": {"vpc_ip": "10.0.0.1"}, "node1": {"vpc_ip": "10.0.0.2"}},
            "head_node_dict": {"mgmt_ip": "node1"},
        }
        scoped = scope_vllm_cluster("single", cluster)
        self.assertEqual(list(scoped["node_dict"]), ["node0"])
        self.assertEqual(scoped["head_node_dict"]["mgmt_ip"], "node0")
        self.assertEqual(list(cluster["node_dict"]), ["node0", "node1"])

    def test_distributed_uses_all_hosts(self):
        variant = _variant(pp="2", env={"NCCL_SOCKET_IFNAME": "eno0"})
        targets, pp = build_vllm_targets("distributed", variant, ["node0", "node1"])
        self.assertEqual(targets, (("node0", "node1"),))
        self.assertEqual(pp, 2)

    def test_distributed_accepts_legacy_ib_netdev_fallback(self):
        targets, pp = build_vllm_targets("distributed", _variant(pp="2", ib_netdev="eth0"), ["node0", "node1"])
        self.assertEqual(targets, (("node0", "node1"),))
        self.assertEqual(pp, 2)

    def test_distributed_uses_all_cluster_hosts(self):
        variant = _variant(pp="4", env={"NCCL_SOCKET_IFNAME": "eno0"})
        hosts = ["node0", "node1", "node2", "node3"]
        targets, pp = build_vllm_targets("distributed", variant, hosts)
        self.assertEqual(targets, (tuple(hosts),))
        self.assertEqual(pp, 4)

    def test_distributed_one_host_delegates_to_single(self):
        # Includes variants the multi-host rules would reject (mp with pp=1, no
        # socket interface) to show none of those rules apply on one host.
        cases = [
            _variant(pp="1"),
            _variant(pp="2"),
            _variant(pp="4", ray=True),
            _variant(pp="2", env={"NCCL_SOCKET_IFNAME": "eno0"}),
        ]
        for variant in cases:
            pp = variant.server_params.pipeline_parallel_size
            with self.subTest(pp=pp, backend=variant.server_params.distributed_executor_backend):
                distributed = resolve_vllm_topology("distributed", variant, ["node0"])
                self.assertEqual(distributed, resolve_vllm_topology("single", variant, ["node0"]))
                self.assertEqual(distributed, EffectiveVllmTopology("single", ("node0",), pp))

    def test_single_keeps_configured_pipeline_parallelism(self):
        for pp in (2, 4):
            with self.subTest(pp=pp):
                topology = resolve_vllm_topology("single", _variant(pp=str(pp)), ["node0"])
                self.assertEqual((topology.mode, topology.nnodes, topology.pipeline_parallel_size), ("single", 1, pp))
                self.assertEqual(build_vllm_targets("single", _variant(pp=str(pp)), ["node0"]), ((("node0",),), pp))

    def test_single_pipeline_parallelism_still_requires_one_host(self):
        with self.assertRaisesRegex(ValueError, "first host"):
            build_vllm_targets("single", _variant(pp="2"), ["node0", "node1"])

    def test_multi_host_distributed_keeps_distributed_mode(self):
        variant = _variant(pp="2", env={"NCCL_SOCKET_IFNAME": "eno0"})
        topology = resolve_vllm_topology("distributed", variant, ["node0", "node1"])
        self.assertEqual(topology, EffectiveVllmTopology("distributed", ("node0", "node1"), 2))

    def test_unknown_mode_is_rejected_even_on_one_host(self):
        with self.assertRaisesRegex(ValueError, "unknown vLLM mode"):
            resolve_vllm_topology("multi", _variant(), ["node0"])

    def test_multi_host_mp_requires_pipeline_parallelism(self):
        with self.assertRaisesRegex(ValueError, "pipeline_parallel_size"):
            build_vllm_targets("distributed", _variant(), ["node0", "node1"])

    def test_multi_host_ray_requires_network_interface(self):
        with self.assertRaisesRegex(ValueError, "NCCL_SOCKET_IFNAME"):
            build_vllm_targets("distributed", _variant(ray=True), ["node0", "node1"])

    def test_cli_discovers_split_suites_only(self):
        tests = ListPlugin.discover_tests()["cvs"]
        self.assertIn("vllm_single", tests)
        self.assertIn("vllm_distributed", tests)
        self.assertNotIn("vllm", tests)
