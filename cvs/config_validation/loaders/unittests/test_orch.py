'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.config_validation.loaders.orch.
'''

import copy
import unittest
from unittest import mock

from cvs.config_validation.loaders.orch import build_orch, cluster_with_hosts
from cvs.config_validation.loaders.suite_config import LoadedSuiteConfig
from cvs.config_validation.schemas.base import SuiteConfig
from cvs.core.orchestrators.baremetal import BaremetalOrchestrator
from cvs.core.orchestrators.container import ContainerOrchestrator
from cvs.core.orchestrators.factory import OrchestratorConfig

MAP_CLUSTER = {
    "username": "u",
    "priv_key_file": "/keys/id",
    "node_dict": {"n0": {"vpc_ip": "v0"}, "n1": {"vpc_ip": "v1"}, "n2": {"vpc_ip": "v2"}},
    "head_node_dict": {"mgmt_ip": "n0", "vpc_ip": "v0"},
}
LIST_CLUSTER = {
    "username": "u",
    "priv_key_file": "/keys/id",
    "node_dict": [{"mgmt_ip": "n0", "vpc_ip": "v0"}, {"mgmt_ip": "n1", "vpc_ip": "v1"}, {"mgmt_ip": "n2"}],
    "head_node_dict": {"mgmt_ip": "n0", "vpc_ip": "v0"},
}


class _Suite(SuiteConfig):
    name: str


class TestClusterWithHosts(unittest.TestCase):
    def _narrow(self, cluster, hosts):
        before = copy.deepcopy(cluster)
        narrowed = cluster_with_hosts(cluster, hosts)
        self.assertEqual(cluster, before, "the input cluster must not change")
        return narrowed

    def test_none_returns_the_cluster_itself(self):
        for cluster in (MAP_CLUSTER, LIST_CLUSTER, {}):
            with self.subTest(cluster=cluster):
                self.assertIs(cluster_with_hosts(cluster, None), cluster)

    def test_a_map_node_dict_stays_a_map_in_its_own_order_whatever_order_hosts_come_in(self):
        narrowed = self._narrow(MAP_CLUSTER, ["n2", "n0", "n2"])
        self.assertIsInstance(narrowed["node_dict"], dict)
        self.assertEqual(list(narrowed["node_dict"].items()), [("n0", {"vpc_ip": "v0"}), ("n2", {"vpc_ip": "v2"})])

    def test_a_list_node_dict_stays_a_list_in_its_own_order_whatever_order_hosts_come_in(self):
        narrowed = self._narrow(LIST_CLUSTER, ["n2", "n1"])
        self.assertEqual(narrowed["node_dict"], [{"mgmt_ip": "n1", "vpc_ip": "v1"}, {"mgmt_ip": "n2"}])

    def test_the_first_kept_node_becomes_head_and_other_head_and_cluster_keys_stay(self):
        for cluster in (MAP_CLUSTER, LIST_CLUSTER):
            with self.subTest(form=type(cluster["node_dict"]).__name__):
                narrowed = self._narrow(cluster, ["n2", "n1"])
                self.assertEqual(narrowed["head_node_dict"], {"mgmt_ip": "n1", "vpc_ip": "v0"})
                self.assertEqual(
                    {key: value for key, value in narrowed.items() if key not in ("node_dict", "head_node_dict")},
                    {"username": "u", "priv_key_file": "/keys/id"},
                )

    def test_selecting_every_node_keeps_node_dict_and_its_first_node_as_head(self):
        narrowed = self._narrow(MAP_CLUSTER, ["n2", "n1", "n0"])
        self.assertEqual(narrowed["node_dict"], MAP_CLUSTER["node_dict"])
        self.assertEqual(narrowed["head_node_dict"]["mgmt_ip"], "n0")

    def test_a_cluster_without_head_node_dict_gets_one_naming_the_first_kept_node(self):
        cluster = {key: value for key, value in MAP_CLUSTER.items() if key != "head_node_dict"}
        self.assertEqual(self._narrow(cluster, ["n1"])["head_node_dict"], {"mgmt_ip": "n1"})

    def test_only_the_hosts_not_in_node_dict_are_refused_and_named(self):
        for cluster, unknown in ((MAP_CLUSTER, "n9"), (LIST_CLUSTER, "n9"), ({"username": "u"}, "n0, n9")):
            with self.subTest(cluster=cluster):
                with self.assertRaises(ValueError) as ctx:
                    self._narrow(cluster, ["n0", "n9"])
                self.assertTrue(str(ctx.exception).endswith(f"node_dict: {unknown}"), str(ctx.exception))

    def test_an_empty_selection_is_refused_rather_than_building_an_orchestrator_with_no_head(self):
        for cluster in (MAP_CLUSTER, LIST_CLUSTER):
            with self.subTest(form=type(cluster["node_dict"]).__name__):
                with self.assertRaisesRegex(ValueError, "no hosts selected"):
                    self._narrow(cluster, [])


@mock.patch("cvs.core.orchestrators.container.RuntimeFactory")
@mock.patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
class TestBuildOrch(unittest.TestCase):
    def setUp(self):
        self.log = mock.Mock()
        self.config = _Suite(name="a")

    def _loaded(self, orch_inputs, cluster=MAP_CLUSTER):
        return LoadedSuiteConfig(_Suite, self.config, cluster, orch_inputs)

    def test_a_baremetal_orchestrator_runs_on_the_given_cluster_and_carries_the_suite_config(self, _pssh, _runtime):
        loaded = self._loaded({"name": "a"})
        orch = build_orch(self.log, loaded, cluster_with_hosts(MAP_CLUSTER, ["n2", "n1"]))
        self.assertIsInstance(orch, BaremetalOrchestrator)
        self.assertNotIsInstance(orch, ContainerOrchestrator)
        self.assertEqual((orch.hosts, orch.head_node), (["n1", "n2"], "n1"))
        self.assertIs(orch.suite_config, self.config)
        self.assertIsInstance(orch.config, OrchestratorConfig)

    def test_orch_inputs_override_the_cluster_file_at_the_top_level(self, _pssh, _runtime):
        cluster = {**MAP_CLUSTER, "container": {"image": "cluster-image", "name": "cluster-ct"}}
        container = {"image": "suite-image", "name": "suite-ct", "runtime": {"name": "docker", "args": {}}}
        loaded = self._loaded({"orchestrator": "container", "container": container}, cluster)
        orch = build_orch(self.log, loaded, cluster)
        self.assertIsInstance(orch, ContainerOrchestrator)
        self.assertEqual((orch.container_config["image"], orch.container_config["name"]), ("suite-image", "suite-ct"))
        self.assertEqual(orch.hosts, ["n0", "n1", "n2"])
        self.assertIs(orch.suite_config, self.config)

    def test_a_list_node_dict_narrowed_to_one_host_builds_on_that_host(self, _pssh, _runtime):
        loaded = self._loaded({}, LIST_CLUSTER)
        orch = build_orch(self.log, loaded, cluster_with_hosts(LIST_CLUSTER, ["n1"]))
        self.assertEqual((orch.hosts, orch.head_node), (["n1"], "n1"))


if __name__ == "__main__":
    unittest.main()
