"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from cvs.lib.cluster import load_cluster_file
from cvs.lib.cluster.pool import NodeDevices, NodeNetwork, NodePaths


class TestLoadClusterFileRejectsUnknownKeys(unittest.TestCase):
    """G4 minimal surface: extra=forbid is wired at five nesting levels
    (top-level ClusterPool, per-Node, NodePaths, NodeDevices, NodeNetwork)
    so an unknown key in a real cluster file fails fast at load_cluster_file
    (no silent acceptance). The NodeNetwork case is the key one -- a typo
    like ``nccl_ib_gid_indx`` would otherwise silently never be exported."""

    def _write(self, payload: dict) -> Path:
        tmp = Path(tempfile.mkdtemp()) / "cluster.json"
        tmp.write_text(json.dumps(payload))
        return tmp

    def test_unknown_top_level_key_rejected(self):
        path = self._write({"nodes": {}, "garbage_top": 1})
        with self.assertRaises(ValidationError):
            load_cluster_file(path)

    def test_unknown_per_node_key_rejected(self):
        path = self._write(
            {"nodes": {"n0": {"ip": "1", "user": "u", "gpus": 8, "labels": ["mi300"], "bogus_field": True}}}
        )
        with self.assertRaises(ValidationError):
            load_cluster_file(path)

    def test_unknown_node_paths_key_rejected(self):
        # NodePaths is the A3 host-path seam; a typo like ``model-cache`` must
        # fail at load rather than silently resolving to None at launch time.
        path = self._write(
            {
                "nodes": {
                    "n0": {
                        "ip": "1",
                        "user": "u",
                        "gpus": 8,
                        "labels": ["mi300"],
                        "paths": {"model_cache": "/data/hf", "model-cache": "/typo"},
                    }
                }
            }
        )
        with self.assertRaises(ValidationError):
            load_cluster_file(path)

    def test_unknown_node_devices_key_rejected(self):
        path = self._write(
            {
                "nodes": {
                    "n0": {
                        "ip": "1",
                        "user": "u",
                        "gpus": 8,
                        "labels": ["mi300"],
                        "devices": {"ib": ["/dev/infiniband/uverbs0"], "bogus_dev_kind": ["/dev/foo"]},
                    }
                }
            }
        )
        with self.assertRaises(ValidationError):
            load_cluster_file(path)

    def test_unknown_node_network_key_rejected(self):
        # The headline case: a typo in a typed NCCL knob (here
        # ``nccl_ib_gid_indx``) silently never reaches NCCL_IB_GID_INDEX
        # without extra=forbid. Load-time rejection is the whole point.
        path = self._write(
            {
                "nodes": {
                    "n0": {
                        "ip": "1",
                        "user": "u",
                        "gpus": 8,
                        "labels": ["mi300"],
                        "network": {"nccl_ib_gid_index": "3", "nccl_ib_gid_indx": "4"},
                    }
                }
            }
        )
        with self.assertRaises(ValidationError):
            load_cluster_file(path)


class TestNodePathsDefaults(unittest.TestCase):
    """G4 minimal surface: Node.paths defaults to an empty NodePaths so existing
    cluster files (no ``paths`` key) keep loading; populated paths round-trip."""

    def test_node_without_paths_defaults_to_empty_node_paths(self):
        path = Path(tempfile.mkdtemp()) / "cluster.json"
        path.write_text(json.dumps({"nodes": {"n0": {"ip": "1", "user": "u", "gpus": 8, "labels": ["mi300"]}}}))
        pool = load_cluster_file(path)
        self.assertEqual(pool.nodes["n0"].paths, NodePaths())
        self.assertIsNone(pool.nodes["n0"].paths.model_cache)

    def test_node_paths_roundtrip(self):
        path = Path(tempfile.mkdtemp()) / "cluster.json"
        path.write_text(
            json.dumps(
                {
                    "nodes": {
                        "n0": {
                            "ip": "1",
                            "user": "u",
                            "gpus": 8,
                            "labels": ["mi300"],
                            "paths": {"model_cache": "/data/hf-cache", "dataset_root": "/data/datasets"},
                        }
                    }
                }
            )
        )
        pool = load_cluster_file(path)
        self.assertEqual(pool.nodes["n0"].paths.model_cache, "/data/hf-cache")
        self.assertEqual(pool.nodes["n0"].paths.dataset_root, "/data/datasets")


class TestNodeDevicesAndNetworkRoundtrip(unittest.TestCase):
    """G4 minimal surface: Node.devices / Node.network default empty so cluster
    files predating these fields keep loading; populated values round-trip with
    typed Optional[str] / List[str] / Dict[str, str] shapes intact. G5b is
    responsible for merging these into ContainerSpec at launch time."""

    def test_devices_and_network_default_empty(self):
        path = Path(tempfile.mkdtemp()) / "cluster.json"
        path.write_text(json.dumps({"nodes": {"n0": {"ip": "1", "user": "u", "gpus": 8, "labels": ["mi300"]}}}))
        pool = load_cluster_file(path)
        self.assertEqual(pool.nodes["n0"].devices, NodeDevices())
        self.assertEqual(pool.nodes["n0"].network, NodeNetwork())
        self.assertEqual(pool.nodes["n0"].devices.ib, [])
        self.assertEqual(pool.nodes["n0"].network.extra_env, {})

    def test_devices_and_network_populated_roundtrip(self):
        path = Path(tempfile.mkdtemp()) / "cluster.json"
        path.write_text(
            json.dumps(
                {
                    "nodes": {
                        "n0": {
                            "ip": "1",
                            "user": "u",
                            "gpus": 8,
                            "labels": ["mi300", "thor2"],
                            "devices": {
                                "ib": [f"/dev/infiniband/uverbs{i}" for i in range(8)] + ["/dev/infiniband/rdma_cm"],
                                "gpu": ["/dev/kfd", "/dev/dri/renderD128"],
                                "extra": [],
                            },
                            "network": {
                                "nccl_socket_ifname": "bond0",
                                "nccl_ib_hca": "bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7",
                                "nccl_ib_gid_index": "3",
                                "nccl_ib_sl": "3",
                                "ucx_net_devices": "bnxt_re0:1,bnxt_re1:1",
                                "gloo_socket_ifname": "bond0",
                                "extra_env": {"NCCL_IB_TC": "106", "NCCL_NET_GDR_LEVEL": "PHB"},
                            },
                        }
                    }
                }
            )
        )
        pool = load_cluster_file(path)
        n = pool.nodes["n0"]
        self.assertEqual(len(n.devices.ib), 9)
        self.assertIn("/dev/kfd", n.devices.gpu)
        self.assertEqual(n.network.nccl_socket_ifname, "bond0")
        self.assertEqual(n.network.nccl_ib_gid_index, "3")
        self.assertEqual(n.network.extra_env["NCCL_IB_TC"], "106")


if __name__ == "__main__":
    unittest.main()
