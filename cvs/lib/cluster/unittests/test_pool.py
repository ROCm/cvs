"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

ClusterPool / Node round-trip + extra=forbid typo gates.

Pre-DTNI revert: Node = {vpc_ip, bmc_ip}; credentials (username,
priv_key_file) and env_vars are pool-level; container devices / NCCL fabric
/ weka paths are on the workload YAML, not here. NodePaths / NodeDevices /
NodeNetwork classes were deleted.
"""

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from cvs.lib.cluster import load_cluster_file
from cvs.lib.cluster.pool import ClusterPool


def _write(payload: dict) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "cluster.json"
    tmp.write_text(json.dumps(payload))
    return tmp


class TestLoadClusterFileRejectsUnknownKeys(unittest.TestCase):
    """extra=forbid at both nesting levels so a typo in either the
    pool-level fields or a per-Node field fails fast at load."""

    def _base(self):
        return {
            "username": "test_user",
            "priv_key_file": "/home/test_user/.ssh/id_rsa",
            "nodes": {"n0": {"vpc_ip": "10.0.0.1"}},
        }

    def test_unknown_top_level_key_rejected(self):
        payload = self._base()
        payload["garbage"] = 1
        with self.assertRaises(ValidationError):
            load_cluster_file(_write(payload))

    def test_unknown_per_node_key_rejected(self):
        payload = self._base()
        payload["nodes"]["n0"]["gpus"] = 8  # legacy field, no longer on schema
        with self.assertRaises(ValidationError):
            load_cluster_file(_write(payload))


class TestClusterPoolRoundtrip(unittest.TestCase):
    """Pool-level fields (username, priv_key_file, env_vars, head_node,
    container) round-trip; Node carries only addressing."""

    def test_minimal_pool(self):
        pool = load_cluster_file(
            _write(
                {
                    "username": "test_user",
                    "priv_key_file": "/home/test_user/.ssh/id_rsa",
                    "nodes": {"n0": {"vpc_ip": "10.0.0.1"}},
                }
            )
        )
        self.assertEqual(pool.username, "test_user")
        self.assertEqual(pool.priv_key_file, "/home/test_user/.ssh/id_rsa")
        self.assertEqual(pool.env_vars, {})
        self.assertIsNone(pool.head_node)
        self.assertIsNone(pool.container)
        self.assertEqual(pool.nodes["n0"].vpc_ip, "10.0.0.1")
        self.assertIsNone(pool.nodes["n0"].bmc_ip)

    def test_full_pool_roundtrip(self):
        pool = load_cluster_file(
            _write(
                {
                    "username": "test_user",
                    "priv_key_file": "/home/test_user/.ssh/id_rsa",
                    "env_vars": {"PATH": "/opt/rocm/bin:$PATH"},
                    "head_node": "n0",
                    "nodes": {
                        "n0": {"vpc_ip": "10.0.0.1", "bmc_ip": "10.0.1.1"},
                        "n1": {"vpc_ip": "10.0.0.2", "bmc_ip": "10.0.1.2"},
                    },
                    "container": {
                        "enabled": True,
                        "launch": True,
                        "image": "rocm/cvs:latest",
                        "name": "cvs_container",
                        "runtime": {
                            "name": "docker",
                            "args": {"network": "host", "ipc": "host", "privileged": True},
                        },
                    },
                }
            )
        )
        self.assertEqual(pool.env_vars["PATH"], "/opt/rocm/bin:$PATH")
        self.assertEqual(pool.head_node, "n0")
        self.assertEqual(pool.nodes["n0"].bmc_ip, "10.0.1.1")
        self.assertTrue(pool.container.enabled)
        self.assertEqual(pool.container.image, "rocm/cvs:latest")
        self.assertEqual(pool.container.runtime.name, "docker")
        self.assertEqual(pool.container.runtime.args["network"], "host")


class TestHeadNodeMustReferenceKnownNode(unittest.TestCase):
    """head_node names the orchestrator host for distributed bootstraps;
    a typo would silently break MASTER_ADDR resolution. Fail fast at load."""

    def test_head_node_not_in_nodes_rejected(self):
        with self.assertRaises(ValidationError):
            ClusterPool.model_validate(
                {
                    "username": "u",
                    "priv_key_file": "/k",
                    "head_node": "n9",
                    "nodes": {"n0": {"vpc_ip": "1.1.1.1"}},
                }
            )

    def test_head_node_absent_is_fine(self):
        pool = ClusterPool.model_validate(
            {
                "username": "u",
                "priv_key_file": "/k",
                "nodes": {"n0": {"vpc_ip": "1.1.1.1"}},
            }
        )
        self.assertIsNone(pool.head_node)


if __name__ == "__main__":
    unittest.main()
