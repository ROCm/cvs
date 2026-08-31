"""Unit tests for cluster.json schema (cvs/schema/cluster_file/cluster.py)."""

import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from cvs.schema.cluster_file.cluster import ClusterConfigFile

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_SAMPLE_CLUSTER = _PACKAGE_ROOT / "input" / "cluster_file" / "cluster.json"


class TestClusterConfigFile(unittest.TestCase):
    def test_sample_cluster_json_validates(self):
        raw = json.loads(_SAMPLE_CLUSTER.read_text())
        config = ClusterConfigFile.model_validate(raw)
        self.assertGreater(len(config.node_dict), 0)
        first_node = next(iter(config.node_dict.values()))
        self.assertTrue(first_node.vpc_ip)

    def test_minimal_node_dict_required(self):
        with self.assertRaises(ValidationError):
            ClusterConfigFile.model_validate({"node_dict": {}})

    def test_extra_top_level_keys_allowed(self):
        config = ClusterConfigFile.model_validate(
            {
                "username": "testuser",
                "priv_key_file": "/home/testuser/.ssh/id_rsa",
                "node_dict": {"host1": {"vpc_ip": "10.0.0.1"}},
                "orchestrator": {"type": "baremetal"},
                "env_vars": {"PATH": "/bin"},
            }
        )
        self.assertIn("host1", config.node_dict)

    def test_node_requires_vpc_ip(self):
        with self.assertRaises(ValidationError):
            ClusterConfigFile.model_validate(
                {
                    "username": "testuser",
                    "priv_key_file": "/home/testuser/.ssh/id_rsa",
                    "node_dict": {"host1": {"bmc_ip": "1.2.3.4"}},
                }
            )


if __name__ == "__main__":
    unittest.main()
