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
from cvs.lib.cluster.pool import NodePaths


class TestLoadClusterFileRejectsUnknownKeys(unittest.TestCase):
    """G4 minimal surface: extra=forbid is wired at three nesting levels
    (top-level ClusterPool, per-Node, and per-NodePaths) so an unknown key in
    a real cluster file fails fast at load_cluster_file (no silent acceptance)."""

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


if __name__ == "__main__":
    unittest.main()
