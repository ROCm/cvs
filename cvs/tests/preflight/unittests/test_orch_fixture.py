"""Unit tests for the preflight orch fixture's parallel-handle overrides."""

import json
import os
import tempfile
import unittest

from cvs.tests.preflight.conftest import preflight_hosts_per_shard, preflight_parallel_handle_overrides


class TestPreflightHostsPerShard(unittest.TestCase):
    def _write(self, payload):
        handle = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
        json.dump(payload, handle)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_default_when_preflight_section_missing(self):
        path = self._write({"node_check": {}})
        self.assertEqual(preflight_hosts_per_shard(path), 32)

    def test_nodes_per_full_mesh_group_wins(self):
        path = self._write(
            {
                "preflight": {
                    "connectivity_check": {
                        "rdma": {"nodes_per_full_mesh_group": 8, "parallel_group_size": 16},
                    },
                    "parallelism": {"parallel_group_size": 64},
                }
            }
        )
        self.assertEqual(preflight_hosts_per_shard(path), 8)

    def test_parallel_group_size_fallback(self):
        path = self._write({"preflight": {"connectivity_check": {"rdma": {"parallel_group_size": 16}}}})
        self.assertEqual(preflight_hosts_per_shard(path), 16)

    def test_parallelism_section_fallback(self):
        path = self._write({"preflight": {"parallelism": {"parallel_group_size": 64}}})
        self.assertEqual(preflight_hosts_per_shard(path), 64)

    def test_overrides_include_hardcoded_ssh_settings(self):
        path = self._write({"preflight": {"connectivity_check": {"rdma": {"nodes_per_full_mesh_group": 8}}}})
        overrides = preflight_parallel_handle_overrides(path)
        self.assertEqual(
            overrides,
            {
                'parallel_handle': {
                    'config': {'hosts_per_shard': 8},
                    'transport_kwargs': {'timeout': 60, 'num_retries': 2, 'retry_delay': 2},
                }
            },
        )
