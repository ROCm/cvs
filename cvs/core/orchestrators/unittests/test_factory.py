'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/core/orchestrators/factory.py: OrchestratorFactory.create_orchestrator
# dispatch and OrchestratorConfig construction / from_configs validation. Mocks Pssh and
# RuntimeFactory so tests run without SSH or container runtime.

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.core.orchestrators.baremetal import BaremetalOrchestrator
from cvs.core.orchestrators.container import ContainerOrchestrator


def _make_orch_config(orchestrator="baremetal", with_container=False, launch=False, enabled=True):
    """Minimal OrchestratorConfig that satisfies BaremetalOrchestrator and (optionally)
    ContainerOrchestrator __init__ without touching disk or SSH."""
    kwargs = dict(
        orchestrator=orchestrator,
        node_dict={"10.0.0.1": {}, "10.0.0.2": {}},
        username="testuser",
        priv_key_file="/dev/null",
        password=None,
        head_node_dict={"mgmt_ip": "10.0.0.1"},
        container={},
    )
    if with_container or orchestrator == "container":
        kwargs["container"] = {
            "enabled": enabled,
            "launch": launch,
            "image": "rocm/cvs:test",
            "name": "cvs_iter_test",
            "runtime": {"name": "docker", "args": {}},
        }
    return OrchestratorConfig(**kwargs)


class TestOrchestratorFactory(unittest.TestCase):
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_create_returns_baremetal_for_baremetal_string(self, _mock_pssh):
        cfg = _make_orch_config(orchestrator="baremetal")
        orch = OrchestratorFactory.create_orchestrator(MagicMock(), cfg)
        self.assertIsInstance(orch, BaremetalOrchestrator)

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_create_returns_container_for_container_string(self, _mock_pssh, _mock_rf):
        cfg = _make_orch_config(orchestrator="container")
        orch = OrchestratorFactory.create_orchestrator(MagicMock(), cfg)
        self.assertIsInstance(orch, ContainerOrchestrator)

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_create_raises_for_unsupported_string(self, _mock_pssh):
        # Bypass OrchestratorConfig.from_configs validation by constructing
        # the config object directly with an unsupported orchestrator value.
        cfg = _make_orch_config(orchestrator="slurm")
        with self.assertRaises(ValueError):
            OrchestratorFactory.create_orchestrator(MagicMock(), cfg)

    def test_create_raises_typeerror_for_non_config(self):
        with self.assertRaises(TypeError):
            OrchestratorFactory.create_orchestrator(MagicMock(), {"orchestrator": "baremetal"})

    def test_get_supported_backends_lists_both(self):
        backends = OrchestratorFactory.get_supported_backends()
        self.assertIn("baremetal", backends)
        self.assertIn("container", backends)


class TestOrchestratorConfig(unittest.TestCase):
    def test_get_returns_default_for_missing_key(self):
        cfg = _make_orch_config()
        self.assertEqual(cfg.get("nonexistent_key", "fallback"), "fallback")

    def test_get_returns_value_for_existing_attr(self):
        cfg = _make_orch_config()
        self.assertEqual(cfg.get("username"), "testuser")

    def test_from_configs_defaults_orchestrator_to_baremetal_when_omitted(self):
        cluster = {"node_dict": {"1.1.1.1": {}}, "username": "u", "priv_key_file": "/dev/null"}
        cfg = OrchestratorConfig.from_configs(cluster)
        self.assertEqual(cfg.orchestrator, "baremetal")

    def test_from_configs_merges_testsuite_over_cluster(self):
        cluster = {
            "orchestrator": "baremetal",
            "node_dict": {"1.1.1.1": {}},
            "username": "cluster_user",
            "priv_key_file": "/dev/null",
        }
        testsuite = {"username": "testsuite_user"}
        cfg = OrchestratorConfig.from_configs(cluster, testsuite)
        self.assertEqual(cfg.username, "testsuite_user")

    def test_from_configs_raises_when_node_dict_missing(self):
        cluster = {"username": "u", "priv_key_file": "/dev/null"}
        with self.assertRaises(ValueError):
            OrchestratorConfig.from_configs(cluster)

    def test_from_configs_raises_when_username_missing(self):
        cluster = {"node_dict": {"1.1.1.1": {}}, "priv_key_file": "/dev/null"}
        with self.assertRaises(ValueError):
            OrchestratorConfig.from_configs(cluster)

    def test_from_configs_raises_when_priv_key_file_missing(self):
        cluster = {"node_dict": {"1.1.1.1": {}}, "username": "u"}
        with self.assertRaises(ValueError):
            OrchestratorConfig.from_configs(cluster)

    def test_init_raises_valueerror_for_missing_required_key(self):
        # Direct construction with a missing required key should surface a clear
        # ValueError listing what's missing, not a bare KeyError.
        with self.assertRaises(ValueError) as ctx:
            OrchestratorConfig(
                orchestrator="baremetal",
                node_dict={"1.1.1.1": {}},
                username="u",
                # priv_key_file deliberately omitted
            )
        self.assertIn("priv_key_file", str(ctx.exception))

    def test_from_configs_reads_from_file_path(self):
        # Round-trip a real JSON file through from_configs to exercise the
        # path-handling branch (the rvs_cvs.py orch fixture goes through this).
        cluster = {
            "orchestrator": "baremetal",
            "node_dict": {"1.1.1.1": {}, "1.1.1.2": {}},
            "username": "u",
            "priv_key_file": "/dev/null",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(cluster, f)
            path = f.name
        try:
            cfg = OrchestratorConfig.from_configs(path)
            self.assertEqual(cfg.orchestrator, "baremetal")
            self.assertEqual(len(cfg.node_dict), 2)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
