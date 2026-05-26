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

    @patch.dict(os.environ, {"USER": "alice", "LOGNAME": "", "USERNAME": ""}, clear=False)
    def test_from_configs_reads_from_file_path(self):
        # Round-trip a real JSON file through from_configs to exercise the
        # path-handling branch (the rvs_cvs.py orch fixture goes through this).
        # Also pins that {user-id} placeholders are resolved when input came from disk.
        cluster = {
            "orchestrator": "baremetal",
            "node_dict": {"1.1.1.1": {}, "1.1.1.2": {}},
            "username": "{user-id}",
            "priv_key_file": "/home/{user-id}/.ssh/id_rsa",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(cluster, f)
            path = f.name
        try:
            cfg = OrchestratorConfig.from_configs(path)
            self.assertEqual(cfg.orchestrator, "baremetal")
            self.assertEqual(len(cfg.node_dict), 2)
            self.assertEqual(cfg.username, "alice")
            self.assertEqual(cfg.priv_key_file, "/home/alice/.ssh/id_rsa")
        finally:
            os.unlink(path)

    # ------------------------------------------------------------------
    # Placeholder resolution at the from_configs boundary.
    #
    # Pins that {user-id} in the cluster file (and any merged testsuite
    # overrides) is substituted before the values reach BaremetalOrchestrator /
    # Pssh / docker run. Env is patched so the substituted value is
    # deterministic regardless of host $USER (CI / sudo / containers).
    # ------------------------------------------------------------------

    @patch.dict(os.environ, {"USER": "alice", "LOGNAME": "", "USERNAME": ""}, clear=False)
    def test_from_configs_resolves_user_id_in_username_and_priv_key(self):
        cluster = {
            "node_dict": {"1.1.1.1": {}},
            "username": "{user-id}",
            "priv_key_file": "/home/{user-id}/.ssh/id_rsa",
        }
        cfg = OrchestratorConfig.from_configs(cluster)
        self.assertEqual(cfg.username, "alice")
        self.assertEqual(cfg.priv_key_file, "/home/alice/.ssh/id_rsa")

    @patch.dict(os.environ, {"USER": "alice", "LOGNAME": "", "USERNAME": ""}, clear=False)
    def test_from_configs_resolves_user_id_throughout_container_block(self):
        # The recursive walk must descend into container at every nesting shape:
        # top-level string scalar (image, name), and list item under a nested dict
        # (runtime.args.volumes).
        cluster = {
            "orchestrator": "container",
            "node_dict": {"1.1.1.1": {}},
            "username": "{user-id}",
            "priv_key_file": "/home/{user-id}/.ssh/id_rsa",
            "container": {
                "enabled": True,
                "image": "rocm/{user-id}:test",
                "name": "{user-id}_cvs",
                "runtime": {
                    "name": "docker",
                    "args": {"volumes": ["/home/{user-id}/data:/data"]},
                },
            },
        }
        cfg = OrchestratorConfig.from_configs(cluster)
        self.assertEqual(cfg.container["image"], "rocm/alice:test")
        self.assertEqual(cfg.container["name"], "alice_cvs")
        self.assertEqual(
            cfg.container["runtime"]["args"]["volumes"],
            ["/home/alice/data:/data"],
        )

    @patch.dict(os.environ, {"USER": "alice", "LOGNAME": "", "USERNAME": ""}, clear=False)
    def test_from_configs_resolves_user_id_in_testsuite_override(self):
        # Pins that resolution runs AFTER the cluster+testsuite merge. If a future
        # refactor moves the resolver call before the merge, the testsuite override
        # would leak {user-id} verbatim and this test would fail.
        cluster = {
            "node_dict": {"1.1.1.1": {}},
            "username": "literal_user",
            "priv_key_file": "/dev/null",
        }
        testsuite = {"username": "{user-id}"}
        cfg = OrchestratorConfig.from_configs(cluster, testsuite)
        self.assertEqual(cfg.username, "alice")

    def test_from_configs_exits_on_unresolved_changeme(self):
        # Pins that the resolver's <changeme> guard propagates through from_configs.
        # Defends against a future change that disables the guard and silently leaks
        # <changeme> into Pssh / docker run.
        cluster = {
            "node_dict": {"1.1.1.1": {}},
            "username": "<changeme>",
            "priv_key_file": "/dev/null",
        }
        with self.assertRaises(SystemExit) as ctx:
            OrchestratorConfig.from_configs(cluster)
        self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
