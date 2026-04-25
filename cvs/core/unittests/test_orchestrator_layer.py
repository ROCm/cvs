'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for the orchestrator layer (factory, OrchestratorConfig, BaremetalOrchestrator,
# ContainerOrchestrator) at base 7f44134, scoped to methods the migrated rvs_cvs.py orch
# fixture relies on. Mocks Pssh and RuntimeFactory so tests run with no SSH or container
# runtime. Notable: pins the existing teardown_containers() short-circuit semantic at
# cvs/core/orchestrators/container.py:439 so the bare-minimum phdl -> orch migration in
# rvs_cvs.py inherits a known, asserted behavior.

import json
import unittest
from unittest.mock import MagicMock, patch

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.core.orchestrators.baremetal import BaremetalOrchestrator
from cvs.core.orchestrators.container import ContainerOrchestrator


def _make_orch_config(orchestrator="baremetal", with_container=False, launch=False, enabled=True):
    """Build a minimal OrchestratorConfig that satisfies BaremetalOrchestrator and
    (optionally) ContainerOrchestrator __init__ without touching disk or SSH."""
    kwargs = dict(
        orchestrator=orchestrator,
        node_dict={"10.0.0.1": {}, "10.0.0.2": {}},
        username="atnair",
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


# ---------------------------------------------------------------------------
# OrchestratorFactory
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# OrchestratorConfig
# ---------------------------------------------------------------------------

class TestOrchestratorConfig(unittest.TestCase):

    def test_get_returns_default_for_missing_key(self):
        cfg = _make_orch_config()
        self.assertEqual(cfg.get("nonexistent_key", "fallback"), "fallback")

    def test_get_returns_value_for_existing_attr(self):
        cfg = _make_orch_config()
        self.assertEqual(cfg.get("username"), "atnair")

    def test_from_configs_defaults_orchestrator_to_baremetal_when_omitted(self):
        # In-memory dict path (no disk IO).
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

    def test_from_configs_reads_from_file_path(self):
        # Round-trip a real JSON file through from_configs to exercise the
        # path-handling branch (the rvs_cvs.py orch fixture goes through this).
        import tempfile
        import os
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


# ---------------------------------------------------------------------------
# BaremetalOrchestrator
# ---------------------------------------------------------------------------

class TestBaremetalOrchestrator(unittest.TestCase):

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_init_constructs_pssh_handles(self, mock_pssh):
        BaremetalOrchestrator(MagicMock(), _make_orch_config())
        # __init__ creates two Pssh handles: self.head and self.all.
        self.assertEqual(mock_pssh.call_count, 2)

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_init_sets_orchestrator_type(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        self.assertEqual(orch.orchestrator_type, "baremetal")

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_init_picks_first_node_as_head(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        # _make_orch_config inserts 10.0.0.1 first.
        self.assertEqual(orch.head_node, "10.0.0.1")

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_exec_delegates_to_all_when_targeting_full_set(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch.all = MagicMock()
        orch.all.exec.return_value = {"10.0.0.1": "ok", "10.0.0.2": "ok"}
        result = orch.exec("ls", timeout=5)
        orch.all.exec.assert_called_once_with("ls", timeout=5)
        self.assertEqual(result, {"10.0.0.1": "ok", "10.0.0.2": "ok"})

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_exec_on_head_delegates_to_head_handle(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch.head = MagicMock()
        orch.head.exec.return_value = {"10.0.0.1": "ok"}
        result = orch.exec_on_head("hostname", timeout=10)
        orch.head.exec.assert_called_once_with("hostname", timeout=10)
        self.assertEqual(result, {"10.0.0.1": "ok"})

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_cleanup_returns_true(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        self.assertTrue(orch.cleanup(orch.hosts))


# ---------------------------------------------------------------------------
# ContainerOrchestrator
# ---------------------------------------------------------------------------

class TestContainerOrchestrator(unittest.TestCase):

    def _make(self, _mock_pssh, _mock_rf, launch=False, enabled=True):
        cfg = _make_orch_config(orchestrator="container", launch=launch, enabled=enabled)
        runtime = MagicMock(name="docker_runtime")
        _mock_rf.create.return_value = runtime
        orch = ContainerOrchestrator(MagicMock(), cfg)
        return orch, runtime

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_init_creates_runtime_via_factory(self, _mock_pssh, mock_rf):
        orch, runtime = self._make(_mock_pssh, mock_rf)
        self.assertIs(orch.runtime, runtime)
        # docker is the default runtime when container.runtime.name == "docker".
        mock_rf.create.assert_called_once()
        self.assertEqual(mock_rf.create.call_args[0][0], "docker")

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_init_sets_orchestrator_type(self, _mock_pssh, mock_rf):
        orch, _ = self._make(_mock_pssh, mock_rf)
        self.assertEqual(orch.orchestrator_type, "container")

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_init_overrides_ssh_port_to_container_sshd(self, _mock_pssh, mock_rf):
        orch, _ = self._make(_mock_pssh, mock_rf)
        self.assertEqual(orch.ssh_port, 2224)

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_setup_containers_launch_true_delegates_to_runtime(self, _mock_pssh, mock_rf):
        orch, runtime = self._make(_mock_pssh, mock_rf, launch=True)
        runtime.setup_containers.return_value = True
        result = orch.setup_containers()
        self.assertTrue(result)
        runtime.setup_containers.assert_called_once()

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_setup_containers_short_circuits_when_disabled(self, _mock_pssh, mock_rf):
        orch, runtime = self._make(_mock_pssh, mock_rf, enabled=False)
        result = orch.setup_containers()
        self.assertTrue(result)  # Disabled is treated as success / no-op.
        runtime.setup_containers.assert_not_called()

    # teardown_containers short-circuit logic (cvs/core/orchestrators/container.py:439).
    # The bare-minimum migration in rvs_cvs.py inherits this behavior; pinning it here
    # ensures any future change has a loud canary.

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_teardown_containers_short_circuits_when_launch_true(self, _mock_pssh, mock_rf):
        # launch:true means containers are externally managed; teardown is a no-op.
        orch, runtime = self._make(_mock_pssh, mock_rf, launch=True)
        orch.container_id = "test_container"
        self.assertTrue(orch.teardown_containers())
        runtime.teardown_containers.assert_not_called()

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_teardown_containers_calls_runtime_when_launch_false(self, _mock_pssh, mock_rf):
        orch, runtime = self._make(_mock_pssh, mock_rf, launch=False)
        orch.container_id = "test_container"
        runtime.teardown_containers.return_value = True
        self.assertTrue(orch.teardown_containers())
        runtime.teardown_containers.assert_called_once_with("test_container")
        # container_id cleared on successful teardown.
        self.assertIsNone(orch.container_id)

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_teardown_containers_short_circuits_when_disabled(self, _mock_pssh, mock_rf):
        orch, runtime = self._make(_mock_pssh, mock_rf, enabled=False)
        orch.container_id = "test_container"
        self.assertTrue(orch.teardown_containers())
        runtime.teardown_containers.assert_not_called()

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_teardown_containers_short_circuits_when_no_container_id(self, _mock_pssh, mock_rf):
        # container_id stays None when prepare/setup never ran; teardown is a no-op.
        orch, runtime = self._make(_mock_pssh, mock_rf, launch=False)
        self.assertIsNone(orch.container_id)
        self.assertTrue(orch.teardown_containers())
        runtime.teardown_containers.assert_not_called()

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_init_requires_container_config(self, _mock_pssh, _mock_rf):
        # ContainerOrchestrator raises if 'container' config is empty. Construct
        # the config directly to bypass the helper's auto-population.
        cfg = OrchestratorConfig(
            orchestrator="container",
            node_dict={"10.0.0.1": {}},
            username="atnair",
            priv_key_file="/dev/null",
            container={},
        )
        with self.assertRaises(ValueError):
            ContainerOrchestrator(MagicMock(), cfg)


if __name__ == "__main__":
    unittest.main()
