'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

"""Unit tests for the additive Orchestrator ABC surface introduced for the
multi-orch RVS suite migration: privileged_prefix(), prepare(), dispose(),
and the abstract host_all/host_head properties.

Scope: behavioral contract verification only. Cluster-level integration is
covered separately under cvs/tests/health/.
"""

import unittest
from unittest.mock import MagicMock, patch

from cvs.core.orchestrators.base import Orchestrator
from cvs.core.orchestrators.baremetal import BaremetalOrchestrator
from cvs.core.orchestrators.container import ContainerOrchestrator
from cvs.core.orchestrators.factory import OrchestratorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_orch_config(orchestrator="baremetal", with_container=False):
    """Build a minimal OrchestratorConfig that satisfies BaremetalOrchestrator
    and (optionally) ContainerOrchestrator __init__."""
    kwargs = dict(
        orchestrator=orchestrator,
        node_dict={"10.0.0.1": {}, "10.0.0.2": {}},
        username="atnair",
        priv_key_file="/dev/null",
        password=None,
        head_node_dict={"mgmt_ip": "10.0.0.1"},
        container={},
    )
    if with_container:
        kwargs["container"] = {
            "enabled": True,
            "launch": True,
            "image": "rocm/cvs:test",
            "name": "cvs_iter_test",
            "runtime": {"name": "docker", "args": {}},
        }
    return OrchestratorConfig(**kwargs)


# ---------------------------------------------------------------------------
# ABC default behavior
# ---------------------------------------------------------------------------

class TestOrchestratorABCDefaults(unittest.TestCase):
    """Verify the ABC's default implementations are what the plan promises:
    privileged_prefix() == 'sudo ', prepare() and dispose() are no-op
    True-returning hooks. We can't instantiate Orchestrator directly because
    of abstract methods, so we exercise the defaults via a minimal subclass
    that fills in the abstract methods with stubs."""

    def _make_minimal_subclass(self):
        class _Stub(Orchestrator):
            def exec(self, cmd, hosts=None, timeout=None):
                return {}

            def exec_on_head(self, cmd, timeout=None):
                return {}

            def setup_env(self, hosts, env_script=None):
                return True

            def cleanup(self, hosts):
                return True

            @property
            def host_all(self):
                return MagicMock(name="host_all")

            @property
            def host_head(self):
                return MagicMock(name="host_head")

        return _Stub(MagicMock(), MagicMock())

    def test_privileged_prefix_default_is_sudo(self):
        orch = self._make_minimal_subclass()
        self.assertEqual(orch.privileged_prefix(), "sudo ")

    def test_prepare_default_returns_true(self):
        orch = self._make_minimal_subclass()
        self.assertTrue(orch.prepare())

    def test_dispose_default_returns_true(self):
        orch = self._make_minimal_subclass()
        self.assertTrue(orch.dispose())

    def test_abstract_host_handles_required(self):
        """An Orchestrator subclass that does NOT implement host_all/host_head
        cannot be instantiated."""

        class _Missing(Orchestrator):
            def exec(self, cmd, hosts=None, timeout=None):
                return {}

            def exec_on_head(self, cmd, timeout=None):
                return {}

            def setup_env(self, hosts, env_script=None):
                return True

            def cleanup(self, hosts):
                return True

        with self.assertRaises(TypeError):
            _Missing(MagicMock(), MagicMock())


# ---------------------------------------------------------------------------
# BaremetalOrchestrator overrides
# ---------------------------------------------------------------------------

class TestBaremetalOrchestratorABC(unittest.TestCase):

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_host_all_aliases_self_all(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        self.assertIs(orch.host_all, orch.all)

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_host_head_aliases_self_head(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        self.assertIs(orch.host_head, orch.head)

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_privileged_prefix_is_sudo(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        self.assertEqual(orch.privileged_prefix(), "sudo ")

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_dispose_calls_cleanup_on_hosts(self, _mock_pssh):
        log = MagicMock()
        orch = BaremetalOrchestrator(log, _make_orch_config())
        # Replace cleanup with a sentinel so we can assert it was called.
        orch.cleanup = MagicMock(return_value=True)
        self.assertTrue(orch.dispose())
        orch.cleanup.assert_called_once_with(orch.hosts)

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_dispose_returns_false_on_cleanup_exception(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch.cleanup = MagicMock(side_effect=RuntimeError("ssh died"))
        self.assertFalse(orch.dispose())


# ---------------------------------------------------------------------------
# ContainerOrchestrator overrides
# ---------------------------------------------------------------------------

class TestContainerOrchestratorABC(unittest.TestCase):

    def _make(self, _mock_pssh, _mock_runtime_factory):
        cfg = _make_orch_config(orchestrator="container", with_container=True)
        runtime = MagicMock(name="docker_runtime")
        _mock_runtime_factory.create.return_value = runtime
        log = MagicMock()
        orch = ContainerOrchestrator(log, cfg)
        return orch, runtime

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_privileged_prefix_is_empty(self, _mock_pssh, _mock_rf):
        orch, _ = self._make(_mock_pssh, _mock_rf)
        self.assertEqual(orch.privileged_prefix(), "")

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_host_handles_inherit_from_baremetal(self, _mock_pssh, _mock_rf):
        """ContainerOrchestrator does NOT route host_all / host_head through
        the container runtime; they remain the underlying host SSH handles."""
        orch, _ = self._make(_mock_pssh, _mock_rf)
        self.assertIs(orch.host_all, orch.all)
        self.assertIs(orch.host_head, orch.head)

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_prepare_success_path(self, _mock_pssh, _mock_rf):
        orch, _ = self._make(_mock_pssh, _mock_rf)
        orch.setup_containers = MagicMock(return_value=True)
        orch.setup_sshd = MagicMock(return_value=True)
        self.assertTrue(orch.prepare())
        orch.setup_containers.assert_called_once()
        orch.setup_sshd.assert_called_once()

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_prepare_rolls_back_on_setup_containers_failure(self, _mock_pssh, _mock_rf):
        orch, runtime = self._make(_mock_pssh, _mock_rf)
        orch.setup_containers = MagicMock(return_value=False)
        orch.setup_sshd = MagicMock(return_value=True)
        # Simulate that setup_containers managed to register a container_id
        # before failing (e.g., partial success across a multi-node fanout).
        orch.container_id = "cvs_iter_test"
        self.assertFalse(orch.prepare())
        # Rollback path MUST stop the container we partially launched, even
        # though launch:true would normally short-circuit teardown_containers.
        runtime.teardown_containers.assert_called_once_with("cvs_iter_test")
        orch.setup_sshd.assert_not_called()
        # container_id cleared so dispose() is a no-op.
        self.assertIsNone(orch.container_id)

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_prepare_rolls_back_on_setup_sshd_failure(self, _mock_pssh, _mock_rf):
        orch, runtime = self._make(_mock_pssh, _mock_rf)
        orch.setup_containers = MagicMock(return_value=True)
        orch.setup_sshd = MagicMock(return_value=False)
        orch.container_id = "cvs_iter_test"
        self.assertFalse(orch.prepare())
        runtime.teardown_containers.assert_called_once_with("cvs_iter_test")
        self.assertIsNone(orch.container_id)

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_prepare_swallows_exception_and_rolls_back(self, _mock_pssh, _mock_rf):
        orch, runtime = self._make(_mock_pssh, _mock_rf)
        orch.setup_containers = MagicMock(side_effect=RuntimeError("boom"))
        orch.container_id = "cvs_iter_test"
        # Exceptions inside prepare() must NOT propagate; finalizers depend on
        # a clean True/False return.
        self.assertFalse(orch.prepare())
        runtime.teardown_containers.assert_called_once_with("cvs_iter_test")

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_dispose_tears_down_then_cleans_up(self, _mock_pssh, _mock_rf):
        orch, runtime = self._make(_mock_pssh, _mock_rf)
        orch.container_id = "cvs_iter_test"
        orch.cleanup = MagicMock(return_value=True)
        runtime.teardown_containers.return_value = True
        self.assertTrue(orch.dispose())
        # Bypasses the existing teardown_containers() launch:true short-circuit
        # by calling the runtime directly.
        runtime.teardown_containers.assert_called_once_with("cvs_iter_test")
        orch.cleanup.assert_called_once_with(orch.hosts)
        self.assertIsNone(orch.container_id)

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_dispose_safe_when_prepare_never_called(self, _mock_pssh, _mock_rf):
        """Even if prepare() never ran (so container_id is None), dispose()
        must not raise; cleanup() still runs against the hosts."""
        orch, runtime = self._make(_mock_pssh, _mock_rf)
        orch.cleanup = MagicMock(return_value=True)
        # container_id starts as None per __init__.
        self.assertTrue(orch.dispose())
        runtime.teardown_containers.assert_not_called()
        orch.cleanup.assert_called_once_with(orch.hosts)

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_dispose_returns_false_when_teardown_fails(self, _mock_pssh, _mock_rf):
        orch, runtime = self._make(_mock_pssh, _mock_rf)
        orch.container_id = "cvs_iter_test"
        orch.cleanup = MagicMock(return_value=True)
        runtime.teardown_containers.return_value = False
        self.assertFalse(orch.dispose())
        # cleanup() still attempted for best-effort host hygiene.
        orch.cleanup.assert_called_once_with(orch.hosts)

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_dispose_returns_false_when_cleanup_fails(self, _mock_pssh, _mock_rf):
        orch, runtime = self._make(_mock_pssh, _mock_rf)
        orch.container_id = "cvs_iter_test"
        runtime.teardown_containers.return_value = True
        orch.cleanup = MagicMock(return_value=False)
        self.assertFalse(orch.dispose())

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_dispose_swallows_teardown_exception(self, _mock_pssh, _mock_rf):
        orch, runtime = self._make(_mock_pssh, _mock_rf)
        orch.container_id = "cvs_iter_test"
        orch.cleanup = MagicMock(return_value=True)
        runtime.teardown_containers.side_effect = RuntimeError("docker daemon dead")
        # Finalizers must not raise.
        self.assertFalse(orch.dispose())
        # cleanup still attempted for best-effort host hygiene.
        orch.cleanup.assert_called_once_with(orch.hosts)
        self.assertIsNone(orch.container_id)


if __name__ == "__main__":
    unittest.main()
