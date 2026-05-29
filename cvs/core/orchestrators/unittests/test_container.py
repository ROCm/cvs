'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/core/orchestrators/container.py: ContainerOrchestrator construction
# (incl. SSH port override and runtime wiring) and the setup_containers / teardown_containers
# lifetime branching inherited by the rvs_cvs.py orch fixture. Mocks Pssh and RuntimeFactory
# so tests run with no SSH or container runtime.
#
# The per-lifetime setup/teardown behavior is pinned here so any future change to the
# external / per_run / persistent contract has a loud canary.

import unittest
import warnings
from unittest.mock import MagicMock, patch

from cvs.core.orchestrators.factory import OrchestratorConfig, _resolve_container_lifetime
from cvs.core.orchestrators.container import ContainerOrchestrator


def _make_orch_config(lifetime="per_run"):
    """Minimal OrchestratorConfig that satisfies ContainerOrchestrator.__init__
    without touching disk or SSH."""
    return OrchestratorConfig(
        orchestrator="container",
        node_dict={"10.0.0.1": {}, "10.0.0.2": {}},
        username="testuser",
        priv_key_file="/dev/null",
        password=None,
        head_node_dict={"mgmt_ip": "10.0.0.1"},
        container={
            "lifetime": lifetime,
            "image": "rocm/cvs:test",
            "name": "cvs_iter_test",
            "runtime": {"name": "docker", "args": {}},
        },
    )


class TestContainerOrchestrator(unittest.TestCase):
    def _make(self, _mock_pssh, _mock_rf, lifetime="per_run"):
        cfg = _make_orch_config(lifetime=lifetime)
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

    # ------------------------------------------------------------------
    # setup_containers lifetime branching
    # ------------------------------------------------------------------

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_setup_containers_per_run_delegates_to_runtime(self, _mock_pssh, mock_rf):
        orch, runtime = self._make(_mock_pssh, mock_rf, lifetime="per_run")
        runtime.setup_containers.return_value = True
        result = orch.setup_containers()
        self.assertTrue(result)
        runtime.setup_containers.assert_called_once()

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_setup_containers_external_verifies_only(self, _mock_pssh, mock_rf):
        # external never starts a container; it verifies and sets container_id.
        orch, runtime = self._make(_mock_pssh, mock_rf, lifetime="external")
        runtime.is_running.return_value = {
            "10.0.0.1": {"running": True, "exit_code": 0, "name": "cvs_iter_test"},
            "10.0.0.2": {"running": True, "exit_code": 0, "name": "cvs_iter_test"},
        }
        self.assertTrue(orch.setup_containers())
        runtime.setup_containers.assert_not_called()
        self.assertEqual(orch.container_id, "cvs_iter_test")

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_setup_containers_persistent_attaches_when_running(self, _mock_pssh, mock_rf):
        orch, runtime = self._make(_mock_pssh, mock_rf, lifetime="persistent")
        runtime.is_running.return_value = {
            "10.0.0.1": {"running": True, "exit_code": 0, "name": "cvs_iter_test"},
            "10.0.0.2": {"running": True, "exit_code": 0, "name": "cvs_iter_test"},
        }
        runtime.image_sha_status.return_value = {
            "10.0.0.1": {"container_sha": "sha:abc", "image_sha": "sha:abc", "exit_code": 0},
            "10.0.0.2": {"container_sha": "sha:abc", "image_sha": "sha:abc", "exit_code": 0},
        }
        self.assertTrue(orch.setup_containers())
        # Attach path: no new container started.
        runtime.setup_containers.assert_not_called()
        self.assertEqual(orch.container_id, "cvs_iter_test")

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_setup_containers_persistent_starts_when_not_running(self, _mock_pssh, mock_rf):
        orch, runtime = self._make(_mock_pssh, mock_rf, lifetime="persistent")
        runtime.is_running.return_value = {
            "10.0.0.1": {"running": False, "exit_code": 0, "name": ""},
            "10.0.0.2": {"running": False, "exit_code": 0, "name": ""},
        }
        runtime.setup_containers.return_value = True
        self.assertTrue(orch.setup_containers())
        runtime.setup_containers.assert_called_once()
        # No SHA check when launching fresh.
        runtime.image_sha_status.assert_not_called()

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_setup_containers_persistent_idempotent_on_resetup(self, _mock_pssh, mock_rf):
        # Re-running setup against an already-running persistent container is a
        # no-op attach both times -- never starts a new container.
        orch, runtime = self._make(_mock_pssh, mock_rf, lifetime="persistent")
        runtime.is_running.return_value = {
            "10.0.0.1": {"running": True, "exit_code": 0, "name": "cvs_iter_test"},
        }
        runtime.image_sha_status.return_value = {
            "10.0.0.1": {"container_sha": "sha:abc", "image_sha": "sha:abc", "exit_code": 0},
        }
        self.assertTrue(orch.setup_containers())
        self.assertTrue(orch.setup_containers())
        runtime.setup_containers.assert_not_called()

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_setup_containers_persistent_cross_host_sha_skew_errors(self, _mock_pssh, mock_rf):
        # Nodes running different image SHAs is a correctness error, not staleness.
        orch, runtime = self._make(_mock_pssh, mock_rf, lifetime="persistent")
        runtime.is_running.return_value = {
            "10.0.0.1": {"running": True, "exit_code": 0, "name": "cvs_iter_test"},
            "10.0.0.2": {"running": True, "exit_code": 0, "name": "cvs_iter_test"},
        }
        runtime.image_sha_status.return_value = {
            "10.0.0.1": {"container_sha": "sha:abc", "image_sha": "sha:abc", "exit_code": 0},
            "10.0.0.2": {"container_sha": "sha:def", "image_sha": "sha:def", "exit_code": 0},
        }
        self.assertFalse(orch.setup_containers())

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_setup_containers_persistent_stale_overlay_warns_but_passes(self, _mock_pssh, mock_rf):
        # Per-host staleness (container older than local image tag) warns, does not fail.
        orch, runtime = self._make(_mock_pssh, mock_rf, lifetime="persistent")
        runtime.is_running.return_value = {
            "10.0.0.1": {"running": True, "exit_code": 0, "name": "cvs_iter_test"},
        }
        runtime.image_sha_status.return_value = {
            "10.0.0.1": {"container_sha": "sha:old", "image_sha": "sha:new", "exit_code": 0},
        }
        self.assertTrue(orch.setup_containers())

    # ------------------------------------------------------------------
    # teardown_containers lifetime branching
    # ------------------------------------------------------------------

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_teardown_containers_short_circuits_when_lifetime_external(self, _mock_pssh, mock_rf):
        # external means containers are externally managed; teardown is a no-op.
        orch, runtime = self._make(_mock_pssh, mock_rf, lifetime="external")
        orch.container_id = "test_container"
        self.assertTrue(orch.teardown_containers())
        runtime.teardown_containers.assert_not_called()

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_teardown_containers_persistent_is_noop(self, _mock_pssh, mock_rf):
        # persistent leaves the container running for the next run.
        orch, runtime = self._make(_mock_pssh, mock_rf, lifetime="persistent")
        orch.container_id = "test_container"
        self.assertTrue(orch.teardown_containers())
        runtime.teardown_containers.assert_not_called()

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_teardown_containers_calls_runtime_when_lifetime_per_run(self, _mock_pssh, mock_rf):
        # per_run means CVS owns the container lifecycle; teardown delegates to runtime.
        orch, runtime = self._make(_mock_pssh, mock_rf, lifetime="per_run")
        orch.container_id = "test_container"
        runtime.teardown_containers.return_value = True
        self.assertTrue(orch.teardown_containers())
        runtime.teardown_containers.assert_called_once_with("test_container")
        # container_id cleared on successful teardown.
        self.assertIsNone(orch.container_id)

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_teardown_containers_short_circuits_when_no_container_id(self, _mock_pssh, mock_rf):
        # container_id stays None when setup never ran; per_run teardown is a no-op
        # since there is nothing to tear down.
        orch, runtime = self._make(_mock_pssh, mock_rf, lifetime="per_run")
        self.assertIsNone(orch.container_id)
        self.assertTrue(orch.teardown_containers())
        runtime.teardown_containers.assert_not_called()

    # ------------------------------------------------------------------
    # setup_sshd idempotency (required by lifetime: persistent)
    # ------------------------------------------------------------------

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_setup_sshd_idempotent_when_already_running(self, _mock_pssh, mock_rf):
        # When the pgrep precheck reports sshd already on 2224 for every host, the
        # start commands must NOT run (would re-bind the port and fail).
        orch, runtime = self._make(_mock_pssh, mock_rf, lifetime="persistent")
        orch.container_id = "cvs_iter_test"
        runtime.exec.return_value = {
            "10.0.0.1": {"exit_code": 0, "output": ""},
            "10.0.0.2": {"exit_code": 0, "output": ""},
        }
        self.assertTrue(orch.setup_sshd())
        # Only the precheck exec happened; no setup commands were issued.
        runtime.exec.assert_called_once()

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_init_requires_container_config(self, _mock_pssh, _mock_rf):
        # ContainerOrchestrator raises if 'container' config is empty.
        cfg = OrchestratorConfig(
            orchestrator="container",
            node_dict={"10.0.0.1": {}},
            username="testuser",
            priv_key_file="/dev/null",
            container={},
        )
        with self.assertRaises(ValueError):
            ContainerOrchestrator(MagicMock(), cfg)


class TestResolveContainerLifetime(unittest.TestCase):
    """One assertion per row of the lifetime resolution table."""

    def test_enabled_present_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _resolve_container_lifetime({"enabled": True, "image": "x"})
        self.assertIn("enabled", str(ctx.exception))

    def test_enabled_false_also_raises(self):
        # Any value of the removed field is a hard error, including False.
        with self.assertRaises(ValueError):
            _resolve_container_lifetime({"enabled": False, "image": "x"})

    def test_explicit_lifetime_kept_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would fail the test
            out = _resolve_container_lifetime({"lifetime": "persistent", "image": "x"})
        self.assertEqual(out["lifetime"], "persistent")

    def test_invalid_lifetime_raises(self):
        with self.assertRaises(ValueError):
            _resolve_container_lifetime({"lifetime": "forever", "image": "x"})

    def test_launch_true_maps_to_per_run_with_deprecation(self):
        with self.assertWarns(DeprecationWarning):
            out = _resolve_container_lifetime({"launch": True, "image": "x"})
        self.assertEqual(out["lifetime"], "per_run")
        self.assertNotIn("launch", out)

    def test_launch_false_maps_to_external_with_deprecation(self):
        with self.assertWarns(DeprecationWarning):
            out = _resolve_container_lifetime({"launch": False, "image": "x"})
        self.assertEqual(out["lifetime"], "external")
        self.assertNotIn("launch", out)

    def test_no_policy_keys_defaults_to_per_run(self):
        out = _resolve_container_lifetime({"image": "x"})
        self.assertEqual(out["lifetime"], "per_run")

    def test_empty_block_untouched(self):
        # Baremetal path: empty container block stays empty (no lifetime injected).
        self.assertEqual(_resolve_container_lifetime({}), {})

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_orchestratorconfig_init_resolves_launch_alias(self, _mock_pssh):
        # Direct construction routes through __init__ -> the same helper, so a
        # legacy launch flag is resolved identically to from_configs.
        with self.assertWarns(DeprecationWarning):
            cfg = OrchestratorConfig(
                orchestrator="container",
                node_dict={"1.1.1.1": {}},
                username="u",
                priv_key_file="/dev/null",
                container={"launch": True, "image": "x"},
            )
        self.assertEqual(cfg.container["lifetime"], "per_run")


if __name__ == "__main__":
    unittest.main()
