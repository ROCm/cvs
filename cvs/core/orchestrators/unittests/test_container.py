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
# no_launch / per_run / persistent contract has a loud canary. Pssh + RuntimeFactory are
# patched once in setUp (not per method); _make() returns a fresh orch + runtime mock.

import unittest
from unittest.mock import MagicMock, patch

from cvs.core.orchestrators.factory import OrchestratorConfig, _resolve_container_lifetime
from cvs.core.orchestrators.container import ContainerOrchestrator


# Reusable runtime.is_running fixtures (two-host cluster).
_RUNNING = {
    "10.0.0.1": {"running": True, "exit_code": 0, "name": "cvs_iter_test"},
    "10.0.0.2": {"running": True, "exit_code": 0, "name": "cvs_iter_test"},
}
_NOT_RUNNING = {
    "10.0.0.1": {"running": False, "exit_code": 0, "name": ""},
    "10.0.0.2": {"running": False, "exit_code": 0, "name": ""},
}
# Probe itself failed on every host (non-zero exit: SSH/sudo/docker error or
# timeout). 'running' is False but it is NOT trustworthy -- state is unknown.
_PROBE_FAILED = {
    "10.0.0.1": {"running": False, "exit_code": 1, "name": ""},
    "10.0.0.2": {"running": False, "exit_code": -1, "name": ""},
}
# Container up on one host; probe aborted (SSH timeout, exit_code -1) on the
# other. The aborted host must be treated as unknown, never as "absent".
_RUNNING_PLUS_PROBE_FAILED = {
    "10.0.0.1": {"running": True, "exit_code": 0, "name": "cvs_iter_test"},
    "10.0.0.2": {"running": False, "exit_code": -1, "name": ""},
}


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
    def setUp(self):
        # Patch SSH transport + runtime factory for every test (replaces the
        # per-method @patch decorators). Mocks are torn down via addCleanup.
        p_pssh = patch("cvs.core.orchestrators.baremetal.Pssh")
        p_rf = patch("cvs.core.orchestrators.container.RuntimeFactory")
        self.mock_pssh = p_pssh.start()
        self.mock_rf = p_rf.start()
        self.addCleanup(p_pssh.stop)
        self.addCleanup(p_rf.stop)

    def _make(self, lifetime="per_run"):
        cfg = _make_orch_config(lifetime=lifetime)
        runtime = MagicMock(name="docker_runtime")
        self.mock_rf.create.return_value = runtime
        orch = ContainerOrchestrator(MagicMock(), cfg)
        return orch, runtime

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def test_init_creates_runtime_via_factory(self):
        orch, runtime = self._make()
        self.assertIs(orch.runtime, runtime)
        # docker is the default runtime when container.runtime.name == "docker".
        self.mock_rf.create.assert_called_once()
        self.assertEqual(self.mock_rf.create.call_args[0][0], "docker")

    def test_init_sets_orchestrator_type(self):
        orch, _ = self._make()
        self.assertEqual(orch.orchestrator_type, "container")

    def test_init_overrides_ssh_port_to_container_sshd(self):
        orch, _ = self._make()
        self.assertEqual(orch.ssh_port, 2224)

    def test_init_requires_container_config(self):
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

    # ------------------------------------------------------------------
    # setup_containers lifetime branching
    # ------------------------------------------------------------------

    def test_setup_containers_per_run_delegates_to_runtime(self):
        orch, runtime = self._make(lifetime="per_run")
        runtime.setup_containers.return_value = True
        self.assertTrue(orch.setup_containers())
        runtime.setup_containers.assert_called_once()

    def test_setup_containers_no_launch_verifies_only(self):
        # no_launch never starts a container; it verifies and sets container_id.
        orch, runtime = self._make(lifetime="no_launch")
        runtime.is_running.return_value = _RUNNING
        self.assertTrue(orch.setup_containers())
        runtime.setup_containers.assert_not_called()
        self.assertEqual(orch.container_id, "cvs_iter_test")

    def test_setup_containers_no_launch_not_running_fails(self):
        # no_launch + container not actually running -> verification fails; nothing
        # is launched.
        orch, runtime = self._make(lifetime="no_launch")
        runtime.is_running.return_value = {
            "10.0.0.1": {"running": False, "exit_code": 0, "name": ""},
        }
        self.assertFalse(orch.setup_containers())
        runtime.setup_containers.assert_not_called()

    def test_setup_containers_persistent_attaches_when_running(self):
        orch, runtime = self._make(lifetime="persistent")
        runtime.is_running.return_value = _RUNNING
        self.assertTrue(orch.setup_containers())
        # Attach path: no new container started.
        runtime.setup_containers.assert_not_called()
        self.assertEqual(orch.container_id, "cvs_iter_test")

    def test_setup_containers_persistent_starts_when_not_running(self):
        orch, runtime = self._make(lifetime="persistent")
        runtime.is_running.return_value = _NOT_RUNNING
        runtime.setup_containers.return_value = True
        self.assertTrue(orch.setup_containers())
        runtime.setup_containers.assert_called_once()

    def test_setup_containers_persistent_partial_running_refuses(self):
        # Running on some hosts but not all must NOT auto-relaunch (that would
        # force-remove and rebuild the still-running hosts, destroying their
        # overlay). It fails loudly and starts nothing.
        orch, runtime = self._make(lifetime="persistent")
        runtime.is_running.return_value = {
            "10.0.0.1": {"running": True, "exit_code": 0, "name": "cvs_iter_test"},
            "10.0.0.2": {"running": False, "exit_code": 0, "name": ""},
        }
        self.assertFalse(orch.setup_containers())
        runtime.setup_containers.assert_not_called()

    def test_setup_containers_persistent_probe_failed_refuses_and_does_not_launch(self):
        # Probe failed on every host (non-zero exit). The container's true state
        # is unknown -- treating it as "absent" and cold-starting would force-remove
        # a possibly-running container and destroy its overlay. Refuse, launch nothing.
        orch, runtime = self._make(lifetime="persistent")
        runtime.is_running.return_value = _PROBE_FAILED
        self.assertFalse(orch.setup_containers())
        runtime.setup_containers.assert_not_called()

    def test_setup_containers_persistent_running_plus_probe_failed_refuses(self):
        # Up on one host, probe aborted on the other. The aborted host must not be
        # bucketed as "absent" (which would trip the partial-relaunch path); an
        # untrustworthy probe is its own hard stop. Nothing is launched.
        orch, runtime = self._make(lifetime="persistent")
        runtime.is_running.return_value = _RUNNING_PLUS_PROBE_FAILED
        self.assertFalse(orch.setup_containers())
        runtime.setup_containers.assert_not_called()

    def test_setup_containers_persistent_host_dropped_from_status_refuses(self):
        # A host expected by node_dict is absent from is_running's result entirely
        # (pruned as unreachable by an earlier command). It must surface as a probe
        # failure, not be silently ignored into a false "running on all" attach.
        orch, runtime = self._make(lifetime="persistent")
        runtime.is_running.return_value = {
            "10.0.0.1": {"running": True, "exit_code": 0, "name": "cvs_iter_test"},
            # 10.0.0.2 missing
        }
        self.assertFalse(orch.setup_containers())
        runtime.setup_containers.assert_not_called()
        self.assertIsNone(orch.container_id)

    def test_partition_by_status_three_buckets(self):
        # Direct test of the shared partition helper: one host per bucket.
        orch, _ = self._make(lifetime="persistent")
        orch.hosts = ["h_run", "h_absent", "h_failed"]
        status = {
            "h_run": {"running": True, "exit_code": 0, "name": "cvs_iter_test"},
            "h_absent": {"running": False, "exit_code": 0, "name": ""},
            "h_failed": {"running": False, "exit_code": 1, "name": ""},
        }
        running, absent, probe_failed = orch._partition_by_status(status)
        self.assertEqual(running, ["h_run"])
        self.assertEqual(absent, ["h_absent"])
        self.assertEqual(probe_failed, ["h_failed"])

    def test_partition_by_status_missing_host_is_probe_failed(self):
        # A host expected in self.hosts but absent from status -> probe_failed.
        orch, _ = self._make(lifetime="persistent")
        orch.hosts = ["h_run", "h_gone"]
        status = {"h_run": {"running": True, "exit_code": 0, "name": "cvs_iter_test"}}
        running, absent, probe_failed = orch._partition_by_status(status)
        self.assertEqual(running, ["h_run"])
        self.assertEqual(absent, [])
        self.assertEqual(probe_failed, ["h_gone"])

    # ------------------------------------------------------------------
    # setup_sshd single-node guard
    # ------------------------------------------------------------------

    def test_setup_sshd_single_node_skips_and_returns_true(self):
        # The in-container sshd is only needed for multinode MPI. On a single-host
        # cluster setup_sshd must short-circuit: no exec into the container, no
        # dependency on the image shipping /usr/sbin/sshd.
        orch, runtime = self._make(lifetime="per_run")
        orch.hosts = ["10.0.0.1"]
        orch.container_id = "cvs_iter_test"
        self.assertTrue(orch.setup_sshd())
        runtime.exec.assert_not_called()

    def test_setup_sshd_requires_container_even_single_node(self):
        # The container_id precondition is checked BEFORE the single-node guard,
        # so a single-node orch with no running container still raises rather than
        # silently returning True.
        orch, _ = self._make(lifetime="per_run")
        orch.hosts = ["10.0.0.1"]
        self.assertIsNone(orch.container_id)
        with self.assertRaises(RuntimeError):
            orch.setup_sshd()

    @patch("time.sleep", lambda *_a, **_k: None)
    def test_setup_sshd_multinode_attempts_setup(self):
        # The guard must NOT skip a genuine multinode run: every setup command and
        # the final validation probe are exec'd into the container.
        orch, runtime = self._make(lifetime="per_run")
        orch.container_id = "cvs_iter_test"
        runtime.exec.return_value = {
            "10.0.0.1": {"exit_code": 0},
            "10.0.0.2": {"exit_code": 0},
        }
        self.assertTrue(orch.setup_sshd())
        self.assertTrue(runtime.exec.called)

    # ------------------------------------------------------------------
    # teardown_containers lifetime branching
    # ------------------------------------------------------------------

    def test_teardown_containers_short_circuits_when_lifetime_no_launch(self):
        # no_launch means CVS did not launch the container; teardown is a no-op.
        orch, runtime = self._make(lifetime="no_launch")
        orch.container_id = "test_container"
        self.assertTrue(orch.teardown_containers())
        runtime.teardown_containers.assert_not_called()

    def test_teardown_containers_persistent_is_noop(self):
        # persistent leaves the container running for the next run.
        orch, runtime = self._make(lifetime="persistent")
        orch.container_id = "test_container"
        self.assertTrue(orch.teardown_containers())
        runtime.teardown_containers.assert_not_called()

    def test_teardown_containers_calls_runtime_when_lifetime_per_run(self):
        # per_run means CVS owns the container lifecycle; teardown delegates to runtime.
        orch, runtime = self._make(lifetime="per_run")
        orch.container_id = "test_container"
        runtime.teardown_containers.return_value = True
        self.assertTrue(orch.teardown_containers())
        runtime.teardown_containers.assert_called_once_with("test_container")
        # container_id cleared on successful teardown.
        self.assertIsNone(orch.container_id)

    def test_teardown_containers_short_circuits_when_no_container_id(self):
        # container_id stays None when setup never ran; per_run teardown is a no-op
        # since there is nothing to tear down.
        orch, runtime = self._make(lifetime="per_run")
        self.assertIsNone(orch.container_id)
        self.assertTrue(orch.teardown_containers())
        runtime.teardown_containers.assert_not_called()


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

    def test_explicit_lifetime_kept(self):
        out = _resolve_container_lifetime({"lifetime": "persistent", "image": "x"})
        self.assertEqual(out["lifetime"], "persistent")

    def test_invalid_lifetime_raises(self):
        with self.assertRaises(ValueError):
            _resolve_container_lifetime({"lifetime": "forever", "image": "x"})

    def test_launch_present_raises(self):
        # launch is a removed field: any value is a hard error (no silent mapping).
        with self.assertRaises(ValueError) as ctx:
            _resolve_container_lifetime({"launch": True, "image": "x"})
        self.assertIn("launch", str(ctx.exception))

    def test_launch_false_also_raises(self):
        with self.assertRaises(ValueError):
            _resolve_container_lifetime({"launch": False, "image": "x"})

    def test_launch_alongside_lifetime_raises(self):
        # A stale launch flag next to an explicit lifetime must fail loudly rather
        # than being silently retained/ignored.
        with self.assertRaises(ValueError):
            _resolve_container_lifetime({"lifetime": "per_run", "launch": True, "image": "x"})

    def test_no_policy_keys_defaults_to_per_run(self):
        out = _resolve_container_lifetime({"image": "x"})
        self.assertEqual(out["lifetime"], "per_run")

    def test_empty_block_untouched(self):
        # Baremetal path: empty container block stays empty (no lifetime injected).
        self.assertEqual(_resolve_container_lifetime({}), {})

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_orchestratorconfig_init_rejects_launch(self, _mock_pssh):
        # Direct construction routes through __init__ -> the same helper, so a
        # legacy launch flag is rejected identically to from_configs.
        with self.assertRaises(ValueError):
            OrchestratorConfig(
                orchestrator="container",
                node_dict={"1.1.1.1": {}},
                username="u",
                priv_key_file="/dev/null",
                container={"launch": True, "image": "x"},
            )


if __name__ == "__main__":
    unittest.main()
