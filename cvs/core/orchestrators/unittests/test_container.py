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
# external / per_run / persistent contract has a loud canary. Pssh + RuntimeFactory are
# patched once in setUp (not per method); _make() returns a fresh orch + runtime mock.

import base64
import unittest
import warnings
from unittest.mock import MagicMock, mock_open, patch

from cvs.core.orchestrators.factory import OrchestratorConfig, _resolve_container_lifetime
from cvs.core.orchestrators.container import ContainerOrchestrator, MAX_INLINE_SETUP_SCRIPT_BYTES


# Reusable runtime.exec / is_running / image_sha_status fixtures (two-host cluster).
_OK = {
    "10.0.0.1": {"exit_code": 0, "output": ""},
    "10.0.0.2": {"exit_code": 0, "output": ""},
}
_RUNNING = {
    "10.0.0.1": {"running": True, "exit_code": 0, "name": "cvs_iter_test"},
    "10.0.0.2": {"running": True, "exit_code": 0, "name": "cvs_iter_test"},
}
_NOT_RUNNING = {
    "10.0.0.1": {"running": False, "exit_code": 0, "name": ""},
    "10.0.0.2": {"running": False, "exit_code": 0, "name": ""},
}
_SHA_MATCH = {
    "10.0.0.1": {"container_sha": "sha:abc", "image_sha": "sha:abc", "exit_code": 0},
    "10.0.0.2": {"container_sha": "sha:abc", "image_sha": "sha:abc", "exit_code": 0},
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
        # The fresh-launch path also provisions via self.exec (-> runtime.exec).
        runtime.exec.return_value = _OK
        self.assertTrue(orch.setup_containers())
        runtime.setup_containers.assert_called_once()

    def test_setup_containers_external_verifies_only(self):
        # external never starts a container; it verifies and sets container_id.
        orch, runtime = self._make(lifetime="external")
        runtime.is_running.return_value = _RUNNING
        self.assertTrue(orch.setup_containers())
        runtime.setup_containers.assert_not_called()
        self.assertEqual(orch.container_id, "cvs_iter_test")

    def test_setup_containers_external_not_running_fails(self):
        # external + container not actually running -> verification fails; nothing
        # is launched and nothing is provisioned.
        orch, runtime = self._make(lifetime="external")
        runtime.is_running.return_value = {
            "10.0.0.1": {"running": False, "exit_code": 0, "name": ""},
        }
        self.assertFalse(orch.setup_containers())
        runtime.setup_containers.assert_not_called()
        runtime.exec.assert_not_called()

    def test_setup_containers_persistent_attaches_when_running(self):
        orch, runtime = self._make(lifetime="persistent")
        runtime.is_running.return_value = _RUNNING
        runtime.image_sha_status.return_value = _SHA_MATCH
        self.assertTrue(orch.setup_containers())
        # Attach path: no new container started.
        runtime.setup_containers.assert_not_called()
        self.assertEqual(orch.container_id, "cvs_iter_test")

    def test_setup_containers_persistent_starts_when_not_running(self):
        orch, runtime = self._make(lifetime="persistent")
        runtime.is_running.return_value = _NOT_RUNNING
        runtime.setup_containers.return_value = True
        runtime.exec.return_value = _OK
        self.assertTrue(orch.setup_containers())
        runtime.setup_containers.assert_called_once()
        # No SHA check when launching fresh.
        runtime.image_sha_status.assert_not_called()

    def test_setup_containers_persistent_idempotent_on_resetup(self):
        # Re-running setup against an already-running persistent container is a
        # no-op attach both times -- never starts a new container.
        orch, runtime = self._make(lifetime="persistent")
        runtime.is_running.return_value = {
            "10.0.0.1": {"running": True, "exit_code": 0, "name": "cvs_iter_test"},
        }
        runtime.image_sha_status.return_value = {
            "10.0.0.1": {"container_sha": "sha:abc", "image_sha": "sha:abc", "exit_code": 0},
        }
        self.assertTrue(orch.setup_containers())
        self.assertTrue(orch.setup_containers())
        runtime.setup_containers.assert_not_called()

    def test_setup_containers_persistent_cross_host_sha_skew_errors(self):
        # Nodes running different image SHAs is a correctness error, not staleness.
        orch, runtime = self._make(lifetime="persistent")
        runtime.is_running.return_value = _RUNNING
        runtime.image_sha_status.return_value = {
            "10.0.0.1": {"container_sha": "sha:abc", "image_sha": "sha:abc", "exit_code": 0},
            "10.0.0.2": {"container_sha": "sha:def", "image_sha": "sha:def", "exit_code": 0},
        }
        self.assertFalse(orch.setup_containers())

    def test_setup_containers_persistent_stale_overlay_warns_but_passes(self):
        # Per-host staleness (container older than local image tag) warns, does not fail.
        orch, runtime = self._make(lifetime="persistent")
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

    def test_teardown_containers_short_circuits_when_lifetime_external(self):
        # external means containers are externally managed; teardown is a no-op.
        orch, runtime = self._make(lifetime="external")
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

    # ------------------------------------------------------------------
    # setup_sshd (required by lifetime: persistent)
    # ------------------------------------------------------------------

    def test_setup_sshd_idempotent_when_already_running(self):
        # When the pgrep precheck reports sshd already on 2224 for every host, the
        # start commands must NOT run (would re-bind the port and fail).
        orch, runtime = self._make(lifetime="persistent")
        orch.container_id = "cvs_iter_test"
        runtime.exec.return_value = _OK
        self.assertTrue(orch.setup_sshd())
        # Only the precheck exec happened; no setup commands were issued.
        runtime.exec.assert_called_once()
        # Guard the subtle self-match fix: the precheck must use the `[s]shd`
        # trick, not a literal 'sshd.*2224' (which matches pgrep's own parent
        # shell and makes this precheck always report "already running").
        precheck_cmd = runtime.exec.call_args[0][1]
        self.assertIn("[s]shd.*2224", precheck_cmd)
        self.assertNotIn("'sshd.*2224'", precheck_cmd)

    @patch("time.sleep")
    def test_setup_sshd_installs_and_validates_when_not_running(self, _mock_sleep):
        # The other setup_sshd branch: precheck reports NO sshd -> the start
        # commands run (incl. /usr/sbin/sshd -p2224) and the post-start validation
        # runs. Both pgrep sites must use the non-self-matching [s]shd pattern.
        orch, runtime = self._make(lifetime="per_run")
        orch.container_id = "cvs_iter_test"
        need = {
            "10.0.0.1": {"exit_code": 1, "output": ""},
            "10.0.0.2": {"exit_code": 1, "output": ""},
        }
        # precheck (needs sshd) -> 6 ssh_setup_commands -> post-start validation.
        runtime.exec.side_effect = [need] + [_OK] * 6 + [_OK]
        self.assertTrue(orch.setup_sshd())
        issued = [c[0][1] for c in runtime.exec.call_args_list]
        self.assertIn("/usr/sbin/sshd -p2224", issued)
        # precheck (first) and post-start validation (last) both use the trick.
        self.assertIn("[s]shd.*2224", issued[0])
        self.assertIn("[s]shd.*2224", issued[-1])

    # ------------------------------------------------------------------
    # Container provisioning (setup_script) -- runs only on fresh launch.
    # ------------------------------------------------------------------

    def test_provisioning_runs_only_on_fresh_launch(self):
        # Dispatch matrix: provisioning (one exec) happens only on a fresh launch
        # (per_run, persistent cold-start); external and persistent-attach skip it.
        # (name, lifetime, is_running, image_sha, expect_provision)
        cases = [
            ("per_run", "per_run", None, None, True),
            ("persistent_cold", "persistent", _NOT_RUNNING, None, True),
            ("external", "external", _RUNNING, None, False),
            ("persistent_attach", "persistent", _RUNNING, _SHA_MATCH, False),
        ]
        for name, lifetime, is_running, image_sha, expect in cases:
            with self.subTest(name):
                orch, runtime = self._make(lifetime=lifetime)
                runtime.setup_containers.return_value = True
                runtime.exec.return_value = _OK
                if is_running is not None:
                    runtime.is_running.return_value = is_running
                if image_sha is not None:
                    runtime.image_sha_status.return_value = image_sha
                self.assertTrue(orch.setup_containers())
                if expect:
                    runtime.exec.assert_called_once()
                    provision_cmd = runtime.exec.call_args[0][1]
                    self.assertIn("base64 -d", provision_cmd)
                    self.assertIn("| bash", provision_cmd)
                else:
                    runtime.exec.assert_not_called()

    def test_provisioning_size_guard(self):
        # Strict `>`: exactly MAX bytes proceeds; MAX+1 is rejected before any exec.
        cases = [
            ("at_limit_ok", b"x" * MAX_INLINE_SETUP_SCRIPT_BYTES, True),
            ("over_limit_rejected", b"x" * (MAX_INLINE_SETUP_SCRIPT_BYTES + 1), False),
        ]
        for name, payload, expect_ok in cases:
            with self.subTest(name):
                orch, runtime = self._make(lifetime="per_run")
                orch.container_id = "cvs_iter_test"
                runtime.exec.return_value = {"10.0.0.1": {"exit_code": 0, "output": ""}}
                with patch("builtins.open", mock_open(read_data=payload)):
                    self.assertEqual(orch._provision_container(), expect_ok)
                if expect_ok:
                    runtime.exec.assert_called_once()
                else:
                    runtime.exec.assert_not_called()

    def test_provisioning_failure_fails_launch(self):
        # A non-zero provisioning exit on any host fails setup_containers.
        orch, runtime = self._make(lifetime="per_run")
        runtime.setup_containers.return_value = True
        runtime.exec.return_value = {
            "10.0.0.1": {"exit_code": 0, "output": ""},
            "10.0.0.2": {"exit_code": 100, "output": "apt failed"},
        }
        self.assertFalse(orch.setup_containers())

    def test_provisioning_not_attempted_when_runtime_launch_fails(self):
        # If the container failed to start, provisioning must not be attempted.
        orch, runtime = self._make(lifetime="per_run")
        runtime.setup_containers.return_value = False
        self.assertFalse(orch.setup_containers())
        runtime.exec.assert_not_called()

    def test_provisioning_noop_when_no_setup_script(self):
        # Defensive skip: a hand-built config with no setup_script provisions nothing.
        orch, runtime = self._make(lifetime="per_run")
        orch.container_id = "cvs_iter_test"
        orch.container_config["setup_script"] = None
        self.assertTrue(orch._provision_container())
        runtime.exec.assert_not_called()

    def test_provisioning_failure_logs_every_host_with_detail(self):
        # F1: all failing hosts are logged (not just the first), each with its
        # captured stderr/stdout so the failure is diagnosable from the log.
        orch, runtime = self._make(lifetime="per_run")
        orch.container_id = "cvs_iter_test"
        runtime.exec.return_value = {
            "10.0.0.1": {"exit_code": 100, "output": "apt boom one"},
            "10.0.0.2": {"exit_code": 5, "output": "apt boom two"},
        }
        self.assertFalse(orch._provision_container())
        logged = " ".join(str(c) for c in orch.log.error.call_args_list)
        self.assertIn("10.0.0.1", logged)
        self.assertIn("apt boom one", logged)
        self.assertIn("10.0.0.2", logged)
        self.assertIn("apt boom two", logged)

    def test_provisioning_ships_the_user_supplied_script(self):
        # A user-supplied setup_script (not just the default) is the payload shipped.
        orch, runtime = self._make(lifetime="per_run")
        orch.container_id = "cvs_iter_test"
        custom = b"#!/bin/bash\necho custom-provision\napt-get install -y foo\n"
        runtime.exec.return_value = {"10.0.0.1": {"exit_code": 0, "output": ""}}
        with patch("builtins.open", mock_open(read_data=custom)):
            self.assertTrue(orch._provision_container())
        cmd = runtime.exec.call_args[0][1]
        encoded = cmd.split("echo ", 1)[1].split(" | base64", 1)[0]
        self.assertEqual(base64.b64decode(encoded), custom)

    def test_provisioning_payload_is_the_resolved_script(self):
        # Pin that the base64 actually carries the resolved setup_script bytes,
        # not just that the command contains "base64 -d" (guards a broken encoder).
        orch, runtime = self._make(lifetime="per_run")
        runtime.setup_containers.return_value = True
        runtime.exec.return_value = _OK
        self.assertTrue(orch.setup_containers())
        cmd = runtime.exec.call_args[0][1]
        encoded = cmd.split("echo ", 1)[1].split(" | base64", 1)[0]
        with open(orch.container_config["setup_script"], "rb") as f:
            self.assertEqual(base64.b64decode(encoded), f.read())

    def test_provisioning_read_failure_returns_false(self):
        # An unreadable setup_script fails the launch and never reaches docker exec.
        orch, runtime = self._make(lifetime="per_run")
        orch.container_id = "cvs_iter_test"
        with patch("builtins.open", side_effect=OSError("boom")):
            self.assertFalse(orch._provision_container())
        runtime.exec.assert_not_called()


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
