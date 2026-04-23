"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

import logging
import unittest
from unittest.mock import MagicMock

from cvs.core.lifecycle.base import PhaseError
from cvs.core.lifecycle.phases.multinode_ssh import (
    SETUP_CMDS,
    SSHD_CHECK_CMD,
    SSHD_STOP_CMD,
    MultinodeSshPhase,
)
from cvs.core.scope import ExecResult, ExecScope, ExecTarget


def _ok(host="h1"):
    return ExecResult(host=host, output="", exit_code=0)


def _fail(host="h1", msg="boom"):
    return ExecResult(host=host, output=msg, exit_code=1)


def _stub_orch(capabilities, exec_results_per_call):
    """Build a stub orchestrator with a runtime that has the given capabilities,
    and an exec method that returns the next entry from exec_results_per_call
    on each call.
    """
    runtime = MagicMock()
    runtime.capabilities = capabilities
    orch = MagicMock()
    orch.runtime = runtime
    orch.hosts = ["h1", "h2"]
    orch.log = logging.getLogger("test_multinode_ssh_phase")
    orch.exec = MagicMock(side_effect=list(exec_results_per_call))
    return orch


class TestMultinodeSshPhaseAppliesTo(unittest.TestCase):
    def test_applies_when_runtime_advertises_in_namespace_sshd(self):
        phase = MultinodeSshPhase()
        orch = MagicMock()
        orch.runtime.capabilities = {"in_namespace_sshd"}
        self.assertTrue(phase.applies_to(orch))

    def test_does_not_apply_when_capability_absent(self):
        phase = MultinodeSshPhase()
        orch = MagicMock()
        orch.runtime.capabilities = set()
        self.assertFalse(phase.applies_to(orch))

    def test_does_not_apply_for_unrelated_capabilities(self):
        phase = MultinodeSshPhase()
        orch = MagicMock()
        orch.runtime.capabilities = {"exposes_gpus", "isolates_workload"}
        self.assertFalse(phase.applies_to(orch))


class TestMultinodeSshPhaseHappyPath(unittest.TestCase):
    def test_runs_setup_commands_then_validates_with_pgrep(self):
        phase = MultinodeSshPhase()
        # 6 setup commands + 1 sshd check; all succeed on both hosts.
        all_ok = {h: _ok(h) for h in ["h1", "h2"]}
        results = [all_ok] * (len(SETUP_CMDS) + 1)
        orch = _stub_orch({"in_namespace_sshd"}, results)

        artifact: dict = {}
        phase.run(orch, artifact)

        # Each setup command was called with scope=ALL, target=RUNTIME.
        all_calls = orch.exec.call_args_list
        self.assertEqual(len(all_calls), len(SETUP_CMDS) + 1)
        for i, cmd in enumerate(SETUP_CMDS):
            self.assertEqual(all_calls[i].args[0], cmd)
            self.assertEqual(all_calls[i].kwargs["scope"], ExecScope.ALL)
            self.assertEqual(all_calls[i].kwargs["target"], ExecTarget.RUNTIME)
        # The validation call uses the SSHD_CHECK_CMD.
        self.assertEqual(all_calls[-1].args[0], SSHD_CHECK_CMD)

        # Artifact records the port + hosts the sshd was started on.
        self.assertEqual(artifact["port"], 2224)
        self.assertEqual(artifact["hosts"], ["h1", "h2"])


class TestMultinodeSshPhaseFailureAtSetup(unittest.TestCase):
    def test_failed_setup_command_raises_PhaseError_with_host_and_cmd(self):
        phase = MultinodeSshPhase()
        # First setup command fails on h2.
        bad = {"h1": _ok("h1"), "h2": _fail("h2", "permission denied")}
        results = [bad]
        orch = _stub_orch({"in_namespace_sshd"}, results)

        with self.assertRaises(PhaseError) as ctx:
            phase.run(orch, {})

        self.assertIn("h2", str(ctx.exception))
        self.assertIn(SETUP_CMDS[0], str(ctx.exception))


class TestMultinodeSshPhaseValidationFailure(unittest.TestCase):
    def test_pgrep_failure_after_setup_raises_PhaseError(self):
        phase = MultinodeSshPhase()
        all_ok = {h: _ok(h) for h in ["h1", "h2"]}
        check_failed = {"h1": _ok("h1"), "h2": _fail("h2", "no sshd")}
        results = [all_ok] * len(SETUP_CMDS) + [check_failed]
        orch = _stub_orch({"in_namespace_sshd"}, results)

        with self.assertRaises(PhaseError) as ctx:
            phase.run(orch, {})
        self.assertIn("sshd:2224", str(ctx.exception))
        self.assertIn("h2", str(ctx.exception))


class TestMultinodeSshPhaseUndo(unittest.TestCase):
    def test_undo_runs_pkill_via_runtime(self):
        phase = MultinodeSshPhase()
        all_ok = {h: _ok(h) for h in ["h1", "h2"]}
        orch = _stub_orch({"in_namespace_sshd"}, [all_ok])

        phase.undo(orch, {"port": 2224})

        orch.exec.assert_called_once_with(
            SSHD_STOP_CMD, scope=ExecScope.ALL, target=ExecTarget.RUNTIME, timeout=10
        )


if __name__ == "__main__":
    unittest.main()
