"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

import logging
import unittest
from typing import Optional

from cvs.core.orchestrator import Orchestrator
from cvs.core.scope import ExecResult, ExecScope, ExecTarget


class _FakeTransport:
    """Records every exec call and returns a synthetic ExecResult per host."""

    def __init__(self, hosts, head_node):
        self.hosts = list(hosts)
        self.head_node = head_node
        self.env_prefix = ""
        self.calls: list[dict] = []

    def exec(self, cmd, scope, *, subset=None, timeout=None):
        self.calls.append(
            {"cmd": cmd, "scope": scope, "subset": subset, "timeout": timeout}
        )
        if scope is ExecScope.ALL:
            targets = self.hosts
        elif scope is ExecScope.HEAD:
            targets = [self.head_node]
        else:
            targets = list(subset or [])
        return {h: ExecResult(host=h, output=f"out:{cmd}", exit_code=0) for h in targets}

    def scp(self, src, dst, scope, *, subset=None):  # pragma: no cover
        pass


class _FakeRuntime:
    name = "fake"
    capabilities: set = set()

    def __init__(self):
        self.wrap_calls: list[str] = []
        self.setup_calls = 0
        self.teardown_calls = 0

    def setup(self, transport):
        self.setup_calls += 1

    def teardown(self, transport):
        self.teardown_calls += 1

    def wrap_cmd(self, cmd: str) -> str:
        self.wrap_calls.append(cmd)
        return f"WRAPPED({cmd})"

    def workload_ssh_port(self) -> int:
        return 9999

    def workload_hostfile_path(self) -> str:
        return "/tmp/fake"


def _make_orch(hosts=("h1", "h2", "h3"), head="h1"):
    transport = _FakeTransport(hosts=hosts, head_node=head)
    runtime = _FakeRuntime()
    log = logging.getLogger("test_orch_dispatch")
    return Orchestrator(transport, runtime, {}, log=log), transport, runtime


class TestOrchestratorTargetSwitch(unittest.TestCase):
    """target=HOST bypasses runtime.wrap_cmd; target=RUNTIME goes through it."""

    def test_target_host_skips_wrap_cmd(self):
        orch, transport, runtime = _make_orch()
        results = orch.exec("echo hi", target=ExecTarget.HOST)
        self.assertEqual(len(results), 3)
        self.assertEqual(transport.calls[0]["cmd"], "echo hi")
        self.assertEqual(runtime.wrap_calls, [])

    def test_target_runtime_calls_wrap_cmd(self):
        orch, transport, runtime = _make_orch()
        orch.exec("echo hi", target=ExecTarget.RUNTIME)
        self.assertEqual(transport.calls[0]["cmd"], "WRAPPED(echo hi)")
        self.assertEqual(runtime.wrap_calls, ["echo hi"])

    def test_default_target_is_runtime(self):
        orch, transport, runtime = _make_orch()
        orch.exec("hostname")
        self.assertEqual(transport.calls[0]["cmd"], "WRAPPED(hostname)")


class TestOrchestratorScopeRouting(unittest.TestCase):
    """scope ALL/HEAD/SUBSET routes the right hosts list to the transport."""

    def test_scope_all(self):
        orch, transport, _ = _make_orch()
        results = orch.exec("x", scope=ExecScope.ALL, target=ExecTarget.HOST)
        self.assertEqual(set(results.keys()), {"h1", "h2", "h3"})
        self.assertEqual(transport.calls[0]["scope"], ExecScope.ALL)

    def test_scope_head(self):
        orch, transport, _ = _make_orch()
        results = orch.exec("x", scope=ExecScope.HEAD, target=ExecTarget.HOST)
        self.assertEqual(set(results.keys()), {"h1"})
        self.assertEqual(transport.calls[0]["scope"], ExecScope.HEAD)

    def test_scope_subset(self):
        orch, transport, _ = _make_orch()
        results = orch.exec(
            "x", scope=ExecScope.SUBSET, subset=["h2"], target=ExecTarget.HOST
        )
        self.assertEqual(set(results.keys()), {"h2"})
        self.assertEqual(transport.calls[0]["scope"], ExecScope.SUBSET)
        self.assertEqual(transport.calls[0]["subset"], ["h2"])


class TestOrchestratorSetupCleanup(unittest.TestCase):
    """setup() runs runtime.setup + PREPARE_PIPELINE; cleanup() rolls back."""

    def test_setup_calls_runtime_setup(self):
        orch, _, runtime = _make_orch()
        orch.setup()
        self.assertEqual(runtime.setup_calls, 1)
        # PREPARE_PIPELINE recorded an artifact dict (skipped because the
        # fake runtime has empty capabilities, so MultinodeSshPhase skips).
        self.assertIn("prepare", orch.artifacts)
        self.assertEqual(orch.artifacts["prepare"]["multinode_ssh"]["status"], "skipped")

    def test_cleanup_calls_runtime_teardown(self):
        orch, _, runtime = _make_orch()
        orch.setup()
        orch.cleanup()
        self.assertEqual(runtime.teardown_calls, 1)


class TestOrchestratorTimeoutPasses(unittest.TestCase):
    def test_timeout_kwarg_threads_through(self):
        orch, transport, _ = _make_orch()
        orch.exec("x", target=ExecTarget.HOST, timeout=30)
        self.assertEqual(transport.calls[0]["timeout"], 30)


if __name__ == "__main__":
    unittest.main()
