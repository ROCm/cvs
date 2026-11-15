'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs/lib/utils/container_stages.py: each stage records its label under the
test's node id, a failed launch or sshd stage sets lifecycle.failed so later stages skip,
sshd skips its port probe on managed compute and on one host, and teardown records its
label on every call. Launch and teardown run a real ContainerOrchestrator with SSH and the
container runtime mocked.
'''

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cvs.core.orchestrators.container import ContainerOrchestrator
from cvs.core.orchestrators.factory import OrchestratorConfig
from cvs.lib.inference.unittests.fake_orch import FakeOrch
from cvs.lib.inference.utils.inference_suite_lifecycle import InferenceLifecycle
from cvs.lib.utils import container_stages

_HOSTS = ("10.0.0.1", "10.0.0.2")
_RUNNING = {host: {"running": True, "exit_code": 0, "name": "cvs_stage_test"} for host in _HOSTS}
_NOT_RUNNING = {host: {"running": False, "exit_code": 0, "name": ""} for host in _HOSTS}


def _request(nodeid):
    return SimpleNamespace(node=SimpleNamespace(nodeid=nodeid))


def _labels(lifecycle, nodeid):
    return [label for label, _value, _unit in lifecycle.report.get(nodeid, [])]


class _ContainerOrchTestCase(unittest.TestCase):
    def setUp(self):
        p_pssh = patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
        p_rf = patch("cvs.core.orchestrators.container.RuntimeFactory")
        p_pssh.start()
        self.mock_rf = p_rf.start()
        self.addCleanup(p_pssh.stop)
        self.addCleanup(p_rf.stop)
        self.runtime = MagicMock(name="docker_runtime")
        self.mock_rf.create.return_value = self.runtime
        cfg = OrchestratorConfig(
            orchestrator="container",
            node_dict={host: {} for host in _HOSTS},
            username="testuser",
            priv_key_file="/dev/null",
            password=None,
            head_node_dict={"mgmt_ip": _HOSTS[0]},
            container={
                "lifetime": "per_run",
                "image": "rocm/cvs:test",
                "name": "cvs_stage_test",
                "runtime": {"name": "docker", "args": {}},
            },
        )
        self.orch = ContainerOrchestrator(MagicMock(), cfg)
        self.lifecycle = InferenceLifecycle()


class TestLaunchContainer(_ContainerOrchTestCase):
    nodeid = "atom.py::test_launch_container"

    def _launch(self):
        container_stages.test_launch_container(self.orch, None, self.lifecycle, _request(self.nodeid))

    def test_success_records_both_launch_labels_with_one_duration(self):
        self.runtime.setup_containers.return_value = True
        self.runtime.is_running.return_value = _RUNNING
        self._launch()
        self.assertFalse(self.lifecycle.failed)
        rows = self.lifecycle.report[self.nodeid]
        self.assertEqual(
            [(label, unit) for label, _, unit in rows], [("container_launch", "s"), ("server.container_launch_s", "s")]
        )
        self.assertEqual(rows[0][1], rows[1][1])
        self.assertGreaterEqual(rows[0][1], 0)
        self.runtime.is_running.assert_called_once_with("cvs_stage_test")

    def test_setup_false_fails_and_marks_lifecycle_failed_without_verifying(self):
        self.runtime.setup_containers.return_value = False
        with self.assertRaises(pytest.fail.Exception) as ctx:
            self._launch()
        self.assertIn("setup_containers() returned False for cvs_stage_test", str(ctx.exception))
        self.assertTrue(self.lifecycle.failed)
        self.assertEqual(_labels(self.lifecycle, self.nodeid), ["container_launch", "server.container_launch_s"])
        self.runtime.is_running.assert_not_called()

    def test_container_not_running_after_setup_fails_and_marks_lifecycle_failed(self):
        self.runtime.setup_containers.return_value = True
        self.runtime.is_running.return_value = _NOT_RUNNING
        with self.assertRaises(pytest.fail.Exception) as ctx:
            self._launch()
        self.assertIn("container cvs_stage_test not running after setup_containers()", str(ctx.exception))
        self.assertTrue(self.lifecycle.failed)


class TestSetupSshd(unittest.TestCase):
    nodeid = "atom.py::test_setup_sshd"

    def _orch(self, hosts, exec_return=None, sshd_ok=True):
        orch = FakeOrch(exec_return=exec_return, hosts=hosts)
        orch.setup_sshd = MagicMock(return_value=sshd_ok)
        return orch

    def _sshd(self, orch, lifecycle, managed=False):
        with patch("cvs.core.scheduler.is_managed_compute", return_value=managed):
            container_stages.test_setup_sshd(orch, lifecycle, _request(self.nodeid))

    def test_prior_failure_skips_before_setup(self):
        orch = self._orch(["node0", "node1"])
        lifecycle = InferenceLifecycle()
        lifecycle.failed = True
        with self.assertRaises(pytest.skip.Exception):
            self._sshd(orch, lifecycle)
        orch.setup_sshd.assert_not_called()
        self.assertEqual(lifecycle.report, {})

    def test_setup_false_fails_and_marks_lifecycle_failed_without_probing(self):
        orch = self._orch(["node0", "node1"], sshd_ok=False)
        lifecycle = InferenceLifecycle()
        with self.assertRaises(pytest.fail.Exception):
            self._sshd(orch, lifecycle)
        self.assertTrue(lifecycle.failed)
        self.assertEqual(_labels(lifecycle, self.nodeid), ["sshd_setup"])
        self.assertEqual(orch.commands, [])

    def test_managed_multinode_does_not_probe_listen_port(self):
        orch = self._orch(["node0", "node1"])
        lifecycle = InferenceLifecycle()
        self._sshd(orch, lifecycle, managed=True)
        self.assertFalse(lifecycle.failed)
        self.assertEqual(_labels(lifecycle, self.nodeid), ["sshd_setup"])
        self.assertEqual(orch.commands, [])

    def test_single_host_does_not_probe_listen_port(self):
        orch = self._orch(["node0"])
        lifecycle = InferenceLifecycle()
        self._sshd(orch, lifecycle)
        self.assertFalse(lifecycle.failed)
        self.assertEqual(orch.commands, [])

    def test_baremetal_multinode_probes_the_orch_ssh_port(self):
        orch = self._orch(["node0", "node1"], exec_return={"node0": "OK", "node1": "OK"})
        orch.ssh_port = 2299
        lifecycle = InferenceLifecycle()
        self._sshd(orch, lifecycle)
        self.assertFalse(lifecycle.failed)
        self.assertEqual(len(orch.commands), 1)
        self.assertIn(":2299", orch.commands[0][0])

    def test_baremetal_multinode_still_requires_listen_port(self):
        orch = self._orch(["node0", "node1"], exec_return={"node0": "NO", "node1": "NO"})
        lifecycle = InferenceLifecycle()
        with self.assertRaises(pytest.fail.Exception):
            self._sshd(orch, lifecycle)
        self.assertTrue(lifecycle.failed)
        self.assertIn(":2224", orch.commands[0][0])

    def test_empty_probe_result_fails_and_marks_lifecycle_failed(self):
        orch = self._orch(["node0", "node1"], exec_return={})
        lifecycle = InferenceLifecycle()
        with self.assertRaises(pytest.fail.Exception):
            self._sshd(orch, lifecycle)
        self.assertTrue(lifecycle.failed)


class TestTeardown(_ContainerOrchTestCase):
    nodeid = "atom.py::test_teardown"

    def _teardown(self):
        container_stages.test_teardown(self.orch, self.lifecycle, _request(self.nodeid))

    def test_removed_container_records_teardown_and_marks_torn_down(self):
        self.orch.container_id = "cvs_stage_test"
        self.runtime.teardown_containers.return_value = True
        self.runtime.is_running.return_value = _NOT_RUNNING
        self._teardown()
        self.runtime.teardown_containers.assert_called_once_with("cvs_stage_test")
        self.assertTrue(self.lifecycle.torn_down)
        self.assertFalse(self.lifecycle.failed)
        self.assertEqual(_labels(self.lifecycle, self.nodeid), ["teardown"])

    def test_container_still_running_fails_without_marking_torn_down_or_failed(self):
        self.orch.container_id = "cvs_stage_test"
        self.runtime.teardown_containers.return_value = False
        self.runtime.is_running.return_value = _RUNNING
        with self.assertRaises(pytest.fail.Exception) as ctx:
            self._teardown()
        self.assertIn("container cvs_stage_test still running after teardown_containers()", str(ctx.exception))
        self.assertFalse(self.lifecycle.torn_down)
        self.assertFalse(self.lifecycle.failed)
        self.assertEqual(_labels(self.lifecycle, self.nodeid), ["teardown"])

    def test_each_call_records_teardown(self):
        self.runtime.is_running.return_value = _NOT_RUNNING
        self._teardown()
        self._teardown()
        self.assertTrue(self.lifecycle.torn_down)
        self.assertEqual(_labels(self.lifecycle, self.nodeid), ["teardown", "teardown"])


if __name__ == "__main__":
    unittest.main()
