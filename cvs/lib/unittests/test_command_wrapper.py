"""Unit tests for the CommandWrapper hierarchy in cvs/lib/parallel_ssh_lib.py
(CVS docker-mode P4)."""

import os
import unittest
from unittest.mock import patch

from cvs.lib.parallel_ssh_lib import (
    CommandWrapper,
    DockerExecWrapper,
    NoOpWrapper,
    _maybe_docker_wrap,
    _resolve_default_wrapper,
    wrapper_for_cluster,
)


class TestNoOpWrapper(unittest.TestCase):
    def test_identity(self):
        w = NoOpWrapper()
        self.assertEqual(w.wrap("echo hello"), "echo hello")

    def test_empty(self):
        self.assertEqual(NoOpWrapper().wrap(""), "")

    def test_preserves_special_chars(self):
        # The whole point: byte-exact passthrough so existing CVS unit tests
        # like test_parallel_ssh_lib.py keep asserting on identical strings.
        cmd = "sudo bash -c 'echo $X | grep \"foo\" && exit 0'"
        self.assertEqual(NoOpWrapper().wrap(cmd), cmd)


class TestDockerExecWrapper(unittest.TestCase):
    def test_basic_wrap(self):
        w = DockerExecWrapper("cvs-runner")
        out = w.wrap("rocminfo")
        self.assertTrue(out.startswith("docker exec cvs-runner bash -lc "))
        self.assertIn("rocminfo", out)

    def test_strips_leading_sudo(self):
        out = DockerExecWrapper("cvs-runner").wrap("sudo /opt/rocm/bin/rocminfo")
        # The bash -lc payload must NOT contain `sudo` since the container is root.
        self.assertNotIn("sudo", out.split("bash -lc ")[1])
        self.assertIn("/opt/rocm/bin/rocminfo", out)

    def test_strips_only_leading_sudo(self):
        # `sudo` deeper in the command (e.g. inside a heredoc or after a pipe)
        # should NOT be stripped. Only a leading `^\s*sudo\s+`.
        out = DockerExecWrapper("cvs-runner").wrap("echo sudo pretend && true")
        self.assertIn("sudo pretend", out)

    def test_container_envs(self):
        w = DockerExecWrapper(
            "cvs-runner", container_envs={"NCCL_DEBUG": "INFO", "FOO": "bar baz"}
        )
        out = w.wrap("ls")
        # Both env vars present, both quoted (bar baz contains a space)
        self.assertIn("-e NCCL_DEBUG=INFO", out)
        self.assertIn("-e FOO='bar baz'", out)

    def test_invalid_container_name(self):
        with self.assertRaises(ValueError):
            DockerExecWrapper("")
        with self.assertRaises(ValueError):
            DockerExecWrapper(None)

    def test_subclass_of_command_wrapper(self):
        self.assertIsInstance(DockerExecWrapper("cvs-runner"), CommandWrapper)
        self.assertIsInstance(NoOpWrapper(), CommandWrapper)


class TestResolveDefaultWrapper(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=False)
    def test_no_env_var_returns_noop(self):
        os.environ.pop("CVS_DOCKER_CONTAINER", None)
        self.assertIsInstance(_resolve_default_wrapper(), NoOpWrapper)

    def test_env_var_set_returns_docker_wrapper(self):
        with patch.dict(os.environ, {"CVS_DOCKER_CONTAINER": "cvs-runner"}):
            w = _resolve_default_wrapper()
            self.assertIsInstance(w, DockerExecWrapper)
            self.assertEqual(w.container_name, "cvs-runner")

    def test_env_var_whitespace_treated_as_unset(self):
        with patch.dict(os.environ, {"CVS_DOCKER_CONTAINER": "   "}):
            self.assertIsInstance(_resolve_default_wrapper(), NoOpWrapper)


class TestLegacyShim(unittest.TestCase):
    """The pre-P4 _maybe_docker_wrap function is kept as a compatibility shim
    so any out-of-tree caller keeps working. Behavior must match P1: identity
    when env unset, wrap when set."""

    @patch.dict(os.environ, {}, clear=False)
    def test_env_unset_identity(self):
        os.environ.pop("CVS_DOCKER_CONTAINER", None)
        self.assertEqual(_maybe_docker_wrap("echo hello"), "echo hello")

    def test_env_set_wraps(self):
        with patch.dict(os.environ, {"CVS_DOCKER_CONTAINER": "cvs-runner"}):
            out = _maybe_docker_wrap("sudo /opt/rocm/bin/rocminfo")
            self.assertTrue(out.startswith("docker exec cvs-runner bash -lc "))
            self.assertNotIn("sudo", out.split("bash -lc ")[1])


class TestWrapperForCluster(unittest.TestCase):
    """Resolves a wrapper from a parsed cluster.json dict via the P3 parser."""

    def test_no_runtime_block_yields_noop(self):
        cluster = {"username": "u", "node_dict": {"h": {}}}
        w = wrapper_for_cluster(cluster)
        self.assertIsInstance(w, NoOpWrapper)

    def test_host_mode_explicit_yields_noop(self):
        w = wrapper_for_cluster({"runtime": {"mode": "host"}})
        self.assertIsInstance(w, NoOpWrapper)

    def test_docker_mode_yields_docker_wrapper(self):
        cluster = {
            "runtime": {
                "mode": "docker",
                "image": "ghcr.io/x/cvs-runner:1",
                "container_name": "my-runner",
                "container_envs": {"FOO": "bar"},
            }
        }
        w = wrapper_for_cluster(cluster)
        self.assertIsInstance(w, DockerExecWrapper)
        self.assertEqual(w.container_name, "my-runner")
        self.assertEqual(w.container_envs, {"FOO": "bar"})

    def test_docker_mode_default_container_name(self):
        cluster = {
            "runtime": {
                "mode": "docker",
                "image": "ghcr.io/x/cvs-runner:1",
            }
        }
        w = wrapper_for_cluster(cluster)
        self.assertEqual(w.container_name, "cvs-runner")
        self.assertEqual(w.container_envs, {})


if __name__ == "__main__":
    unittest.main()
