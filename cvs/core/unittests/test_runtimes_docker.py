"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

import unittest

from cvs.core.errors import OrchestratorConfigError
from cvs.core.runtimes.docker import DockerRuntime


class TestDockerRuntimeParseConfig(unittest.TestCase):
    def test_missing_image_raises_with_clear_message(self):
        with self.assertRaises(OrchestratorConfigError) as ctx:
            DockerRuntime.parse_config({})
        self.assertIn("image", str(ctx.exception))
        self.assertIn("docker", str(ctx.exception))

    def test_minimal_config_image_only(self):
        rt = DockerRuntime.parse_config({"image": "rocm/cvs:latest"})
        self.assertEqual(rt.image, "rocm/cvs:latest")
        self.assertEqual(rt.container_name, "cvs-runner")
        self.assertEqual(rt.network, "host")
        self.assertEqual(rt.ipc, "host")
        self.assertTrue(rt.privileged)

    def test_user_overrides_replace_defaults(self):
        rt = DockerRuntime.parse_config(
            {
                "image": "rocm/cvs:latest",
                "container_name": "my-runner",
                "network": "bridge",
                "ipc": "shareable",
                "privileged": False,
                "env": {"FOO": "bar", "GPUS": "4"},  # overrides default GPUS=8
            }
        )
        self.assertEqual(rt.container_name, "my-runner")
        self.assertEqual(rt.network, "bridge")
        self.assertEqual(rt.ipc, "shareable")
        self.assertFalse(rt.privileged)
        self.assertEqual(rt.env["FOO"], "bar")
        self.assertEqual(rt.env["GPUS"], "4")
        self.assertEqual(rt.env["MULTINODE"], "true")  # default still present


class TestDockerRuntimeCapabilities(unittest.TestCase):
    """The capability flag is what makes MultinodeSshPhase opt in."""

    def test_default_capabilities_include_in_namespace_sshd(self):
        rt = DockerRuntime.parse_config({"image": "rocm/cvs:latest"})
        self.assertIn("in_namespace_sshd", rt.capabilities)

    def test_capabilities_can_be_overridden(self):
        rt = DockerRuntime(
            image="rocm/cvs:latest", capabilities={"custom_tag"}
        )
        self.assertEqual(rt.capabilities, {"custom_tag"})


class TestDockerRuntimeWorkloadFacts(unittest.TestCase):
    """workload_ssh_port=2224 + in-container hostfile path -- the two facts
    MpiLauncher needs that differ between docker and hostshell."""

    def test_workload_ssh_port_is_in_container_2224(self):
        rt = DockerRuntime.parse_config({"image": "rocm/cvs:latest"})
        self.assertEqual(rt.workload_ssh_port(), 2224)

    def test_workload_hostfile_is_in_container_tmp(self):
        rt = DockerRuntime.parse_config({"image": "rocm/cvs:latest"})
        self.assertEqual(rt.workload_hostfile_path(), "/tmp/mpi_hosts.txt")


class TestDockerRuntimeWrapCmd(unittest.TestCase):
    def test_wrap_cmd_produces_docker_exec_with_bash_lc(self):
        rt = DockerRuntime.parse_config(
            {"image": "rocm/cvs:latest", "container_name": "test", "env": {"K": "V"}}
        )
        wrapped = rt.wrap_cmd("echo hello")
        self.assertIn("sudo docker exec", wrapped)
        self.assertIn("test", wrapped)
        self.assertIn("bash -lc", wrapped)
        self.assertIn("echo hello", wrapped)
        # Env is forwarded via -e flags so the workload sees it.
        self.assertIn("-e K=V", wrapped)

    def test_wrap_cmd_quotes_command_via_shlex(self):
        rt = DockerRuntime.parse_config(
            {"image": "rocm/cvs:latest", "container_name": "test"}
        )
        wrapped = rt.wrap_cmd("echo 'hello world' && ls /opt/rocm")
        # The whole inner command should be a single shell-quoted argument.
        self.assertIn("bash -lc", wrapped)
        # The single-quote escape pattern indicates shlex.quote handled it.
        self.assertIn("'\"'\"'", wrapped)  # literal "'\''" pattern produced by shlex


class TestDockerRuntimeRunArgs(unittest.TestCase):
    def test_run_args_include_default_devices_caps_and_ulimits(self):
        rt = DockerRuntime.parse_config({"image": "rocm/cvs:latest"})
        args = rt._docker_run_args()
        self.assertIn("--network host", args)
        self.assertIn("--ipc host", args)
        self.assertIn("--privileged", args)
        self.assertIn("--device /dev/kfd", args)
        self.assertIn("--cap-add IPC_LOCK", args)
        self.assertIn("--ulimit memlock=-1", args)
        self.assertIn("/dev/infiniband/", args)  # the per-host glob expansion

    def test_run_args_drops_infiniband_glob_when_disabled(self):
        rt = DockerRuntime(
            image="rocm/cvs:latest", expand_infiniband_devices=False
        )
        self.assertNotIn("/dev/infiniband/*", rt._docker_run_args())


if __name__ == "__main__":
    unittest.main()
