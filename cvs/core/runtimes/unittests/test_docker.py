'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/core/runtimes/docker.py: DockerRuntime.setup_containers
# command-rendering invariants. Mocks the orchestrator's Pssh handle so we can
# capture the rendered `docker run` command without touching SSH or docker.
#
# Pinned invariants:
#   - User-supplied runtime.args.volumes must NOT be double-listed.
#   - --gpus all must NEVER be emitted: CVS is AMD-only and AMD GPU access
#     comes from the auto-injected --device /dev/kfd /dev/dri /dev/infiniband
#     in DEFAULT_CONTAINER_ARGS. The flag is NVIDIA/CDI-specific and breaks
#     AMD-only docker without the AMD container toolkit.
#   - Every privileged command is built from a single deterministic
#     orchestrator.sudo_prefix() prefix -- never the old `cmd || sudo -n cmd`
#     retry, which double-runs the caller's payload whenever it fails for any
#     reason (not just permission-denied).

import unittest
from unittest.mock import MagicMock, patch

from cvs.core.orchestrators.baremetal import BaremetalOrchestrator
from cvs.core.orchestrators.factory import OrchestratorConfig
from cvs.core.runtimes.docker import DockerRuntime


def _make_runtime(captured, sudo_prefix=""):
    """DockerRuntime wired to a MagicMock orchestrator that captures the
    `docker run` cmd string into the supplied list."""

    def _fake_exec(cmd, timeout=None, detailed=False, print_console=True):
        # The first call DockerRuntime.setup_containers makes is `docker rm -f`
        # for a stale container; the second is the actual `docker run`. We only
        # care about the run cmd.
        if "docker run" in cmd:
            captured.append(cmd)
        # Mock a successful detailed result so setup_containers returns True.
        return {"host1": {"output": "", "exit_code": 0}}

    orchestrator = MagicMock()
    orchestrator.hosts = ["host1"]
    orchestrator.all.exec.side_effect = _fake_exec
    orchestrator.sudo_prefix.return_value = sudo_prefix

    log = MagicMock()
    return DockerRuntime(log, orchestrator)


def _container_config(image="img:test", lifetime="per_run", extra_runtime_args=None, **extra):
    """Minimal container config dict for setup_containers.

    lifetime is carried for parity with a real config block, but the runtime never
    reads it: lifecycle policy is resolved and branched on by the orchestrator,
    which only calls runtime.setup_containers on start paths. The runtime is
    policy-free, so the value here has no effect on its behavior.
    """
    cfg = {
        "image": image,
        "name": "cvs_iter_test",
        "lifetime": lifetime,
        "runtime": {"name": "docker", "args": dict(extra_runtime_args or {})},
    }
    cfg.update(extra)
    return cfg


class TestDockerRuntimeSetupContainers(unittest.TestCase):
    def test_setup_containers_always_proceeds(self):
        # The legacy `if not launch: return True` short-circuit was removed. The
        # orchestrator only calls runtime.setup_containers on start paths, so the
        # runtime must always render a `docker run` when invoked.
        captured = []
        rt = _make_runtime(captured)
        result = rt.setup_containers(
            container_config=_container_config(),
            container_name="cvs_iter_test",
            volumes=["/home/u:/workspace"],
        )
        self.assertTrue(result)
        self.assertEqual(len(captured), 1)
        self.assertIn("docker run", captured[0])

    def test_user_volume_listed_once(self):
        # With runtime.args.volumes=['/foo:/bar'] AND positional volumes that
        # already include '/foo:/bar' (mirroring what
        # ContainerOrchestrator.get_volumes() produces), the rendered docker
        # run cmd must contain '-v /foo:/bar' EXACTLY ONCE -- vol_args is the
        # single source of truth; _build_runtime_args must not re-emit it.
        captured = []
        rt = _make_runtime(captured)
        cfg = _container_config(extra_runtime_args={"volumes": ["/foo:/bar"]})

        rt.setup_containers(
            container_config=cfg,
            container_name="cvs_iter_test",
            volumes=["/home/u:/workspace", "/home/u/.ssh:/host_ssh", "/foo:/bar"],
        )

        self.assertEqual(len(captured), 1, f"Expected 1 docker run cmd, got: {captured}")
        cmd = captured[0]
        self.assertEqual(
            cmd.count("-v /foo:/bar"),
            1,
            f"User volume '/foo:/bar' must appear exactly once in:\n{cmd}",
        )

    def test_cmd_never_contains_gpus_all(self):
        # CVS is AMD-only. The rendered docker run cmd must never contain
        # '--gpus all', regardless of any container_config knob a future caller
        # might re-introduce. Asserts in three scenarios:
        #   1. minimal config, no GPU-related keys
        #   2. legacy config that sets gpu_passthrough=True (must be ignored)
        #   3. config with extra runtime args
        # AMD GPU access is provided by --device /dev/kfd /dev/dri /dev/infiniband
        # via DEFAULT_CONTAINER_ARGS -- not by --gpus all.
        for label, cfg in [
            ("minimal", _container_config()),
            ("legacy_gpu_passthrough_true", _container_config(gpu_passthrough=True)),
            ("with_extra_runtime_args", _container_config(extra_runtime_args={"network": "host"})),
        ]:
            with self.subTest(scenario=label):
                captured = []
                rt = _make_runtime(captured)
                rt.setup_containers(
                    container_config=cfg,
                    container_name="cvs_iter_test",
                    volumes=["/home/u:/workspace"],
                )
                self.assertEqual(len(captured), 1)
                self.assertNotIn(
                    "--gpus all",
                    captured[0],
                    f"[{label}] '--gpus all' must never appear in docker cmd:\n{captured[0]}",
                )


class TestDockerRuntimeExecCmdList(unittest.TestCase):
    def test_exec_cmd_list_uses_sudo_prefix_not_unconditional_sudo(self):
        # exec_cmd_list must build every rendered command from a single
        # sudo_prefix() read (not one probe per list item -- sudo_prefix() is a
        # cached property read, not a per-call probe), never an unconditional
        # `sudo docker exec ...` and never the old `cmd || sudo -n cmd` retry.
        for label, sudo_prefix in [("no_sudo", ""), ("passwordless_sudo", "sudo -n ")]:
            with self.subTest(scenario=label):
                orchestrator = MagicMock()
                orchestrator.hosts = ["host1", "host2"]
                orchestrator.all.exec_cmd_list.return_value = {
                    "host1": {"output": "", "exit_code": 0},
                    "host2": {"output": "", "exit_code": 0},
                }
                orchestrator.sudo_prefix.return_value = sudo_prefix
                rt = DockerRuntime(MagicMock(), orchestrator)

                rt.exec_cmd_list("cvs_iter_test", ["echo one", "echo two"], timeout=None)

                rendered = orchestrator.all.exec_cmd_list.call_args[0][0]
                self.assertEqual(len(rendered), 2)
                for cmd in rendered:
                    self.assertNotIn("||", cmd, f"[{label}] must not use the old fallback form: {cmd!r}")
                    self.assertEqual(
                        cmd, f"{sudo_prefix}docker exec cvs_iter_test bash -c {cmd.split('bash -c ', 1)[1]}"
                    )
                    self.assertTrue(cmd.startswith(f"{sudo_prefix}docker exec cvs_iter_test bash -c "))

    def test_exec_cmd_list_reads_sudo_prefix_once_not_per_item(self):
        # sudo_prefix() is now a cached probe-once read, not a per-command
        # retry -- rendering a list of N commands must call it once, not N
        # times, to prove the migration didn't reintroduce a per-item probe.
        orchestrator = MagicMock()
        orchestrator.hosts = ["host1", "host2"]
        orchestrator.all.exec_cmd_list.return_value = {
            "host1": {"output": "", "exit_code": 0},
            "host2": {"output": "", "exit_code": 0},
        }
        orchestrator.sudo_prefix.return_value = "sudo -n "
        rt = DockerRuntime(MagicMock(), orchestrator)

        rt.exec_cmd_list("cvs_iter_test", ["echo one", "echo two", "echo three"], timeout=None)

        orchestrator.sudo_prefix.assert_called_once()


class TestDockerRuntimeExec(unittest.TestCase):
    def test_exec_uses_sudo_prefix(self):
        # exec() must render its command from a single sudo_prefix() prefix,
        # never the old `cmd || sudo -n cmd` retry -- a plain, un-sudo'd
        # `docker exec` would fail outright on hosts where the SSH user isn't
        # in the docker group.
        for label, sudo_prefix in [("no_sudo", ""), ("passwordless_sudo", "sudo -n ")]:
            with self.subTest(scenario=label):
                orchestrator = MagicMock()
                orchestrator.all.exec.return_value = {"host1": {"output": "", "exit_code": 0}}
                orchestrator.sudo_prefix.return_value = sudo_prefix
                rt = DockerRuntime(MagicMock(), orchestrator)

                rt.exec("cvs_iter_test", "echo hi")

                rendered = orchestrator.all.exec.call_args[0][0]
                self.assertNotIn("||", rendered, f"[{label}] must not use the old fallback form: {rendered!r}")
                self.assertEqual(rendered, f"{sudo_prefix}docker exec cvs_iter_test bash -c 'echo hi'")

    def test_exec_with_hosts_subset_uses_sudo_prefix(self):
        # The hosts-subset branch builds its own Pssh and must render the
        # same sudo_prefix()-derived command as the default (all-hosts) branch.
        orchestrator = MagicMock()
        orchestrator.log = MagicMock()
        orchestrator.user = "u"
        orchestrator.password = None
        orchestrator.pkey = None
        orchestrator.stop_on_errors = False
        orchestrator.sudo_prefix.return_value = "sudo -n "
        rt = DockerRuntime(MagicMock(), orchestrator)

        with patch("cvs.lib.parallel_ssh_lib.Pssh") as mock_pssh_cls:
            mock_pssh = MagicMock()
            mock_pssh.exec.return_value = {"host1": {"output": "", "exit_code": 0}}
            mock_pssh_cls.return_value = mock_pssh

            rt.exec("cvs_iter_test", "echo hi", hosts=["host1"])

            rendered = mock_pssh.exec.call_args[0][0]
            self.assertNotIn("||", rendered, f"must not use the old fallback form: {rendered!r}")
            self.assertTrue(rendered.startswith("sudo -n docker exec cvs_iter_test bash -c "))

    def test_exec_on_head_uses_sudo_prefix(self):
        orchestrator = MagicMock()
        orchestrator.head.exec.return_value = {"output": "", "exit_code": 0}
        orchestrator.sudo_prefix.return_value = ""
        rt = DockerRuntime(MagicMock(), orchestrator)

        rt.exec_on_head("cvs_iter_test", "echo hi")

        rendered = orchestrator.head.exec.call_args[0][0]
        self.assertNotIn("||", rendered, f"must not use the old fallback form: {rendered!r}")
        self.assertTrue(rendered.startswith("docker exec cvs_iter_test bash -c "))


class TestDockerRuntimeSudoProbeCachedAcrossCalls(unittest.TestCase):
    """Regression test for the bug being fixed: with a REAL BaremetalOrchestrator
    (not a bare MagicMock) as DockerRuntime's orchestrator, the underlying
    passwordless-sudo probe must fire once total across multiple exec-family
    calls, not once per call."""

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_sudo_probe_fires_once_across_exec_and_exec_on_head(self, mock_pssh):
        pssh_instance = MagicMock()
        pssh_instance.exec.return_value = {"10.0.0.1": "0", "10.0.0.2": "0"}
        mock_pssh.return_value = pssh_instance

        config = OrchestratorConfig(
            orchestrator="baremetal",
            node_dict={"10.0.0.1": {}, "10.0.0.2": {}},
            username="testuser",
            priv_key_file="/dev/null",
            password=None,
            head_node_dict={"mgmt_ip": "10.0.0.1"},
            container={},
        )
        orchestrator = BaremetalOrchestrator(MagicMock(), config)
        rt = DockerRuntime(MagicMock(), orchestrator)

        rt.exec("cvs_iter_test", "echo hi")
        rt.exec_on_head("cvs_iter_test", "echo hi")

        probe_calls = [
            c for c in pssh_instance.exec.call_args_list if c[0] and c[0][0] == "sudo -n true >/dev/null 2>&1; echo $?"
        ]
        self.assertEqual(len(probe_calls), 1, f"probe must fire once total, calls: {pssh_instance.exec.call_args_list}")


if __name__ == "__main__":
    unittest.main()
