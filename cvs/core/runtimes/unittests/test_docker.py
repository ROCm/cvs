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

import unittest
from unittest.mock import MagicMock

from cvs.core.runtimes.docker import DockerRuntime


def _make_runtime(captured):
    """DockerRuntime wired to a MagicMock orchestrator that captures the
    `docker run` cmd string into the supplied list."""

    def _fake_exec(cmd, timeout=None, detailed=False, print_console=True):
        # The first call DockerRuntime.setup_containers makes is `docker rm -f`
        # for a stale container; the second is the actual `docker run`. We only
        # care about the run cmd.
        if cmd.startswith("docker run"):
            captured.append(cmd)
        # Mock a successful detailed result so setup_containers returns True.
        return {"host1": {"output": "", "exit_code": 0}}

    orchestrator = MagicMock()
    orchestrator.hosts = ["host1"]
    orchestrator.all.exec.side_effect = _fake_exec

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
        # captured[0] is the `cmd || sudo -n cmd` fallback form -- check a
        # single invocation (the part before the fallback), since the
        # invariant under test is about vol_args duplication within one
        # docker run rendering, not the (expected) sudo-fallback duplication.
        cmd = captured[0].split(" || ")[0]
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


class TestDockerRuntimeRegistryLogin(unittest.TestCase):
    def test_registry_login_requires_username_and_password_file(self):
        captured = []
        rt = _make_runtime(captured)
        self.assertFalse(rt.registry_login({}))
        self.assertFalse(rt.registry_login({"username": "u"}))
        self.assertFalse(rt.registry_login({"password_file": "/tmp/pw"}))

    def test_registry_login_sudo_fallback_covers_entire_pipeline(self):
        # Regression test: `docker login` is the SECOND stage of the
        # `cat pwfile | docker login ...` pipe. Naively passing that pipe
        # string straight into with_sudo_fallback() renders
        # `cat f | docker login ... || sudo -n cat f | docker login ...` --
        # `||` binds tighter than `|`, so `sudo -n` only prefixes `cat`, never
        # `docker login`, leaving the "fallback" branch just as unprivileged as
        # the one it was supposed to fix. Caught via a live docker daemon where
        # the plain command failed with a permission error and the fallback
        # failed identically. Fix: wrap the whole pipeline (e.g. `sh -c '...'`)
        # before with_sudo_fallback() so sudo applies to all of it.
        login_cmds = []

        def _fake_exec(cmd, timeout=None, detailed=False, print_console=True):
            login_cmds.append(cmd)
            return {"host1": {"output": "", "exit_code": 0}}

        orchestrator = MagicMock()
        orchestrator.hosts = ["host1"]
        orchestrator.all.exec.side_effect = _fake_exec
        rt = DockerRuntime(MagicMock(), orchestrator)

        rt.registry_login({"username": "u", "password_file": "/tmp/pw"})

        self.assertEqual(len(login_cmds), 1)
        cmd = login_cmds[0]
        _, _, fallback = cmd.partition(" || ")
        self.assertTrue(fallback.startswith("sudo -n "), f"fallback branch: {fallback!r}")
        # The pipeline must be wrapped as a single command (e.g. `sh -c '...'`)
        # immediately after `sudo -n`, so sudo governs the whole pipeline --
        # not `sudo -n cat ... | docker login ...` with docker login left as a
        # trailing, unprivileged pipe stage outside sudo's reach.
        self.assertRegex(fallback, r"^sudo -n (sh|bash) -c ")

    def test_registry_login_pipes_password_file_via_stdin_not_command_line(self):
        # The password/token must never appear as a --password flag: only the
        # *path* to the credential file appears in the rendered command, piped
        # into `docker login --password-stdin`.
        login_cmds = []

        def _fake_exec(cmd, timeout=None, detailed=False, print_console=True):
            login_cmds.append(cmd)
            return {"host1": {"output": "", "exit_code": 0}}

        orchestrator = MagicMock()
        orchestrator.hosts = ["host1"]
        orchestrator.all.exec.side_effect = _fake_exec
        rt = DockerRuntime(MagicMock(), orchestrator)

        result = rt.registry_login({"username": "myuser", "password_file": "/home/myuser/.docker_token"})

        self.assertTrue(result)
        self.assertEqual(len(login_cmds), 1)
        cmd = login_cmds[0]
        self.assertIn("cat /home/myuser/.docker_token", cmd)
        self.assertIn("--password-stdin", cmd)
        self.assertIn("--username myuser", cmd)
        self.assertNotIn("--password ", cmd)

    def test_registry_login_reports_failure_on_bad_exit_code(self):
        orchestrator = MagicMock()
        orchestrator.hosts = ["host1"]
        orchestrator.all.exec.return_value = {"host1": {"output": "", "exit_code": 1}}
        rt = DockerRuntime(MagicMock(), orchestrator)

        self.assertFalse(rt.registry_login({"username": "u", "password_file": "/tmp/pw"}))

    def test_setup_containers_logs_in_before_pull_when_registry_configured(self):
        calls = []

        def _fake_exec(cmd, timeout=None, detailed=False, print_console=True):
            calls.append(cmd)
            return {"host1": {"output": "", "exit_code": 0}}

        orchestrator = MagicMock()
        orchestrator.hosts = ["host1"]
        orchestrator.all.exec.side_effect = _fake_exec
        rt = DockerRuntime(MagicMock(), orchestrator)

        cfg = _container_config(
            image="rocm/ufb-private:latest",
            extra_runtime_args={"registry": {"username": "myuser", "password_file": "/tmp/token"}},
        )
        result = rt.setup_containers(container_config=cfg, container_name="cvs_iter_test", volumes=[])

        self.assertTrue(result)
        login_calls = [c for c in calls if "docker login" in c]
        run_calls = [c for c in calls if c.startswith("docker run")]
        self.assertEqual(len(login_calls), 1)
        self.assertEqual(len(run_calls), 1)
        # Login must happen before the run that needs the pull.
        self.assertLess(calls.index(login_calls[0]), calls.index(run_calls[0]))

    def test_setup_containers_skips_login_when_image_tar_present(self):
        # A tar load never needs a registry pull, so no login should be attempted
        # even if 'registry' is configured (e.g. left over from another profile).
        calls = []

        def _fake_exec(cmd, timeout=None, detailed=False, print_console=True):
            calls.append(cmd)
            return {"host1": {"output": "", "exit_code": 0}}

        orchestrator = MagicMock()
        orchestrator.hosts = ["host1"]
        orchestrator.all.exec.side_effect = _fake_exec
        rt = DockerRuntime(MagicMock(), orchestrator)

        cfg = _container_config(
            extra_runtime_args={"registry": {"username": "myuser", "password_file": "/tmp/token"}},
            image_tar="/tmp/image.tar",
        )
        rt.setup_containers(container_config=cfg, container_name="cvs_iter_test", volumes=[])

        self.assertFalse(any("docker login" in c for c in calls))


class TestDockerRuntimeExecCmdList(unittest.TestCase):
    def test_exec_cmd_list_uses_sudo_fallback_not_unconditional_sudo(self):
        # exec_cmd_list must follow the same `cmd || sudo -n cmd` fallback
        # convention as every other docker invocation in this file, not an
        # unconditional `sudo docker exec ...` -- otherwise a rootless docker
        # setup with no sudo available at all would fail outright instead of
        # running the plain (already-permitted) command.
        orchestrator = MagicMock()
        orchestrator.hosts = ["host1", "host2"]
        orchestrator.all.exec_cmd_list.return_value = {
            "host1": {"output": "", "exit_code": 0},
            "host2": {"output": "", "exit_code": 0},
        }
        rt = DockerRuntime(MagicMock(), orchestrator)

        rt.exec_cmd_list("cvs_iter_test", ["echo one", "echo two"], timeout=None)

        rendered = orchestrator.all.exec_cmd_list.call_args[0][0]
        self.assertEqual(len(rendered), 2)
        for cmd in rendered:
            self.assertNotRegex(
                cmd,
                r"^sudo docker exec",
                f"exec_cmd_list must not hardcode unconditional sudo:\n{cmd}",
            )
            self.assertIn(" || sudo -n ", cmd)
            self.assertTrue(cmd.startswith("docker exec cvs_iter_test bash -c "))


if __name__ == "__main__":
    unittest.main()
