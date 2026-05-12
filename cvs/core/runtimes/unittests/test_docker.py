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
        if cmd.startswith("sudo docker run"):
            captured.append(cmd)
        # Mock a successful detailed result so setup_containers returns True.
        return {"host1": {"output": "", "exit_code": 0}}

    orchestrator = MagicMock()
    orchestrator.hosts = ["host1"]
    orchestrator.all.exec.side_effect = _fake_exec

    log = MagicMock()
    return DockerRuntime(log, orchestrator)


def _container_config(launch=True, image="img:test", extra_runtime_args=None, **extra):
    """Minimal container config dict for setup_containers."""
    cfg = {
        "enabled": True,
        "launch": launch,
        "image": image,
        "name": "cvs_iter_test",
        "runtime": {"name": "docker", "args": dict(extra_runtime_args or {})},
    }
    cfg.update(extra)
    return cfg


class TestDockerRuntimeSetupContainers(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
