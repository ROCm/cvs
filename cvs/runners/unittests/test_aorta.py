"""
Unit tests for ``AortaRunner`` helpers that do not require a live cluster.

The networked container/SSH paths are not exercised here; see
``cvs/tests/benchmark/test_aorta.py`` for the end-to-end pytest suite that runs
against a real cluster.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cvs.runners.aorta as aorta_mod
from cvs.runners.aorta import (
    AortaConfig,
    AortaDockerConfig,
    AortaEnvironment,
    AortaRunner,
    RcclConfig,
)


def _make_runner(
    *,
    nodes,
    aorta_path,
    base_config="config/distributed.yaml",
    experiment_script="scripts/launch_rocm.sh",
    **config_overrides,
):
    cfg = AortaConfig(
        nodes=list(nodes),
        username="testuser",
        pkey="/home/testuser/.ssh/id_rsa",
        aorta_path=Path(aorta_path),
        base_config=base_config,
        docker=AortaDockerConfig(),
        rccl=RcclConfig(),
        environment=AortaEnvironment(),
        build_script="scripts/launch_rocm.sh",
        experiment_script=experiment_script,
        gpus_per_node=8,
        **config_overrides,
    )
    # The runner's __init__ aborts when the docker SDK is unavailable. None of
    # the helpers under test actually call into docker, so flip the module flag
    # for the duration of this call. This keeps the unit tests runnable in
    # minimal CI environments without the docker package.
    with patch.object(aorta_mod, "DOCKER_SDK_AVAILABLE", True):
        return AortaRunner(cfg)


class TestBuildBaseEnv(unittest.TestCase):
    def test_rccl_paths_are_exported(self):
        runner = _make_runner(nodes=["a"], aorta_path="/tmp/aorta")
        env = runner._build_base_env()
        self.assertIn("LD_LIBRARY_PATH", env)
        self.assertEqual(env["rccl_path"], runner.config.rccl.build_path)
        # Existing NCCL knobs should still be there.
        self.assertEqual(env["NCCL_MAX_NCHANNELS"], "112")

    def test_no_override_var_without_training_overrides(self):
        runner = _make_runner(nodes=["a"], aorta_path="/tmp/aorta")
        self.assertNotIn("AORTA_OVERRIDE_ARGS", runner._build_base_env())

    def test_training_overrides_become_env_var(self):
        runner = _make_runner(nodes=["a"], aorta_path="/tmp/aorta")
        runner.config.training_overrides = {"training.max_steps": 5}
        env = runner._build_base_env()
        self.assertIn("AORTA_OVERRIDE_ARGS", env)
        self.assertIn("training.max_steps", env["AORTA_OVERRIDE_ARGS"])

    def test_multi_key_overrides_share_one_override_group(self):
        # Aorta train.py uses argparse(--override, nargs="*"); multiple
        # `--override` groups would silently keep only the last group's values.
        # Guarantee a single group regardless of how many keys are configured.
        runner = _make_runner(nodes=["a"], aorta_path="/tmp/aorta")
        runner.config.training_overrides = {
            "training.max_steps": 5,
            "training.batch_size": 8,
            "profiling.active": 3,
        }
        env = runner._build_base_env()
        self.assertEqual(env["AORTA_OVERRIDE_ARGS"].count("--override"), 1)
        for key in runner.config.training_overrides:
            self.assertIn(key, env["AORTA_OVERRIDE_ARGS"])


class TestLaunchContainerGpuAccess(unittest.TestCase):
    def _launch(self):
        runner = _make_runner(nodes=["a"], aorta_path="/tmp/aorta")
        client = Mock()
        client.containers.run.return_value.status = "running"
        with patch.object(aorta_mod, "docker", Mock()):
            runner._launch_container(client, "a")
        return client.containers.run.call_args.kwargs

    def test_container_runs_as_root(self):
        # Without this the container inherits the image's default UID, which on
        # the validation cluster could not open /dev/kfd even with --privileged.
        self.assertEqual(self._launch()["user"], "root")

    def test_render_group_is_added(self):
        self.assertIn("render", self._launch()["group_add"])
        self.assertIn("video", self._launch()["group_add"])


if __name__ == "__main__":
    unittest.main()
