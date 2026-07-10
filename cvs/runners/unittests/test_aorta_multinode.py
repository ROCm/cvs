"""
Unit tests for the multi-node disaggregated launch path of ``AortaRunner``.

These tests exercise pure helpers on the runner (command construction, launch
mode resolution, port selection, head-node trace collection). The networked
container/SSH paths are not exercised here; see ``test_aorta.py`` for the
end-to-end pytest suite that runs against a real cluster.
"""

import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cvs.runners.aorta as aorta_mod
from cvs.runners.aorta import (
    AortaConfig,
    AortaDockerConfig,
    AortaEnvironment,
    AortaMultiNodeConfig,
    AortaRunner,
    RcclConfig,
    combined_traces_in,
)


def _make_runner(
    *,
    nodes,
    aorta_path,
    multi_node=None,
    base_config="config/distributed.yaml",
    experiment_script="scripts/launch_rocm.sh",
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
        multi_node=multi_node or AortaMultiNodeConfig(),
        build_script="scripts/launch_rocm.sh",
        experiment_script=experiment_script,
        gpus_per_node=8,
    )
    # The runner's __init__ aborts when the docker SDK is unavailable. None of
    # the helpers under test actually call into docker, so flip the module flag
    # for the duration of this call. This keeps the unit tests runnable in
    # minimal CI environments without the docker package.
    with patch.object(aorta_mod, "DOCKER_SDK_AVAILABLE", True):
        return AortaRunner(cfg)


class TestResolveLaunchMode(unittest.TestCase):
    def test_auto_resolves_to_script_for_single_node(self):
        r = _make_runner(nodes=["10.0.0.1"], aorta_path="/tmp/aorta")
        self.assertEqual(r._resolve_launch_mode(), "script")

    def test_auto_resolves_to_torchrun_for_multi_node(self):
        r = _make_runner(nodes=["10.0.0.1", "10.0.0.2"], aorta_path="/tmp/aorta")
        self.assertEqual(r._resolve_launch_mode(), "torchrun")

    def test_explicit_script_mode_is_respected(self):
        r = _make_runner(
            nodes=["10.0.0.1", "10.0.0.2"],
            aorta_path="/tmp/aorta",
            multi_node=AortaMultiNodeConfig(master_launch_mode="script"),
        )
        self.assertEqual(r._resolve_launch_mode(), "script")

    def test_explicit_torchrun_mode_is_respected_single_node(self):
        r = _make_runner(
            nodes=["10.0.0.1"],
            aorta_path="/tmp/aorta",
            multi_node=AortaMultiNodeConfig(master_launch_mode="torchrun"),
        )
        self.assertEqual(r._resolve_launch_mode(), "torchrun")


class TestPickMasterPort(unittest.TestCase):
    def test_returns_configured_port_when_set(self):
        mn = AortaMultiNodeConfig(master_port=29501)
        r = _make_runner(nodes=["10.0.0.1", "10.0.0.2"], aorta_path="/tmp/aorta", multi_node=mn)
        self.assertEqual(r._pick_master_port(), 29501)

    def test_returns_free_port_in_valid_range_when_unset(self):
        r = _make_runner(nodes=["10.0.0.1", "10.0.0.2"], aorta_path="/tmp/aorta")
        port = r._pick_master_port()
        self.assertIsInstance(port, int)
        self.assertGreater(port, 0)
        self.assertLess(port, 65536)
        # Port should be bindable right after we picked it (best-effort sanity).
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("", port))
            except OSError:
                # Race acceptable; we just want the value to look plausible.
                pass


class TestBuildTorchrunCommand(unittest.TestCase):
    def setUp(self):
        self.runner = _make_runner(
            nodes=["10.0.0.1", "10.0.0.2"],
            aorta_path="/tmp/aorta",
            multi_node=AortaMultiNodeConfig(),
            base_config="config/distributed_multinode.yaml",
        )

    def test_command_contains_required_torchrun_flags(self):
        cmd = self.runner._build_torchrun_command(
            node_rank=1,
            nnodes=2,
            master_addr="10.0.0.1",
            master_port=29500,
            nproc_per_node=8,
        )
        self.assertIn("torchrun", cmd)
        self.assertIn("--nnodes=2", cmd)
        self.assertIn("--node_rank=1", cmd)
        self.assertIn("--nproc_per_node=8", cmd)
        self.assertIn("--master_addr=10.0.0.1", cmd)
        self.assertIn("--master_port=29500", cmd)

    def test_command_uses_container_mount_paths(self):
        cmd = self.runner._build_torchrun_command(
            node_rank=0,
            nnodes=2,
            master_addr="10.0.0.1",
            master_port=29500,
            nproc_per_node=8,
        )
        self.assertIn("/mnt/train.py", cmd)
        self.assertIn("--config /mnt/config/distributed_multinode.yaml", cmd)

    def test_command_propagates_training_overrides(self):
        runner = _make_runner(nodes=["10.0.0.1", "10.0.0.2"], aorta_path="/tmp/aorta")
        runner.config.training_overrides = {"training.max_steps": 15, "profiling.active": 6}
        cmd = runner._build_torchrun_command(
            node_rank=0,
            nnodes=2,
            master_addr="10.0.0.1",
            master_port=29500,
            nproc_per_node=8,
        )
        self.assertIn("--override", cmd)
        self.assertIn("training.max_steps=15", cmd)
        self.assertIn("profiling.active=6", cmd)
        # All overrides must share a single `--override` group -- aorta's
        # argparse(nargs="*") silently drops earlier groups otherwise.
        self.assertEqual(cmd.count("--override"), 1)

    def test_extra_torchrun_and_train_args_are_appended(self):
        mn = AortaMultiNodeConfig(
            extra_torchrun_args=["--rdzv_backend=c10d"],
            extra_train_args=["--enable-rocm-metrics"],
        )
        runner = _make_runner(nodes=["a", "b"], aorta_path="/tmp/aorta", multi_node=mn)
        cmd = runner._build_torchrun_command(
            node_rank=0,
            nnodes=2,
            master_addr="a",
            master_port=29500,
            nproc_per_node=8,
        )
        self.assertIn("--rdzv_backend=c10d", cmd)
        self.assertIn("--enable-rocm-metrics", cmd)


class TestBuildBaseEnv(unittest.TestCase):
    def test_extra_env_is_merged_in(self):
        mn = AortaMultiNodeConfig(extra_env={"NCCL_SOCKET_IFNAME": "bond0", "MY_FLAG": "1"})
        runner = _make_runner(nodes=["a", "b"], aorta_path="/tmp/aorta", multi_node=mn)
        env = runner._build_base_env()
        self.assertEqual(env["NCCL_SOCKET_IFNAME"], "bond0")
        self.assertEqual(env["MY_FLAG"], "1")
        # Existing NCCL knobs should still be there.
        self.assertEqual(env["NCCL_MAX_NCHANNELS"], "112")
        self.assertIn("LD_LIBRARY_PATH", env)

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


class TestCombinedTracesIn(unittest.TestCase):
    def test_returns_true_when_under_combined_traces(self):
        root = Path("/aorta")
        self.assertTrue(combined_traces_in(root / "combined_traces" / "node_0" / "torch_profiler", root))

    def test_returns_false_for_real_run_artifacts(self):
        root = Path("/aorta")
        self.assertFalse(combined_traces_in(root / "artifacts" / "run1" / "torch_profiler", root))

    def test_returns_false_for_path_outside_root(self):
        root = Path("/aorta")
        self.assertFalse(combined_traces_in(Path("/elsewhere/torch_profiler"), root))


class TestCopyLocalTorchProfilers(unittest.TestCase):
    def test_copies_torch_profiler_trees_and_skips_combined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Real run artifact
            (root / "artifacts" / "run1" / "torch_profiler" / "rank_0").mkdir(parents=True)
            (root / "artifacts" / "run1" / "torch_profiler" / "rank_0" / "trace.json").write_text("{}")

            # Pre-existing combined traces (must be skipped to avoid recursion)
            (root / "combined_traces" / "node_0" / "torch_profiler").mkdir(parents=True)
            (root / "combined_traces" / "node_0" / "torch_profiler" / "trace.json").write_text("{}")

            dest = root / "combined_traces" / "node_0_new"
            dest.mkdir()

            runner = _make_runner(nodes=["a"], aorta_path=str(root))
            copied = runner._copy_local_torch_profilers(root, dest)

            self.assertTrue(copied)
            target = dest / "artifacts" / "run1" / "torch_profiler" / "rank_0" / "trace.json"
            self.assertTrue(target.exists(), f"Expected {target} to exist")
            # Combined traces tree itself must NOT have been re-copied under dest
            self.assertFalse((dest / "combined_traces").exists())

    def test_returns_false_when_no_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / "out"
            dest.mkdir()
            runner = _make_runner(nodes=["a"], aorta_path=str(root))
            self.assertFalse(runner._copy_local_torch_profilers(root, dest))


class TestCollectMultiNodeTracesHeadOnly(unittest.TestCase):
    """
    End-to-end happy path for trace collection where every node is the head
    (no SSH involved) so we can exercise the directory layout logic without a
    real cluster.
    """

    def test_layout_matches_combined_traces_node_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts" / "torch_profiler" / "rank_0").mkdir(parents=True)
            (root / "artifacts" / "torch_profiler" / "rank_0" / "trace.json").write_text("{}")

            # Single-node "cluster" so the head-node fast path is used for both ranks.
            runner = _make_runner(nodes=[socket.gethostname()], aorta_path=str(root))
            result = runner._collect_multi_node_traces([socket.gethostname()])

            self.assertIsNotNone(result)
            self.assertEqual(result, root / "combined_traces")
            self.assertTrue(
                (
                    root / "combined_traces" / "node_0" / "artifacts" / "torch_profiler" / "rank_0" / "trace.json"
                ).exists()
            )


class TestValidateConfigChecksTrainScriptInTorchrunMode(unittest.TestCase):
    def test_torchrun_mode_requires_train_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Set up a minimal aorta_path layout missing train.py
            (root / "config").mkdir()
            (root / "config" / "distributed.yaml").write_text("dummy: 1\n")
            (root / "scripts").mkdir()
            (root / "scripts" / "launch_rocm.sh").write_text("#!/bin/bash\n")

            runner = _make_runner(
                nodes=["a", "b"],
                aorta_path=str(root),
                multi_node=AortaMultiNodeConfig(master_launch_mode="torchrun"),
            )
            errors = runner.validate_config()
            self.assertTrue(
                any("train_script does not exist" in e for e in errors),
                f"Expected a train_script error, got: {errors}",
            )

    def test_script_mode_does_not_require_train_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config").mkdir()
            (root / "config" / "distributed.yaml").write_text("dummy: 1\n")
            (root / "scripts").mkdir()
            (root / "scripts" / "launch_rocm.sh").write_text("#!/bin/bash\n")

            runner = _make_runner(
                nodes=["a"],
                aorta_path=str(root),
                multi_node=AortaMultiNodeConfig(master_launch_mode="script"),
            )
            errors = runner.validate_config()
            self.assertFalse(
                any("train_script" in e for e in errors),
                f"train_script should not be required in script mode, got: {errors}",
            )


class TestSchemaMultiNodeBlock(unittest.TestCase):
    def test_extra_keys_under_multi_node_are_rejected(self):
        from cvs.parsers.schemas import AortaBenchmarkConfigFile
        from pydantic import ValidationError

        raw = {
            "aorta_path": "/tmp/aorta",
            "multi_node": {"bogus_key": "value"},
        }
        with self.assertRaises(ValidationError):
            AortaBenchmarkConfigFile.model_validate(raw)

    def test_invalid_master_launch_mode_rejected(self):
        from cvs.parsers.schemas import AortaBenchmarkConfigFile
        from pydantic import ValidationError

        raw = {
            "aorta_path": "/tmp/aorta",
            "multi_node": {"master_launch_mode": "magic"},
        }
        with self.assertRaises(ValidationError):
            AortaBenchmarkConfigFile.model_validate(raw)

    def test_default_multi_node_block_has_auto_mode(self):
        from cvs.parsers.schemas import AortaBenchmarkConfigFile

        raw = {"aorta_path": "/tmp/aorta"}
        cfg = AortaBenchmarkConfigFile.model_validate(raw)
        self.assertEqual(cfg.multi_node.master_launch_mode, "auto")
        self.assertTrue(cfg.multi_node.collect_traces)
        self.assertEqual(cfg.multi_node.train_script, "train.py")


if __name__ == "__main__":
    unittest.main()
