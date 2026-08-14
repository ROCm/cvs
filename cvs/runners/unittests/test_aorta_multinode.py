"""
Unit tests for the multi-node disaggregated launch path of ``AortaRunner``.

These tests exercise pure helpers on the runner (launch mode resolution, port
and address selection, torchrun command construction). The networked
container/SSH paths are not exercised here; see ``cvs/tests/benchmark/test_aorta.py``
for the end-to-end pytest suite that runs against a real cluster.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import subprocess
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
)
from cvs.runners.unittests.test_aorta import _make_runner


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
    def test_returns_configured_port_when_set_without_ssh(self):
        mn = AortaMultiNodeConfig(master_port=29501)
        r = _make_runner(nodes=["10.0.0.1", "10.0.0.2"], aorta_path="/tmp/aorta", multi_node=mn)
        with patch.object(aorta_mod.subprocess, "run") as mock_run:
            self.assertEqual(r._pick_master_port(), 29501)
        mock_run.assert_not_called()

    def test_picks_free_port_on_head_node_via_ssh(self):
        # The port must be free on the node running the rendezvous, not on
        # whichever host happens to be running the orchestrator.
        r = _make_runner(nodes=["10.0.0.1", "10.0.0.2"], aorta_path="/tmp/aorta")
        fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="29502\n", stderr="")
        with patch.object(aorta_mod.subprocess, "run", return_value=fake_result) as mock_run:
            port = r._pick_master_port()
        self.assertEqual(port, 29502)
        cmd = mock_run.call_args[0][0]
        self.assertIn("testuser@10.0.0.1", cmd)
        self.assertNotIn("testuser@10.0.0.2", cmd)

    def test_port_pick_ssh_uses_configured_pkey(self):
        r = _make_runner(nodes=["10.0.0.1"], aorta_path="/tmp/aorta")
        fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="29502\n", stderr="")
        with patch.object(aorta_mod.subprocess, "run", return_value=fake_result) as mock_run:
            r._pick_master_port()
        cmd = mock_run.call_args[0][0]
        self.assertIn("-i", cmd)
        self.assertIn("/home/testuser/.ssh/id_rsa", cmd)

    def test_port_pick_ssh_omits_identity_flag_without_pkey(self):
        cfg = AortaConfig(
            nodes=["10.0.0.1"],
            username="testuser",
            aorta_path=Path("/tmp/aorta"),
            base_config="config/distributed.yaml",
            docker=AortaDockerConfig(),
            rccl=RcclConfig(),
            environment=AortaEnvironment(),
            multi_node=AortaMultiNodeConfig(),
            build_script="scripts/launch_rocm.sh",
            experiment_script="scripts/launch_rocm.sh",
            gpus_per_node=8,
        )
        with patch.object(aorta_mod, "DOCKER_SDK_AVAILABLE", True):
            r = AortaRunner(cfg)
        fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="29502\n", stderr="")
        with patch.object(aorta_mod.subprocess, "run", return_value=fake_result) as mock_run:
            r._pick_master_port()
        cmd = mock_run.call_args[0][0]
        self.assertNotIn("-i", cmd)

    def test_raises_when_ssh_port_pick_fails(self):
        r = _make_runner(nodes=["10.0.0.1"], aorta_path="/tmp/aorta")
        fake_result = subprocess.CompletedProcess(args=[], returncode=255, stdout="", stderr="Connection refused")
        with patch.object(aorta_mod.subprocess, "run", return_value=fake_result):
            with self.assertRaises(RuntimeError):
                r._pick_master_port()

    def test_port_pick_remote_command_is_a_single_quoted_argument(self):
        # ssh joins every trailing argv element after the destination into one
        # remote-shell string; passing "python3", "-c", snippet as three
        # separate elements lets the remote shell reinterpret the snippet's
        # semicolons as its own command separators.
        r = _make_runner(nodes=["10.0.0.1"], aorta_path="/tmp/aorta")
        fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="29502\n", stderr="")
        with patch.object(aorta_mod.subprocess, "run", return_value=fake_result) as mock_run:
            r._pick_master_port()
        cmd = mock_run.call_args[0][0]
        dest_index = cmd.index("testuser@10.0.0.1")
        self.assertEqual(len(cmd) - dest_index - 1, 1)
        self.assertTrue(cmd[-1].startswith("python3 -c "))


class TestResolveMasterAddr(unittest.TestCase):
    def test_uses_explicit_override_when_set(self):
        mn = AortaMultiNodeConfig(master_addr="explicit.example.com")
        r = _make_runner(nodes=["10.0.0.1", "10.0.0.2"], aorta_path="/tmp/aorta", multi_node=mn)
        self.assertEqual(r._resolve_master_addr(), "explicit.example.com")

    def test_falls_back_to_head_node_when_no_vpc_ip_known(self):
        r = _make_runner(nodes=["10.0.0.1", "10.0.0.2"], aorta_path="/tmp/aorta")
        self.assertEqual(r._resolve_master_addr(), "10.0.0.1")

    def test_prefers_head_node_vpc_ip_when_known(self):
        # Other nodes must rendezvous over the RDMA fabric, not the mgmt/SSH
        # address, which may only be reachable from the orchestrator.
        r = _make_runner(nodes=["10.0.0.1", "10.0.0.2"], aorta_path="/tmp/aorta")
        r.config.node_vpc_ips = {"10.0.0.1": "192.168.100.1", "10.0.0.2": "192.168.100.2"}
        self.assertEqual(r._resolve_master_addr(), "192.168.100.1")


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


class TestBuildBaseEnvExtraEnv(unittest.TestCase):
    def test_extra_env_is_merged_in(self):
        mn = AortaMultiNodeConfig(extra_env={"NCCL_SOCKET_IFNAME": "bond0", "MY_FLAG": "1"})
        runner = _make_runner(nodes=["a", "b"], aorta_path="/tmp/aorta", multi_node=mn)
        env = runner._build_base_env()
        self.assertEqual(env["NCCL_SOCKET_IFNAME"], "bond0")
        self.assertEqual(env["MY_FLAG"], "1")
        # Existing NCCL knobs should still be there.
        self.assertEqual(env["NCCL_MAX_NCHANNELS"], "112")
        self.assertIn("LD_LIBRARY_PATH", env)


class TestValidateConfigChecksTrainScriptInTorchrunMode(unittest.TestCase):
    @staticmethod
    def _minimal_aorta_tree(root: Path) -> None:
        """An aorta_path layout that satisfies every check except train.py."""
        (root / "config").mkdir()
        (root / "config" / "distributed.yaml").write_text("dummy: 1\n")
        (root / "scripts").mkdir()
        (root / "scripts" / "launch_rocm.sh").write_text("#!/bin/bash\n")

    def test_torchrun_mode_requires_train_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._minimal_aorta_tree(root)

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
            self._minimal_aorta_tree(root)

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


if __name__ == "__main__":
    unittest.main()
