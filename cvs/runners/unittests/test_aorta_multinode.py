"""
Unit tests for the multi-node disaggregated launch path of ``AortaRunner``.

These tests exercise pure helpers on the runner (launch mode resolution, port
and address selection, torchrun command construction). The networked
container/SSH paths are not exercised here; see ``cvs/tests/benchmark/test_aorta.py``
for the end-to-end pytest suite that runs against a real cluster.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cvs.runners.aorta as aorta_mod
from cvs.runners._base_runner import RunStatus
from cvs.runners.aorta import (
    AortaConfig,
    AortaDockerConfig,
    AortaEnvironment,
    AortaMultiNodeConfig,
    AortaRunner,
    RcclConfig,
    combined_traces_in,
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


class TestRunBoundedParallel(unittest.TestCase):
    def test_all_tasks_succeed(self):
        tasks = {"a": lambda: 1, "b": lambda: 2}
        results, errors, timed_out = AortaRunner._run_bounded_parallel(tasks, timeout_seconds=5)
        self.assertEqual(results, {"a": 1, "b": 2})
        self.assertEqual(errors, {})
        self.assertEqual(timed_out, [])

    def test_task_exception_is_captured_per_key(self):
        def boom():
            raise ValueError("bad")

        tasks = {"good": lambda: "ok", "bad": boom}
        results, errors, timed_out = AortaRunner._run_bounded_parallel(tasks, timeout_seconds=5)
        self.assertEqual(results, {"good": "ok"})
        self.assertIsInstance(errors["bad"], ValueError)
        self.assertEqual(timed_out, [])

    def test_hung_task_times_out_without_blocking_caller(self):
        never_set = threading.Event()

        def hang():
            never_set.wait()
            return "unreachable"

        tasks = {"fast": lambda: "ok", "stuck": hang}
        start = time.time()
        results, errors, timed_out = AortaRunner._run_bounded_parallel(tasks, timeout_seconds=0.05)
        elapsed = time.time() - start

        self.assertLess(elapsed, 0.5)
        self.assertEqual(results, {"fast": "ok"})
        self.assertEqual(errors, {})
        self.assertEqual(timed_out, ["stuck"])
        never_set.set()

    def test_hung_task_runs_on_daemon_thread(self):
        # Proves a stuck node cannot hang the whole process at interpreter exit
        # (CPython's atexit joins every non-daemon thread regardless of any
        # shutdown(wait=False) call the caller might make on an executor).
        captured = []
        never_set = threading.Event()

        def hang():
            captured.append(threading.current_thread())
            never_set.wait()

        AortaRunner._run_bounded_parallel({"stuck": hang}, timeout_seconds=0.05)

        self.assertEqual(len(captured), 1)
        self.assertTrue(captured[0].daemon)
        self.assertTrue(captured[0].is_alive())
        never_set.set()
        captured[0].join(timeout=1)


class TestRunMultiNodeTimeout(unittest.TestCase):
    def test_hung_node_times_out_without_blocking_run(self):
        r = _make_runner(nodes=["10.0.0.1", "10.0.0.2"], aorta_path="/tmp/aorta")
        r.config.timeout_seconds = 0.05

        def fake_run_single_node(*, node, node_rank, launch_cmd, env):
            if node == "10.0.0.2":
                time.sleep(0.3)
            return (node, 0, "ok")

        with (
            patch.object(r, "_run_single_node", side_effect=fake_run_single_node),
            patch.object(r, "_pick_master_port", return_value=29500),
            patch.object(r, "_collect_multi_node_traces", return_value=None),
        ):
            start = time.time()
            result = r.run()
            elapsed = time.time() - start

        self.assertLess(elapsed, 0.25)
        self.assertEqual(result.status, RunStatus.TIMEOUT)
        self.assertEqual(result.exit_codes["10.0.0.1"], 0)
        self.assertEqual(result.exit_codes["10.0.0.2"], -1)
        self.assertIn("Timed out", result.stdout["10.0.0.2"])


class TestSetupTimeout(unittest.TestCase):
    def test_hung_node_times_out_without_blocking_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = _make_runner(nodes=["10.0.0.1", "10.0.0.2"], aorta_path=tmp)
            r.config.timeout_seconds = 0.05

            def fake_setup_single_node(node, cancel_event):
                if node == "10.0.0.2":
                    time.sleep(0.3)
                return (node, True, None)

            with patch.object(r, "_setup_single_node", side_effect=fake_setup_single_node):
                start = time.time()
                ok = r.setup()
                elapsed = time.time() - start

            self.assertFalse(ok)
            self.assertLess(elapsed, 0.25)


class TestSetupSingleNodeCancelledLate(unittest.TestCase):
    def test_container_launched_after_cancel_is_torn_down_not_registered(self):
        r = _make_runner(nodes=["10.0.0.1"], aorta_path="/tmp/aorta")
        fake_container = Mock()
        cancel_event = threading.Event()
        cancel_event.set()

        with (
            patch.object(r, "_connect_docker", return_value=Mock()),
            patch.object(r, "_cleanup_existing_containers"),
            patch.object(r, "_launch_container", return_value=fake_container),
        ):
            node, success, error = r._setup_single_node("10.0.0.1", cancel_event)

        self.assertFalse(success)
        self.assertIn("timed out", error.lower())
        self.assertNotIn("10.0.0.1", r._containers)
        fake_container.stop.assert_called_once()
        fake_container.remove.assert_called_once()

    def test_container_launched_before_cancel_is_registered_normally(self):
        r = _make_runner(nodes=["10.0.0.1"], aorta_path="/tmp/aorta")
        r.config.skip_rccl_build = True
        fake_container = Mock()
        cancel_event = threading.Event()

        with (
            patch.object(r, "_connect_docker", return_value=Mock()),
            patch.object(r, "_cleanup_existing_containers"),
            patch.object(r, "_launch_container", return_value=fake_container),
        ):
            node, success, error = r._setup_single_node("10.0.0.1", cancel_event)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIs(r._containers["10.0.0.1"], fake_container)
        fake_container.stop.assert_not_called()

    def test_registration_after_teardown_started_is_torn_down_not_registered(self):
        # A straggling setup thread can reach the registration point after
        # teardown() has already taken its snapshot of self._containers, even
        # though cancel_event was never set (e.g. setup() returned True/False
        # for unrelated reasons and execute() moved straight to teardown()).
        # It must notice via self._teardown_started and clean up after itself
        # instead of silently registering into a dict teardown() will never
        # look at again.
        r = _make_runner(nodes=["10.0.0.1"], aorta_path="/tmp/aorta")
        fake_container = Mock()
        cancel_event = threading.Event()
        r._teardown_started = True

        with (
            patch.object(r, "_connect_docker", return_value=Mock()),
            patch.object(r, "_cleanup_existing_containers"),
            patch.object(r, "_launch_container", return_value=fake_container),
        ):
            node, success, error = r._setup_single_node("10.0.0.1", cancel_event)

        self.assertFalse(success)
        self.assertIn("timed out", error.lower())
        self.assertNotIn("10.0.0.1", r._containers)
        fake_container.stop.assert_called_once()
        fake_container.remove.assert_called_once()

    def test_teardown_snapshots_containers_so_concurrent_registration_does_not_crash(self):
        # teardown() must iterate a snapshot, not the live self._containers dict.
        # A straggling setup thread can insert into self._containers while
        # teardown() is mid-iteration; if teardown() iterated the live dict,
        # that mutation would raise "dictionary changed size during iteration".
        r = _make_runner(nodes=["10.0.0.1"], aorta_path="/tmp/aorta")
        existing = Mock()
        existing.stop.side_effect = lambda *a, **k: r._containers.__setitem__("10.0.0.2", Mock())
        r._containers["10.0.0.1"] = existing

        with patch.object(r, "_get_remote_uid_gid", return_value=None):
            r.teardown()  # must not raise

        self.assertTrue(r._teardown_started)


class TestRunPartialNodeFailureStillCollectsTraces(unittest.TestCase):
    def test_rejected_old_traces_are_not_restored_by_discovery_or_legacy_fallback(self):
        for output_name in ("previous_run", "nodes1_rccl_develop_commsCh112_computeCh144"):
            with self.subTest(output_name=output_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                trace = root / output_name / "torch_profiler" / "rank0" / "trace.json"
                trace.parent.mkdir(parents=True)
                trace.write_text("{}")
                os.utime(trace, (1, 1))
                runner = _make_runner(nodes=["head", "worker"], aorta_path=root)

                with (
                    patch.object(
                        runner,
                        "_run_single_node",
                        side_effect=lambda *, node, **kw: (node, 1, "failed before profiling"),
                    ),
                    patch.object(runner, "_pick_master_port", return_value=29500),
                    patch.object(runner, "_copy_remote_torch_profilers", return_value=False),
                ):
                    result = runner.run()

                self.assertEqual(result.status, RunStatus.FAILED)
                self.assertIsNone(result.get_artifact("torch_traces"))
                self.assertEqual(list((root / "combined_traces").rglob("*.json")), [])

    def test_combined_artifact_is_not_replaced_by_a_newer_single_node_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            combined = root / "combined_traces"
            for rank in (0, 8):
                trace = combined / f"node_{rank // 8}" / "torch_profiler" / f"rank{rank}" / "trace.json"
                trace.parent.mkdir(parents=True)
                trace.write_text("{}")
                os.utime(trace, (100, 100))
            source = root / "output" / "torch_profiler" / "rank0" / "trace.json"
            source.parent.mkdir(parents=True)
            source.write_text("{}")
            os.utime(source, (200, 200))
            runner = _make_runner(nodes=["head", "worker"], aorta_path=root)

            with (
                patch.object(runner, "_run_single_node", side_effect=lambda *, node, **kw: (node, 0, "ok")),
                patch.object(runner, "_pick_master_port", return_value=29500),
                patch.object(runner, "_collect_multi_node_traces", return_value=combined),
            ):
                result = runner.run()

            self.assertEqual(result.get_artifact("torch_traces"), combined)

    def test_failed_node_does_not_block_trace_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            aorta_path = Path(tmp)
            combined_root = aorta_path / "combined_traces"
            combined_root.mkdir()
            r = _make_runner(nodes=["10.0.0.1", "10.0.0.2"], aorta_path=aorta_path)

            def fake_run_single_node(*, node, node_rank, launch_cmd, env):
                if node == "10.0.0.2":
                    return (node, 1, "boom")
                return (node, 0, "ok")

            with (
                patch.object(r, "_run_single_node", side_effect=fake_run_single_node),
                patch.object(r, "_pick_master_port", return_value=29500),
                patch.object(r, "_collect_multi_node_traces", return_value=combined_root) as mock_collect,
            ):
                result = r.run()

            mock_collect.assert_called_once()
            self.assertEqual(mock_collect.call_args.args[0], ["10.0.0.1", "10.0.0.2"])
            self.assertIn("min_mtime", mock_collect.call_args.kwargs)
            self.assertEqual(result.status, RunStatus.FAILED)
            self.assertIn("10.0.0.2", result.error_message)
            self.assertEqual(result.get_artifact("torch_traces"), combined_root)

    def test_stale_combined_traces_copy_is_not_discovered_as_trace_dir(self):
        # combined_traces_in must recognize paths nested under combined_traces
        # (e.g. a leftover node_0/torch_profiler/ copy from a prior run in the
        # same aorta_path, with this run's own collection disabled/empty) so
        # the discovery loop never promotes it as trace_dir -- it's a partial,
        # single-node view, not a fresh candidate.
        with tempfile.TemporaryDirectory() as tmp:
            aorta_path = Path(tmp)
            stale_copy = aorta_path / "combined_traces" / "node_0" / "torch_profiler"
            stale_copy.mkdir(parents=True)
            (stale_copy / "trace.json").write_text("{}")
            r = _make_runner(nodes=["10.0.0.1", "10.0.0.2"], aorta_path=aorta_path)
            r.config.multi_node.collect_traces = False

            def fake_run_single_node(*, node, node_rank, launch_cmd, env):
                return (node, 0, "ok")

            with (
                patch.object(r, "_run_single_node", side_effect=fake_run_single_node),
                patch.object(r, "_pick_master_port", return_value=29500),
            ):
                result = r.run()

            self.assertIsNone(result.get_artifact("torch_traces"))


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
    def test_only_fresh_files_are_copied_from_a_reused_profiler_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiler = root / "output" / "torch_profiler"
            for name, mtime in (
                ("rank0/trace_step100.json", 1),
                ("rank0/trace_step10.json", 100.5),
                ("rank8/trace.json", 1),
            ):
                trace = profiler / name
                trace.parent.mkdir(parents=True, exist_ok=True)
                trace.write_text(name)
                os.utime(trace, (mtime, mtime))
            dest = root / "combined_traces" / "node_0"
            runner = _make_runner(nodes=["head"], aorta_path=root)

            self.assertTrue(runner._copy_local_torch_profilers(root, dest, min_mtime=100.5))

            copied = [str(path.relative_to(dest)) for path in dest.rglob("*.json")]
            self.assertEqual(copied, ["output/torch_profiler/rank0/trace_step10.json"])
            self.assertTrue((profiler / "rank0/trace_step100.json").exists())

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


class TestCopyRemoteTorchProfilers(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "torch_profiler" / "aorta checkout"
        self.dest = Path(tmp.name) / "collected"
        self.fresh_paths = [
            "run with spaces/torch_profiler/rank0/trace_step10.json",
            "run with spaces/torch_profiler/rank1/trace with\nnewline.json",
        ]
        for name, mtime in (
            *((name, 100.5) for name in self.fresh_paths),
            ("run with spaces/torch_profiler/rank0/trace_step100.json", 1),
            ("run with spaces/torch_profiler/rank8/trace.json", 1),
            ("combined_traces/node_0/torch_profiler/rank0/trace.json", 200),
            ("unrelated.json", 200),
        ):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
            os.utime(path, (mtime, mtime))
        self.runner = _make_runner(nodes=["head", "worker"], aorta_path=self.root)
        self.run_process = subprocess.run

    def _mock_transport(self, cmd, **kwargs):
        if cmd[0] == "ssh":
            return self.run_process(["sh", "-c", cmd[-1]], capture_output=True, text=True)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def test_rsync_receives_only_fresh_files_with_whitespace_preserved(self):
        with (
            patch.object(aorta_mod.subprocess, "run", side_effect=self._mock_transport) as transport,
            patch("shutil.which", return_value="/usr/bin/rsync"),
        ):
            self.assertTrue(self.runner._copy_remote_torch_profilers("worker", self.dest, min_mtime=100.5))

        self.assertEqual(transport.call_count, 2)
        transfer = transport.call_args
        self.assertEqual(set(transfer.kwargs["input"].split("\0")[:-1]), set(self.fresh_paths))
        cmd = transfer.args[0]
        self.assertIn("--from0", cmd)
        self.assertIn("--files-from=-", cmd)
        self.assertIn("--protect-args", cmd)
        self.assertEqual(cmd[-2:], [f"testuser@worker:{self.root}/", str(self.dest) + "/"])

    def test_scp_fallback_copies_only_selected_files(self):
        with (
            patch.object(aorta_mod.subprocess, "run", side_effect=self._mock_transport) as transport,
            patch("shutil.which", return_value=None),
        ):
            self.assertTrue(self.runner._copy_remote_torch_profilers("worker", self.dest, min_mtime=100.5))

        transfers = [call.args[0] for call in transport.call_args_list[1:]]
        self.assertTrue(all(cmd[0] == "scp" for cmd in transfers))
        self.assertEqual({cmd[-2] for cmd in transfers}, {f"testuser@worker:{self.root}/{p}" for p in self.fresh_paths})
        self.assertEqual({cmd[-1] for cmd in transfers}, {str(self.dest / p) for p in self.fresh_paths})

    def test_no_fresh_remote_files_does_not_start_a_transfer(self):
        with patch.object(aorta_mod.subprocess, "run", side_effect=self._mock_transport) as transport:
            self.assertFalse(self.runner._copy_remote_torch_profilers("worker", self.dest, min_mtime=300))
        transport.assert_called_once()

    def test_failed_listing_is_reported_without_starting_a_transfer(self):
        result = subprocess.CompletedProcess([], 255, stdout="", stderr="connection failed")
        with patch.object(aorta_mod.subprocess, "run", return_value=result) as transport:
            self.assertFalse(self.runner._copy_remote_torch_profilers("worker", self.dest, min_mtime=100))
        transport.assert_called_once()


class TestRunTracelensAnalysisDependencyCheck(unittest.TestCase):
    def test_dependency_check_exception_is_caught_not_raised(self):
        # TraceLens analysis is optional, best-effort post-processing. An
        # exception while merely probing whether the package is importable
        # (e.g. the container exec plumbing itself failing) must not escape
        # and be mistaken by run()'s caller for the run itself having failed.
        r = _make_runner(nodes=["10.0.0.1"], aorta_path="/tmp/aorta")
        with patch.object(r, "_exec_in_container", side_effect=RuntimeError("docker exec blew up")):
            result = r._run_tracelens_analysis(Mock(), Path("/tmp/aorta/some_run"))
        self.assertIsNone(result)

    def test_dependency_missing_skips_without_raising(self):
        r = _make_runner(nodes=["10.0.0.1"], aorta_path="/tmp/aorta")
        with patch.object(r, "_exec_in_container", return_value=(1, "ModuleNotFoundError")):
            result = r._run_tracelens_analysis(Mock(), Path("/tmp/aorta/some_run"))
        self.assertIsNone(result)


class TestRunExceptionPreservesArtifacts(unittest.TestCase):
    def test_exception_after_trace_collection_still_returns_collected_artifacts(self):
        # A failure in later, optional post-processing (e.g. TraceLens analysis)
        # must not discard torch_traces and other artifacts already collected
        # earlier in run() -- those are exactly what a partially-failed run
        # needs to be salvageable.
        with tempfile.TemporaryDirectory() as tmp:
            aorta_path = Path(tmp)
            combined_root = aorta_path / "combined_traces"
            (combined_root / "node_0").mkdir(parents=True)
            r = _make_runner(nodes=["10.0.0.1", "10.0.0.2"], aorta_path=aorta_path)

            def fake_run_single_node(*, node, node_rank, launch_cmd, env):
                return (node, 0, "ok")

            with (
                patch.object(r, "_run_single_node", side_effect=fake_run_single_node),
                patch.object(r, "_pick_master_port", return_value=29500),
                patch.object(r, "_collect_multi_node_traces", return_value=combined_root),
                patch.object(r, "_resolve_analysis_output_dir", side_effect=RuntimeError("boom")),
            ):
                result = r.run()

            self.assertEqual(result.status, RunStatus.FAILED)
            self.assertEqual(result.get_artifact("torch_traces"), combined_root)


class TestResolveAnalysisOutputDir(unittest.TestCase):
    def test_non_combined_trace_dir_is_returned_unchanged(self):
        r = _make_runner(nodes=["10.0.0.1"], aorta_path="/tmp/aorta")
        trace_dir = Path("/tmp/aorta/some_run/torch_profiler")
        output_dir = trace_dir.parent
        self.assertEqual(r._resolve_analysis_output_dir(trace_dir, output_dir), output_dir)

    def test_combined_trace_dir_resolves_to_head_nodes_own_tree(self):
        # trace_dir/output_dir as computed in run() for the combined multi-node
        # case: output_dir is aorta_path itself, which has no torch_profiler
        # child of its own and is useless to the analysis scripts.
        with tempfile.TemporaryDirectory() as tmp:
            aorta_path = Path(tmp)
            head_trace = aorta_path / "combined_traces" / "node_0" / "artifacts" / "torch_profiler"
            head_trace.mkdir(parents=True)

            r = _make_runner(nodes=["10.0.0.1", "10.0.0.2"], aorta_path=aorta_path)
            trace_dir = aorta_path / "combined_traces"
            resolved = r._resolve_analysis_output_dir(trace_dir, aorta_path)

            self.assertEqual(resolved, head_trace.parent)

    def test_combined_trace_dir_with_no_head_tree_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            aorta_path = Path(tmp)
            (aorta_path / "combined_traces" / "node_0").mkdir(parents=True)

            r = _make_runner(nodes=["10.0.0.1", "10.0.0.2"], aorta_path=aorta_path)
            trace_dir = aorta_path / "combined_traces"
            resolved = r._resolve_analysis_output_dir(trace_dir, aorta_path)

            self.assertIsNone(resolved)


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

    def test_stale_trace_from_previous_run_is_cleared_before_recollection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts" / "torch_profiler" / "rank_0").mkdir(parents=True)
            (root / "artifacts" / "torch_profiler" / "rank_0" / "trace.json").write_text("run1")

            runner = _make_runner(nodes=[socket.gethostname()], aorta_path=str(root))
            first = runner._collect_multi_node_traces([socket.gethostname()])
            self.assertIsNotNone(first)
            stale_file = root / "combined_traces" / "node_0" / "artifacts" / "torch_profiler" / "rank_0" / "trace.json"
            self.assertEqual(stale_file.read_text(), "run1")

            # Second run: this node produces no new torch_profiler output this
            # time (e.g. training crashed before profiling started). The prior
            # run's copy must not still be sitting in combined_traces, where
            # it would be mistaken for this run's data.
            shutil.rmtree(root / "artifacts" / "torch_profiler")
            with patch.object(runner, "_copy_remote_torch_profilers", return_value=False):
                second = runner._collect_multi_node_traces([socket.gethostname()])

            self.assertIsNone(second)
            self.assertFalse(stale_file.exists())

    def test_shrunk_cluster_does_not_leave_previous_runs_higher_rank_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts" / "torch_profiler" / "rank_0").mkdir(parents=True)
            (root / "artifacts" / "torch_profiler" / "rank_0" / "trace.json").write_text("{}")

            runner = _make_runner(nodes=[socket.gethostname()], aorta_path=str(root))
            # First run had 3 nodes; node_2 belongs to a node no longer in this run.
            with patch.object(runner, "_copy_remote_torch_profilers", return_value=False) as remote_copy:
                first = runner._collect_multi_node_traces([socket.gethostname(), "worker1", "worker2"])
            self.assertIsNotNone(first)
            self.assertTrue((root / "combined_traces" / "node_2").exists())
            self.assertEqual(remote_copy.call_count, 2)

            # This run only has one node -- the old node_2 must not survive, or a
            # parser walking combined_traces would still see its stale rank data.
            second = runner._collect_multi_node_traces([socket.gethostname()])
            self.assertIsNotNone(second)
            self.assertFalse((root / "combined_traces" / "node_2").exists())
            self.assertFalse((root / "combined_traces" / "node_1").exists())
            self.assertTrue((root / "combined_traces" / "node_0").exists())


class TestCollectMultiNodeTracesSourceFreshness(unittest.TestCase):
    """
    ``min_mtime`` lets collection tell a previous run's leftover torch_profiler
    output apart from this run's, even when the training config reuses the
    same output_dir across runs (so the path alone can't tell them apart).
    """

    def test_source_tree_older_than_min_mtime_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_file = root / "artifacts" / "torch_profiler" / "rank_0" / "trace.json"
            trace_file.parent.mkdir(parents=True)
            trace_file.write_text("stale")
            old_mtime = trace_file.stat().st_mtime - 3600
            os.utime(trace_file, (old_mtime, old_mtime))

            runner = _make_runner(nodes=[socket.gethostname()], aorta_path=str(root))
            # This run "started" after the file above was written, so it must
            # not be mistaken for this run's data.
            with patch.object(runner, "_copy_remote_torch_profilers", return_value=False):
                result = runner._collect_multi_node_traces([socket.gethostname()], min_mtime=old_mtime + 1800)

            self.assertIsNone(result)
            self.assertFalse((root / "combined_traces" / "node_0" / "artifacts").exists())

    def test_source_tree_newer_than_min_mtime_is_collected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_file = root / "artifacts" / "torch_profiler" / "rank_0" / "trace.json"
            trace_file.parent.mkdir(parents=True)
            trace_file.write_text("fresh")
            run_start = trace_file.stat().st_mtime - 60

            runner = _make_runner(nodes=[socket.gethostname()], aorta_path=str(root))
            result = runner._collect_multi_node_traces([socket.gethostname()], min_mtime=run_start)

            self.assertIsNotNone(result)
            self.assertTrue(
                (
                    root / "combined_traces" / "node_0" / "artifacts" / "torch_profiler" / "rank_0" / "trace.json"
                ).exists()
            )

    def test_no_min_mtime_collects_regardless_of_age(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_file = root / "artifacts" / "torch_profiler" / "rank_0" / "trace.json"
            trace_file.parent.mkdir(parents=True)
            trace_file.write_text("old but no floor given")
            old_mtime = trace_file.stat().st_mtime - 3600
            os.utime(trace_file, (old_mtime, old_mtime))

            runner = _make_runner(nodes=[socket.gethostname()], aorta_path=str(root))
            result = runner._collect_multi_node_traces([socket.gethostname()])

            self.assertIsNotNone(result)


class TestRunPassesTraceFreshnessFloor(unittest.TestCase):
    def test_run_passes_min_mtime_derived_from_start_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = _make_runner(nodes=["10.0.0.1", "10.0.0.2"], aorta_path=Path(tmp))

            def fake_run_single_node(*, node, node_rank, launch_cmd, env):
                return (node, 0, "ok")

            before = time.time()
            with (
                patch.object(r, "_run_single_node", side_effect=fake_run_single_node),
                patch.object(r, "_pick_master_port", return_value=29500),
                patch.object(r, "_collect_multi_node_traces", return_value=None) as mock_collect,
            ):
                r.run()
            after = time.time()

            min_mtime = mock_collect.call_args.kwargs["min_mtime"]
            # Derived from run()'s own start_time (bracketed by before/after),
            # offset by exactly the configured skew tolerance -- not some
            # unrelated or hardcoded value.
            self.assertGreaterEqual(min_mtime, before - r._TRACE_FRESHNESS_SKEW_SECONDS - 1)
            self.assertLessEqual(min_mtime, after - r._TRACE_FRESHNESS_SKEW_SECONDS + 1)


if __name__ == "__main__":
    unittest.main()
