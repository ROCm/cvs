"""Aorta execution tests with a mocked orchestrator and real shell wrappers."""

import json
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cvs.lib import globals
from cvs.lib.benchmark.aorta.aorta_artifacts import pack_traces
from cvs.lib.benchmark.aorta.aorta_config_loader import AortaVariantConfig
from cvs.lib.benchmark.aorta.aorta_job import AortaJob, _text
from cvs.lib.benchmark.aorta.unittests.fixtures import variant_dict


class TestAortaJob(unittest.TestCase):
    def setUp(self):
        prior_errors = globals.error_list
        globals.error_list = []
        self.addCleanup(setattr, globals, "error_list", prior_errors)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw = variant_dict()
        self.raw["output_dir"] = self.tmp.name
        self.orch = MagicMock()
        self.orch.hosts = ["node-a", "node-b"]
        self.orch.exec.return_value = {host: {"exit_code": 0, "output": ""} for host in self.orch.hosts}
        self.orch.exec_cmd_list.return_value = {host: "CVS_AORTA_COMMAND_OK\n" for host in self.orch.hosts}

    def job(self, **kwargs):
        return AortaJob(self.orch, AortaVariantConfig.model_validate(self.raw), **kwargs)

    def test_modes_and_empty_cluster(self):
        self.assertEqual(self.job().mode, "torchrun")
        self.orch.hosts = ["node-a"]
        self.assertEqual(self.job().mode, "script")
        self.raw["multi_node"] = {"master_launch_mode": "torchrun"}
        self.assertEqual(self.job().mode, "torchrun")
        self.raw["multi_node"] = {"master_launch_mode": "script"}
        self.orch.hosts = ["node-a", "node-b"]
        with self.assertRaisesRegex(ValueError, "requires one node"):
            self.job()
        self.orch.hosts = []
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.job()

    def test_explicit_and_fabric_rendezvous_precedence(self):
        self.raw["multi_node"] = {"master_port": 29500}
        job = self.job(node_vpc_ips={"node-a": "fabric-a"})
        commands = job.build_launch_cmd()
        argv = shlex.split(shlex.split(commands[1])[2])
        self.assertIn("--node_rank=1", argv)
        self.assertIn("--nnodes=2", argv)
        self.assertIn("--nproc_per_node=2", argv)
        self.assertIn("--master_addr=fabric-a", argv)
        self.assertIn("--master_port=29500", argv)
        self.raw["multi_node"]["master_addr"] = "explicit-master"
        self.assertIn("explicit-master", self.job(node_vpc_ips={"node-a": "fabric-a"}).build_launch_cmd()[0])
        self.orch.exec.assert_not_called()

    def test_port_is_selected_on_head_container(self):
        self.orch.exec.return_value = {"node-a": {"exit_code": 0, "output": "34567\n"}}
        job = self.job()
        job.build_launch_cmd()
        self.assertEqual(job.master_addr, "node-a")
        self.assertEqual(job.master_port, 34567)
        self.assertEqual(self.orch.exec.call_args.kwargs["hosts"], ["node-a"])

    def test_overrides_grouped_once_and_quoted(self):
        self.raw["training_overrides"] = {"training.max_steps": 15, "output_dir": "space and 'quote"}
        job = self.job()
        for command in (
            job._build_experiment_command(),
            shlex.split(job._build_torchrun_command(0, "master", 29500))[2],
        ):
            argv = shlex.split(command)
            self.assertEqual(argv.count("--override"), 1)
            self.assertIn("training.max_steps=15", argv)
            self.assertIn("output_dir=space and 'quote", argv)

    def test_environment_computation_and_extra_values(self):
        self.raw["training_overrides"] = {"training.max_steps": 15, "profiling.active": 6}
        self.raw["multi_node"] = {"extra_env": {"NCCL_MAX_NCHANNELS": "64", "CUSTOM": "a b"}}
        env = self.job()._build_base_env()
        self.assertEqual(env["TENSILE_STREAMK_MAX_CUS"], "192")
        self.assertEqual(env["CUSTOM"], "a b")
        self.assertEqual(env["rccl_path"], "/mnt/rccl")
        self.assertEqual(env["AORTA_OVERRIDE_ARGS"], "--override training.max_steps=15 profiling.active=6")

    def test_checked_commands_detect_missing_and_failed_nodes(self):
        job = self.job()
        for results in (
            {},
            {"node-a": {"exit_code": 0, "output": "ok"}},
            {"node-a": {"exit_code": 1, "output": "failure"}},
        ):
            self.orch.exec.return_value = results
            with self.subTest(results=results), self.assertRaises(RuntimeError):
                job._exec("true")
        self.orch.exec_cmd_list.return_value = {"node-a": "CVS_AORTA_COMMAND_OK", "node-b": "connection failed"}
        with self.assertRaisesRegex(RuntimeError, "node-b"):
            job._exec_list(["true", "true"])
        self.assertEqual(_text({"output": "ok"}), "ok")
        self.assertEqual(_text(None), "")

    def test_prepare_hosts_records_numeric_ownership(self):
        self.orch.exec_on_host.return_value = {
            host: {"exit_code": 0, "output": "1001\n1002\n107\n"} for host in self.orch.hosts
        }
        job = self.job()
        job.prepare_hosts()
        self.assertEqual(job.owners, {"node-a": "1001:1002", "node-b": "1001:1002"})
        self.assertEqual(job.render_gids, {"107"})
        self.assertEqual(job.container_groups(), ["video", "107"])
        self.assertIn("mkdir -p", self.orch.exec_on_host.call_args_list[1].args[0])
        self.assertIn("test -f", self.orch.exec_on_host.call_args_list[2].args[0])

    def test_prepare_hosts_supports_per_host_render_gids(self):
        self.orch.exec_on_host.return_value = {
            "node-a": {"exit_code": 0, "output": "1001\n1002\n107\n"},
            "node-b": {"exit_code": 0, "output": "1001\n1002\n44\n"},
        }
        job = self.job()
        job.prepare_hosts()
        self.assertEqual(job.container_groups(), ["video", "107", "44"])

    def test_auto_clone_runs_on_host_and_verify_runs_in_container(self):
        self.raw["skip_rccl_build"] = True
        self.raw["aorta_auto_clone"] = True
        self.raw["aorta_clone_url"] = "https://example.invalid/aorta.git"
        job = self.job()
        self.orch.exec_on_host.return_value = {
            host: {"exit_code": 0, "output": "1001\n1002\n"} for host in self.orch.hosts
        }
        job.prepare_hosts()
        host_command = self.orch.exec_on_host.call_args_list[1].args[0]
        self.assertIn("git clone --", host_command)
        self.assertIn("https://example.invalid/aorta.git", host_command)
        self.assertIn("test -f", self.orch.exec_on_host.call_args_list[2].args[0])
        job.clone_or_verify_aorta_repo()
        command = self.orch.exec.call_args.args[0]
        self.assertIn("train.py", command)
        self.assertNotIn("git clone", command)
        self.assertNotIn("build_rccl.sh", command)
        self.assertNotIn("rccl_exp.sh", command)

    def test_distributed_setup_checks_configured_devices(self):
        self.raw["container"]["runtime"]["args"]["devices"] = ["/dev/infiniband/rdma_cm"]
        self.job().setup_distributed()
        command = self.orch.exec.call_args.args[0]
        self.assertIn("command -v torchrun", command)
        self.assertIn("test -e /dev/infiniband/rdma_cm", command)

    def test_wrappers_preserve_env_and_report_actual_exit_status(self):
        root = Path(self.tmp.name) / "repo with spaces"
        root.mkdir()
        self.raw["aorta_path"] = str(root)
        self.raw["container_mount_path"] = str(root)
        self.raw["container"]["runtime"]["args"]["volumes"] = [f"{root}:{root}"]
        self.raw["container"]["env"]["CUSTOM"] = "literal $(false) and 'quotes'"
        job = self.job()
        job._start_phase(['printf "%s\\n" "$CUSTOM"', "false"], "benchmark")
        write_commands = self.orch.exec_cmd_list.call_args_list[-2].args[0]
        for rank, wrapped in enumerate(write_commands):
            script = shlex.split(shlex.split(wrapped)[2])[2]
            result = subprocess.run(
                ["bash"],
                input=script,
                text=True,
                capture_output=True,
                env={"PATH": os.defpath, "LD_LIBRARY_PATH": "/existing"},
            )
            self.assertEqual(result.returncode, rank)
            self.assertIn(f"CVS_AORTA_EXIT={rank}", result.stdout)
            if rank == 0:
                self.assertIn("literal $(false) and 'quotes'", result.stdout)
        launches = self.orch.exec_cmd_list.call_args.args[0]
        self.assertTrue(all("nohup setsid bash" in command for command in launches))

    def test_phase_finished_when_every_host_has_recorded_an_exit_code(self):
        job = self.job()
        job.log_paths = ["/tmp/rank0.log", "/tmp/rank1.log"]
        self.orch.exec_cmd_list.return_value = {
            "node-a": "0\nCVS_AORTA_COMMAND_OK",
            "node-b": "\nCVS_AORTA_COMMAND_OK",
        }
        self.assertFalse(job._phase_finished())
        # A recorded failure (non-zero exit) still counts as "finished" -- success
        # is judged separately by _poll_phase, not by _phase_finished.
        self.orch.exec_cmd_list.return_value["node-b"] = "3\nCVS_AORTA_COMMAND_OK"
        self.assertTrue(job._phase_finished())
        self.orch.exec_cmd_list.return_value["node-b"] = "connection failed"
        with self.assertRaises(RuntimeError):
            job._phase_finished()

    @patch("cvs.lib.benchmark.aorta.aorta_job.LogPoller")
    def test_benchmark_polling_updates_status_and_duration(self, poller):
        job = self.job()
        job.start_time = 100
        job.log_paths = ["/tmp/rank0.log", "/tmp/rank1.log"]
        with patch("cvs.lib.benchmark.aorta.aorta_job.time.time", return_value=110):
            job.poll_for_completion()
        self.assertTrue(job.succeeded)
        self.assertEqual(job.duration_seconds, 10)
        self.assertEqual(poller.call_args.args, (self.orch, job.log_paths))
        self.assertEqual(poller.call_args.kwargs["is_complete"].__name__, "is_complete")
        self.assertNotIn("error_patterns", poller.call_args.kwargs)
        poller.return_value.poll.side_effect = RuntimeError("worker failed")
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            job.poll_for_completion()
        self.assertFalse(job.succeeded)
        self.assertEqual(job.error_message, "worker failed")

    @patch("cvs.lib.benchmark.aorta.aorta_job.LogPoller")
    def test_poll_timeout_is_distinguished_from_a_node_failure(self, poller):
        job = self.job()
        job.log_paths = ["/tmp/rank0.log", "/tmp/rank1.log"]
        poller.return_value.poll.side_effect = RuntimeError("Aorta did not complete within 30s")
        with self.assertRaises(TimeoutError):
            job.poll_for_completion()
        self.assertEqual(job.status, "timeout")

    @patch("cvs.lib.benchmark.aorta.aorta_job.LogPoller")
    def test_poll_phase_logs_newly_failed_hosts_without_ending_the_wait(self, poller):
        job = self.job()
        job.log_paths = ["/tmp/rank0.log", "/tmp/rank1.log"]
        with patch.object(job, "_phase_exit_codes", return_value={"node-a": 0, "node-b": 0}):
            job._poll_phase()
        is_complete = poller.call_args.kwargs["is_complete"]
        # is_complete() reads _phase_exit_codes() twice per call (once directly,
        # once via _phase_finished), so each simulated poll tick repeats its value.
        codes = [{"node-a": 1}, {"node-a": 1}, {"node-a": 1, "node-b": 0}, {"node-a": 1, "node-b": 0}]
        with (
            patch.object(job, "_phase_exit_codes", side_effect=codes),
            self.assertLogs("cvs.lib.benchmark.aorta.aorta_job", level="WARNING") as logs,
        ):
            self.assertFalse(is_complete())
            self.assertTrue(is_complete())
        self.assertEqual(sum("node-a" in message for message in logs.output), 1)

    @patch("cvs.lib.benchmark.aorta.aorta_job.LogPoller")
    def test_poll_reports_failed_nodes_only_after_all_nodes_finish(self, poller):
        job = self.job()
        job.log_paths = ["/tmp/rank0.log", "/tmp/rank1.log"]
        with patch.object(job, "_phase_exit_codes", return_value={"node-a": 0, "node-b": 3}):
            with self.assertRaisesRegex(RuntimeError, "node-b.*3"):
                job._poll_phase()
        poller.return_value.poll.assert_called_once()

    @patch("cvs.lib.benchmark.aorta.aorta_job.LogPoller")
    def test_rccl_polling_cleanup_and_skip(self, poller):
        job = self.job()
        poller.return_value.poll.side_effect = RuntimeError("build failed")
        with patch.object(job, "stop_processes") as stop:
            with self.assertRaisesRegex(RuntimeError, "build failed"):
                job.build_rccl()
            stop.assert_called_once()
        self.raw["skip_rccl_build"] = True
        poller.reset_mock()
        self.job().build_rccl()
        poller.assert_not_called()

    def test_build_rccl_preserves_poll_failure_when_stop_also_fails(self):
        job = self.job()
        with (
            patch.object(job, "_start_phase"),
            patch.object(job, "_poll_phase", side_effect=RuntimeError("build failed")),
            patch.object(job, "stop_processes", side_effect=RuntimeError("stop failed")) as stop,
            self.assertLogs("cvs.lib.benchmark.aorta.aorta_job", level="WARNING") as logs,
        ):
            with self.assertRaisesRegex(RuntimeError, "build failed"):
                job.build_rccl()
        stop.assert_called_once()
        self.assertTrue(any("stop failed" in message for message in logs.output))

    def test_start_records_each_nodes_clock(self):
        self.orch.exec.return_value = {
            "node-a": {"exit_code": 0, "output": "1000.5"},
            "node-b": {"exit_code": 0, "output": "2000.5"},
        }
        job = self.job()
        job.launch_commands = ["true", "true"]
        job.start_job()
        self.assertEqual(job.trace_floors, {"node-a": 1000.5, "node-b": 2000.5})
        self.assertTrue(job.started)
        self.assertEqual(job.status, "running")
        self.assertEqual(len(self.orch.exec_cmd_list.call_args.args[0]), 2)

    def test_trace_download_uses_host_mount_and_transport(self):
        job = self.job()
        root = Path(self.tmp.name) / "remote"
        (root / "output/torch_profiler").mkdir(parents=True)
        (root / "output/torch_profiler/rank0.json").write_text('{"traceEvents": []}')
        archive = Path(self.tmp.name) / "download.tar.gz"
        pack_traces(root, archive, 0)
        self.orch.download_file.return_value = {"node-a": str(archive)}
        dest = Path(self.tmp.name) / "collected"
        job._download_archive("node-a", 0, "traces.tar.gz", dest)
        self.assertTrue((dest / "output/torch_profiler/rank0.json").exists())
        remote = self.orch.download_file.call_args.args[0]
        self.assertTrue(remote.startswith("/scratch/tester/repo/.cvs-aorta/"))
        self.assertEqual(self.orch.download_file.call_args.kwargs["hosts"], ["node-a"])
        self.assertFalse(archive.exists())
        self.assertIn("rm -f -- /mnt/.cvs-aorta/", self.orch.exec.call_args.args[0])

    def test_invalid_archive_does_not_publish_partial_traces_or_remove_remote_copy(self):
        job = self.job()
        archive = Path(self.tmp.name) / "invalid.tar.gz"
        archive.write_bytes(b"not an archive")
        self.orch.download_file.return_value = {"node-a": str(archive)}
        destination = job.output_dir / "combined_traces/node_0"
        import tarfile

        with self.assertRaises(tarfile.TarError):
            job._download_archive("node-a", 0, "traces.tar.gz", destination)
        self.assertFalse(destination.exists())
        self.orch.exec.assert_not_called()

    def test_partial_collection_keeps_surviving_node_layout(self):
        job = self.job()
        job.started = True
        job.prepared = True
        job.trace_floors = {host: 1500 for host in self.orch.hosts}
        with (
            patch.object(
                job, "_exec", side_effect=[RuntimeError("node down"), {"node-b": json.dumps(["out/torch_profiler"])}]
            ),
            patch.object(job, "_download_archive") as download,
        ):
            traces = job.collect_traces()
        self.assertEqual(traces.name, "combined_traces")
        self.assertEqual(download.call_args.args[3], traces / "node_1")
        self.assertIn("node-a", job.collection_errors)
        self.assertEqual(job.trace_trees, {"node-b": ["out/torch_profiler"]})

    def test_collect_disabled_fetches_latest_head_tree(self):
        self.raw["multi_node"] = {"collect_traces": False}
        job = self.job()
        job.started = job.prepared = True
        job.trace_floors = {"node-a": 1500}
        with (
            patch.object(job, "_exec", return_value={"node-a": '["out/torch_profiler"]'}) as execute,
            patch.object(job, "_download_archive"),
        ):
            traces = job.collect_traces()
        self.assertTrue(str(traces).endswith("traces/out/torch_profiler"))
        self.assertIn("--latest", execute.call_args.args[0])
        self.assertEqual(execute.call_count, 1)

    def test_optional_analysis_failure_is_not_a_benchmark_failure(self):
        self.raw["analysis"]["enable_tracelens"] = True
        job = self.job()
        job.status = "completed"
        job.trace_trees = {"node-a": ["out/torch_profiler"]}
        with patch.object(job, "_exec", side_effect=RuntimeError("TraceLens missing")) as execute:
            job.run_analysis()
        self.assertTrue(job.succeeded)
        self.assertIn("import TraceLens", execute.call_args.args[0])
        self.assertEqual(execute.call_args.kwargs["hosts"], ["node-a"])

    def test_collect_logs_uses_transport_and_preserves_training_artifact(self):
        job = self.job()
        job.phase_logs = {"benchmark": [str(path / "benchmark.log") for path in job.work_dirs]}
        self.orch.download_file.side_effect = [
            {"node-a": str(Path(self.tmp.name) / "node-a.log")},
            OSError("node-b unavailable"),
        ]
        job.collect_logs()
        self.assertEqual(job.get_artifact("training_log"), Path(self.tmp.name) / "node-a.log")
        self.assertEqual(self.orch.download_file.call_count, 2)
        self.assertIsNone(job.get_artifact("benchmark_log_node_1"))

    def test_collect_logs_tolerates_pruned_unreachable_host(self):
        job = self.job()
        job.phase_logs = {"benchmark": [str(path / "benchmark.log") for path in job.work_dirs]}
        self.orch.download_file.side_effect = ValueError("unreachable host")
        job.collect_logs()
        self.assertIsNone(job.get_artifact("training_log"))

    def test_analysis_uses_original_head_tree_and_configured_gemm_script(self):
        self.raw["analysis"] = {
            "enable_tracelens": False,
            "enable_gemm_analysis": True,
            "gemm_script": "custom/gemm.sh",
            "skip_if_exists": True,
        }
        job = self.job()
        job.trace_trees = {"node-a": ["out/torch_profiler"]}
        job.trace_floors = {"node-a": 1500}
        with patch.object(job, "_exec", return_value={"node-a": ""}) as execute, patch.object(job, "_download_archive"):
            job.run_analysis()
        command = execute.call_args_list[0].args[0]
        self.assertIn("test -d /mnt/out/tracelens_analysis", command)
        self.assertIn("bash /mnt/custom/gemm.sh /mnt/out", command)
        self.assertNotIn("combined_traces", command)
        archive_command = execute.call_args_list[1].args[0]
        self.assertIn("--min-mtime 0", archive_command)

    def test_kernel_scan_is_bounded_and_checks_other_nodes_after_failure(self):
        job = self.job()
        start = {host: "Wed Sep 23 12:00:00\n" for host in self.orch.hosts}
        with patch.object(job, "_exec", return_value=start) as execute:
            job.record_kernel_start()
        execute.assert_called_once_with("date +'%a %b %e %H:%M:%S'", on_host=True)
        ends = {
            "node-a": "Wed Sep 23 12:01:00",
            "node-b": "Wed Sep 23 12:02:00",
        }
        outputs = {"node-a": "GPU reset begin", "node-b": "ordinary kernel event"}
        with (
            patch.dict(os.environ, {"CVS_DMESG_PARSER": "legacy"}),
            patch.object(job, "_exec", return_value=ends) as execute,
        ):
            self.orch.exec_on_host.return_value = outputs
            job.check_kernel_errors()
        self.assertEqual(execute.call_args_list[0].args[0], "date +'%a %b %e %H:%M:%S'")
        self.assertEqual(execute.call_args_list[0].kwargs, {"on_host": True})
        self.assertEqual(len(globals.error_list), 1)
        self.assertIn("GPU reset begin", globals.error_list[0])
        command = self.orch.exec_on_host.call_args.args[0]
        self.assertIn("Wed Sep 23 12:00", command)
        self.assertIn("Wed Sep 23 12:01", command)
        self.assertIn("awk", command)
        self.assertEqual(
            self.orch.exec_on_host.call_args.kwargs,
            {"hosts": self.orch.hosts, "timeout": 120, "print_console": False},
        )

    def test_kernel_scan_reports_host_command_failures(self):
        job = self.job()
        job.kernel_start = {host: "Wed Sep 23 12:00:00" for host in self.orch.hosts}
        ends = {host: "Wed Sep 23 12:01:00" for host in self.orch.hosts}
        with (
            patch.dict(os.environ, {"CVS_DMESG_PARSER": "legacy"}),
            patch.object(job, "_exec", return_value=ends),
        ):
            self.orch.exec_on_host.side_effect = RuntimeError("node-b unreachable")
            with self.assertRaisesRegex(RuntimeError, "node-b unreachable"):
                job.check_kernel_errors()

    def test_teardown_reports_stop_failure_after_ownership_restore(self):
        job = self.job()
        job.owners = {host: "1000:1000" for host in self.orch.hosts}
        job.pid_paths = ["/tmp/aorta0.pid", "/tmp/aorta1.pid"]
        with (
            patch.object(job, "_exec_list", side_effect=[RuntimeError("node down"), {}]) as execute,
            patch.object(job, "collect_logs"),
        ):
            with self.assertRaisesRegex(RuntimeError, "node down"):
                job.teardown()
        self.assertEqual(execute.call_count, 2)
        self.assertIn('kill -TERM -- -"$pid"', execute.call_args_list[0].args[0][0])
        self.assertIn("chown -R 1000:1000 /mnt", execute.call_args_list[-1].args[0][0])
        self.orch.teardown_containers.assert_not_called()

    def test_teardown_raises_when_ownership_restore_fails(self):
        job = self.job()
        job.owners = {host: "1000:1000" for host in self.orch.hosts}
        with (
            patch.object(job, "stop_processes"),
            patch.object(job, "collect_logs"),
            patch.object(job, "_exec_list", side_effect=RuntimeError("chown failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "chown failed"):
                job.teardown()
