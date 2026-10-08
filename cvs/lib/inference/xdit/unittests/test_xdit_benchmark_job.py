import base64
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

from cvs.lib.inference.xdit.xdit_benchmark_job import (
    BenchmarkLaunchPlan,
    PytorchXditBenchmarkJob,
    _ARTIFACT_COLLECT_TIMEOUT_S,
    _REMOTE_COLLECT_SCRIPT,
    _materialize_artifacts,
    cleanup_staged_artifacts,
    stage_remote_benchmark_outputs,
)
from cvs.lib.inference.xdit.xdit_flux import FluxOutputParser
from cvs.lib.inference.xdit.xdit_wan import WanOutputParser
from cvs.lib.inference.xdit.xdit_wan_i2v import WanI2vOutputParser


class _StubBenchmarkJob(PytorchXditBenchmarkJob):
    def validate_parallelism(self):
        return None

    def _build_env_dict(self):
        return {"STUB": "1"}

    def _build_torchrun_cmd(self, *, node_rank, host_output_dir, master_addr, master_port) -> str:
        return f"torchrun --node_rank={node_rank} --master_addr={master_addr}"

    def _host_output_dir(self, output_base_dir: str, hostname: str) -> str:
        return f"{output_base_dir}/stub_{hostname}_outputs"

    def _benchmark_name(self) -> str:
        return "STUB"


def _wire_phdl(phdl, hosts, hostnames=None):
    hostnames = hostnames or {host: f"host-{idx}" for idx, host in enumerate(hosts)}

    def _exec_side(cmd, timeout=None, print_console=False, detailed=False):
        text = str(cmd)
        out = {}
        for host in hosts:
            if "test -e /dev/kfd" in text:
                value = "KFD_OK"
            elif text.strip() == "hostname":
                value = hostnames[host]
            elif "docker run" in text:
                value = "benchmark ok"
                if detailed:
                    out[host] = {"output": value, "exit_code": 0}
                    continue
            elif text.startswith("mkdir -p"):
                value = ""
            else:
                value = ""
            if detailed:
                out[host] = {"output": value, "exit_code": 0}
            else:
                out[host] = value
        return out

    phdl.exec.side_effect = _exec_side
    phdl.exec_cmd_list = MagicMock(return_value={host: "" for host in hosts})


def _make_job(hosts=None, *, distributed=False, cluster_dict=None, nnodes=None, job_cls=_StubBenchmarkJob):
    hosts = hosts or ["10.0.0.1"]
    phdl = MagicMock()
    phdl.host_list = list(hosts)
    _wire_phdl(phdl, hosts)

    inference_dict = {
        "container_image": "test/xdit:latest",
        "container_name": "stub-benchmark",
        "hf_home": "/home/user/.cache/huggingface",
        "output_base_dir": "/home/user/stub_output",
        "container_config": {
            "device_list": ["/dev/dri", "/dev/kfd"],
            "volume_dict": {},
            "env_dict": {},
        },
    }
    if nnodes is not None:
        inference_dict["nnodes"] = nnodes
    return job_cls(
        phdl,
        inference_dict,
        nproc_per_node=8,
        distributed=distributed,
        cluster_dict=cluster_dict or {},
    )


class _FakeContainerOrchestrator:
    def __init__(self, hosts):
        self.hosts = list(hosts)
        self.calls = []

    def exec_on_host(self, cmd, **kwargs):
        raise AssertionError("benchmark jobs must execute inside the container")

    def exec(self, cmd, hosts=None, timeout=None, print_console=False, detailed=False):
        selected = list(hosts or self.hosts)
        self.calls.append((cmd, selected, detailed))
        output = {}
        for host in selected:
            if "test -e /dev/kfd" in cmd:
                value = "KFD_OK"
            elif cmd.strip() == "hostname":
                value = f"container-{host}"
            elif "hostname -I" in cmd:
                value = host
            elif "torchrun" in cmd:
                value = "benchmark ok"
            else:
                value = ""
            output[host] = {"output": value, "exit_code": 0} if detailed else value
        return output


class TestBenchmarkLaunchPlan(unittest.TestCase):
    def test_defaults(self):
        plan = BenchmarkLaunchPlan()
        self.assertEqual(plan.mkdir_cmds, [])
        self.assertEqual(plan.docker_cmds, [])
        self.assertFalse(plan.distributed)
        self.assertEqual(plan.world_size, 0)


class _SpacedEnvJob(_StubBenchmarkJob):
    def _build_env_dict(self):
        return {"PROMPT": "a photo of a cat"}


class TestPytorchXditBenchmarkJob(unittest.TestCase):
    def test_build_env_dict_keeps_spaced_values(self):
        job = _make_job(["10.0.0.1"], job_cls=_SpacedEnvJob)
        job._build_env_args()
        self.assertEqual(job._build_env_dict()["PROMPT"], "a photo of a cat")

    def test_distributed_rejects_nnodes_less_than_2(self):
        cluster = {"node_dict": {"10.0.0.1": {}, "10.0.0.2": {}}}
        with self.assertRaisesRegex(ValueError, "nnodes >= 2"):
            _make_job(["10.0.0.1", "10.0.0.2"], distributed=True, cluster_dict=cluster, nnodes=1)

    def test_distributed_rejects_cluster_smaller_than_nnodes(self):
        cluster = {"node_dict": {"10.0.0.1": {}}}
        with self.assertRaisesRegex(ValueError, "requests 2 nodes"):
            _make_job(["10.0.0.1"], distributed=True, cluster_dict=cluster, nnodes=2)

    def test_check_kfd_all_present(self):
        job = _make_job(["10.0.0.1", "10.0.0.2"])
        self.assertEqual(job.check_kfd(), [])

    def test_build_launch_plan_single_node(self):
        job = _make_job(["10.0.0.1"])
        plan = job.build_launch_plan()
        self.assertEqual(len(plan.docker_cmds), 1)
        self.assertIn("stub_host-0_outputs", plan.output_dirs_by_node["10.0.0.1"])
        self.assertIn("docker run", plan.docker_cmds[0])
        self.assertIn("torchrun", plan.docker_cmds[0])

    def test_token_umask_stays_inside_subshell(self):
        job = _make_job(["10.0.0.1"])
        job.inference_dict["hf_token_file"] = "/home/user/.hf_token"
        cmd = job.build_launch_plan().docker_cmds[0]
        umask_at = cmd.index("(umask 077;")
        subshell_end = cmd.index(");", umask_at)
        torchrun_at = cmd.index("torchrun")
        self.assertLess(subshell_end, torchrun_at)
        self.assertIn("chmod 600", cmd[umask_at:subshell_end])

    def test_store_output_dir_hint_single_node(self):
        job = _make_job(["10.0.0.1"])
        plan = job.build_launch_plan()
        job.store_output_dir_hint(plan)
        self.assertIn("_test_output_dir", job.inference_dict)

    def test_store_output_dir_hint_maps_every_single_node(self):
        job = _make_job(["10.0.0.1", "10.0.0.2"])
        plan = job.build_launch_plan()

        job.store_output_dir_hint(plan)

        self.assertEqual(
            job.inference_dict["_test_output_dirs_by_node"],
            {
                "10.0.0.1": "/home/user/stub_output/stub_host-0_outputs",
                "10.0.0.2": "/home/user/stub_output/stub_host-1_outputs",
            },
        )
        self.assertNotIn("_test_output_dir", job.inference_dict)

    def test_store_output_dir_hint_translates_container_paths_per_node(self):
        orch = _FakeContainerOrchestrator(["10.0.0.1", "10.0.0.2"])
        inference_dict = {
            "container_image": "unused-after-external-setup",
            "container_name": "stub-benchmark",
            "hf_home": "/hf_home",
            "output_base_dir": "/host/results",
            "output_base_dir_container": "/outputs",
            "container_config": {"device_list": [], "volume_dict": {}, "env_dict": {}},
        }
        job = _StubBenchmarkJob(orch, inference_dict, nproc_per_node=8)
        plan = job.build_launch_plan()

        job.store_output_dir_hint(plan)

        self.assertEqual(
            inference_dict["_test_output_dirs_by_node"],
            {
                "10.0.0.1": "/host/results/stub_container-10.0.0.1_outputs",
                "10.0.0.2": "/host/results/stub_container-10.0.0.2_outputs",
            },
        )

    def test_run_success(self):
        job = _make_job(["10.0.0.1"])
        results, plan, errors = job.run(timeout=60)
        self.assertEqual(errors, [])
        self.assertEqual(len(plan.docker_cmds), 1)
        self.assertIn("10.0.0.1", results)

    def test_container_orchestrator_runs_torchrun_without_docker(self):
        orch = _FakeContainerOrchestrator(["10.0.0.1"])
        inference_dict = {
            "container_image": "unused-after-external-setup",
            "container_name": "stub-benchmark",
            "hf_home": "/hf_home",
            "output_base_dir": "/host/results",
            "output_base_dir_container": "/outputs",
            "container_config": {
                "device_list": [],
                "volume_dict": {},
                "env_dict": {},
            },
        }
        job = _StubBenchmarkJob(orch, inference_dict, nproc_per_node=8)

        results, plan, errors = job.run(timeout=60)

        self.assertEqual(errors, [])
        self.assertIn("10.0.0.1", results)
        self.assertNotIn("docker run", plan.docker_cmds[0])
        self.assertIn("torchrun", plan.docker_cmds[0])
        self.assertTrue(any(call[2] for call in orch.calls if "torchrun" in call[0]))
        job.store_output_dir_hint(plan)
        self.assertEqual(
            inference_dict["_test_output_dir"],
            "/host/results/stub_container-10.0.0.1_outputs",
        )


class _RecordingExec:
    def __init__(self, outputs):
        self.host_list = list(outputs)
        self.outputs = outputs
        self.commands = None

    def exec_cmd_list(self, commands, timeout=None, print_console=False):
        self.commands = list(commands)
        self.timeout = timeout
        return {host: self.outputs[host] for host in self.host_list}


def _collect_tree(root):
    saved_argv = sys.argv
    saved_stdout = sys.stdout
    buffer = io.StringIO()
    try:
        sys.argv = ["collect", root]
        sys.stdout = buffer
        exec(_REMOTE_COLLECT_SCRIPT, {"__name__": "__collect__"})
    finally:
        sys.argv = saved_argv
        sys.stdout = saved_stdout
    return buffer.getvalue()


class TestStageRemoteBenchmarkOutputs(unittest.TestCase):
    def tearDown(self):
        cleanup_staged_artifacts()

    def test_cleanup_staged_artifacts_removes_materialized_dirs(self):
        local = _materialize_artifacts([{"rel": "results/timing.json", "b64": base64.b64encode(b"[]").decode("ascii")}])
        self.assertTrue(os.path.isdir(local))
        untouched = tempfile.mkdtemp(prefix="xdit-results-keep-")
        try:
            cleanup_staged_artifacts()
            self.assertFalse(os.path.exists(local))
            self.assertTrue(os.path.isdir(untouched))
            cleanup_staged_artifacts()
            self.assertTrue(os.path.isdir(untouched))
        finally:
            shutil.rmtree(untouched, ignore_errors=True)

    def test_collect_script_reads_flux_and_wan_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            flux = os.path.join(tmp, "results")
            os.makedirs(flux)
            with open(os.path.join(flux, "timing.json"), "w", encoding="utf-8") as handle:
                handle.write(json.dumps([{"pipe_time": 0.97}]))
            with open(os.path.join(flux, "flux_0.png"), "wb") as handle:
                handle.write(b"png")
            with open(os.path.join(tmp, "noise.log"), "w", encoding="utf-8") as handle:
                handle.write("ignore")
            text = _collect_tree(tmp)

        staged = stage_remote_benchmark_outputs(_RecordingExec({"n0": text}), {"n0": "/node/flux_outputs"})
        result, errors = FluxOutputParser(staged["n0"]).parse()
        self.assertIsNotNone(result, errors)
        self.assertAlmostEqual(result.avg_pipe_time_s, 0.97)
        self.assertEqual(result.repetition_count, 1)
        self.assertTrue(result.image_paths)

    def test_each_node_keeps_its_own_timing(self):
        def payload(pipe_time):
            with tempfile.TemporaryDirectory() as tmp:
                results = os.path.join(tmp, "results")
                os.makedirs(results)
                with open(os.path.join(results, "timing.json"), "w", encoding="utf-8") as handle:
                    handle.write(json.dumps([{"pipe_time": pipe_time}, {"pipe_time": pipe_time}]))
                with open(os.path.join(results, "video_i2v.mp4"), "wb") as handle:
                    handle.write(b"mp4")
                return _collect_tree(tmp)

        executor = _RecordingExec({"10.0.0.1": payload(0.9674), "10.0.0.2": payload(0.9735)})
        staged = stage_remote_benchmark_outputs(
            executor,
            {
                "10.0.0.1": "/local/flux_host-a_outputs",
                "10.0.0.2": "/local/flux_host-b_outputs",
            },
        )
        first, _ = WanI2vOutputParser(staged["10.0.0.1"]).parse()
        second, _ = WanI2vOutputParser(staged["10.0.0.2"]).parse()
        self.assertAlmostEqual(first.avg_pipe_time_s, 0.9674)
        self.assertAlmostEqual(second.avg_pipe_time_s, 0.9735)
        self.assertNotEqual(staged["10.0.0.1"], staged["10.0.0.2"])
        self.assertTrue(all("/local/flux_" in cmd for cmd in executor.commands))

    def test_shared_directory_uses_the_node_that_has_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = os.path.join(tmp, "outputs")
            os.makedirs(outputs)
            with open(os.path.join(outputs, "rank0_step0.json"), "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"total_time": 12.5}))
            with open(os.path.join(outputs, "video.mp4"), "wb") as handle:
                handle.write(b"mp4")
            present = _collect_tree(tmp)
        with tempfile.TemporaryDirectory() as empty_dir:
            empty = _collect_tree(empty_dir)
        staged = stage_remote_benchmark_outputs(
            _RecordingExec({"rank0": empty, "rank1": present}),
            {"rank0": "/shared/wan_outputs", "rank1": "/shared/wan_outputs"},
        )
        self.assertEqual(staged["rank0"], staged["rank1"])
        result, errors = WanOutputParser(staged["rank1"]).parse()
        self.assertIsNotNone(result, errors)
        self.assertAlmostEqual(result.avg_total_time_s, 12.5)

    def test_missing_remote_files_leave_no_local_copy(self):
        executor = _RecordingExec({"n0": ""})
        staged = stage_remote_benchmark_outputs(executor, {"n0": "/node/only"})
        self.assertEqual(staged, {})
        self.assertEqual(executor.timeout, _ARTIFACT_COLLECT_TIMEOUT_S)

    def test_store_output_dir_hint_points_parser_at_staged_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = os.path.join(tmp, "results")
            os.makedirs(results)
            with open(os.path.join(results, "timing.json"), "w", encoding="utf-8") as handle:
                handle.write(json.dumps([{"pipe_time": 0.9674}] * 25))
            with open(os.path.join(results, "flux_0.png"), "wb") as handle:
                handle.write(b"png")
            payload = _collect_tree(tmp)
        job = _make_job(["10.0.0.1"])
        job.s_phdl.exec_cmd_list.return_value = {"10.0.0.1": payload}
        plan = job.build_launch_plan()
        job.store_output_dir_hint(plan)
        local = job.inference_dict["_test_output_dir"]
        self.assertNotIn("/home/user/stub_output", local)
        result, errors = FluxOutputParser(local, expected_repetitions=25).parse()
        self.assertIsNotNone(result, errors)
        self.assertEqual(result.repetition_count, 25)
        self.assertAlmostEqual(result.avg_pipe_time_s, 0.9674)


if __name__ == "__main__":
    unittest.main()
