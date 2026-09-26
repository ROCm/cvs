import unittest
from unittest.mock import MagicMock

from cvs.lib.inference.xdit.xdit_benchmark_job import (
    BenchmarkLaunchPlan,
    PytorchXditBenchmarkJob,
)


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


if __name__ == "__main__":
    unittest.main()
