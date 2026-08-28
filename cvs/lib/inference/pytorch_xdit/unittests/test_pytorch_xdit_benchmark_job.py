import unittest
from unittest.mock import MagicMock

from cvs.lib.inference.pytorch_xdit.pytorch_xdit_benchmark_job import (
    BenchmarkLaunchPlan,
    PytorchXditBenchmarkJob,
)


class _StubBenchmarkJob(PytorchXditBenchmarkJob):
    def validate_parallelism(self):
        return None

    def _build_env_args(self) -> str:
        return "-e STUB=1"

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


def _make_job(hosts=None, *, distributed=False, cluster_dict=None):
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
    return _StubBenchmarkJob(
        phdl,
        inference_dict,
        nproc_per_node=8,
        distributed=distributed,
        cluster_dict=cluster_dict or {},
    )


class TestBenchmarkLaunchPlan(unittest.TestCase):
    def test_defaults(self):
        plan = BenchmarkLaunchPlan()
        self.assertEqual(plan.mkdir_cmds, [])
        self.assertEqual(plan.docker_cmds, [])
        self.assertFalse(plan.distributed)
        self.assertEqual(plan.world_size, 0)


class TestPytorchXditBenchmarkJob(unittest.TestCase):
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

    def test_store_output_dir_hint_single_node(self):
        job = _make_job(["10.0.0.1"])
        plan = job.build_launch_plan()
        job.store_output_dir_hint(plan)
        self.assertIn("_test_output_dir", job.inference_dict)

    def test_run_success(self):
        job = _make_job(["10.0.0.1"])
        results, plan, errors = job.run(timeout=60)
        self.assertEqual(errors, [])
        self.assertEqual(len(plan.docker_cmds), 1)
        self.assertIn("10.0.0.1", results)


if __name__ == "__main__":
    unittest.main()
