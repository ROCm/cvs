"""Unit tests for pytorch_xdit_wan_job."""

import unittest
from unittest.mock import patch

from cvs.lib.inference.pytorch_xdit.pytorch_xdit_flux_job import log_benchmark_failure_excerpt
from cvs.lib.inference.pytorch_xdit.pytorch_xdit_wan_job import (
    build_torchrun_cmd,
    parallel_product,
    validate_parallelism,
    validate_wan_parallelism_config,
)


class TestWanParallelism(unittest.TestCase):
    def test_parallel_product(self):
        params = {"ulysses_size": 8, "ring_size": 2, "torchrun_nproc": 8}
        self.assertEqual(parallel_product(params), 16)

    def test_validate_parallelism_pass(self):
        params = {"ulysses_size": 8, "ring_size": 2, "torchrun_nproc": 8}
        world_size, product, err = validate_parallelism(2, params)
        self.assertIsNone(err)
        self.assertEqual(world_size, 16)
        self.assertEqual(product, 16)

    def test_validate_parallelism_fail(self):
        params = {"ulysses_size": 8, "ring_size": 1, "torchrun_nproc": 8}
        _, _, err = validate_parallelism(2, params)
        self.assertIsNotNone(err)
        self.assertIn("Parallel degree product", err)

    def test_validate_wan_parallelism_config_distributed(self):
        cluster_dict = {"node_dict": {"10.0.0.1": {}, "10.0.0.2": {}}}
        inference_dict = {"nnodes": 2}
        benchmark_params = {
            "wan22_i2v_a14b": {
                "ulysses_size": 8,
                "ring_size": 2,
                "torchrun_nproc": 8,
            }
        }
        self.assertIsNone(
            validate_wan_parallelism_config(
                inference_dict,
                benchmark_params,
                distributed=True,
                cluster_dict=cluster_dict,
            )
        )


class TestBuildTorchrunCmd(unittest.TestCase):
    def test_distributed_cmd_includes_rendezvous(self):
        params = {
            "prompt": "test prompt",
            "size": "720*1280",
            "frame_num": 81,
            "num_benchmark_steps": 5,
            "compile": False,
            "torchrun_nproc": 8,
            "ulysses_size": 8,
            "ring_size": 2,
        }
        cmd = build_torchrun_cmd(
            params,
            ckpt_dir="/model",
            distributed=True,
            node_rank=1,
            nnodes=2,
            master_addr="10.0.0.1",
            master_port=29500,
        )
        self.assertIn("--nnodes=2", cmd)
        self.assertIn("--node_rank=1", cmd)
        self.assertIn("--master_addr=10.0.0.1", cmd)
        self.assertIn("--ulysses_size 8", cmd)
        self.assertIn("--ring_size 2", cmd)
        self.assertIn("/app/Wan2.2/run.py", cmd)

    def test_single_node_cmd(self):
        params = {
            "prompt": "test prompt",
            "size": "720*1280",
            "frame_num": 81,
            "num_benchmark_steps": 5,
            "compile": True,
            "torchrun_nproc": 8,
            "ulysses_size": 8,
            "ring_size": 1,
        }
        cmd = build_torchrun_cmd(params, ckpt_dir="/model", distributed=False)
        self.assertIn("torchrun --nproc_per_node=8", cmd)
        self.assertNotIn("--nnodes=", cmd)
        self.assertIn("--compile", cmd)


class TestLogBenchmarkFailureExcerpt(unittest.TestCase):
    @patch("cvs.lib.inference.pytorch_xdit.pytorch_xdit_flux_job.log")
    def test_logs_tail_and_redacts_token(self, mock_log):
        output = "line1\nTraceback (most recent call last):\nHF_TOKEN=hf_secret\n"
        log_benchmark_failure_excerpt("10.0.0.1", output, max_lines=10)

        rendered = []
        for call in mock_log.error.call_args_list:
            args = call.args
            if len(args) == 1:
                rendered.append(str(args[0]))
            elif len(args) >= 2:
                rendered.append(str(args[0]) % args[1:])

        joined = "\n".join(rendered)
        self.assertIn("Benchmark failure excerpt (10.0.0.1", joined)
        self.assertIn("Traceback (most recent call last):", joined)
        self.assertIn("HF_TOKEN=<redacted>", joined)
        self.assertNotIn("hf_secret", joined)


if __name__ == "__main__":
    unittest.main()
