'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for InferenceXAtomJob.parse_results (stock ``results`` artifact -> client.*).
No hardware: a fake orch returns committed fixture text.
'''

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from cvs.lib.inference.inferencex_atom_orch import InferenceXAtomJob
from cvs.lib.inference.unittests.fake_orch import FakeOrch

_HERE = Path(__file__).parent
_FIXTURES = _HERE / "fixtures"
_ISL = 7168
_OSL = 1024
_TP = 8


def _fake_variant(*, driver="vllm"):
    params = SimpleNamespace(
        driver=driver,
        tensor_parallelism=str(_TP),
        port_no="8000",
        random_range_ratio="0.8",
        random_prefix_len="0",
        burstiness="1.0",
        seed="0",
        request_rate="inf",
        tokenizer_mode="auto",
        percentile_metrics="ttft,tpot,itl,e2el",
        metric_percentiles="99",
        base_url="http://0.0.0.0",
        dataset_name="random",
        backend="vllm",
        max_model_length="8192",
        bench_extra_args="",
        result_filename="results",
    )
    roles = SimpleNamespace(server=SimpleNamespace(serve_args={}, atom_args=[], env={}))
    paths = SimpleNamespace(log_dir="/LOGS", models_dir="/models")
    model = SimpleNamespace(id="openai/gpt-oss-120b")
    return SimpleNamespace(params=params, roles=roles, paths=paths, model=model)


class TestInferenceXAtomOrchParse(unittest.TestCase):
    def test_parse_results_maps_client_metrics(self):
        raw = json.loads((_FIXTURES / "vllm_results_sample.json").read_text())
        job = InferenceXAtomJob(
            orch=FakeOrch(exec_return={"node0": json.dumps(raw)}),
            variant=_fake_variant(driver="vllm"),
            hf_token="tok",
            isl=_ISL,
            osl=_OSL,
            concurrency=64,
            num_prompts=100,
        )
        out = job.parse_results()
        metrics = out["node0"]
        w = raw
        self.assertIn("client.output_throughput", metrics)
        self.assertIn("client.mean_ttft_ms", metrics)
        self.assertAlmostEqual(metrics["client.per_gpu_throughput"], w["total_token_throughput"] / _TP)
        self.assertAlmostEqual(metrics["client.output_tput_per_gpu"], w["output_throughput"] / _TP)
        self.assertEqual(metrics["client.p99_ttft_ms"], w["p99_ttft_ms"])

    def test_parse_results_w1_tail_metrics_from_widened_fixture(self):
        raw = json.loads((_FIXTURES / "vllm_results_widened.json").read_text())
        job = InferenceXAtomJob(
            orch=FakeOrch(exec_return={"node0": json.dumps(raw)}),
            variant=_fake_variant(driver="atom"),
            hf_token="tok",
            isl="1024",
            osl="1024",
            concurrency=128,
            num_prompts=100,
        )
        metrics = job.parse_results()["node0"]
        self.assertEqual(metrics["client.p95_tpot_ms"], raw["p95_tpot_ms"])
        self.assertEqual(metrics["client.p99_ttft_ms"], raw["p99_ttft_ms"])

    def test_parse_results_atom_json_suffix(self):
        raw = json.loads((_FIXTURES / "vllm_results_sample.json").read_text())
        job = InferenceXAtomJob(
            orch=FakeOrch(exec_return={"node0": json.dumps(raw)}),
            variant=_fake_variant(driver="atom"),
            hf_token="tok",
            isl="1024",
            osl="1024",
            concurrency=128,
            num_prompts=100,
        )
        self.assertTrue(job._result_artifact.endswith("/results.json"))
        out = job.parse_results()
        self.assertIn("client.output_throughput", out["node0"])

    def test_run_client_clears_stale_result_artifact(self):
        orch = FakeOrch()
        job = InferenceXAtomJob(
            orch=orch,
            variant=_fake_variant(driver="atom"),
            hf_token="tok",
            isl="1024",
            osl="1024",
            concurrency=128,
            num_prompts=1000,
        )
        job.run_client()
        rm_cmds = [c for c in orch.commands if c.startswith("rm -f ")]
        self.assertEqual(len(rm_cmds), 1)
        self.assertIn(job._result_artifact, rm_cmds[0])
        self.assertTrue(any("benchmark_serving" in c for c in orch.commands))

    def test_parse_results_empty_artifact_raises(self):
        job = InferenceXAtomJob(
            orch=FakeOrch(exec_return={"node0": ""}),
            variant=_fake_variant(driver="atom"),
            hf_token="tok",
            isl="1024",
            osl="1024",
            concurrency=128,
            num_prompts=100,
        )
        with self.assertRaisesRegex(RuntimeError, "empty/missing results artifact"):
            job.parse_results()

    def test_parse_results_invalid_json_raises(self):
        job = InferenceXAtomJob(
            orch=FakeOrch(exec_return={"node0": "not-json"}),
            variant=_fake_variant(driver="atom"),
            hf_token="tok",
            isl="1024",
            osl="1024",
            concurrency=128,
            num_prompts=100,
        )
        with self.assertRaisesRegex(RuntimeError, "unparseable results artifact"):
            job.parse_results()

    def test_merged_serve_args_promotes_gpu_memory_util(self):
        variant = _fake_variant(driver="vllm")
        variant.roles.server.env = {"CVS_GPU_MEMORY_UTIL": "0.92"}
        merged = InferenceXAtomJob._merged_serve_args(variant)
        self.assertEqual(merged["gpu-memory-utilization"], "0.92")

    def test_merged_serve_args_skips_promotion_when_flag_present(self):
        variant = _fake_variant(driver="vllm")
        variant.roles.server.serve_args = {"gpu-memory-utilization": "0.75"}
        variant.roles.server.env = {"CVS_GPU_MEMORY_UTIL": "0.92"}
        merged = InferenceXAtomJob._merged_serve_args(variant)
        self.assertEqual(merged["gpu-memory-utilization"], "0.75")

    def test_build_server_cmd_suppresses_gpu_memory_env_vars(self):
        orch = FakeOrch()
        variant = _fake_variant(driver="vllm")
        variant.roles.server.env = {
            "CVS_GPU_MEMORY_UTIL": "0.92",
            "VLLM_GPU_MEMORY_UTIL": "0.91",
            "VLLM_ENFORCE_EAGER": "1",
            "CUSTOM_FLAG": "on",
        }
        job = InferenceXAtomJob(
            orch=orch,
            variant=variant,
            hf_token="tok",
            isl="1024",
            osl="1024",
            concurrency=128,
            num_prompts=100,
        )
        job.build_server_cmd()
        env_cmd = orch.commands[0]
        self.assertNotIn("CVS_GPU_MEMORY_UTIL", env_cmd)
        self.assertNotIn("VLLM_GPU_MEMORY_UTIL", env_cmd)
        self.assertNotIn("VLLM_ENFORCE_EAGER", env_cmd)
        self.assertIn("CUSTOM_FLAG", env_cmd)

    def test_client_log_failures_traceback(self):
        job = InferenceXAtomJob(
            orch=FakeOrch(exec_return={"node0": "Traceback (most recent call last):\n  boom"}),
            variant=_fake_variant(driver="atom"),
            hf_token="tok",
            isl="1024",
            osl="1024",
            concurrency=128,
            num_prompts=100,
        )
        failed = job._client_log_failures()
        self.assertEqual(len(failed), 1)
        self.assertIn("node0", failed[0][0])

    def test_client_log_failures_launch_error(self):
        job = InferenceXAtomJob(
            orch=FakeOrch(exec_return={"node0": "error: argument --foo: invalid choice"}),
            variant=_fake_variant(driver="atom"),
            hf_token="tok",
            isl="1024",
            osl="1024",
            concurrency=128,
            num_prompts=100,
        )
        self.assertEqual(len(job._client_log_failures()), 1)

    def test_client_log_failures_failed_requests_over_cap(self):
        job = InferenceXAtomJob(
            orch=FakeOrch(exec_return={"node0": "Failed requests: 3\n"}),
            variant=_fake_variant(driver="atom"),
            hf_token="tok",
            isl="1024",
            osl="1024",
            concurrency=128,
            num_prompts=100,
        )
        job._bench_max_failed_requests = 0
        failed = job._client_log_failures()
        self.assertEqual(len(failed), 1)
        self.assertIn("Failed requests: 3", failed[0][1])

    def test_client_log_failures_failed_requests_within_cap_warns(self):
        job = InferenceXAtomJob(
            orch=FakeOrch(exec_return={"node0": "Failed requests: 1\n"}),
            variant=_fake_variant(driver="atom"),
            hf_token="tok",
            isl="1024",
            osl="1024",
            concurrency=128,
            num_prompts=100,
        )
        job._bench_max_failed_requests = 2
        failed = job._client_log_failures()
        self.assertEqual(failed, [])

    def test_early_failure_regexes(self):
        job = InferenceXAtomJob(
            orch=FakeOrch(),
            variant=_fake_variant(driver="atom"),
            hf_token="tok",
            isl="1024",
            osl="1024",
            concurrency=128,
            num_prompts=100,
        )
        self.assertTrue(job.FAILED_REQUESTS_RE.search("Failed requests: 2"))
        self.assertTrue(job.CLIENT_CRASH_RE.search("Traceback (most recent call last)"))
        self.assertTrue(job.CLIENT_LAUNCH_FAIL_RE.search("unrecognized arguments: --bad"))
        self.assertTrue(job.EARLY_FAILURE_RE.search("No such file or directory"))


if __name__ == "__main__":
    unittest.main()
