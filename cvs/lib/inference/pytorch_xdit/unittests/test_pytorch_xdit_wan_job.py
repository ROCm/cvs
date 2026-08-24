"""Unit tests for pytorch_xdit_wan_job."""

import unittest
from unittest.mock import patch

from cvs.lib.inference.pytorch_xdit.pytorch_xdit_flux_job import (
    build_nccl_env,
    log_benchmark_failure_excerpt,
)
from cvs.lib.inference.pytorch_xdit.pytorch_xdit_wan_job import (
    RUN_WAN_DIFFUSERS_PATH,
    RUN_WAN_NATIVE_PATH,
    WAN_DIFFUSERS_LAUNCHER_XFUSER,
    WAN_XFUSER_BENCHMARK_OUTPUT_DIR,
    WAN_XFUSER_EXAMPLE_CONTAINER_PATH,
    WAN_MODEL_FORMAT_DIFFUSERS,
    build_run_wan_diffusers_args,
    build_run_wan_native_args,
    build_run_wan_xfuser_example_args,
    build_torchrun_cmd,
    build_wan_output_verify_cmd,
    detect_wan_model_format_from_model_index,
    parallel_product,
    parse_wan_size,
    resolve_host_path_for_container_mount,
    resolve_wan_diffusers_launcher,
    resolve_wan_model_format,
    scan_wan_fatal_output,
    scan_wan_xfuser_benchmark_output,
    summarize_wan_benchmark_log,
    validate_parallelism,
    validate_wan_parallelism_config,
    validate_wan_xfuser_mounts,
)


class TestWanParallelism(unittest.TestCase):
    def test_parallel_product(self):
        params = {"ulysses_size": 8, "ring_size": 2, "torchrun_nproc": 8}
        self.assertEqual(parallel_product(params), 16)

    def test_parallel_product_schema_defaults(self):
        params = {"torchrun_nproc": 8}
        self.assertEqual(parallel_product(params), 8)

    def test_validate_parallelism_single_node_skips_check(self):
        params = {"torchrun_nproc": 8}
        world_size, product, err = validate_parallelism(1, params)
        self.assertIsNone(err)
        self.assertEqual(world_size, 8)
        self.assertEqual(product, 8)

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


class TestWanModelRouting(unittest.TestCase):
    def test_model_index_diffusers(self):
        self.assertEqual(
            detect_wan_model_format_from_model_index({"_class_name": "WanImageToVideoPipeline"}),
            WAN_MODEL_FORMAT_DIFFUSERS,
        )

    def test_resolve_from_repo_name(self):
        self.assertEqual(
            resolve_wan_model_format(None, "Wan-AI/Wan2.2-I2V-A14B-Diffusers"),
            WAN_MODEL_FORMAT_DIFFUSERS,
        )
        self.assertEqual(
            resolve_wan_model_format(None, "Wan-AI/Wan2.2-I2V-A14B"),
            "native",
        )

    def test_parse_wan_size(self):
        self.assertEqual(parse_wan_size("720*1280"), (720, 1280))


class TestBuildRunWanArgs(unittest.TestCase):
    _BASE_PARAMS = {
        "prompt": "test prompt",
        "size": "720*1280",
        "frame_num": 81,
        "num_benchmark_steps": 5,
        "compile": True,
        "torchrun_nproc": 8,
        "ulysses_size": 8,
        "ring_size": 1,
    }

    def test_native_uses_ckpt_dir(self):
        args = build_run_wan_native_args(self._BASE_PARAMS, ckpt_dir="/model")
        self.assertIn("--ckpt_dir /model", args)
        self.assertIn("--task i2v-A14B", args)
        self.assertIn("--ulysses_size 8", args)
        self.assertIn("--compile", args)

    def test_diffusers_uses_model_and_torch_compile(self):
        args = build_run_wan_diffusers_args(self._BASE_PARAMS, model_path="/model")
        self.assertIn("--model /model", args)
        self.assertIn("--task i2v", args)
        self.assertIn("--height 720", args)
        self.assertIn("--width 1280", args)
        self.assertIn("--use_torch_compile", args)
        self.assertIn("--num_repetitions 5", args)

    def test_diffusers_optional_none_seed_uses_default(self):
        params = {**self._BASE_PARAMS, "seed": None, "num_inference_steps": None}
        args = build_run_wan_diffusers_args(params, model_path="/model")
        self.assertIn("--seed 42", args)
        self.assertIn("--num_inference_steps 40", args)

    def test_xfuser_example_uses_input_image(self):
        params = {**self._BASE_PARAMS, "compile": False}
        args = build_run_wan_xfuser_example_args(
            params,
            model_path="/model",
            i2v_image_path="/benchmark/i2v_input.JPG",
        )
        self.assertIn("--input_image /benchmark/i2v_input.JPG", args)
        self.assertIn("--num_repetitions 5", args)
        self.assertIn("--benchmark_output_directory /outputs/outputs", args)
        self.assertIn("--save_video_path /outputs/outputs/video.mp4", args)
        self.assertIn("--output_directory /outputs/outputs", args)
        self.assertIn("--output_type np", args)
        self.assertNotIn("--task i2v", args)

    def test_resolve_xfuser_launcher_from_config(self):
        self.assertEqual(
            resolve_wan_diffusers_launcher(
                {"wan_diffusers_launcher": WAN_DIFFUSERS_LAUNCHER_XFUSER},
            ),
            WAN_DIFFUSERS_LAUNCHER_XFUSER,
        )
        self.assertEqual(
            resolve_wan_diffusers_launcher(
                {"wan_diffusers_run_script": "/benchmark/wan_i2v_example.py"},
            ),
            WAN_DIFFUSERS_LAUNCHER_XFUSER,
        )


class TestBuildTorchrunCmd(unittest.TestCase):
    _BASE_PARAMS = {
        "prompt": "test prompt",
        "size": "720*1280",
        "frame_num": 81,
        "num_benchmark_steps": 5,
        "compile": False,
        "torchrun_nproc": 8,
        "ulysses_size": 8,
        "ring_size": 2,
    }

    def test_distributed_cmd_includes_rendezvous(self):
        cmd = build_torchrun_cmd(
            self._BASE_PARAMS,
            ckpt_dir="/model",
            distributed=True,
            node_rank=1,
            nnodes=2,
            master_addr="10.0.0.1",
            master_port=29500,
            model_repo_hints=["Wan-AI/Wan2.2-I2V-A14B"],
        )
        self.assertIn("--nnodes=2", cmd)
        self.assertIn("--node_rank=1", cmd)
        self.assertIn(RUN_WAN_NATIVE_PATH, cmd)

    def test_single_node_native(self):
        cmd = build_torchrun_cmd(
            {**self._BASE_PARAMS, "compile": True},
            ckpt_dir="/model",
            distributed=False,
            model_repo_hints=["Wan-AI/Wan2.2-I2V-A14B"],
        )
        self.assertIn("torchrun --nproc_per_node=8", cmd)
        self.assertIn(RUN_WAN_NATIVE_PATH, cmd)
        self.assertIn("--compile", cmd)

    def test_single_node_diffusers_xfuser(self):
        cmd = build_torchrun_cmd(
            {
                **self._BASE_PARAMS,
                "wan_diffusers_launcher": WAN_DIFFUSERS_LAUNCHER_XFUSER,
                "wan_diffusers_run_script": WAN_XFUSER_EXAMPLE_CONTAINER_PATH,
                "wan_diffusers_i2v_image": "/benchmark/i2v_input.JPG",
            },
            ckpt_dir="/model",
            distributed=False,
            model_repo_hints=["Wan-AI/Wan2.2-I2V-A14B-Diffusers"],
        )
        self.assertIn(WAN_XFUSER_EXAMPLE_CONTAINER_PATH, cmd)
        self.assertIn("--input_image /benchmark/i2v_input.JPG", cmd)
        self.assertIn("mkdir -p /outputs/outputs", cmd)
        self.assertNotIn(RUN_WAN_DIFFUSERS_PATH, cmd)


class TestWanOutputVerification(unittest.TestCase):
    def test_scan_xfuser_success_output(self):
        self.assertTrue(scan_wan_xfuser_benchmark_output("epoch time: 12.34 sec, memory: 40.00 GB"))
        self.assertFalse(scan_wan_xfuser_benchmark_output("Traceback (most recent call last):"))

    def test_scan_fatal_output_catches_docker_mount_errors(self):
        output = "docker: Error response from daemon: invalid mount config for type bind"
        self.assertTrue(scan_wan_fatal_output(output))

    def test_scan_fatal_output_catches_missing_script(self):
        self.assertTrue(
            scan_wan_fatal_output("python: can't open file '/benchmark/wan_i2v_example.py'")
        )

    def test_build_wan_output_verify_cmd(self):
        cmd = build_wan_output_verify_cmd("/tmp/wan_22_host_outputs")
        self.assertIn("rank0_step*.json", cmd)
        self.assertIn("WAN_OUTPUT_OK", cmd)

    def test_xfuser_benchmark_output_dir_constant(self):
        self.assertEqual(WAN_XFUSER_BENCHMARK_OUTPUT_DIR, "/outputs/outputs")

    def test_summarize_wan_benchmark_log(self):
        self.assertEqual(summarize_wan_benchmark_log(""), "docker benchmark log was empty")
        self.assertIn("line2", summarize_wan_benchmark_log("line1\nline2\nline3"))

    def test_resolve_host_path_for_container_mount(self):
        inference_dict = {
            "container_config": {
                "volume_dict": {
                    "/host/wan_i2v_example.py": "/benchmark/wan_i2v_example.py",
                }
            }
        }
        self.assertEqual(
            resolve_host_path_for_container_mount(inference_dict, "/benchmark/wan_i2v_example.py"),
            "/host/wan_i2v_example.py",
        )

    def test_validate_xfuser_mounts_rejects_placeholders(self):
        class FakePssh:
            host_list = ["10.0.0.1"]

        inference_dict = {
            "container_config": {
                "volume_dict": {
                    "/path/to/cvs/scripts/wan_i2v_example.py": "/benchmark/wan_i2v_example.py",
                    "/path/to/i2v_input.JPG": "/benchmark/i2v_input.JPG",
                }
            }
        }
        wan_params = {
            "wan_diffusers_launcher": WAN_DIFFUSERS_LAUNCHER_XFUSER,
            "wan_diffusers_run_script": "/benchmark/wan_i2v_example.py",
            "wan_diffusers_i2v_image": "/benchmark/i2v_input.JPG",
        }
        errors = validate_wan_xfuser_mounts(FakePssh(), ["10.0.0.1"], inference_dict, wan_params)
        self.assertTrue(any("placeholder" in err.lower() for err in errors))


class TestBuildNcclEnv(unittest.TestCase):
    def test_defaults_include_nccl_proto_simple(self):
        env = build_nccl_env({})
        self.assertEqual(env["NCCL_PROTO"], "Simple")
        self.assertEqual(env["HSA_FORCE_FINE_GRAIN_PCIE"], "1")


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
        self.assertNotIn("hf_secret", joined)


if __name__ == "__main__":
    unittest.main()
