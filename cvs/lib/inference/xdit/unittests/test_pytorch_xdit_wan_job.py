"""Unit tests for pytorch_xdit_wan_job."""

import unittest
from unittest.mock import MagicMock, patch

from cvs.lib.inference.xdit.pytorch_xdit_flux_job import (
    build_nccl_env,
    log_benchmark_failure_excerpt,
)
from cvs.lib.inference.xdit.pytorch_xdit_wan_job import (
    RUN_WAN_DIFFUSERS_PATH,
    RUN_WAN_NATIVE_PATH,
    WAN_DIFFUSERS_LAUNCHER_XFUSER,
    WAN_DISTRIBUTED_BENCHMARK_TIMEOUT_S,
    WAN_XFUSER_AUTO_INPUT_IMAGE,
    WAN_XFUSER_BENCHMARK_OUTPUT_DIR,
    WAN_XFUSER_EXAMPLE_CONTAINER_PATH,
    WAN_XFUSER_PYPACKAGES_ENV,
    WAN_XFUSER_TIMING_JSON_CONTAINER_PATH,
    WAN_XFUSER_VIDEO_CONTAINER_PATH,
    WAN_MODEL_FORMAT_DIFFUSERS,
    WanBenchmarkJob,
    build_run_wan_diffusers_args,
    build_run_wan_native_args,
    build_run_wan_xfuser_example_args,
    build_torchrun_cmd,
    build_wan_xfuser_auto_input_image_cmd,
    build_wan_distributed_container_cleanup_cmds,
    build_wan_output_verify_cmd,
    build_wan_xfuser_output_verify_cmd,
    detect_wan_model_format_from_model_index,
    parallel_product,
    parse_wan_size,
    resolve_host_path_for_container_mount,
    resolve_wan_diffusers_launcher,
    resolve_wan_benchmark_timeout,
    resolve_wan_model_format,
    resolve_wan_xfuser_pypackages_env,
    should_wan_xfuser_auto_generate_input_image,
    scan_wan_fatal_output,
    scan_wan_xfuser_benchmark_output,
    summarize_wan_benchmark_log,
    validate_parallelism,
    validate_wan_parallelism_config,
    validate_wan_xfuser_mounts,
)

_WAN_BENCHMARK_PARAMS = {
    "prompt": "test prompt",
    "size": "720*1280",
    "frame_num": 81,
    "num_benchmark_steps": 5,
    "compile": True,
    "torchrun_nproc": 8,
    "ulysses_size": 8,
    "ring_size": 1,
}


def _wan_benchmark_params(**overrides):
    params = dict(_WAN_BENCHMARK_PARAMS)
    params.update(overrides)
    return {"wan22_i2v_a14b": params}


def _wan_inference_dict(**overrides):
    base = {
        "container_image": "amdsiloai/pytorch-xdit:v25.11.2",
        "container_name": "wan22-benchmark",
        "hf_home": "/home/user/.cache/huggingface",
        "output_base_dir": "/home/user/cvs_outputs",
        "model_repo": "Wan-AI/Wan2.2-I2V-A14B",
        "model_rev": "206a9ee1b7bfaaf8f7e4d81335650533490646a3",
        "container_config": {
            "device_list": ["/dev/dri", "/dev/kfd"],
            "volume_dict": {},
            "env_dict": {"CUSTOM": "1"},
        },
    }
    base.update(overrides)
    return base


def _wire_phdl_exec(phdl, hosts, hostnames=None, *, benchmark_exit_code=0, benchmark_output=None):
    hostnames = hostnames or {host: f"host-{idx}" for idx, host in enumerate(hosts)}
    benchmark_output = benchmark_output or "Initialized process group\nstep 0: epoch time: 12.34 sec"

    def _exec_side(cmd, timeout=None, print_console=False, detailed=False):
        text = str(cmd)
        out = {}
        for host in hosts:
            if "test -e /dev/kfd" in text:
                value = "KFD_OK"
            elif text.strip() == "hostname":
                value = hostnames[host]
            elif "hostname -I" in text:
                value = host
            elif "WAN_MOUNT_OK" in text:
                value = "WAN_MOUNT_OK"
            elif "docker run" in text:
                value = benchmark_output
                if detailed:
                    out[host] = {"output": value, "exit_code": benchmark_exit_code}
                    continue
            else:
                value = ""
            if detailed:
                out[host] = {"output": value, "exit_code": 0}
            else:
                out[host] = value
        return out

    phdl.exec.side_effect = _exec_side


def _make_wan_job(
    hosts=None,
    *,
    distributed=False,
    cluster_dict=None,
    benchmark_overrides=None,
    inference_overrides=None,
):
    hosts = hosts or ["10.0.0.1"]
    phdl = MagicMock()
    phdl.host_list = list(hosts)
    _wire_phdl_exec(phdl, hosts)
    phdl.exec_cmd_list = MagicMock(
        return_value={host: "Initialized process group\nstep 0: epoch time: 12.34 sec" for host in hosts}
    )

    benchmark_params = _wan_benchmark_params(**(benchmark_overrides or {}))
    inference_dict = _wan_inference_dict(**(inference_overrides or {}))
    job = WanBenchmarkJob(
        phdl,
        inference_dict,
        benchmark_params,
        hf_token="hf_test_token",
        distributed=distributed,
        cluster_dict=cluster_dict,
    )
    return job, phdl


class TestWanParallelism(unittest.TestCase):
    def test_resolve_wan_benchmark_timeout(self):
        from cvs.lib.inference.xdit.pytorch_xdit_flux_job import DEFAULT_BENCHMARK_TIMEOUT_S

        self.assertEqual(
            resolve_wan_benchmark_timeout(distributed=True),
            WAN_DISTRIBUTED_BENCHMARK_TIMEOUT_S,
        )
        self.assertEqual(
            resolve_wan_benchmark_timeout(distributed=False),
            DEFAULT_BENCHMARK_TIMEOUT_S,
        )
        self.assertEqual(resolve_wan_benchmark_timeout(distributed=True, explicit_timeout=999), 999)

    def test_build_wan_distributed_container_cleanup_cmds(self):
        cmds = build_wan_distributed_container_cleanup_cmds("wan22-benchmark-dist", 2)
        self.assertEqual(len(cmds), 2)
        self.assertIn("wan22-benchmark-dist-rank0", cmds[0])
        self.assertIn("wan22-benchmark-dist-rank1", cmds[1])

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
        self.assertIn("--output_directory /outputs", args)
        self.assertIn(f"--save_video_path {WAN_XFUSER_VIDEO_CONTAINER_PATH}", args)
        self.assertIn("--timing_json_path results/timing.json", args)
        self.assertIn("--output_type pil", args)
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
                "wan_xfuser_auto_input_image": True,
            },
            ckpt_dir="/model",
            distributed=False,
            model_repo_hints=["Wan-AI/Wan2.2-I2V-A14B-Diffusers"],
        )
        self.assertIn(WAN_XFUSER_EXAMPLE_CONTAINER_PATH, cmd)
        self.assertIn(f"--input_image {WAN_XFUSER_AUTO_INPUT_IMAGE}", cmd)
        self.assertIn("mkdir -p /outputs/results", cmd)
        self.assertIn("imageio", cmd)
        self.assertIn("i2v_input.jpg", cmd)
        self.assertNotIn(RUN_WAN_DIFFUSERS_PATH, cmd)

    def test_auto_input_image_cmd_uses_config_size(self):
        cmd = build_wan_xfuser_auto_input_image_cmd({"size": "720*1280"})
        self.assertIn("1280", cmd)
        self.assertIn("720", cmd)
        self.assertIn(WAN_XFUSER_AUTO_INPUT_IMAGE, cmd)

    def test_should_auto_generate_without_host_mount(self):
        self.assertTrue(
            should_wan_xfuser_auto_generate_input_image(
                {
                    "wan_diffusers_launcher": WAN_DIFFUSERS_LAUNCHER_XFUSER,
                    "wan_xfuser_auto_input_image": True,
                },
                {"container_config": {"volume_dict": {}}},
            )
        )


class TestWanOutputVerification(unittest.TestCase):
    def test_scan_xfuser_success_output(self):
        self.assertTrue(scan_wan_xfuser_benchmark_output("epoch time: 12.34 sec, memory: 40.00 GB"))
        self.assertFalse(scan_wan_xfuser_benchmark_output("Traceback (most recent call last):"))

    def test_scan_fatal_output_catches_docker_mount_errors(self):
        output = "docker: Error response from daemon: invalid mount config for type bind"
        self.assertTrue(scan_wan_fatal_output(output))

    def test_scan_fatal_output_catches_missing_script(self):
        self.assertTrue(scan_wan_fatal_output("python: can't open file '/benchmark/wan_i2v_example.py'"))

    def test_scan_fatal_output_ignores_torchao_module_not_found(self):
        output = (
            "torchao Float8Tensor FSDP2 patches skipped (ModuleNotFoundError): "
            "No module named 'torchao'\n"
            "100%|██████████| 40/40 [02:08<00:00,  3.21s/it]\n"
            "step 0: epoch time: 134.99 sec, memory: 94.91 GB\n"
        )
        self.assertFalse(scan_wan_fatal_output(output))

    def test_scan_fatal_output_still_catches_real_module_not_found(self):
        output = (
            "Traceback (most recent call last):\n"
            "  File \"/benchmark/wan_i2v_example.py\", line 191, in main\n"
            "ModuleNotFoundError: No module named 'xfuser'\n"
        )
        self.assertTrue(scan_wan_fatal_output(output))

    def test_build_wan_output_verify_cmd(self):
        cmd = build_wan_output_verify_cmd("/tmp/wan_22_host_outputs")
        self.assertIn("rank0_step*.json", cmd)
        self.assertIn("WAN_OUTPUT_OK", cmd)

    def test_build_wan_xfuser_output_verify_cmd(self):
        cmd = build_wan_xfuser_output_verify_cmd("/tmp/wan_22_host_outputs")
        self.assertIn("results/timing.json", cmd)
        self.assertIn("WAN_OUTPUT_OK", cmd)

    def test_xfuser_benchmark_output_dir_constant(self):
        self.assertEqual(WAN_XFUSER_BENCHMARK_OUTPUT_DIR, "/outputs")
        self.assertEqual(WAN_XFUSER_VIDEO_CONTAINER_PATH, "/outputs/results/video_i2v.mp4")
        self.assertEqual(WAN_XFUSER_TIMING_JSON_CONTAINER_PATH, "/outputs/results/timing.json")

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

    def test_validate_xfuser_mounts_allows_auto_input(self):
        class FakePssh:
            host_list = ["10.0.0.1"]

            def exec_cmd_list(self, commands, timeout=None, print_console=False):
                return {"10.0.0.1": "WAN_MOUNT_OK"}

        inference_dict = {
            "container_config": {
                "volume_dict": {
                    "/host/wan_i2v_example.py": "/benchmark/wan_i2v_example.py",
                }
            }
        }
        wan_params = {
            "wan_diffusers_launcher": WAN_DIFFUSERS_LAUNCHER_XFUSER,
            "wan_xfuser_auto_input_image": True,
        }
        errors = validate_wan_xfuser_mounts(FakePssh(), ["10.0.0.1"], inference_dict, wan_params)
        self.assertEqual(errors, [])


class TestBuildNcclEnv(unittest.TestCase):
    def test_defaults_include_nccl_proto_simple(self):
        env = build_nccl_env({})
        self.assertEqual(env["NCCL_PROTO"], "Simple")
        self.assertEqual(env["HSA_FORCE_FINE_GRAIN_PCIE"], "1")

    def test_maps_inference_dict_nccl_fields(self):
        env = build_nccl_env(
            {
                "nccl_ib_hca": "mlx5_0",
                "nccl_socket_ifname": "eno0",
                "gloo_socket_ifname": "eno0",
                "nccl_debug": "INFO",
                "nccl_ib_gid_index": 3,
            }
        )
        self.assertEqual(env["NCCL_IB_HCA"], "mlx5_0")
        self.assertEqual(env["NCCL_SOCKET_IFNAME"], "eno0")
        self.assertEqual(env["GLOO_SOCKET_IFNAME"], "eno0")
        self.assertEqual(env["NCCL_DEBUG"], "INFO")
        self.assertEqual(env["NCCL_IB_GID_INDEX"], "3")


class TestWanBenchmarkJob(unittest.TestCase):
    def test_validate_parallelism_single_node_skips_check(self):
        job, _ = _make_wan_job()
        self.assertIsNone(job.validate_parallelism())

    def test_validate_parallelism_distributed_fail(self):
        cluster_dict = {"node_dict": {"10.0.0.1": {}, "10.0.0.2": {}}}
        job, _ = _make_wan_job(
            hosts=["10.0.0.1", "10.0.0.2"],
            distributed=True,
            cluster_dict=cluster_dict,
            inference_overrides={"nnodes": 2, "master_addr": "10.0.0.1"},
        )
        err = job.validate_parallelism()
        self.assertIsNotNone(err)
        self.assertIn("Parallel degree product", err)

    def test_validate_parallelism_distributed_pass(self):
        cluster_dict = {"node_dict": {"10.0.0.1": {}, "10.0.0.2": {}}}
        job, _ = _make_wan_job(
            hosts=["10.0.0.1", "10.0.0.2"],
            distributed=True,
            cluster_dict=cluster_dict,
            inference_overrides={"nnodes": 2, "master_addr": "10.0.0.1"},
            benchmark_overrides={"ring_size": 2},
        )
        self.assertIsNone(job.validate_parallelism())

    @patch.dict("os.environ", {WAN_XFUSER_PYPACKAGES_ENV: ""})
    def test_build_env_args_single_node_omits_nccl(self):
        job, _ = _make_wan_job()
        env_args = job._build_env_args()
        self.assertIn("-e CUSTOM=1", env_args)
        self.assertIn("-e HF_TOKEN=hf_test_token", env_args)
        self.assertIn("-e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7", env_args)
        self.assertNotIn("NCCL_PROTO", env_args)
        self.assertNotIn(WAN_XFUSER_PYPACKAGES_ENV, env_args)

    @patch.dict("os.environ", {WAN_XFUSER_PYPACKAGES_ENV: ""})
    def test_build_env_args_distributed_includes_nccl(self):
        cluster_dict = {"node_dict": {"10.0.0.1": {}, "10.0.0.2": {}}}
        job, _ = _make_wan_job(
            hosts=["10.0.0.1", "10.0.0.2"],
            distributed=True,
            cluster_dict=cluster_dict,
            inference_overrides={
                "nnodes": 2,
                "master_addr": "10.0.0.1",
                "nccl_ib_hca": "mlx5_0",
            },
            benchmark_overrides={"ring_size": 2},
        )
        env_args = job._build_env_args()
        self.assertIn("-e NCCL_PROTO=Simple", env_args)
        self.assertIn("-e NCCL_IB_HCA=mlx5_0", env_args)

    def test_resolve_wan_xfuser_pypackages_prefers_config(self):
        self.assertEqual(
            resolve_wan_xfuser_pypackages_env(
                {WAN_XFUSER_PYPACKAGES_ENV: "/from/config/pypackages"},
                environ={WAN_XFUSER_PYPACKAGES_ENV: "/from/host/pypackages"},
            ),
            "/from/config/pypackages",
        )

    def test_resolve_wan_xfuser_pypackages_uses_process_env(self):
        self.assertEqual(
            resolve_wan_xfuser_pypackages_env(
                {},
                environ={WAN_XFUSER_PYPACKAGES_ENV: "/from/host/pypackages"},
            ),
            "/from/host/pypackages",
        )
        self.assertIsNone(resolve_wan_xfuser_pypackages_env({}, environ={}))

    @patch.dict("os.environ", {WAN_XFUSER_PYPACKAGES_ENV: "/from/host/pypackages"})
    def test_build_env_args_forwards_host_pypackages(self):
        job, _ = _make_wan_job()
        env_args = job._build_env_args()
        self.assertIn(f"-e {WAN_XFUSER_PYPACKAGES_ENV}=/from/host/pypackages", env_args)

    def test_build_docker_cmd_contains_torchrun_and_native_script(self):
        job, _ = _make_wan_job()
        cmd = job._build_docker_cmd(
            node_rank=0,
            host_output_dir="/home/user/cvs_outputs/wan_22_host-0_outputs",
            master_addr="127.0.0.1",
            master_port=29500,
        )
        self.assertIn("docker run", cmd)
        self.assertIn(RUN_WAN_NATIVE_PATH, cmd)
        self.assertIn("--name wan22-benchmark", cmd)

    def test_build_docker_cmd_distributed_uses_ranked_container_name(self):
        cluster_dict = {"node_dict": {"10.0.0.1": {}, "10.0.0.2": {}}}
        job, _ = _make_wan_job(
            hosts=["10.0.0.1", "10.0.0.2"],
            distributed=True,
            cluster_dict=cluster_dict,
            inference_overrides={"nnodes": 2, "master_addr": "10.0.0.1"},
            benchmark_overrides={"ring_size": 2},
        )
        cmd = job._build_docker_cmd(
            node_rank=1,
            host_output_dir="/home/user/cvs_outputs/wan_22_host-0_outputs",
            master_addr="10.0.0.1",
            master_port=29500,
        )
        self.assertIn("--name wan22-benchmark-rank1", cmd)
        self.assertIn("--node_rank=1", cmd)

    def test_build_launch_plan_single_node(self):
        job, _ = _make_wan_job(hosts=["10.0.0.1"])
        plan = job.build_launch_plan()
        self.assertFalse(plan.distributed)
        self.assertEqual(plan.node_order, ["10.0.0.1"])
        self.assertEqual(len(plan.docker_cmds), 1)
        self.assertEqual(len(plan.mkdir_cmds), 1)
        self.assertIn("/outputs", plan.mkdir_cmds[0])
        self.assertIn("/home/user/cvs_outputs/wan_22_host-0_outputs", plan.primary_output_dir)
        self.assertEqual(plan.world_size, 8)

    def test_build_launch_plan_distributed(self):
        cluster_dict = {"node_dict": {"10.0.0.1": {}, "10.0.0.2": {}}}
        job, _ = _make_wan_job(
            hosts=["10.0.0.1", "10.0.0.2"],
            distributed=True,
            cluster_dict=cluster_dict,
            inference_overrides={"nnodes": 2, "master_addr": "10.0.0.1"},
            benchmark_overrides={"ring_size": 2},
        )
        plan = job.build_launch_plan()
        self.assertTrue(plan.distributed)
        self.assertEqual(plan.node_order, ["10.0.0.1", "10.0.0.2"])
        self.assertEqual(len(plan.docker_cmds), 2)
        self.assertEqual(plan.world_size, 16)
        self.assertIn("wan_22_host-0_outputs", plan.primary_output_dir)

    def test_run_fails_on_parallelism_mismatch(self):
        cluster_dict = {"node_dict": {"10.0.0.1": {}, "10.0.0.2": {}}}
        job, phdl = _make_wan_job(
            hosts=["10.0.0.1", "10.0.0.2"],
            distributed=True,
            cluster_dict=cluster_dict,
            inference_overrides={"nnodes": 2, "master_addr": "10.0.0.1"},
        )
        results, plan, errors = job.run()
        self.assertEqual(results, {})
        self.assertEqual(plan.docker_cmds, [])
        self.assertTrue(errors)
        phdl.exec_cmd_list.assert_not_called()

    def test_run_fails_on_missing_kfd(self):
        job, phdl = _make_wan_job()

        def _missing_kfd(cmd, timeout=None, print_console=False, detailed=False):
            if "test -e /dev/kfd" in str(cmd):
                return {"10.0.0.1": "KFD_MISSING"}
            return {"10.0.0.1": "host-0"}

        phdl.exec.side_effect = _missing_kfd
        results, plan, errors = job.run()
        self.assertEqual(results, {})
        self.assertTrue(any("/dev/kfd" in err for err in errors))
        phdl.exec_cmd_list.assert_not_called()

    def test_run_success_single_node(self):
        job, phdl = _make_wan_job()
        results, plan, errors = job.run()
        self.assertEqual(errors, [])
        self.assertEqual(set(results.keys()), {"10.0.0.1"})
        self.assertEqual(len(plan.docker_cmds), 1)
        self.assertEqual(phdl.exec_cmd_list.call_count, 1)
        phdl.exec.assert_called()
        benchmark_call = phdl.exec.call_args_list[-1]
        self.assertTrue(benchmark_call.kwargs.get("detailed"))
        self.assertEqual(
            benchmark_call.kwargs.get("timeout"),
            resolve_wan_benchmark_timeout(distributed=False),
        )

    def test_run_fails_on_nonzero_exit_without_traceback(self):
        job, phdl = _make_wan_job()
        _wire_phdl_exec(phdl, ["10.0.0.1"], benchmark_exit_code=1, benchmark_output="HIP error: invalid device")
        results, plan, errors = job.run()
        self.assertEqual(set(results.keys()), {"10.0.0.1"})
        self.assertTrue(errors)
        self.assertTrue(any("10.0.0.1" in err for err in errors))
        self.assertNotIn("Traceback", results["10.0.0.1"])


class TestLogBenchmarkFailureExcerpt(unittest.TestCase):
    @patch("cvs.lib.inference.xdit.pytorch_xdit_flux_job.log")
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
