import threading
import unittest
from unittest.mock import MagicMock, patch

from cvs.lib.inference.xdit.pytorch_xdit_flux_job import (
    FLUX2_DEFAULT_HF_REPO,
    FLUX2_EXAMPLE_MOUNT_PATH,
    FLUX2_EXAMPLE_PATH,
    RUN_USP_PATH,
    FluxBenchmarkJob,
    build_flux2_chat_template_host_check_cmd,
    build_flux2_ensure_chat_template_cmd,
    build_flux2_example_args,
    build_flux2_example_host_check_cmd,
    build_flux2_example_image_probe_cmd,
    build_nccl_env,
    build_run_usp_args,
    build_torchrun_cmd,
    default_flux2_example_host_path,
    detect_flux_model_type_from_model_index,
    ensure_flux2_example_available,
    is_flux2_model,
    parallel_product,
    resolve_flux2_example_host_mount,
    resolve_flux2_hf_repo_id,
    resolve_flux_model_type,
    validate_flux_parallelism_config,
    validate_parallelism,
    _exec_cmd_list_on_nodes,
    _exec_result_exit_code,
    _exec_result_output,
)

_FLUX_BENCHMARK_PARAMS = {
    "prompt": "A small cat",
    "seed": 42,
    "num_inference_steps": 25,
    "max_sequence_length": 256,
    "no_use_resolution_binning": True,
    "use_torch_compile": True,
    "warmup_steps": 1,
    "warmup_calls": 5,
    "num_repetitions": 25,
    "height": 1024,
    "width": 1024,
    "ulysses_degree": 8,
    "ring_degree": 1,
    "torchrun_nproc": 8,
}


def _flux_benchmark_params(**overrides):
    params = dict(_FLUX_BENCHMARK_PARAMS)
    params.update(overrides)
    return {"flux1_dev_t2i": params}


def _flux_inference_dict(**overrides):
    base = {
        "container_image": "amdsiloai/pytorch-xdit:v25.11.2",
        "container_name": "flux-benchmark",
        "hf_home": "/home/user/.cache/huggingface",
        "output_base_dir": "/home/user/cvs_flux_output",
        "model_repo": "black-forest-labs/FLUX.1-dev",
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
    benchmark_output = benchmark_output or "Initialized process group\nepoch time: 1.0 sec"

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
            elif "FLUX2_EXAMPLE_PRESENT" in text:
                value = "FLUX2_EXAMPLE_PRESENT"
            elif "FLUX2_EXAMPLE_HOST_OK" in text:
                value = "FLUX2_EXAMPLE_HOST_OK"
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


def _make_flux_job(
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
        return_value={host: "Initialized process group\nepoch time: 1.0 sec" for host in hosts}
    )

    benchmark_params = _flux_benchmark_params(**(benchmark_overrides or {}))
    inference_dict = _flux_inference_dict(**(inference_overrides or {}))
    job = FluxBenchmarkJob(
        phdl,
        inference_dict,
        benchmark_params,
        hf_token="hf_test_token",
        distributed=distributed,
        cluster_dict=cluster_dict,
    )
    return job, phdl


class TestFluxModelRouting(unittest.TestCase):
    def test_is_flux2_model(self):
        self.assertTrue(is_flux2_model("flux2"))
        self.assertTrue(is_flux2_model("flux2_klein"))
        self.assertFalse(is_flux2_model(None))
        self.assertFalse(is_flux2_model("flux_kontext"))

    def test_model_index_flux2(self):
        self.assertEqual(
            detect_flux_model_type_from_model_index({"_class_name": "Flux2Pipeline"}),
            "flux2",
        )


class TestResolveFluxModelType(unittest.TestCase):
    def test_model_mount_uses_original_repo_hint(self):
        self.assertEqual(
            resolve_flux_model_type(None, "/model", "/data/black-forest-labs/FLUX.2-dev"),
            "flux2",
        )


class TestResolveFlux2HfRepoId(unittest.TestCase):
    def test_local_path_defaults_to_flux2_dev(self):
        self.assertEqual(
            resolve_flux2_hf_repo_id("flux2", "/data/models/FLUX.2-dev"),
            FLUX2_DEFAULT_HF_REPO,
        )

    def test_hf_repo_hint_is_used(self):
        self.assertEqual(
            resolve_flux2_hf_repo_id("flux2", "/model", ["black-forest-labs/FLUX.2-dev"]),
            "black-forest-labs/FLUX.2-dev",
        )


class TestFlux2ChatTemplate(unittest.TestCase):
    def test_host_check_cmd(self):
        cmd = build_flux2_chat_template_host_check_cmd("/data/models/FLUX.2-dev")
        self.assertIn("chat_template.jinja", cmd)

    def test_container_ensure_cmd_fetches_from_hub(self):
        cmd = build_flux2_ensure_chat_template_cmd(hf_repo_id=FLUX2_DEFAULT_HF_REPO)
        self.assertIn("hf_hub_download", cmd)
        self.assertIn("chat_template.jinja", cmd)


class TestBuildRunUspArgs(unittest.TestCase):
    _BASE_PARAMS = {
        "prompt": "A small cat",
        "seed": 42,
        "num_inference_steps": 25,
        "max_sequence_length": 256,
        "no_use_resolution_binning": True,
        "use_torch_compile": True,
        "warmup_steps": 1,
        "warmup_calls": 5,
        "num_repetitions": 25,
        "height": 1024,
        "width": 1024,
        "ulysses_degree": 8,
        "ring_degree": 1,
        "torchrun_nproc": 8,
    }

    def test_flux1_uses_run_usp_flags(self):
        args = build_run_usp_args(
            self._BASE_PARAMS,
            model_repo="black-forest-labs/FLUX.1-dev",
        )
        self.assertIn("--use-torch-compile", args)
        self.assertIn("--benchmark_output_directory", args)
        self.assertNotIn("--model_type", args)


class TestBuildFlux2ExampleArgs(unittest.TestCase):
    _BASE_PARAMS = {
        "prompt": "A small cat",
        "seed": 42,
        "num_inference_steps": 50,
        "max_sequence_length": 512,
        "no_use_resolution_binning": True,
        "use_torch_compile": True,
        "warmup_steps": 5,
        "warmup_calls": 5,
        "num_repetitions": 25,
        "height": 1024,
        "width": 1024,
        "ulysses_degree": 8,
        "ring_degree": 1,
        "torchrun_nproc": 8,
    }

    def test_flux2_uses_xfuser_torch_compile_flag(self):
        args = build_flux2_example_args(
            self._BASE_PARAMS,
            model_repo="/model",
            model_type="flux2",
        )
        self.assertIn("--use_torch_compile", args)
        self.assertNotIn("--use-torch-compile", args)
        self.assertIn("--guidance_scale 4.0", args)
        self.assertIn("--output_type pil", args)


class TestBuildTorchrunCmd(unittest.TestCase):
    _FLUX1_PARAMS = {
        "prompt": "A small cat",
        "seed": 42,
        "num_inference_steps": 25,
        "max_sequence_length": 256,
        "no_use_resolution_binning": True,
        "use_torch_compile": True,
        "warmup_steps": 1,
        "warmup_calls": 5,
        "num_repetitions": 25,
        "height": 1024,
        "width": 1024,
        "ulysses_degree": 8,
        "ring_degree": 1,
        "torchrun_nproc": 8,
    }

    def test_flux1_uses_run_usp(self):
        cmd = build_torchrun_cmd(
            self._FLUX1_PARAMS,
            model_repo="black-forest-labs/FLUX.1-dev",
            distributed=False,
        )
        self.assertIn(RUN_USP_PATH, cmd)
        self.assertNotIn(FLUX2_EXAMPLE_PATH, cmd)

    def test_flux2_uses_flux2_example_wrapper(self):
        params = {
            **self._FLUX1_PARAMS,
            "num_inference_steps": 50,
            "max_sequence_length": 512,
        }
        cmd = build_torchrun_cmd(
            params,
            model_repo="/model",
            model_repo_hints=["black-forest-labs/FLUX.2-dev"],
            distributed=False,
        )
        self.assertIn(FLUX2_EXAMPLE_PATH, cmd)
        self.assertNotIn(RUN_USP_PATH, cmd)
        self.assertIn("results/timing.json", cmd)
        self.assertIn("--use_torch_compile", cmd)
        self.assertNotIn("for _ in range(reps)", cmd)
        self.assertNotIn("FLUX2_RUN_CMD", cmd)

    def test_flux1_distributed_uses_run_usp(self):
        cmd = build_torchrun_cmd(
            self._FLUX1_PARAMS,
            model_repo="black-forest-labs/FLUX.1-dev",
            distributed=True,
            node_rank=1,
            nnodes=2,
            master_addr="10.0.0.1",
            master_port=29500,
        )
        self.assertIn(RUN_USP_PATH, cmd)
        self.assertNotIn(FLUX2_EXAMPLE_PATH, cmd)
        self.assertIn("--nnodes=2", cmd)
        self.assertIn("--node_rank=1", cmd)
        self.assertIn("--benchmark_output_directory", cmd)

    def test_flux2_distributed_last_node_writes_timing(self):
        params = {
            **self._FLUX1_PARAMS,
            "num_inference_steps": 50,
            "max_sequence_length": 512,
        }
        cmd = build_torchrun_cmd(
            params,
            model_repo="/model",
            model_repo_hints=["black-forest-labs/FLUX.2-dev"],
            distributed=True,
            node_rank=1,
            nnodes=2,
            master_addr="10.0.0.1",
            master_port=29500,
        )
        self.assertIn(FLUX2_EXAMPLE_PATH, cmd)
        self.assertIn("--node_rank=1", cmd)
        self.assertIn("results/timing.json", cmd)

    def test_flux2_distributed_non_last_node_skips_timing_wrapper(self):
        params = {
            **self._FLUX1_PARAMS,
            "num_inference_steps": 50,
            "max_sequence_length": 512,
        }
        cmd = build_torchrun_cmd(
            params,
            model_repo="/model",
            model_repo_hints=["black-forest-labs/FLUX.2-dev"],
            distributed=True,
            node_rank=0,
            nnodes=2,
            master_addr="10.0.0.1",
            master_port=29500,
        )
        self.assertIn(FLUX2_EXAMPLE_PATH, cmd)
        self.assertIn("--node_rank=0", cmd)
        self.assertNotIn("results/timing.json", cmd)

    def test_flux2_cmd_ensures_chat_template_before_torchrun(self):
        params = {
            **self._FLUX1_PARAMS,
            "num_inference_steps": 50,
            "max_sequence_length": 512,
        }
        cmd = build_torchrun_cmd(
            params,
            model_repo="/model",
            model_repo_hints=["black-forest-labs/FLUX.2-dev"],
            distributed=False,
        )
        self.assertIn("hf_hub_download", cmd)
        self.assertLess(cmd.index("hf_hub_download"), cmd.index(FLUX2_EXAMPLE_PATH))

    def test_flux2_uses_volume_dict_mount_path(self):
        params = {
            **self._FLUX1_PARAMS,
            "num_inference_steps": 50,
            "max_sequence_length": 512,
        }
        cmd = build_torchrun_cmd(
            params,
            model_repo="/model",
            model_repo_hints=["black-forest-labs/FLUX.2-dev"],
            distributed=False,
            flux2_example_path=FLUX2_EXAMPLE_MOUNT_PATH,
        )
        self.assertIn(FLUX2_EXAMPLE_MOUNT_PATH, cmd)
        self.assertNotIn(FLUX2_EXAMPLE_PATH, cmd)


class TestFlux2ExampleFallbackMount(unittest.TestCase):
    def test_default_host_path_points_at_cvs_script(self):
        host_path = default_flux2_example_host_path()
        self.assertTrue(host_path.endswith("scripts/flux2_example.py") or host_path.endswith(r"scripts\flux2_example.py"))
        self.assertTrue(host_path.replace("\\", "/").endswith("lib/inference/xdit/scripts/flux2_example.py"))

    def test_image_probe_cmd_checks_in_image_path(self):
        cmd = build_flux2_example_image_probe_cmd("rocm/ufb-private:xdit")
        self.assertIn("--entrypoint test", cmd)
        self.assertIn(FLUX2_EXAMPLE_PATH, cmd)
        self.assertIn("FLUX2_EXAMPLE_PRESENT", cmd)
        self.assertIn("FLUX2_EXAMPLE_MISSING", cmd)

    def test_host_check_cmd(self):
        cmd = build_flux2_example_host_check_cmd("/host/flux2_example.py")
        self.assertIn("test -e", cmd)
        self.assertIn("/host/flux2_example.py", cmd)

    def test_flux1_does_not_probe_or_mount(self):
        job, phdl = _make_flux_job()
        errors = ensure_flux2_example_available(
            phdl,
            job.server_nodes,
            job.inference_dict,
            job.flux_params,
        )
        self.assertEqual(errors, [])
        self.assertEqual(job.inference_dict["container_config"]["volume_dict"], {})
        probe_cmds = [str(call.args[0]) for call in phdl.exec.call_args_list]
        self.assertFalse(any("FLUX2_EXAMPLE_PRESENT" in cmd for cmd in probe_cmds))

    def test_present_in_image_skips_bind_mount(self):
        job, phdl = _make_flux_job(
            inference_overrides={
                "model_repo": "black-forest-labs/FLUX.2-dev",
                "_resolved_flux_model_type": "flux2",
            }
        )
        errors = ensure_flux2_example_available(
            phdl,
            job.server_nodes,
            job.inference_dict,
            job.flux_params,
        )
        self.assertEqual(errors, [])
        self.assertEqual(job.inference_dict["container_config"]["volume_dict"], {})
        self.assertIsNone(resolve_flux2_example_host_mount(job.inference_dict))
        self.assertEqual(job.inference_dict["_flux2_example_container_path"], FLUX2_EXAMPLE_PATH)

    def test_missing_in_image_bind_mounts_cvs_script(self):
        job, phdl = _make_flux_job(
            inference_overrides={
                "model_repo": "black-forest-labs/FLUX.2-dev",
                "_resolved_flux_model_type": "flux2",
            }
        )

        def _missing_in_image(cmd, timeout=None, print_console=False, detailed=False):
            text = str(cmd)
            if "FLUX2_EXAMPLE_PRESENT" in text:
                return {"10.0.0.1": "FLUX2_EXAMPLE_MISSING"}
            if "FLUX2_EXAMPLE_HOST_OK" in text:
                return {"10.0.0.1": "FLUX2_EXAMPLE_HOST_OK"}
            return {"10.0.0.1": ""}

        phdl.exec.side_effect = _missing_in_image
        errors = ensure_flux2_example_available(
            phdl,
            job.server_nodes,
            job.inference_dict,
            job.flux_params,
        )
        self.assertEqual(errors, [])
        host_path = default_flux2_example_host_path()
        self.assertEqual(
            job.inference_dict["container_config"]["volume_dict"][host_path],
            FLUX2_EXAMPLE_MOUNT_PATH,
        )
        self.assertEqual(
            resolve_flux2_example_host_mount(job.inference_dict),
            (host_path, FLUX2_EXAMPLE_MOUNT_PATH),
        )
        self.assertEqual(
            job.inference_dict["_flux2_example_container_path"],
            FLUX2_EXAMPLE_MOUNT_PATH,
        )

    def test_existing_volume_dict_skips_image_probe(self):
        job, phdl = _make_flux_job(
            inference_overrides={
                "model_repo": "black-forest-labs/FLUX.2-dev",
                "_resolved_flux_model_type": "flux2",
                "container_config": {
                    "device_list": ["/dev/dri", "/dev/kfd"],
                    "volume_dict": {
                        "/home/user/cvs/cvs/lib/inference/xdit/scripts/flux2_example.py": (
                            FLUX2_EXAMPLE_MOUNT_PATH
                        )
                    },
                    "env_dict": {},
                },
            }
        )
        errors = ensure_flux2_example_available(
            phdl,
            job.server_nodes,
            job.inference_dict,
            job.flux_params,
        )
        self.assertEqual(errors, [])
        probe_cmds = [str(call.args[0]) for call in phdl.exec.call_args_list]
        self.assertFalse(any("FLUX2_EXAMPLE_PRESENT" in cmd for cmd in probe_cmds))
        self.assertTrue(any("FLUX2_EXAMPLE_HOST_OK" in cmd for cmd in probe_cmds))
        self.assertEqual(
            job.inference_dict["_flux2_example_container_path"],
            FLUX2_EXAMPLE_MOUNT_PATH,
        )

    def test_missing_host_script_returns_error(self):
        job, phdl = _make_flux_job(
            inference_overrides={
                "model_repo": "black-forest-labs/FLUX.2-dev",
                "_resolved_flux_model_type": "flux2",
            }
        )

        def _missing_host(cmd, timeout=None, print_console=False, detailed=False):
            text = str(cmd)
            if "FLUX2_EXAMPLE_PRESENT" in text:
                return {"10.0.0.1": "FLUX2_EXAMPLE_MISSING"}
            if "FLUX2_EXAMPLE_HOST_OK" in text:
                return {"10.0.0.1": "FLUX2_EXAMPLE_HOST_MISSING"}
            return {"10.0.0.1": ""}

        phdl.exec.side_effect = _missing_host
        errors = ensure_flux2_example_available(
            phdl,
            job.server_nodes,
            job.inference_dict,
            job.flux_params,
        )
        self.assertTrue(errors)
        self.assertIn("flux2_example.py", errors[0])
        self.assertEqual(job.inference_dict["container_config"]["volume_dict"], {})

    def test_launch_plan_docker_cmd_includes_fallback_mount(self):
        job, phdl = _make_flux_job(
            inference_overrides={
                "model_repo": "black-forest-labs/FLUX.2-dev",
                "_resolved_flux_model_type": "flux2",
            }
        )

        def _missing_in_image(cmd, timeout=None, print_console=False, detailed=False):
            text = str(cmd)
            if "test -e /dev/kfd" in text:
                return {"10.0.0.1": "KFD_OK"}
            if text.strip() == "hostname":
                return {"10.0.0.1": "host-0"}
            if "FLUX2_EXAMPLE_PRESENT" in text:
                return {"10.0.0.1": "FLUX2_EXAMPLE_MISSING"}
            if "FLUX2_EXAMPLE_HOST_OK" in text:
                return {"10.0.0.1": "FLUX2_EXAMPLE_HOST_OK"}
            if "docker run" in text:
                return {"10.0.0.1": "ok"}
            return {"10.0.0.1": ""}

        phdl.exec.side_effect = _missing_in_image
        plan = job.build_launch_plan()
        host_path = default_flux2_example_host_path()
        self.assertIn(host_path, plan.docker_cmds[0])
        self.assertIn(f"target={FLUX2_EXAMPLE_MOUNT_PATH}", plan.docker_cmds[0])
        self.assertIn(FLUX2_EXAMPLE_MOUNT_PATH, plan.docker_cmds[0])
        self.assertEqual(job._pre_launch_validation(plan), [])


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


class TestFluxParallelism(unittest.TestCase):
    def test_parallel_product_includes_all_degrees(self):
        params = {
            "ulysses_degree": 2,
            "ring_degree": 2,
            "pipefusion_parallel_degree": 2,
            "tensor_parallel_degree": 1,
            "data_parallel_degree": 1,
            "torchrun_nproc": 4,
        }
        self.assertEqual(parallel_product(params), 8)

    def test_validate_parallelism_pass(self):
        params = {**_FLUX_BENCHMARK_PARAMS, "ring_degree": 2}
        world_size, product, err = validate_parallelism(2, params)
        self.assertIsNone(err)
        self.assertEqual(world_size, 16)
        self.assertEqual(product, 16)

    def test_validate_parallelism_fail(self):
        _, _, err = validate_parallelism(2, _FLUX_BENCHMARK_PARAMS)
        self.assertIsNotNone(err)
        self.assertIn("Parallel degree product", err)

    def test_validate_flux_parallelism_config_distributed(self):
        cluster_dict = {"node_dict": {"10.0.0.1": {}, "10.0.0.2": {}}}
        inference_dict = {"nnodes": 2}
        benchmark_params = _flux_benchmark_params(ring_degree=2)
        self.assertIsNone(
            validate_flux_parallelism_config(
                inference_dict,
                benchmark_params,
                distributed=True,
                cluster_dict=cluster_dict,
            )
        )


class TestFluxBenchmarkJob(unittest.TestCase):
    def test_validate_parallelism_single_node_pass(self):
        job, _ = _make_flux_job()
        self.assertIsNone(job.validate_parallelism())

    def test_validate_parallelism_distributed_fail(self):
        cluster_dict = {"node_dict": {"10.0.0.1": {}, "10.0.0.2": {}}}
        job, _ = _make_flux_job(
            hosts=["10.0.0.1", "10.0.0.2"],
            distributed=True,
            cluster_dict=cluster_dict,
            inference_overrides={"nnodes": 2, "master_addr": "10.0.0.1"},
        )
        err = job.validate_parallelism()
        self.assertIsNotNone(err)
        self.assertIn("Parallel degree product", err)

    def test_build_env_args_single_node_omits_nccl(self):
        job, _ = _make_flux_job()
        env_args = job._build_env_args()
        self.assertIn("-e CUSTOM=1", env_args)
        self.assertIn("-e HF_TOKEN=hf_test_token", env_args)
        self.assertIn("-e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7", env_args)
        self.assertNotIn("NCCL_PROTO", env_args)

    def test_build_env_args_distributed_includes_nccl(self):
        cluster_dict = {"node_dict": {"10.0.0.1": {}, "10.0.0.2": {}}}
        job, _ = _make_flux_job(
            hosts=["10.0.0.1", "10.0.0.2"],
            distributed=True,
            cluster_dict=cluster_dict,
            inference_overrides={
                "nnodes": 2,
                "master_addr": "10.0.0.1",
                "nccl_ib_hca": "mlx5_0",
            },
            benchmark_overrides={"ring_degree": 2},
        )
        env_args = job._build_env_args()
        self.assertIn("-e NCCL_PROTO=Simple", env_args)
        self.assertIn("-e NCCL_IB_HCA=mlx5_0", env_args)

    def test_build_env_args_flux2_adds_hf_repo_id(self):
        job, _ = _make_flux_job(
            inference_overrides={
                "model_repo": "black-forest-labs/FLUX.2-dev",
                "_resolved_flux_model_type": "flux2",
            }
        )
        env_args = job._build_env_args()
        self.assertIn("-e FLUX2_HF_REPO_ID=black-forest-labs/FLUX.2-dev", env_args)

    def test_build_docker_cmd_contains_torchrun_and_run_usp(self):
        job, _ = _make_flux_job()
        cmd = job._build_docker_cmd(
            node_rank=0,
            host_output_dir="/home/user/cvs_flux_output/flux_host-0_outputs",
            master_addr="127.0.0.1",
            master_port=29500,
        )
        self.assertIn("docker run", cmd)
        self.assertIn(RUN_USP_PATH, cmd)
        self.assertIn("--name flux-benchmark", cmd)

    def test_build_docker_cmd_distributed_uses_ranked_container_name(self):
        cluster_dict = {"node_dict": {"10.0.0.1": {}, "10.0.0.2": {}}}
        job, _ = _make_flux_job(
            hosts=["10.0.0.1", "10.0.0.2"],
            distributed=True,
            cluster_dict=cluster_dict,
            inference_overrides={"nnodes": 2, "master_addr": "10.0.0.1"},
            benchmark_overrides={"ring_degree": 2},
        )
        cmd = job._build_docker_cmd(
            node_rank=1,
            host_output_dir="/home/user/cvs_flux_output/flux_host-0_outputs",
            master_addr="10.0.0.1",
            master_port=29500,
        )
        self.assertIn("--name flux-benchmark-rank1", cmd)
        self.assertIn("--node_rank=1", cmd)

    def test_build_launch_plan_single_node(self):
        job, _ = _make_flux_job(hosts=["10.0.0.1"])
        plan = job.build_launch_plan()
        self.assertFalse(plan.distributed)
        self.assertEqual(plan.node_order, ["10.0.0.1"])
        self.assertEqual(len(plan.docker_cmds), 1)
        self.assertEqual(len(plan.mkdir_cmds), 1)
        self.assertIn("/home/user/cvs_flux_output/flux_host-0_outputs", plan.primary_output_dir)
        self.assertEqual(plan.world_size, 8)

    def test_build_launch_plan_distributed(self):
        cluster_dict = {"node_dict": {"10.0.0.1": {}, "10.0.0.2": {}}}
        job, _ = _make_flux_job(
            hosts=["10.0.0.1", "10.0.0.2"],
            distributed=True,
            cluster_dict=cluster_dict,
            inference_overrides={"nnodes": 2, "master_addr": "10.0.0.1"},
            benchmark_overrides={"ring_degree": 2},
        )
        plan = job.build_launch_plan()
        self.assertTrue(plan.distributed)
        self.assertEqual(plan.node_order, ["10.0.0.1", "10.0.0.2"])
        self.assertEqual(len(plan.docker_cmds), 2)
        self.assertEqual(plan.world_size, 16)
        self.assertIn("flux_host-0_outputs", plan.primary_output_dir)

    def test_run_fails_on_parallelism_mismatch(self):
        cluster_dict = {"node_dict": {"10.0.0.1": {}, "10.0.0.2": {}}}
        job, phdl = _make_flux_job(
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
        job, phdl = _make_flux_job()

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
        job, phdl = _make_flux_job()
        results, plan, errors = job.run()
        self.assertEqual(errors, [])
        self.assertEqual(set(results.keys()), {"10.0.0.1"})
        self.assertEqual(len(plan.docker_cmds), 1)
        self.assertEqual(phdl.exec_cmd_list.call_count, 1)
        phdl.exec.assert_called()
        benchmark_call = phdl.exec.call_args_list[-1]
        self.assertTrue(benchmark_call.kwargs.get("detailed"))
        self.assertIn("Initialized process group", results["10.0.0.1"])

    def test_run_fails_on_nonzero_exit_without_traceback(self):
        job, phdl = _make_flux_job()
        _wire_phdl_exec(phdl, ["10.0.0.1"], benchmark_exit_code=137, benchmark_output="Killed")
        results, plan, errors = job.run()
        self.assertEqual(set(results.keys()), {"10.0.0.1"})
        self.assertTrue(errors)
        self.assertTrue(any("10.0.0.1" in err for err in errors))
        self.assertNotIn("Traceback", results["10.0.0.1"])


class TestExecResultHelpers(unittest.TestCase):
    def test_exec_result_output_from_detailed_dict(self):
        self.assertEqual(_exec_result_output({"output": "hello", "exit_code": 0}), "hello")

    def test_exec_result_exit_code_from_detailed_dict(self):
        self.assertEqual(_exec_result_exit_code({"output": "hello", "exit_code": 2}), 2)

    def test_exec_result_exit_code_defaults_for_plain_string(self):
        self.assertEqual(_exec_result_exit_code("hello"), 0)


class TestExecCmdListOnNodes(unittest.TestCase):
    @patch("cvs.lib.inference.xdit.pytorch_xdit_flux_job._exec_on_single_node")
    def test_detailed_multi_node_runs_in_parallel(self, mock_single):
        barrier = threading.Barrier(2, timeout=2)

        def _wait_and_return(*_args, **_kwargs):
            barrier.wait()
            return {"output": "ok", "exit_code": 0}

        mock_single.side_effect = _wait_and_return

        phdl = MagicMock()
        phdl.host_list = ["10.0.0.1", "10.0.0.2"]
        results = _exec_cmd_list_on_nodes(
            phdl,
            ["10.0.0.1", "10.0.0.2"],
            ["docker run rank0", "docker run rank1"],
            detailed=True,
        )

        self.assertEqual(mock_single.call_count, 2)
        self.assertEqual(results["10.0.0.1"]["exit_code"], 0)
        self.assertEqual(results["10.0.0.2"]["exit_code"], 0)
        phdl.exec_cmd_list.assert_not_called()

    def test_non_detailed_uses_exec_cmd_list_when_host_order_matches(self):
        phdl = MagicMock()
        phdl.host_list = ["10.0.0.1", "10.0.0.2"]
        phdl.exec_cmd_list.return_value = {"10.0.0.1": "ok", "10.0.0.2": "ok"}

        results = _exec_cmd_list_on_nodes(
            phdl,
            ["10.0.0.1", "10.0.0.2"],
            ["mkdir -p /a", "mkdir -p /a"],
        )

        self.assertEqual(results, {"10.0.0.1": "ok", "10.0.0.2": "ok"})
        phdl.exec_cmd_list.assert_called_once()


class TestPhdlConnectionKwargs(unittest.TestCase):
    def test_reads_credentials_from_inner_pssh(self):
        from cvs.lib.inference.xdit.pytorch_xdit_flux_job import _phdl_connection_kwargs

        class _Inner:
            user = "ubuntu"
            password = None
            pkey = "/home/ubuntu/.ssh/id_rsa"
            env_vars = {"FOO": "bar"}

        class _Wrapper:
            pssh = _Inner()
            host_list = ["10.0.0.1", "10.0.0.2"]
            env_vars = None

        kwargs = _phdl_connection_kwargs(_Wrapper())
        self.assertEqual(kwargs["user"], "ubuntu")
        self.assertEqual(kwargs["pkey"], "/home/ubuntu/.ssh/id_rsa")
        self.assertEqual(kwargs["env_vars"], {"FOO": "bar"})


if __name__ == "__main__":
    unittest.main()
