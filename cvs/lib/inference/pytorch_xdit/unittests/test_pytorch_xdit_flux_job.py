import unittest

from cvs.lib.inference.pytorch_xdit.pytorch_xdit_flux_job import (
    FLUX2_EXAMPLE_PATH,
    RUN_USP_PATH,
    build_flux2_example_args,
    build_run_usp_args,
    build_torchrun_cmd,
    detect_flux_model_type_from_model_index,
    infer_flux_model_type,
    is_flux2_model,
    resolve_flux_guidance_scale,
    resolve_flux_model_type,
)


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

    def test_flux2_distributed_rank0_writes_timing(self):
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
        self.assertIn("results/timing.json", cmd)

    def test_flux2_distributed_worker_skips_timing_wrapper(self):
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
        self.assertNotIn("results/timing.json", cmd)


if __name__ == "__main__":
    unittest.main()
