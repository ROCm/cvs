import unittest

from cvs.lib.inference.pytorch_xdit.pytorch_xdit_flux_job import (
    build_run_usp_args,
    build_torchrun_cmd,
    infer_flux_model_type,
    resolve_flux_guidance_scale,
)


class TestInferFluxModelType(unittest.TestCase):
    def test_explicit_model_type_wins(self):
        self.assertEqual(
            infer_flux_model_type("black-forest-labs/FLUX.1-dev", "flux2"),
            "flux2",
        )

    def test_infers_flux2_from_repo(self):
        self.assertEqual(
            infer_flux_model_type("black-forest-labs/FLUX.2-dev"),
            "flux2",
        )
        self.assertEqual(
            infer_flux_model_type("/models/black-forest-labs/FLUX.2-dev"),
            "flux2",
        )

    def test_infers_flux2_klein(self):
        self.assertEqual(
            infer_flux_model_type("black-forest-labs/FLUX.2-klein-9B"),
            "flux2_klein",
        )

    def test_infers_kontext(self):
        self.assertEqual(
            infer_flux_model_type("black-forest-labs/FLUX.1-Kontext-dev"),
            "flux_kontext",
        )

    def test_flux1_returns_none(self):
        self.assertIsNone(infer_flux_model_type("black-forest-labs/FLUX.1-dev"))


class TestResolveFluxGuidanceScale(unittest.TestCase):
    def test_flux2_default(self):
        self.assertEqual(resolve_flux_guidance_scale("flux2", None), 4.0)

    def test_flux1_no_default(self):
        self.assertIsNone(resolve_flux_guidance_scale(None, None))

    def test_explicit_overrides_default(self):
        self.assertEqual(resolve_flux_guidance_scale("flux2", 3.5), 3.5)


class TestBuildRunUspArgs(unittest.TestCase):
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

    def test_flux2_includes_model_type_and_default_guidance_scale(self):
        args = build_run_usp_args(
            self._BASE_PARAMS,
            model_repo="black-forest-labs/FLUX.2-dev",
        )
        self.assertIn("--model_type flux2", args)
        self.assertIn("--guidance_scale 4.0", args)
        self.assertIn("--max_sequence_length 512", args)

    def test_flux1_omits_model_type_and_guidance_scale(self):
        params = {
            **self._BASE_PARAMS,
            "num_inference_steps": 25,
            "max_sequence_length": 256,
        }
        args = build_run_usp_args(
            params,
            model_repo="black-forest-labs/FLUX.1-dev",
        )
        self.assertNotIn("--model_type", args)
        self.assertNotIn("--guidance_scale", args)


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

    def test_flux1_uses_run_usp_without_model_type(self):
        cmd = build_torchrun_cmd(
            self._FLUX1_PARAMS,
            model_repo="black-forest-labs/FLUX.1-dev",
            distributed=False,
        )
        self.assertIn("/app/Flux/run_usp.py", cmd)
        self.assertNotIn("--model_type", cmd)

    def test_flux2_uses_run_usp_with_model_type(self):
        params = {
            **self._FLUX1_PARAMS,
            "num_inference_steps": 50,
            "max_sequence_length": 512,
        }
        cmd = build_torchrun_cmd(
            params,
            model_repo="black-forest-labs/FLUX.2-dev",
            distributed=False,
        )
        self.assertIn("/app/Flux/run_usp.py", cmd)
        self.assertIn("--model_type flux2", cmd)
        self.assertIn("--guidance_scale 4.0", cmd)


if __name__ == "__main__":
    unittest.main()
