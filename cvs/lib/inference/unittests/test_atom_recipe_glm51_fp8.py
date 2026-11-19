'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Behavior checks for the GLM-5.1-FP8 vLLM recipe stems.
'''

import unittest
from pathlib import Path

from cvs.lib.inference.atom.atom_config_loader import AtomVariantConfig, atom_config_from_raw, atom_variant_raw
from cvs.lib.utils.config_loader import substitute_config


def _load(name):
    """Load a shipped template, which still holds its <changeme> markers."""
    path = Path(__file__).resolve().parents[3] / "input/config_file/inference/atom" / name
    raw, thresholds = substitute_config(path, {"username": "testuser"})
    config, gpu_arch = atom_config_from_raw(raw, thresholds, path)
    return AtomVariantConfig(**atom_variant_raw(config, gpu_arch))


class TestAtomRecipeGlm51Fp8(unittest.TestCase):
    def test_vllm_single_uses_recipe_checkpoint_and_tp8(self):
        variant = _load("mi3xx_atom_vllm_glm-5.1_fp8_single.json")
        self.assertEqual(variant.params.driver, "vllm_atom")
        self.assertEqual(variant.model.id, "zai-org/GLM-5.1-FP8")
        self.assertEqual(variant.gpu_arch, "mi3xx")
        self.assertEqual(variant.params.tensor_parallelism, "8")
        self.assertEqual(variant.params.pipeline_parallel_size, "1")
        self.assertEqual(variant.params.nnodes, "1")
        self.assertEqual(variant.roles.server.serve_args["kv-cache-dtype"], "fp8")
        self.assertEqual(variant.roles.server.serve_args["gpu-memory-utilization"], "0.9")
        self.assertTrue(variant.roles.server.serve_args["trust-remote-code"])
        self.assertEqual(variant.roles.server.env["AITER_QUICK_REDUCE_QUANTIZATION"], "INT4")
        self.assertEqual(variant.container.image, "<changeme>")
        for cell in variant.expected_cells():
            self.assertIn(cell, variant.thresholds)
        self.assertFalse(variant.enforce_thresholds)

    def test_vllm_distributed_keeps_tp8_and_pp2(self):
        variant = _load("mi3xx_atom_vllm_glm-5.1_fp8_distributed.json")
        self.assertEqual(variant.params.driver, "vllm_atom")
        self.assertEqual(variant.model.id, "zai-org/GLM-5.1-FP8")
        self.assertEqual(variant.params.tensor_parallelism, "8")
        self.assertEqual(variant.params.pipeline_parallel_size, "2")
        self.assertEqual(variant.params.nnodes, "2")
        self.assertFalse(variant.enforce_thresholds)
        for cell in variant.expected_cells():
            self.assertIn(",PP=2,", cell)
            self.assertIn(cell, variant.thresholds)


if __name__ == "__main__":
    unittest.main()
