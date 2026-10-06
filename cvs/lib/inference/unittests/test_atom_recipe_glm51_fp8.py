'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Behavior checks for the GLM-5.1-FP8 vLLM recipe stems.
'''

import unittest
from pathlib import Path

from cvs.lib.inference.atom.atom_config_loader import load_variant


def _load(name):
    root = Path(__file__).resolve().parents[3]
    return load_variant(
        root / "input/config_file/inference/atom" / name,
        {"username": "testuser"},
    )


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

    def test_vllm_distributed_keeps_tp8_and_pp2(self):
        variant = _load("mi3xx_atom_vllm_glm-5.1_fp8_distributed.json")
        self.assertEqual(variant.params.driver, "vllm_atom")
        self.assertEqual(variant.model.id, "zai-org/GLM-5.1-FP8")
        self.assertEqual(variant.params.tensor_parallelism, "8")
        self.assertEqual(variant.params.pipeline_parallel_size, "2")
        self.assertEqual(variant.params.nnodes, "2")
        for cell in variant.expected_cells():
            self.assertIn(",PP=2,", cell)
            self.assertIn(cell, variant.thresholds)


if __name__ == "__main__":
    unittest.main()
