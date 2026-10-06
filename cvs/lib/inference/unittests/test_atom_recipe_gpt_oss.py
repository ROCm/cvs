'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Behavior checks for the GPT-OSS-120B recipe stems.
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


class TestAtomRecipeGptOss(unittest.TestCase):
    def test_native_single_gpu_uses_recipe_utilization(self):
        variant = _load("mi3xx_atom_gpt-oss-120b_mxfp4_single.json")
        self.assertEqual(variant.params.driver, "atom")
        self.assertEqual(variant.model.id, "openai/gpt-oss-120b")
        self.assertEqual(variant.gpu_arch, "mi3xx")
        self.assertEqual(variant.params.tensor_parallelism, "1")
        args = variant.roles.server.atom_args
        self.assertEqual(args[args.index("--gpu-memory-utilization") + 1], "0.5")
        self.assertEqual(args[args.index("--kv_cache_dtype") + 1], "fp8")
        self.assertEqual(variant.container.image, "<changeme>")
        for cell in variant.expected_cells():
            self.assertIn(cell, variant.thresholds)

    def test_vllm_distributed_pairs_with_shipped_tp4_single(self):
        single = _load("mi3xx_atom_vllm_gpt-oss-120b_mxfp4_single.json")
        distributed = _load("mi3xx_atom_vllm_gpt-oss-120b_mxfp4_distributed.json")
        self.assertEqual(single.model.id, "openai/gpt-oss-120b")
        self.assertEqual(single.params.tensor_parallelism, "4")
        self.assertEqual(single.params.nnodes, "1")
        self.assertEqual(distributed.params.tensor_parallelism, "4")
        self.assertEqual(distributed.params.nnodes, "2")
        self.assertEqual(distributed.params.pipeline_parallel_size, "2")
        for cell in distributed.expected_cells():
            self.assertIn(",PP=2,", cell)
            self.assertIn(cell, distributed.thresholds)


if __name__ == "__main__":
    unittest.main()
