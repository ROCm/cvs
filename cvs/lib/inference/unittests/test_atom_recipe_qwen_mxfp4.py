'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Behavior checks for the Qwen3.5-397B MXFP4 vLLM recipe stems.
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


class TestAtomRecipeQwenMxfp4(unittest.TestCase):
    def test_vllm_single_uses_mxfp4_checkpoint_and_tp4(self):
        variant = _load("mi3xx_atom_vllm_qwen3.5-397b-a17b_mxfp4_single.json")
        self.assertEqual(variant.params.driver, "vllm_atom")
        self.assertEqual(variant.model.id, "amd/Qwen3.5-397B-A17B-MXFP4")
        self.assertEqual(variant.gpu_arch, "mi355x")
        self.assertEqual(variant.params.tensor_parallelism, "4")
        self.assertEqual(variant.params.nnodes, "1")
        self.assertEqual(variant.params.pipeline_parallel_size, "1")
        self.assertEqual(variant.roles.server.serve_args["kv-cache-dtype"], "fp8")
        self.assertEqual(variant.roles.server.serve_args["gpu-memory-utilization"], "0.9")
        self.assertEqual(variant.roles.server.env["ATOM_USE_CUSTOM_ALL_GATHER"], "0")
        self.assertEqual(variant.container.image, "<changeme>")
        for cell in variant.expected_cells():
            self.assertIn(cell, variant.thresholds)

    def test_vllm_distributed_keeps_tp4_and_pp2(self):
        variant = _load("mi3xx_atom_vllm_qwen3.5-397b-a17b_mxfp4_distributed.json")
        self.assertEqual(variant.gpu_arch, "mi355x")
        self.assertEqual(variant.params.tensor_parallelism, "4")
        self.assertEqual(variant.params.nnodes, "2")
        self.assertEqual(variant.params.pipeline_parallel_size, "2")
        for cell in variant.expected_cells():
            self.assertIn(cell, variant.thresholds)


if __name__ == "__main__":
    unittest.main()
