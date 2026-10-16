'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Behavior checks for the GLM-5.2-FP8 recipe stems.
'''

import unittest
from pathlib import Path

from cvs.lib.inference.atom.atom_config_loader import load_variant


def _load(name, profile=None):
    root = Path(__file__).resolve().parents[3]
    return load_variant(
        root / "input/config_file/inference/atom" / name,
        {"username": "testuser"},
        profile=profile,
    )


class TestAtomRecipeGlm52Fp8(unittest.TestCase):
    def test_native_perf_omits_mtp(self):
        variant = _load("mi3xx_atom_glm-5.2_fp8_single.json")
        self.assertEqual(variant.params.driver, "atom")
        self.assertEqual(variant.model.id, "zai-org/GLM-5.2-FP8")
        self.assertEqual(variant.gpu_arch, "mi3xx")
        self.assertEqual(variant.params.tensor_parallelism, "8")
        self.assertNotIn("--method", variant.roles.server.atom_args)
        self.assertEqual(variant.container.image, "<changeme>")
        for cell in variant.expected_cells():
            self.assertIn(cell, variant.thresholds)

    def test_native_mtp3_adds_three_speculative_tokens(self):
        variant = _load("mi3xx_atom_glm-5.2_fp8_single.json", profile="mtp3")
        args = variant.roles.server.atom_args
        self.assertEqual(variant.model.id, "zai-org/GLM-5.2-FP8")
        self.assertEqual(args[args.index("--method") + 1], "mtp")
        self.assertEqual(args[args.index("--num-speculative-tokens") + 1], "3")

    def test_vllm_and_sglang_keep_single_and_pp2(self):
        single = _load("mi3xx_atom_vllm_glm-5.2_fp8_single.json")
        distributed = _load("mi3xx_atom_vllm_glm-5.2_fp8_distributed.json")
        self.assertEqual(single.params.nnodes, "1")
        self.assertEqual(single.params.pipeline_parallel_size, "1")
        self.assertEqual(distributed.params.nnodes, "2")
        self.assertEqual(distributed.params.pipeline_parallel_size, "2")
        self.assertEqual(single.params.tensor_parallelism, distributed.params.tensor_parallelism)
        self.assertIn("ptpc_fp8", single.roles.server.serve_args["additional-config"])
        sglang = _load("mi3xx_atom_sglang_glm-5.2_fp8_single.json")
        sglang_pp = _load("mi3xx_atom_sglang_glm-5.2_fp8_distributed.json")
        self.assertEqual(sglang.params.driver, "sglang")
        self.assertEqual(sglang.params.nnodes, "1")
        self.assertEqual(sglang_pp.params.nnodes, "2")
        self.assertEqual(sglang_pp.params.pipeline_parallel_size, "2")
        self.assertEqual(sglang.roles.server.env["SGLANG_EXTERNAL_MODEL_PACKAGE"], "atom.plugin.sglang.models")


if __name__ == "__main__":
    unittest.main()
