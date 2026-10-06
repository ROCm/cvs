'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Behavior checks for the DeepSeek-R1 MXFP4 recipe stems.
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


class TestAtomRecipeR1Mxfp4(unittest.TestCase):
    def test_native_perf_and_mtp3_use_recipe_checkpoints(self):
        perf = _load("mi3xx_atom_deepseek-r1_mxfp4_single.json")
        mtp = _load("mi3xx_atom_deepseek-r1_mxfp4_single.json", profile="mtp3")
        self.assertEqual(perf.model.id, "amd/DeepSeek-R1-0528-MXFP4")
        self.assertEqual(perf.gpu_arch, "mi355x")
        self.assertEqual(perf.params.tensor_parallelism, "8")
        self.assertIn("--kv_cache_dtype", perf.roles.server.atom_args)
        self.assertNotIn("--method", perf.roles.server.atom_args)
        self.assertEqual(mtp.model.id, "amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4")
        args = mtp.roles.server.atom_args
        self.assertEqual(args[args.index("--method") + 1], "mtp")
        self.assertEqual(args[args.index("--num-speculative-tokens") + 1], "3")
        self.assertEqual(perf.container.image, "<changeme>")
        for cell in perf.expected_cells():
            self.assertIn(cell, perf.thresholds)

    def test_vllm_and_sglang_use_their_recipe_checkpoints(self):
        vllm = _load("mi3xx_atom_vllm_deepseek-r1_mxfp4_single.json")
        vllm_pp = _load("mi3xx_atom_vllm_deepseek-r1_mxfp4_distributed.json")
        sglang = _load("mi3xx_atom_sglang_deepseek-r1_mxfp4_single.json")
        sglang_pp = _load("mi3xx_atom_sglang_deepseek-r1_mxfp4_distributed.json")
        self.assertEqual(vllm.model.id, "amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4")
        self.assertEqual(sglang.model.id, "amd/DeepSeek-R1-0528-MXFP4-v2")
        self.assertEqual(vllm.gpu_arch, "mi355x")
        self.assertEqual(sglang.gpu_arch, "mi355x")
        self.assertEqual(vllm.params.nnodes, "1")
        self.assertEqual(vllm_pp.params.nnodes, "2")
        self.assertEqual(vllm_pp.params.pipeline_parallel_size, "2")
        self.assertEqual(sglang_pp.params.pipeline_parallel_size, "2")
        self.assertEqual(vllm.params.tensor_parallelism, "8")
        self.assertEqual(sglang.params.tensor_parallelism, "8")
        for cell in sglang_pp.expected_cells():
            self.assertIn(cell, sglang_pp.thresholds)


if __name__ == "__main__":
    unittest.main()
