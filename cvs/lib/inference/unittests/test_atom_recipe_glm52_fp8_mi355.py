'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Behavior checks for the MI355 GLM-5.2-FP8 recipe stems.
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


class TestAtomRecipeGlm52Fp8Mi355(unittest.TestCase):
    def test_native_keeps_mi355_block_fp8_experts(self):
        perf = _load("mi355x_atom_glm-5.2_fp8_single.json")
        mtp = _load("mi355x_atom_glm-5.2_fp8_single.json", profile="mtp3")
        self.assertEqual(perf.gpu_arch, "mi355x")
        self.assertEqual(perf.model.id, "zai-org/GLM-5.2-FP8")
        self.assertEqual(perf.params.driver, "atom")
        self.assertEqual(perf.params.tensor_parallelism, "4")
        self.assertNotIn("--method", perf.roles.server.atom_args)
        quant = perf.roles.server.atom_args[perf.roles.server.atom_args.index("--online_quant_config") + 1]
        self.assertIn("per_block_fp8", quant)
        self.assertIn("model.layers.*.mlp.experts", quant)
        args = mtp.roles.server.atom_args
        self.assertEqual(args[args.index("--method") + 1], "mtp")
        self.assertEqual(args[args.index("--num-speculative-tokens") + 1], "3")
        self.assertEqual(mtp.model.id, "zai-org/GLM-5.2-FP8")
        self.assertEqual(perf.container.image, "<changeme>")
        self.assertNotIn("ATOM_USE_TRITON_MOE", perf.roles.server.env)
        for cell in perf.expected_cells():
            self.assertIn(cell, perf.thresholds)

    def test_engines_keep_tp4_and_pp2(self):
        for backend in ("vllm", "sglang"):
            single = _load(f"mi355x_atom_{backend}_glm-5.2_fp8_single.json")
            distributed = _load(f"mi355x_atom_{backend}_glm-5.2_fp8_distributed.json")
            self.assertEqual(single.gpu_arch, "mi355x")
            self.assertEqual(single.model.id, "zai-org/GLM-5.2-FP8")
            self.assertEqual(single.params.tensor_parallelism, "4")
            self.assertEqual(single.params.nnodes, "1")
            self.assertEqual(single.params.pipeline_parallel_size, "1")
            self.assertEqual(distributed.params.nnodes, "2")
            self.assertEqual(distributed.params.pipeline_parallel_size, "2")
            self.assertEqual(distributed.params.tensor_parallelism, "4")
            for cell in distributed.expected_cells():
                self.assertIn(cell, distributed.thresholds)
        vllm = _load("mi355x_atom_vllm_glm-5.2_fp8_single.json")
        self.assertIn("per_block_fp8", vllm.roles.server.serve_args["additional-config"])
        sglang = _load("mi355x_atom_sglang_glm-5.2_fp8_single.json")
        self.assertEqual(sglang.roles.server.env["SGLANG_ROCM_ARCH"], "gfx950")
        self.assertNotIn("SGLANG_AITER_FP8_PREFILL_ATTN", sglang.roles.server.env)
        self.assertIn("--disable-radix-cache", sglang.roles.server.sglang_args)


if __name__ == "__main__":
    unittest.main()
