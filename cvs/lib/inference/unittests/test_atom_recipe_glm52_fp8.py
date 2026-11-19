'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Behavior checks for the GLM-5.2-FP8 recipe stems.
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


class TestAtomRecipeGlm52Fp8(unittest.TestCase):
    def test_native_perf_omits_mtp(self):
        variant = _load("mi3xx_atom_glm-5.2_fp8_single_perf.json")
        self.assertEqual(variant.params.driver, "atom")
        self.assertEqual(variant.model.id, "zai-org/GLM-5.2-FP8")
        self.assertEqual(variant.gpu_arch, "mi3xx")
        self.assertEqual(variant.params.tensor_parallelism, "8")
        self.assertNotIn("--method", variant.roles.server.atom_args)
        self.assertEqual(variant.container.image, "<changeme>")
        self.assertFalse(variant.enforce_thresholds)
        self.assertEqual(variant.roles.server.env["AITER_QUICK_REDUCE_QUANTIZATION"], "INT4")
        for cell in variant.expected_cells():
            self.assertIn(cell, variant.thresholds)

    def test_native_mtp3_adds_three_speculative_tokens(self):
        variant = _load("mi3xx_atom_glm-5.2_fp8_single_mtp3.json")
        args = variant.roles.server.atom_args
        self.assertEqual(variant.model.id, "zai-org/GLM-5.2-FP8")
        self.assertEqual(args[args.index("--method") + 1], "mtp")
        self.assertEqual(args[args.index("--num-speculative-tokens") + 1], "3")

    def test_vllm_single_and_distributed_topology(self):
        single = _load("mi3xx_atom_vllm_glm-5.2_fp8_single.json")
        distributed = _load("mi3xx_atom_vllm_glm-5.2_fp8_distributed.json")
        self.assertEqual(single.params.nnodes, "1")
        self.assertEqual(single.params.pipeline_parallel_size, "1")
        self.assertEqual(distributed.params.nnodes, "2")
        self.assertEqual(distributed.params.pipeline_parallel_size, "2")
        self.assertEqual(single.params.tensor_parallelism, distributed.params.tensor_parallelism)
        self.assertIn("ptpc_fp8", single.roles.server.serve_args["additional-config"])

    def test_sglang_single_and_distributed_topology(self):
        sglang = _load("mi3xx_atom_sglang_glm-5.2_fp8_single.json")
        sglang_pp = _load("mi3xx_atom_sglang_glm-5.2_fp8_distributed.json")
        self.assertEqual(sglang.params.driver, "sglang")
        self.assertEqual(sglang.params.nnodes, "1")
        self.assertEqual(sglang_pp.params.nnodes, "2")
        self.assertEqual(sglang_pp.params.pipeline_parallel_size, "2")
        self.assertEqual(
            sglang.roles.server.env["SGLANG_EXTERNAL_MODEL_PACKAGE"],
            "atom.plugin.sglang.models",
        )
        for stem in (sglang, sglang_pp):
            args = stem.roles.server.sglang_args
            self.assertEqual(args[args.index("--context-length") + 1], "16384")


if __name__ == "__main__":
    unittest.main()
