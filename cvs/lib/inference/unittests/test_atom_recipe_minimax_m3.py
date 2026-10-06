'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Behavior checks for the MiniMax-M3 MXFP4 recipe stems.
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


class TestAtomRecipeMinimaxM3(unittest.TestCase):
    def test_native_perf_is_mxfp4_and_eagle3_adds_draft(self):
        perf = _load("mi3xx_atom_minimax-m3_mxfp4_single.json")
        eagle = _load("mi3xx_atom_minimax-m3_mxfp4_single.json", profile="eagle3")
        self.assertEqual(perf.model.id, "amd/MiniMax-M3-MXFP4")
        self.assertEqual(perf.gpu_arch, "mi355x")
        self.assertEqual(perf.params.tensor_parallelism, "4")
        self.assertNotIn("--method", perf.roles.server.atom_args)
        args = eagle.roles.server.atom_args
        self.assertEqual(args[args.index("--method") + 1], "eagle3")
        self.assertEqual(args[args.index("--draft-model") + 1], "Inferact/MiniMax-M3-EAGLE3")
        self.assertEqual(eagle.model.id, "amd/MiniMax-M3-MXFP4")
        self.assertEqual(perf.container.image, "<changeme>")
        for cell in perf.expected_cells():
            self.assertIn(cell, perf.thresholds)

    def test_vllm_omits_trust_remote_code_and_pp2_keeps_tp4(self):
        single = _load("mi3xx_atom_vllm_minimax-m3_mxfp4_single.json")
        distributed = _load("mi3xx_atom_vllm_minimax-m3_mxfp4_distributed.json")
        self.assertTrue(single.roles.server.serve_args["no-trust-remote-code"])
        self.assertEqual(single.params.nnodes, "1")
        self.assertEqual(distributed.params.nnodes, "2")
        self.assertEqual(distributed.params.pipeline_parallel_size, "2")
        self.assertEqual(distributed.params.tensor_parallelism, "4")
        sglang = _load("mi3xx_atom_sglang_minimax-m3_mxfp4_distributed.json")
        self.assertEqual(sglang.params.driver, "sglang")
        self.assertEqual(sglang.gpu_arch, "mi355x")
        self.assertEqual(sglang.params.pipeline_parallel_size, "2")


if __name__ == "__main__":
    unittest.main()
