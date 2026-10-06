'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Behavior checks for the GLM-5.2-MXFP4 recipe stems.
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


class TestAtomRecipeGlm52Mxfp4(unittest.TestCase):
    def test_native_keeps_mi355_arch_and_mtp3_quant(self):
        perf = _load("mi3xx_atom_glm-5.2_mxfp4_single.json")
        mtp = _load("mi3xx_atom_glm-5.2_mxfp4_single.json", profile="mtp3")
        self.assertEqual(perf.gpu_arch, "mi355x")
        self.assertEqual(perf.model.id, "amd/GLM-5.2-MXFP4")
        self.assertEqual(perf.params.tensor_parallelism, "4")
        self.assertNotIn("--method", perf.roles.server.atom_args)
        quant = mtp.roles.server.atom_args[mtp.roles.server.atom_args.index("--online_quant_config") + 1]
        self.assertIn("model.layers.[0-9].mlp.*expert*", quant)
        self.assertEqual(mtp.roles.server.atom_args[mtp.roles.server.atom_args.index("--method") + 1], "mtp")
        self.assertEqual(perf.container.image, "<changeme>")

    def test_engines_keep_tp4_and_pp2(self):
        for backend in ("vllm", "sglang"):
            single = _load(f"mi3xx_atom_{backend}_glm-5.2_mxfp4_single.json")
            distributed = _load(f"mi3xx_atom_{backend}_glm-5.2_mxfp4_distributed.json")
            self.assertEqual(single.gpu_arch, "mi355x")
            self.assertEqual(single.model.id, "amd/GLM-5.2-MXFP4")
            self.assertEqual(single.params.tensor_parallelism, "4")
            self.assertEqual(single.params.nnodes, "1")
            self.assertEqual(distributed.params.nnodes, "2")
            self.assertEqual(distributed.params.pipeline_parallel_size, "2")
            for cell in distributed.expected_cells():
                self.assertIn(cell, distributed.thresholds)


if __name__ == "__main__":
    unittest.main()
