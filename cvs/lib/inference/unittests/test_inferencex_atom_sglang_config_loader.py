'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.inference.utils.inferencex_atom_sglang_config_loader.
'''

import unittest
from pathlib import Path

from cvs.lib.inference.utils.inferencex_atom_sglang_config_loader import load_variant


def _cluster_dict():
    return {"username": "testuser"}


class TestInferenceXAtomSglangConfigLoader(unittest.TestCase):
    def test_load_w1_sglang_parity_variant(self):
        root = Path(__file__).resolve().parents[3]
        config = root / (
            "input/config_file/inference/inferencex_atom_sglang_single/"
            "deepseek_r1_fp8_mi300x_atom_sglang_perf/deepseek_r1_fp8_mi300x_atom_sglang_perf_config.json"
        )
        variant = load_variant(config, _cluster_dict())
        self.assertEqual(variant.framework, "inferencex_atom_sglang_single")
        self.assertEqual(variant.model.id, "deepseek-ai/DeepSeek-R1-0528")
        self.assertIn("--kv-cache-dtype", variant.roles.server.sglang_args)
        self.assertEqual(
            variant.expected_cells(),
            ["ISL=1024,OSL=1024,TP=8,CONC=128", "ISL=1024,OSL=1024,TP=8,CONC=256"],
        )


if __name__ == "__main__":
    unittest.main()
