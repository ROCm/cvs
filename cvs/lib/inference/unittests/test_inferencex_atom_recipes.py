'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import unittest

from cvs.lib.inference.utils.inferencex_atom_recipes import apply_ix_recipe, get_recipe


class TestInferenceXAtomRecipes(unittest.TestCase):
    def test_dsr1_fp8_mi300x_recipe_args(self):
        recipe = get_recipe("dsr1-fp8-mi300x-atom")
        self.assertEqual(recipe["gpu_arch"], "mi300x")
        self.assertIn("-tp", recipe["atom_args"])
        self.assertIn("--kv_cache_dtype", recipe["atom_args"])

    def test_apply_ix_recipe_merges_server_args(self):
        raw = {
            "gpu_arch": "mi300x",
            "ix_recipe_id": "dsr1-fp8-mi300x-atom",
            "model": {"id": "deepseek-ai/DeepSeek-R1-0528"},
            "roles": {"server": {}},
            "params": {},
        }
        out = apply_ix_recipe(raw)
        self.assertTrue(out["roles"]["server"]["atom_args"])

    def test_mtp3_recipe_bench_extra_args(self):
        raw = {
            "gpu_arch": "mi355x",
            "ix_recipe_id": "dsr1-fp8-mi355x-atom-mtp3",
            "model": {"id": "deepseek-ai/DeepSeek-R1-0528"},
            "roles": {"server": {}},
            "params": {},
        }
        out = apply_ix_recipe(raw)
        self.assertIn("--method", out["roles"]["server"]["atom_args"])
        self.assertIn("--use-chat-template", out["params"]["bench_extra_args"])

    def test_gpu_arch_mismatch_raises(self):
        raw = {
            "gpu_arch": "mi355x",
            "ix_recipe_id": "dsr1-fp8-mi300x-atom",
            "model": {"id": "deepseek-ai/DeepSeek-R1-0528"},
        }
        with self.assertRaises(ValueError):
            apply_ix_recipe(raw)


if __name__ == "__main__":
    unittest.main()
