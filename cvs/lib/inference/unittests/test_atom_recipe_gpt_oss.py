'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Behavior checks for the GPT-OSS-120B vLLM PP=2 stem.
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
