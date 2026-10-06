'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Behavior checks for the MiMo-V2.5-Pro native recipe stem.
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


class TestAtomRecipeMimo(unittest.TestCase):
    def test_native_perf_is_tp8_on_mi355(self):
        variant = _load("mi3xx_atom_mimo-v2.5-pro_single.json")
        self.assertEqual(variant.params.driver, "atom")
        self.assertEqual(variant.model.id, "XiaomiMiMo/MiMo-V2.5-Pro")
        self.assertEqual(variant.gpu_arch, "mi355x")
        self.assertEqual(variant.params.tensor_parallelism, "8")
        args = variant.roles.server.atom_args
        self.assertEqual(args[args.index("--kv_cache_dtype") + 1], "fp8")
        self.assertIn("--trust-remote-code", args)
        self.assertNotIn("--method", args)
        self.assertEqual(variant.container.image, "<changeme>")
        for cell in variant.expected_cells():
            self.assertIn(cell, variant.thresholds)

    def test_mtp1_requests_one_speculative_token(self):
        variant = _load("mi3xx_atom_mimo-v2.5-pro_single.json", profile="mtp1")
        args = variant.roles.server.atom_args
        self.assertEqual(variant.model.id, "XiaomiMiMo/MiMo-V2.5-Pro")
        self.assertEqual(args[args.index("--method") + 1], "mtp")
        self.assertEqual(args[args.index("--num-speculative-tokens") + 1], "1")


if __name__ == "__main__":
    unittest.main()
