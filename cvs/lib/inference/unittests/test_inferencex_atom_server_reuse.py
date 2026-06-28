'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for InferenceX ATOM server-reuse helpers and sweep parametrization.
'''

import unittest
from types import SimpleNamespace

from cvs.lib.inference.utils.inferencex_atom_config_loader import (
    expand_sweep_parametrize,
    reuse_server_flag,
    server_session_key,
)
from cvs.lib.inference.utils.inferencex_atom_parsing import METRIC_TIER_ORDER


class TestServerReuseHelpers(unittest.TestCase):
    def test_reuse_server_flag_truthy_values(self):
        for raw in ("true", "1", "yes", "TRUE", " Yes "):
            params = SimpleNamespace(reuse_server_across_sweep=raw)
            self.assertTrue(reuse_server_flag(params), raw)

    def test_reuse_server_flag_falsey_values(self):
        for raw in ("false", "0", "no", "", "maybe"):
            params = SimpleNamespace(reuse_server_across_sweep=raw)
            self.assertFalse(reuse_server_flag(params), raw)

    def test_reuse_server_flag_defaults_false_when_missing(self):
        self.assertFalse(reuse_server_flag(SimpleNamespace()))

    def test_server_session_key_differs_for_model(self):
        base = SimpleNamespace(
            model=SimpleNamespace(id="model-a"),
            params=SimpleNamespace(driver="atom", tensor_parallelism="8"),
            ix_recipe_id="recipe-a",
        )
        other = SimpleNamespace(
            model=SimpleNamespace(id="model-b"),
            params=SimpleNamespace(driver="atom", tensor_parallelism="8"),
            ix_recipe_id="recipe-a",
        )
        k1 = server_session_key(base, "1024", "1024")
        k2 = server_session_key(other, "1024", "1024")
        self.assertNotEqual(k1, k2)

    def test_server_session_key_differs_for_shape(self):
        variant = SimpleNamespace(
            model=SimpleNamespace(id="model-a"),
            params=SimpleNamespace(driver="atom", tensor_parallelism="8"),
            ix_recipe_id="recipe-a",
        )
        self.assertNotEqual(
            server_session_key(variant, "1024", "1024"),
            server_session_key(variant, "2048", "2048"),
        )


class TestExpandSweepParametrize(unittest.TestCase):
    def test_metric_tier_expansion_multiplies_cases(self):
        sweep = {
            "sequence_combinations": [{"name": "w1_1k_1k", "isl": "1024", "osl": "1024"}],
            "runs": [{"combo": "w1_1k_1k", "concurrency": 128}],
        }
        spec = expand_sweep_parametrize(sweep, ("metric_tier",))
        argnames, argvalues, ids = spec
        self.assertEqual(argnames, "seq_combo,concurrency,metric_tier")
        self.assertEqual(len(argvalues), len(METRIC_TIER_ORDER))
        self.assertEqual(len(ids), len(METRIC_TIER_ORDER))
        self.assertEqual(ids[0], "w1_1k_1k-conc128-throughput")

    def test_inference_only_parametrize_without_metric_tier(self):
        sweep = {
            "sequence_combinations": [{"name": "w1_1k_1k", "isl": "1024", "osl": "1024"}],
            "runs": [
                {"combo": "w1_1k_1k", "concurrency": 128},
                {"combo": "w1_1k_1k", "concurrency": 256},
            ],
        }
        _, argvalues, ids = expand_sweep_parametrize(
            sweep, ("seq_combo", "concurrency")
        )
        self.assertEqual(len(argvalues), 2)
        self.assertEqual(ids, ["w1_1k_1k-conc128", "w1_1k_1k-conc256"])


if __name__ == "__main__":
    unittest.main()
