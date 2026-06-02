"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

# Unit tests for cvs/lib/config/frameworks/vllm.py: vLLM-specific schema rules.
# Generic spine behavior (loader/sweep/threshold/hash) is proven in
# cvs/lib/config/unittests/ over every framework base; this file pins only the
# invariants that live on the vLLM config classes themselves.
#
# Pinned invariants:
#   - vLLM dispatches to an inference-kind config and its concurrency x
#     sequence_combinations sweep expands to the expected cell ids.
#   - VllmConfig._tp_consistent_with_topology rejects a tensor_parallelism sweep
#     value that no role's gpus_per_node can satisfy (the topology is fixed).
#   - VllmSweepParams enforces positive concurrency and a non-empty
#     sequence_combinations list.

import unittest

from cvs.lib.config import expand_sweep, parse_config
from cvs.lib.config.loader import ConfigError
from cvs.lib.config.unittests._fixtures import make_base


class TestVllmConfig(unittest.TestCase):
    def test_dispatches_to_inference_kind(self):
        cfg = parse_config(make_base("vllm"))
        self.assertEqual(cfg.framework, "vllm")
        self.assertEqual(cfg.workload_kind, "inference")

    def test_sweep_cell_ids(self):
        # End-to-end through VllmSweepParams: the concurrency cartesian axis x the
        # named sequence_combinations bundle yields one cell per concurrency.
        cfg = parse_config(make_base("vllm"))
        cells = expand_sweep(cfg.sweep)
        self.assertEqual(len(cells), 2)
        self.assertEqual({c.id for c in cells}, {"concurrency16-balanced", "concurrency64-balanced"})


class TestVllmTpConsistency(unittest.TestCase):
    def test_tp_matching_gpus_per_node_accepted(self):
        # The base server role has gpus_per_node=8, so a TP sweep of [8] is valid.
        base = make_base("vllm")
        base["sweep"]["tensor_parallelism"] = [8]
        cfg = parse_config(base)
        self.assertEqual(cfg.sweep.tensor_parallelism, [8])

    def test_tp_diverging_from_topology_rejected(self):
        # TP=4 matches no role's gpus_per_node (8); the fixed topology cannot
        # satisfy it, so the config is rejected at load.
        base = make_base("vllm")
        base["sweep"]["tensor_parallelism"] = [4]
        with self.assertRaises(ConfigError):
            parse_config(base)


class TestVllmSweepAxes(unittest.TestCase):
    def test_concurrency_must_be_positive(self):
        for bad in ([0], [-1], [16, -4]):
            base = make_base("vllm")
            base["sweep"]["concurrency"] = bad
            with self.assertRaises(ConfigError):
                parse_config(base)

    def test_sequence_combinations_required_nonempty(self):
        base = make_base("vllm")
        base["sweep"]["sequence_combinations"] = []
        with self.assertRaises(ConfigError):
            parse_config(base)


if __name__ == "__main__":
    unittest.main()
