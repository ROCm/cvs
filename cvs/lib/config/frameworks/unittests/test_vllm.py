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

from cvs.lib.config import parse_config
from cvs.lib.config.unittests._fixtures import make_base


class TestVllmConfig(unittest.TestCase):
    def test_dispatches_to_inference_kind(self):
        cfg = parse_config(make_base("vllm"))
        self.assertEqual(cfg.framework, "vllm")
        self.assertEqual(cfg.workload_kind, "inference")


if __name__ == "__main__":
    unittest.main()
