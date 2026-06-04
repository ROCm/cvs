"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

# Unit tests for cvs/lib/config/base.py: the framework-agnostic schema primitives
# (Role/Topology/ContainerSpec/BaseTestConfig). Behaviors that live on the base
# are parametrized over every framework fixture via iter_bases(), so a new
# framework re-proves them automatically; vLLM-specific rules live in
# frameworks/unittests/test_vllm.py.
#
# Pinned invariants:
#   - Hash identity: workload_hash ignores container.env (the HF token is not
#     workload-defining) while config_hash retains it; seed IS workload-defining.
#   - ContainerSpec is fail-closed at load: it stringifies str/int tokens but
#     rejects None/bool/list/float rather than emitting a malformed docker arg.
#   - Topology must declare at least one role.

import unittest
from pathlib import Path

from pydantic import ValidationError

from cvs.lib.config import ContainerSpec, parse_config
from cvs.lib.config.loader import ConfigError

from ._fixtures import iter_bases


class TestConfigHashing(unittest.TestCase):
    def test_workload_hash_includes_container(self):
        # Pre-DTNI revert: container IS workload-defining (image, devices,
        # env all materially change the run). Two configs differing only in
        # container.env (e.g. a different HF_TOKEN, a different VLLM_USE_V1)
        # must hash differently for reuse-manifest soundness.
        for framework, base in iter_bases():
            with self.subTest(framework=framework):
                base_with_image = {
                    **base,
                    "container": {**base.get("container", {}), "image": "rocm/vllm-dev:nightly"},
                }
                a = parse_config(base_with_image)
                b = parse_config(
                    {
                        **base_with_image,
                        "container": {
                            **base_with_image["container"],
                            "env": {"HF_TOKEN": "different"},
                        },
                    }
                )
                self.assertNotEqual(a.workload_hash(), b.workload_hash())
                # config_hash still distinguishes (it always did); verification
                # is unaffected because thresholds/benchmarks did not change.
                self.assertNotEqual(a.config_hash(), b.config_hash())
                self.assertEqual(a.verification_hash(), b.verification_hash())

    def test_workload_hash_includes_fabric(self):
        # Pre-DTNI: fabric (NCCL/UCX/Gloo) is workload-defining. A different
        # NCCL_IB_HCA selection is materially a different workload (different
        # collective performance, potentially different correctness on
        # heterogeneous fabrics).
        for framework, base in iter_bases():
            with self.subTest(framework=framework):
                a = parse_config(base)
                b = parse_config({**base, "fabric": {"nccl_ib_hca": "bnxt_re0"}})
                self.assertNotEqual(a.workload_hash(), b.workload_hash())

    def test_seed_is_workload_defining(self):
        # A different seed yields different numerical results, so it must change
        # workload identity (otherwise two runs collide in any hash-keyed cache).
        for framework, base in iter_bases():
            with self.subTest(framework=framework):
                a = parse_config(base)
                b = parse_config({**base, "seed": 7})
                self.assertNotEqual(a.workload_hash(), b.workload_hash())
                self.assertNotEqual(a.config_hash(), b.config_hash())


class TestContainerSpec(unittest.TestCase):
    def test_stringifies_non_str_values(self):
        spec = ContainerSpec(ports={8888: 8888}, volume_dict={Path("/weights"): "/models"}, device_list=[0, 1])
        self.assertEqual(spec.ports, {"8888": "8888"})
        self.assertEqual(spec.volume_dict, {"/weights": "/models"})
        self.assertEqual(spec.device_list, ["0", "1"])

    def test_rejects_non_token_values(self):
        # Fail-closed: None/bool/list must error at load, never become
        # "None"/"True"/"['x']" and emit a malformed docker arg.
        for bad in ({"X": None}, {"X": True}, {"X": ["a"]}):
            with self.assertRaises(ValidationError):
                ContainerSpec(env=bad)
        with self.assertRaises(ValidationError):
            ContainerSpec(device_list=[None])

    def test_float_token_rejected(self):
        # A float port (8888.0) would stringify to a malformed "-p 8888.0:..".
        with self.assertRaises(ValidationError):
            ContainerSpec(ports={8888.0: 8888})
        with self.assertRaises(ValidationError):
            ContainerSpec(env={"RATIO": 0.8})
        with self.assertRaises(ValidationError):
            ContainerSpec(device_list=[0.5])


class TestTopology(unittest.TestCase):
    def test_empty_roles_rejected(self):
        for framework, base in iter_bases():
            with self.subTest(framework=framework):
                with self.assertRaises(ConfigError):
                    parse_config({**base, "topology": {"roles": {}}})


if __name__ == "__main__":
    unittest.main()
