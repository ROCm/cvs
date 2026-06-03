"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

import random
import unittest

from cvs.lib.cluster import ClusterPool, bind
from cvs.lib.config.base import Role, Topology


def _pool_from_order(hostnames):
    return ClusterPool.model_validate(
        {
            "nodes": {
                h: {"ip": f"10.0.0.{i}", "user": "u", "gpus": 8, "labels": ["mi300"]} for i, h in enumerate(hostnames)
            }
        }
    )


class TestBinderDeterminism(unittest.TestCase):
    """G4 surface: same pool contents -> same BindResult regardless of how the
    caller assembled the node dict. Three shuffled orderings is the minimal
    proof per addendum gate row (the spike-blob x100 paranoia is dropped per
    addendum 6.1)."""

    def test_deterministic_across_3_randomized_orderings(self):
        base = ["n0", "n1", "n2", "n3"]
        topo = Topology(roles={"server": Role(count=2, gpus_per_node=8, selector="mi300")})

        rng = random.Random(0)
        orderings = []
        while len(orderings) < 3:
            shuffled = base[:]
            rng.shuffle(shuffled)
            if shuffled not in orderings:
                orderings.append(shuffled)

        results = {repr(bind(topo, _pool_from_order(order)).bindings) for order in orderings}
        self.assertEqual(len(results), 1, f"non-deterministic bindings across orderings: {results}")


class TestBinderSkipWithReason(unittest.TestCase):
    """G4 surface: an under-resourced cell is reported skipped with a concrete
    insufficient_nodes reason (not a hard error -- small dev clusters get
    partial coverage)."""

    def test_under_resourced_skipped_with_insufficient_nodes_reason(self):
        pool = ClusterPool.model_validate(
            {"nodes": {f"n{i}": {"ip": f"10.0.0.{i}", "user": "u", "gpus": 8, "labels": ["mi300"]} for i in range(3)}}
        )
        topo = Topology(roles={"d": Role(count=8, gpus_per_node=8, selector="mi300")})
        res = bind(topo, pool)
        self.assertEqual(res.status, "skipped")
        self.assertIsNotNone(res.reason)
        self.assertIn("insufficient_nodes", res.reason)


class TestBinderSelectorMismatchSkips(unittest.TestCase):
    """G4 surface: a role whose selector matches no node yields skipped (the
    binder-driven complement to test_topology.test_node_matches)."""

    def test_selector_mismatch_skips(self):
        pool = ClusterPool.model_validate(
            {"nodes": {"n0": {"ip": "10.0.0.0", "user": "u", "gpus": 8, "labels": ["mi300"]}}}
        )
        topo = Topology(roles={"s": Role(count=1, gpus_per_node=8, selector="mi355x")})
        res = bind(topo, pool)
        self.assertEqual(res.status, "skipped")


class TestBinderMultiRoleNoOverlap(unittest.TestCase):
    """G4 surface: the binder's actual non-trivial logic is claim-tracking
    across roles -- two roles bound from the same pool must not share a
    hostname. This is the test that catches a regression in claimed.update()
    or the ``if hostname in claimed: continue`` guard."""

    def test_multi_role_no_overlap(self):
        pool = ClusterPool.model_validate(
            {"nodes": {f"n{i}": {"ip": f"10.0.0.{i}", "user": "u", "gpus": 8, "labels": ["mi300"]} for i in range(4)}}
        )
        topo = Topology(
            roles={
                "prefill": Role(count=2, gpus_per_node=8, selector="mi300"),
                "decode": Role(count=2, gpus_per_node=8, selector="mi300"),
            }
        )
        res = bind(topo, pool)
        self.assertEqual(res.status, "bound")
        prefill = set(res.bindings["prefill"])
        decode = set(res.bindings["decode"])
        self.assertEqual(len(prefill), 2)
        self.assertEqual(len(decode), 2)
        self.assertEqual(prefill & decode, set(), f"role overlap: {prefill & decode}")


if __name__ == "__main__":
    unittest.main()
