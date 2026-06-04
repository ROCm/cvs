"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Binder: deterministic first-N lexicographic claim across roles.

Pre-DTNI revert: no selector, no gpus_per_node gate. The cluster file is
"what hosts I can reach", and the binder just partitions the first
``sum(role.count)`` hostnames in lexicographic order across roles in
declaration order.
"""

import random
import unittest

from cvs.lib.cluster import ClusterPool, bind
from cvs.lib.config.base import Role, Topology


def _pool(hostnames):
    return ClusterPool.model_validate(
        {
            "username": "u",
            "priv_key_file": "/k",
            "nodes": {h: {"vpc_ip": f"10.0.0.{i}"} for i, h in enumerate(hostnames)},
        }
    )


class TestBinderDeterminism(unittest.TestCase):
    """Same pool contents -> same BindResult regardless of dict assembly
    order. Three shuffled orderings is the minimal proof per addendum gate."""

    def test_deterministic_across_3_randomized_orderings(self):
        base = ["n0", "n1", "n2", "n3"]
        topo = Topology(roles={"server": Role(count=2)})

        rng = random.Random(0)
        orderings = []
        while len(orderings) < 3:
            shuffled = base[:]
            rng.shuffle(shuffled)
            if shuffled not in orderings:
                orderings.append(shuffled)

        results = {repr(bind(topo, _pool(order)).bindings) for order in orderings}
        self.assertEqual(len(results), 1, f"non-deterministic bindings across orderings: {results}")


class TestBinderClaimsLexicographicFirstN(unittest.TestCase):
    """First N hostnames (sorted) go to the first role, next M to the
    second, etc."""

    def test_single_role_picks_lex_first_n(self):
        pool = _pool(["n3", "n0", "n2", "n1"])  # insertion order shuffled
        topo = Topology(roles={"server": Role(count=2)})
        res = bind(topo, pool)
        self.assertEqual(res.status, "bound")
        # Lex-sorted: n0, n1, n2, n3 -> first 2 = [n0, n1].
        self.assertEqual(res.bindings["server"], ["n0", "n1"])

    def test_multi_role_partitions_in_declaration_order(self):
        pool = _pool([f"n{i}" for i in range(4)])
        topo = Topology(
            roles={
                "prefill": Role(count=2),
                "decode": Role(count=2),
            }
        )
        res = bind(topo, pool)
        self.assertEqual(res.status, "bound")
        self.assertEqual(res.bindings["prefill"], ["n0", "n1"])
        self.assertEqual(res.bindings["decode"], ["n2", "n3"])


class TestBinderMultiRoleNoOverlap(unittest.TestCase):
    """The binder's load-bearing logic is no-overlap across roles."""

    def test_multi_role_no_overlap(self):
        pool = _pool([f"n{i}" for i in range(4)])
        topo = Topology(roles={"prefill": Role(count=2), "decode": Role(count=2)})
        res = bind(topo, pool)
        prefill = set(res.bindings["prefill"])
        decode = set(res.bindings["decode"])
        self.assertEqual(prefill & decode, set(), f"role overlap: {prefill & decode}")


class TestBinderSkipsWhenUnderResourced(unittest.TestCase):
    """When the pool has fewer hosts than ``sum(role.count)``, the binding
    is skipped with a concrete insufficient_nodes reason -- not raised,
    so small dev clusters get useful partial coverage."""

    def test_under_resourced_skipped_with_reason(self):
        pool = _pool([f"n{i}" for i in range(3)])
        topo = Topology(roles={"worker": Role(count=8)})
        res = bind(topo, pool)
        self.assertEqual(res.status, "skipped")
        self.assertIn("insufficient_nodes", res.reason)
        self.assertIn("need 8", res.reason)
        self.assertIn("pool has 3", res.reason)


if __name__ == "__main__":
    unittest.main()
