"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Topology shorthand: ``{nnodes: N}`` expands to
``{roles: {server: {count: N}}}`` at load time. The explicit ``roles``
form continues to load unchanged; mixing the two is rejected.

Pre-DTNI revert: ``gpus_per_node`` and ``selector`` are no longer on the
``Role`` schema (GPU gating is runtime via rocm-smi probe; selector gating
dropped entirely). Shorthand is just ``nnodes``.
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from cvs.lib.config.base import Topology


class TestTopologyShorthand(unittest.TestCase):
    def test_shorthand_expands_to_single_server_role(self):
        t = Topology.model_validate({"nnodes": 1})
        self.assertIn("server", t.roles)
        self.assertEqual(t.roles["server"].count, 1)

    def test_shorthand_and_explicit_roles_equal_in_memory(self):
        short = Topology.model_validate({"nnodes": 1})
        full = Topology.model_validate({"roles": {"server": {"count": 1}}})
        self.assertEqual(short.model_dump(), full.model_dump())

    def test_explicit_roles_still_load(self):
        t = Topology.model_validate({"roles": {"master": {"count": 1}, "worker": {"count": 3}}})
        self.assertEqual(t.roles["master"].count, 1)
        self.assertEqual(t.roles["worker"].count, 3)

    def test_mixing_shorthand_and_roles_rejected(self):
        with self.assertRaises(ValidationError):
            Topology.model_validate({"nnodes": 1, "roles": {"server": {"count": 1}}})

    def test_shorthand_nnodes_greater_than_one(self):
        t = Topology.model_validate({"nnodes": 4})
        self.assertEqual(t.roles["server"].count, 4)


if __name__ == "__main__":
    unittest.main()
