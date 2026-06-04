"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

B3 regression: ``Topology`` accepts a single-role shorthand
``{nnodes, gpus_per_node, selector}`` that expands to
``{roles: {server: {count, gpus_per_node, selector}}}`` at load time. The
explicit ``roles`` form continues to load unchanged; mixing the two is
rejected.
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from cvs.lib.config.base import Topology


class TestTopologyShorthand(unittest.TestCase):
    def test_shorthand_expands_to_single_server_role(self):
        t = Topology.model_validate({"nnodes": 1, "gpus_per_node": 8, "selector": "mi300"})
        self.assertIn("server", t.roles)
        self.assertEqual(t.roles["server"].count, 1)
        self.assertEqual(t.roles["server"].gpus_per_node, 8)
        self.assertEqual(t.roles["server"].selector, "mi300")

    def test_shorthand_and_explicit_roles_equal_in_memory(self):
        short = Topology.model_validate({"nnodes": 1, "gpus_per_node": 8, "selector": "mi300"})
        full = Topology.model_validate({"roles": {"server": {"count": 1, "gpus_per_node": 8, "selector": "mi300"}}})
        self.assertEqual(short.model_dump(), full.model_dump())

    def test_explicit_roles_still_load(self):
        t = Topology.model_validate({"roles": {"server": {"count": 1, "gpus_per_node": 8, "selector": "mi300"}}})
        self.assertEqual(t.roles["server"].count, 1)

    def test_mixing_shorthand_and_roles_rejected(self):
        with self.assertRaises(ValidationError):
            Topology.model_validate(
                {
                    "nnodes": 1,
                    "roles": {"server": {"count": 1, "gpus_per_node": 8}},
                }
            )

    def test_shorthand_defaults(self):
        # selector-less shorthand still works (no GPU class filter at the binder)
        t = Topology.model_validate({"nnodes": 2, "gpus_per_node": 4})
        self.assertEqual(t.roles["server"].count, 2)
        self.assertIsNone(t.roles["server"].selector)


if __name__ == "__main__":
    unittest.main()
