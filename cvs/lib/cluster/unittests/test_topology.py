"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

import unittest

from cvs.lib.cluster import node_matches
from cvs.lib.cluster.pool import Node


def _node(labels):
    return Node(ip="1", user="u", gpus=8, labels=labels)


class TestNodeMatches(unittest.TestCase):
    """G4 minimal surface: selector grammar is whitespace/comma-split required
    labels (logical AND); empty/None matches any node."""

    def test_empty_selector_matches_any(self):
        self.assertTrue(node_matches(_node(["mi300"]), None))
        self.assertTrue(node_matches(_node([]), ""))

    def test_single_label_required(self):
        self.assertTrue(node_matches(_node(["mi300", "ib"]), "mi300"))
        self.assertFalse(node_matches(_node(["mi355x"]), "mi300"))

    def test_multi_label_logical_and(self):
        n = _node(["mi300", "ib", "site42"])
        self.assertTrue(node_matches(n, "mi300 ib"))
        self.assertTrue(node_matches(n, "mi300,ib"))
        self.assertFalse(node_matches(n, "mi300 rocm7"))  # rocm7 absent -> no match


if __name__ == "__main__":
    unittest.main()
