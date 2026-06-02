"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

# Unit tests for cvs/lib/config/sweep.py: the expand_sweep engine. expand_sweep
# accepts a plain dict, so these exercise the framework-agnostic expansion rules
# directly with neutral axis names; a framework's concrete axes (vLLM's
# concurrency x sequence_combinations) are covered in that framework's own suite.
#
# Pinned invariants:
#   - A scalar list is one cartesian axis; a list-of-objects is a paired bundle
#     whose fields co-vary as one option and cross cartesian-style with scalars.
#   - Cell ids are unique: named bundles use their name, unnamed ones get a
#     positional token (never the literal "None"), and a duplicate id raises.

import unittest

from cvs.lib.config import expand_sweep


class TestSweepExpansion(unittest.TestCase):
    def test_cartesian_times_paired(self):
        # Axis keys are sorted for deterministic ids: "scale" < "shape", so the
        # scalar token leads. One named bundle x two scalars -> two cells.
        cells = expand_sweep(
            {
                "scale": [16, 64],
                "shape": [{"isl": 1024, "osl": 1024, "name": "balanced"}],
            }
        )
        self.assertEqual(len(cells), 2)
        self.assertEqual({c.id for c in cells}, {"scale16-balanced", "scale64-balanced"})

    def test_paired_bundle_fields_covary(self):
        cells = {
            c.id: c.params
            for c in expand_sweep(
                {
                    "scale": [16],
                    "shape": [
                        {"isl": 1024, "osl": 1024, "name": "balanced"},
                        {"isl": 8192, "osl": 1024, "name": "long_context"},
                    ],
                }
            )
        }
        self.assertEqual(cells["scale16-long_context"]["isl"], 8192)

    def test_unnamed_combos_get_unique_positional_tokens(self):
        ids = [
            c.id
            for c in expand_sweep(
                {
                    "scale": [16],
                    "shape": [{"isl": 1, "osl": 1}, {"isl": 2, "osl": 2}],
                }
            )
        ]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn("scale16-None", ids)

    def test_duplicate_cell_ids_raise(self):
        with self.assertRaises(ValueError):
            expand_sweep({"scale": [16, 16], "shape": [{"isl": 1, "osl": 1, "name": "a"}]})


if __name__ == "__main__":
    unittest.main()
