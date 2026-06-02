"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

# Unit tests for cvs/lib/failure_taxonomy.py: the closed five-category failure
# model and category_of dispatch.
#
# Pinned invariants:
#   - The category set and its priority order are fixed (lower priority wins
#     when several conditions hold), so a value typo or mid-list reorder fails.
#   - Every exception subclass binds its own category; the base defaults to
#     setup, and category_of maps an unclassified exception to setup.

import unittest

from cvs.lib.failure_taxonomy import (
    FailureCategory,
    FailurePatternMatched,
    LivenessFailure,
    SafetyViolation,
    SetupFailure,
    VerificationFailure,
    WorkloadFailure,
    category_of,
)


class TestFailureTaxonomy(unittest.TestCase):
    def test_five_disjoint_categories(self):
        # The taxonomy is closed and the values are exactly these five: pinning
        # the set catches a value typo, not just a count change.
        self.assertEqual(
            [c.value for c in FailureCategory],
            [
                "setup_failure",
                "safety_violation",
                "failure_pattern_matched",
                "liveness_failure",
                "verification_failure",
            ],
        )

    def test_priority_order(self):
        # Pin the exact category order and their priorities. This is what ranks
        # severity when several conditions hold at once (lower wins), so a
        # mid-list reorder must fail here rather than pass tautologically.
        self.assertEqual(
            [c.value for c in FailureCategory],
            [
                "setup_failure",
                "safety_violation",
                "failure_pattern_matched",
                "liveness_failure",
                "verification_failure",
            ],
        )
        self.assertEqual([c.priority for c in FailureCategory], [0, 1, 2, 3, 4])

    def test_exceptions_carry_category(self):
        # Every subclass binds its category (so the driver records exc.category
        # directly rather than guessing), and the base defaults to setup.
        self.assertEqual(SetupFailure("x").category, FailureCategory.SETUP_FAILURE)
        self.assertEqual(SafetyViolation("x").category, FailureCategory.SAFETY_VIOLATION)
        self.assertEqual(FailurePatternMatched("x").category, FailureCategory.FAILURE_PATTERN_MATCHED)
        self.assertEqual(LivenessFailure("x").category, FailureCategory.LIVENESS_FAILURE)
        self.assertEqual(VerificationFailure("x").category, FailureCategory.VERIFICATION_FAILURE)
        self.assertEqual(WorkloadFailure("x").category, FailureCategory.SETUP_FAILURE)

    def test_category_of_unknown_is_setup(self):
        # An exception raised outside a classified boundary (or a bare
        # WorkloadFailure) is a setup/harness failure; a classified one keeps the
        # category it declares.
        self.assertEqual(category_of(ValueError("oops")), FailureCategory.SETUP_FAILURE)
        self.assertEqual(category_of(WorkloadFailure("x")), FailureCategory.SETUP_FAILURE)
        self.assertEqual(category_of(SafetyViolation("x")), FailureCategory.SAFETY_VIOLATION)


if __name__ == "__main__":
    unittest.main()
