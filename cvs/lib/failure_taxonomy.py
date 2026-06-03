"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import enum
from typing import Optional


class FailureCategory(str, enum.Enum):
    """The five disjoint failure categories, in priority order.

    Failures are classified *at the boundary where they originate* (the raise
    site), never by post-hoc inspection of a stack trace or by scanning console
    output for the word "error". When more than one condition is present, the
    earliest member in this declaration order wins.
    """

    SETUP_FAILURE = "setup_failure"
    SAFETY_VIOLATION = "safety_violation"
    FAILURE_PATTERN_MATCHED = "failure_pattern_matched"
    LIVENESS_FAILURE = "liveness_failure"
    VERIFICATION_FAILURE = "verification_failure"


class WorkloadFailure(Exception):
    """Base for all classified workload failures.

    Each subclass binds a single :class:`FailureCategory`. The ``Job`` driver
    catches ``WorkloadFailure`` and records ``exc.category`` directly; it never
    guesses the category from the message.
    """

    category: FailureCategory = FailureCategory.SETUP_FAILURE

    def __init__(self, message: str, *, detail: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class SetupFailure(WorkloadFailure):
    """``prepare`` or ``launch`` raised before the workload started running."""

    category = FailureCategory.SETUP_FAILURE


class SafetyViolation(WorkloadFailure):
    """The progress predicate broke mid-run (server died, role crashed, ...)."""

    category = FailureCategory.SAFETY_VIOLATION


class FailurePatternMatched(WorkloadFailure):
    """A pattern from ``failure_patterns.yaml`` hit a monitored log stream."""

    category = FailureCategory.FAILURE_PATTERN_MATCHED


class LivenessFailure(WorkloadFailure):
    """``await_completion`` timed out without the progress predicate breaking."""

    category = FailureCategory.LIVENESS_FAILURE


class VerificationFailure(WorkloadFailure):
    """At least one threshold evaluated to False at end-of-test."""

    category = FailureCategory.VERIFICATION_FAILURE


def category_of(exc: BaseException) -> FailureCategory:
    """Map an exception to a category.

    A :class:`WorkloadFailure` carries its category explicitly. Any other
    exception escaping the lifecycle is treated as a setup failure (it occurred
    outside a classified boundary, so it is, by definition, a harness/setup
    problem rather than a workload verdict).
    """
    if isinstance(exc, WorkloadFailure):
        return exc.category
    return FailureCategory.SETUP_FAILURE
