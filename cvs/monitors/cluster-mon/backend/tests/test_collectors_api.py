"""
Tests for GET /api/collectors/status endpoint logic.
"""
import pytest
from unittest.mock import MagicMock

from app.api.collectors import _compute_overall_status
from app.collectors.base import CollectorResult, CollectorState


def _make_result(state: CollectorState, name: str = "gpu") -> CollectorResult:
    return CollectorResult(
        collector_name=name,
        timestamp="2026-01-01T00:00:00+00:00",
        state=state,
        data={},
    )


def test_overall_status_healthy_when_all_ok():
    results = {
        "gpu": _make_result(CollectorState.OK, "gpu"),
        "nic": _make_result(CollectorState.OK, "nic"),
    }
    meta = {"gpu": {"critical": True}, "nic": {"critical": True}}
    assert _compute_overall_status(results, meta) == "healthy"


def test_overall_status_healthy_when_no_results():
    assert _compute_overall_status({}, {}) == "healthy"


def test_overall_status_healthy_with_no_service():
    """NO_SERVICE (e.g., no RCCL job) is not an error — still healthy."""
    results = {
        "gpu": _make_result(CollectorState.OK, "gpu"),
        "rccl": _make_result(CollectorState.NO_SERVICE, "rccl"),
    }
    meta = {"gpu": {"critical": True}, "rccl": {"critical": False}}
    assert _compute_overall_status(results, meta) == "healthy"


def test_overall_status_critical_when_critical_collector_errors():
    results = {
        "gpu": _make_result(CollectorState.ERROR, "gpu"),
        "nic": _make_result(CollectorState.OK, "nic"),
    }
    meta = {"gpu": {"critical": True}, "nic": {"critical": True}}
    assert _compute_overall_status(results, meta) == "critical"


def test_overall_status_degraded_when_non_critical_errors():
    results = {
        "gpu": _make_result(CollectorState.OK, "gpu"),
        "rccl": _make_result(CollectorState.ERROR, "rccl"),
    }
    meta = {"gpu": {"critical": True}, "rccl": {"critical": False}}
    assert _compute_overall_status(results, meta) == "degraded"


def test_overall_status_critical_on_unreachable_critical():
    results = {
        "gpu": _make_result(CollectorState.UNREACHABLE, "gpu"),
    }
    meta = {"gpu": {"critical": True}}
    assert _compute_overall_status(results, meta) == "critical"
