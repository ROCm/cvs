"""verdict.evaluate_threshold: kind handling, boundary equality, missing metric."""

import pytest

from cvs.lib.verdict import all_passed, evaluate_all, evaluate_threshold


def test_min_passes_at_equality():
    v = evaluate_threshold("tput", {"kind": "min", "value": 100.0}, 100.0)
    assert v.passed


def test_min_fails_below():
    v = evaluate_threshold("tput", {"kind": "min", "value": 100.0}, 99.999)
    assert not v.passed


def test_max_ms_passes_at_equality():
    v = evaluate_threshold("ttft", {"kind": "max_ms", "value": 200.0}, 200.0)
    assert v.passed


def test_max_ms_fails_above():
    v = evaluate_threshold("ttft", {"kind": "max_ms", "value": 200.0}, 200.0001)
    assert not v.passed


def test_within_both_bounds_inclusive():
    spec = {"kind": "within", "lo": 1.0, "hi": 2.0}
    assert evaluate_threshold("x", spec, 1.0).passed
    assert evaluate_threshold("x", spec, 2.0).passed
    assert not evaluate_threshold("x", spec, 0.999).passed
    assert not evaluate_threshold("x", spec, 2.001).passed


def test_missing_metric_fails_with_note():
    v = evaluate_threshold("absent", {"kind": "min", "value": 1.0}, None)
    assert not v.passed
    assert "not produced" in (v.note or "")


def test_unknown_kind_rejected():
    with pytest.raises(ValueError, match="unknown kind"):
        evaluate_threshold("x", {"kind": "gibberish", "value": 1}, 1.0)


def test_all_passed_aggregator():
    thresholds = {
        "a": {"kind": "min", "value": 10.0},
        "b": {"kind": "max_ms", "value": 100.0},
    }
    out = evaluate_all(thresholds, {"a": 11.0, "b": 50.0})
    assert all_passed(out)
    out2 = evaluate_all(thresholds, {"a": 11.0, "b": 200.0})
    assert not all_passed(out2)
