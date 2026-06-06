"""Unit tests for cvs.lib.benchmarks.registry."""

from __future__ import annotations

import pytest

from cvs.lib.benchmarks.registry import (
    BENCHMARK_REGISTRY,
    BenchmarkSpec,
    list_benchmarks,
    lookup,
)


def test_registry_contains_v1_accuracy_set():
    assert set(BENCHMARK_REGISTRY) == {"mmlu", "gsm8k"}


def test_specs_are_well_formed():
    for bid, spec in BENCHMARK_REGISTRY.items():
        assert isinstance(spec, BenchmarkSpec)
        assert spec.id == bid
        assert spec.harness == "lm-eval-harness"
        assert spec.dataset_id
        assert spec.score_metric
        assert spec.shots >= 0
        # frozen dataclass: extra is a dict but immutable spec
        assert isinstance(spec.extra, dict)


def test_lookup_returns_spec():
    spec = lookup("mmlu")
    assert spec.score_metric == "acc"
    assert spec.shots == 5
    assert spec.extra["task"] == "mmlu"


def test_lookup_unknown_raises_with_known_list():
    with pytest.raises(KeyError) as exc_info:
        lookup("mmmlu")
    msg = str(exc_info.value)
    assert "mmmlu" in msg
    assert "gsm8k" in msg
    assert "mmlu" in msg


def test_list_benchmarks_is_stable():
    ids = list_benchmarks()
    assert isinstance(ids, tuple)
    assert set(ids) == set(BENCHMARK_REGISTRY)
