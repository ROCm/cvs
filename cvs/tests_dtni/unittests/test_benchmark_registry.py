"""Unit tests for cvs.lib.benchmarks.registry."""

from __future__ import annotations

import pytest

from cvs.lib.benchmarks.registry import (
    BENCHMARK_REGISTRY,
    BenchmarkSpec,
    list_benchmarks,
    lookup,
)


def test_registry_contains_accuracy_and_perf_sets():
    keys = set(BENCHMARK_REGISTRY)
    # accuracy
    assert {"mmlu", "gsm8k"} <= keys
    # perf (vllm-bench-serve harness)
    assert {"serve_synth_short", "serve_synth_long"} <= keys


def test_specs_are_well_formed():
    for bid, spec in BENCHMARK_REGISTRY.items():
        assert isinstance(spec, BenchmarkSpec)
        assert spec.id == bid
        assert spec.harness in {"lm-eval-harness", "vllm-bench-serve"}
        assert spec.shots >= 0
        assert isinstance(spec.extra, dict)


def test_lookup_returns_spec():
    spec = lookup("mmlu")
    assert spec.score_metric == "acc"
    assert spec.shots == 5
    assert spec.extra["task"] == "mmlu"


def test_perf_spec_has_no_score_metric():
    # Perf benchmarks don't use the lm-eval scoring fields.
    spec = lookup("serve_synth_short")
    assert spec.score_metric is None
    assert spec.extra["dataset_name"] == "random"
    assert spec.extra["num_prompts"] == 200


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
