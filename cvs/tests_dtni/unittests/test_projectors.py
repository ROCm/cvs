"""Unit tests for cvs.lib.benchmarks.projectors."""

from __future__ import annotations

import pytest

from cvs.lib.benchmarks.projectors import _project_lm_eval, _project_vllm_bench_serve, project
from cvs.lib.benchmarks.registry import BenchmarkSpec, lookup


# ---- lm-eval --------------------------------------------------------------


def test_lm_eval_bare_metric_key():
    payload = {"results": {"mmlu": {"acc": 0.732, "acc_stderr": 0.011}}}
    out = _project_lm_eval(lookup("mmlu"), payload)
    assert out == {"mmlu": 0.732, "mmlu_stderr": 0.011}


def test_lm_eval_prefers_filter_suffix_for_gsm8k():
    payload = {"results": {"gsm8k": {
        "exact_match,strict-match": 0.812,
        "exact_match_stderr,strict-match": 0.009,
        "exact_match,none": 0.99,  # red herring
    }}}
    out = _project_lm_eval(lookup("gsm8k"), payload)
    assert out == {"gsm8k": 0.812, "gsm8k_stderr": 0.009}


def test_lm_eval_missing_task_returns_empty():
    out = _project_lm_eval(lookup("mmlu"), {"results": {}})
    assert out == {}


def test_lm_eval_skips_non_numeric_metric():
    payload = {"results": {"mmlu": {"acc": "n/a"}}}
    out = _project_lm_eval(lookup("mmlu"), payload)
    assert out == {}


# ---- vllm-bench-serve -----------------------------------------------------


def _serve_spec(id: str = "serve_x") -> BenchmarkSpec:
    return BenchmarkSpec(
        id=id,
        harness="vllm-bench-serve",
        extra={"dataset_name": "random", "num_prompts": 10,
               "random_input_len": 64, "random_output_len": 64},
    )


def test_vllm_serve_scalars_namespaced_by_benchmark_id():
    payload = {
        "completed": 200,
        "request_throughput": 12.5,
        "request_goodput": 11.8,
        "output_throughput": 4321.0,
        "total_token_throughput": 6543.2,
        "mean_ttft_ms": 71.5,
        "median_ttft_ms": 73.8,
        "std_ttft_ms": 2.1,
        "mean_tpot_ms": 7.91,
    }
    out = _project_vllm_bench_serve(_serve_spec("svc"), payload)
    assert out["svc.completed"] == 200.0
    assert out["svc.request_throughput"] == 12.5
    assert out["svc.request_goodput"] == 11.8
    assert out["svc.output_throughput"] == 4321.0
    assert out["svc.total_token_throughput"] == 6543.2
    assert out["svc.mean_ttft_ms"] == 71.5
    assert out["svc.median_ttft_ms"] == 73.8
    assert out["svc.std_ttft_ms"] == 2.1
    assert out["svc.mean_tpot_ms"] == 7.91
    # spec.id prefix means two invocations don't collide.
    assert all(k.startswith("svc.") for k in out)


def test_vllm_serve_percentile_lists_unrolled_to_named_scalars():
    payload = {
        "percentiles_ttft_ms": [[50, 73.8], [95, 84.2], [99, 91.0]],
        "percentiles_tpot_ms": [[50, 7.96], [99, 8.03]],
        "percentiles_itl_ms":  [[95, 9.5]],
        "percentiles_e2el_ms": [[50, 2100.0], [95, 2400.0]],
    }
    out = _project_vllm_bench_serve(_serve_spec("svc"), payload)
    assert out["svc.ttft_p50_ms"] == 73.8
    assert out["svc.ttft_p95_ms"] == 84.2
    assert out["svc.ttft_p99_ms"] == 91.0
    assert out["svc.tpot_p50_ms"] == 7.96
    assert out["svc.tpot_p99_ms"] == 8.03
    assert out["svc.itl_p95_ms"] == 9.5
    assert out["svc.e2el_p50_ms"] == 2100.0
    assert out["svc.e2el_p95_ms"] == 2400.0


def test_vllm_serve_skips_non_numeric_and_malformed_pctls():
    payload = {
        "completed": "abc",                       # non-numeric → dropped
        "percentiles_ttft_ms": [
            [50, "not-a-number"],                  # bad value
            ["bogus", 100],                        # bad percentile
            [95, 84.2],                            # good
            "not-a-tuple",                         # bad entry
        ],
        "percentiles_tpot_ms": "not-a-list",      # bad container
    }
    out = _project_vllm_bench_serve(_serve_spec("svc"), payload)
    assert "svc.completed" not in out
    assert out == {"svc.ttft_p95_ms": 84.2}


def test_vllm_serve_empty_payload_returns_empty_dict():
    assert _project_vllm_bench_serve(_serve_spec(), {}) == {}


def test_vllm_serve_bool_field_is_not_coerced_to_scalar():
    # bool is a subclass of int in Python — `isinstance(True, int)` is True.
    # Without an explicit bool guard the threshold engine would see 1.0/0.0
    # as a real measurement.
    payload = {
        "completed": True,
        "percentiles_ttft_ms": [[95, False]],
    }
    out = _project_vllm_bench_serve(_serve_spec("svc"), payload)
    assert out == {}


def test_lm_eval_bool_field_is_not_coerced_to_scalar():
    payload = {"results": {"mmlu": {"acc": True}}}
    out = _project_lm_eval(lookup("mmlu"), payload)
    assert out == {}


def test_vllm_serve_flat_pNN_metric_ms_keys_are_unrolled():
    # Current vllm bench serve writes percentiles as flat keys, not lists.
    # Both p50_ttft_ms and p95_ttft_ms should land in the namespaced form.
    payload = {
        "p50_ttft_ms": 138.0,
        "p95_ttft_ms": 226.1,
        "p99_tpot_ms": 7.94,
        "p95_e2el_ms": 2400.0,
        # red herrings -- must be ignored
        "p50_unknown_ms": 999.0,        # metric not in known set
        "p1000_ttft_ms": True,           # bool value, must be skipped
    }
    out = _project_vllm_bench_serve(_serve_spec("svc"), payload)
    assert out == {
        "svc.ttft_p50_ms": 138.0,
        "svc.ttft_p95_ms": 226.1,
        "svc.tpot_p99_ms": 7.94,
        "svc.e2el_p95_ms": 2400.0,
    }


# ---- dispatch -------------------------------------------------------------


def test_dispatch_routes_by_harness():
    payload = {"results": {"mmlu": {"acc": 0.7}}}
    out = project(lookup("mmlu"), payload)
    assert out == {"mmlu": 0.7}


def test_dispatch_unknown_harness_raises():
    spec = BenchmarkSpec("fake", "no-such")
    with pytest.raises(KeyError):
        project(spec, {})
