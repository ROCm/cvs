'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Pure parsers for vLLM benchmark artifacts.

This module owns the *vocabulary and math* of vLLM metrics -- the mapping from
a raw benchmark artifact to the namespaced metric dict that downstream code
(threshold files, the per-metric HTML rows, `evaluate_all`) keys on. It is
deliberately free of I/O and orchestration: callers fetch the bytes (each job
lays its artifacts out differently -- single-node `cat`, disagg prefill+decode,
distributed rank-0 vs all-ranks) and hand the parsed/raw payload in here.

Keeping the transforms pure makes them reusable across jobs (single-node,
distributed, disaggregated, InferenceMax) and unit-testable with plain
dict/string fixtures -- no fake orchestrator required.

Namespacing contract:
  - `client.*`  -- scalars measured by the load generator (`vllm bench serve`).
  - `server.*`  -- scalars scraped from the live server `/metrics` endpoint
                   (future; `to_server_metrics` will live here too).
'''

from __future__ import annotations


def _safe_div(num, den):
    """num/den, or None if either is missing/None or the divisor is 0.

    Derived metrics must degrade to None (a clean "not available"), never a
    bogus 0 or a KeyError that would crash an otherwise-good run.
    """
    if num is None or den is None:
        return None
    try:
        den = float(den)
        if den == 0:
            return None
        return float(num) / den
    except (TypeError, ValueError):
        return None


def to_client_metrics(raw, *, tp, isl):
    """Map a stock `vllm bench serve` results dict to the `client.*` namespace.

    `raw` is the already-parsed JSON the load generator writes to its
    `--result-dir` `results` artifact. Every stock scalar is namespaced
    `client.<key>` 1:1, then a few derived metrics are appended. Pure: no I/O,
    no orchestration -- the caller is responsible for fetching and json-loading
    the artifact and for raising on missing/unparseable input.

    `tp` (tensor parallelism) and `isl` (input sequence length) are the only
    out-of-band scalars the derivations need.
    """
    m = {f"client.{k}": v for k, v in raw.items()}
    # Friendly alias: stock's request_goodput -> client.goodput
    # (the name the results table and threshold file reference).
    m["client.goodput"] = raw.get("request_goodput")

    # Derived metrics. _safe_div guards every divisor (0/None/missing).
    ttot = raw.get("total_token_throughput")
    mean_ttft = raw.get("mean_ttft_ms")
    p99_itl = raw.get("p99_itl_ms")
    p50_itl = raw.get("p50_itl_ms")
    median_tpot = raw.get("median_tpot_ms")
    completed = raw.get("completed")
    failed = raw.get("failed")

    m["client.per_gpu_throughput"] = _safe_div(ttot, tp)
    m["client.normalized_ttft_ms_per_tok"] = _safe_div(mean_ttft, isl)
    m["client.decode_latency_ratio"] = _safe_div(p99_itl, p50_itl)
    m["client.decode_throughput_p50"] = _safe_div(1000.0, median_tpot)
    total_req = None if completed is None or failed is None else completed + failed
    m["client.success_rate"] = _safe_div(completed, total_req)
    return m


# Numeric client.* metrics surfaced as one pytest test (= one HTML row) each.
# (short_name, unit); the value is looked up as "client.<short_name>" in the
# dict to_client_metrics returns. request_rate is omitted (stock emits the
# string "inf"). This is the *display surface* of the client.* vocabulary above
# -- it lives here, beside to_client_metrics, so every vLLM flavor (single-node,
# distributed, disaggregated, InferenceMax) shares one definition instead of
# each suite re-listing the rows.
CLIENT_METRICS = [
    ("max_concurrency", "-"),
    ("max_concurrent_requests", "-"),
    ("num_prompts", "-"),
    ("completed", "-"),
    ("failed", "-"),
    ("success_rate", "-"),
    ("duration", "s"),
    ("request_throughput", "req/s"),
    ("goodput", "req/s"),
    ("output_throughput", "tok/s"),
    ("total_token_throughput", "tok/s"),
    ("per_gpu_throughput", "tok/s"),
    ("decode_throughput_p50", "tok/s"),
    ("max_output_tokens_per_s", "tok/s"),
    ("total_input_tokens", "-"),
    ("total_output_tokens", "-"),
    ("mean_ttft_ms", "ms"),
    ("median_ttft_ms", "ms"),
    ("p90_ttft_ms", "ms"),
    ("p95_ttft_ms", "ms"),
    ("p99_ttft_ms", "ms"),
    ("normalized_ttft_ms_per_tok", "ms/tok"),
    ("mean_tpot_ms", "ms"),
    ("median_tpot_ms", "ms"),
    ("p90_tpot_ms", "ms"),
    ("p95_tpot_ms", "ms"),
    ("p99_tpot_ms", "ms"),
    ("mean_itl_ms", "ms"),
    ("median_itl_ms", "ms"),
    ("p95_itl_ms", "ms"),
    ("p99_itl_ms", "ms"),
    ("decode_latency_ratio", "-"),
    ("mean_e2el_ms", "ms"),
    ("median_e2el_ms", "ms"),
    ("p90_e2el_ms", "ms"),
    ("p95_e2el_ms", "ms"),
    ("p99_e2el_ms", "ms"),
]
CLIENT_METRIC_UNITS = dict(CLIENT_METRICS)
