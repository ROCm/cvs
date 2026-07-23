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
distributed, disaggregated, InferenceX ATOM) and unit-testable with plain
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


def _gpu_count(tp, pp):
    """int(tp) * int(pp), or None if either is missing/None/non-numeric.

    Pure, never raises. Degrades to None on anything int() can't coerce
    (missing, empty string, non-numeric string, wrong type) so callers can
    feed the result straight into `_safe_div`'s existing None/0 guard.
    """
    try:
        return int(tp) * int(pp)
    except (TypeError, ValueError):
        return None


def to_client_metrics(raw, *, tp, isl, pp="1"):
    """Map a stock `vllm bench serve` results dict to the `client.*` namespace.

    `raw` is the already-parsed JSON the load generator writes to its
    `--result-dir` `results` artifact. Every stock scalar is namespaced
    `client.<key>` 1:1, then a few derived metrics are appended. Pure: no I/O,
    no orchestration -- the caller is responsible for fetching and json-loading
    the artifact and for raising on missing/unparseable input.

    `tp` (tensor parallelism), `isl` (input sequence length), and `pp`
    (pipeline parallelism) are the only out-of-band scalars the derivations
    need. `pp` defaults to `"1"` for callers with no pipeline-parallel concept
    (e.g. InferenceX ATOM); vLLM call sites should still pass it explicitly.
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
    num_prompts = raw.get("num_prompts")
    # ATOM benchmark_serving omits `failed` when every request succeeds; derive it
    # so success_rate and gated health metrics remain assertable.
    if failed is None and completed is not None and num_prompts is not None:
        try:
            failed = max(0, int(num_prompts) - int(completed))
        except (TypeError, ValueError):
            failed = None
        if failed is not None:
            m["client.failed"] = failed

    m["client.per_gpu_throughput"] = _safe_div(ttot, _gpu_count(tp, pp))
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
# distributed, disaggregated, InferenceX ATOM) shares one definition instead of
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

# The perf+health SLO contract: the subset of CLIENT_METRICS a calibrated run
# must *assert*, not merely display. The loader's coverage check requires a
# threshold spec for every name here in every present cell, so a gated metric
# can never silently fall through to a zero-assertion record-only row.
#
# Membership = "out of range means FAILURE", not "informational". Gated:
# throughput (total + per-request output), the FULL latency distribution
# (mean/median/p90/p95/p99 for ttft, tpot, itl, e2el -- itl has no p90), and run
# health (success_rate floor, failed ceiling). Record-only by design: inputs
# (num_prompts), totals (total_*_tokens), secondary throughputs (per_gpu_*,
# decode_throughput_p50, max_output_tokens_per_s, request_throughput, goodput),
# and derived diagnostics (normalized_ttft_ms_per_tok, decode_latency_ratio).
#
# Closed-world default: a NEW metric added to CLIENT_METRICS is record-only
# until its name is added here. Add a metric to this set the moment it becomes
# a pass/fail criterion -- the loader then forces a spec for it in every cell
# before the suite can run green. Every name must also appear in CLIENT_METRICS
# (a gated metric with no producer would gate a '-'); a unit test pins that.
GATED_METRICS = {
    # throughput
    "total_token_throughput",
    "output_throughput",
    # latency -- full distribution per family (mean/median/p90/p95/p99;
    # itl has no p90 producer). Every emitted quantile is a pass/fail gate.
    "mean_ttft_ms",
    "median_ttft_ms",
    "p90_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p90_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p90_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
    # run health
    "success_rate",
    "failed",
}

# `(label, client.* key)` columns for the report's results table and the
# console table in `_shared.py::test_print_results_table` -- kept in sync
# with that table's headers. First 7 are the fixed positional columns
# `inference_payload.build_results_table` always emits (Model, GPU, ISL, OSL,
# Policy, Conc, Host); only `metric_keys[7:]` are looked up per host.
VLLM_RESULTS_COLUMNS = (
    ("Model", None),
    ("GPU", None),
    ("ISL", None),
    ("OSL", None),
    ("Policy", None),
    ("Conc", None),
    ("Host", None),
    ("Req/s", "client.request_throughput"),
    ("Total tok/s", "client.total_token_throughput"),
    ("Mean TTFT (ms)", "client.mean_ttft_ms"),
    ("P95 TTFT (ms)", "client.p95_ttft_ms"),
    ("Mean TPOT (ms)", "client.mean_tpot_ms"),
    ("P95 TPOT (ms)", "client.p95_tpot_ms"),
    ("P99 ITL (ms)", "client.p99_itl_ms"),
    ("Goodput (req/s)", "client.goodput"),
)

# Report gate-matrix tiers. Membership is seeded from GATED_METRICS so the
# report's tier partition exactly mirrors what the suite enforces: every name
# in GATED_METRICS must land in exactly one non-record tier (pinned by a unit
# test), and `set(METRIC_TIERS) <= set(METRIC_TIER_ORDER)` must hold (true by
# construction below) or `cell_build.tier_status` would silently drop metrics
# whose tier isn't iterated.
METRIC_TIERS: dict[str, tuple[str, ...]] = {
    "throughput": (
        "total_token_throughput",
        "output_throughput",
    ),
    "ttft": (
        "mean_ttft_ms",
        "median_ttft_ms",
        "p90_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
    ),
    "tpot": (
        "mean_tpot_ms",
        "median_tpot_ms",
        "p90_tpot_ms",
        "p95_tpot_ms",
        "p99_tpot_ms",
    ),
    "latency": (
        "mean_itl_ms",
        "median_itl_ms",
        "p95_itl_ms",
        "p99_itl_ms",
        "mean_e2el_ms",
        "median_e2el_ms",
        "p90_e2el_ms",
        "p95_e2el_ms",
        "p99_e2el_ms",
    ),
    "health": (
        "success_rate",
        "failed",
    ),
}

METRIC_TIER_ORDER: tuple[str, ...] = tuple(METRIC_TIERS.keys()) + ("record",)

_tiered = {m for names in METRIC_TIERS.values() for m in names}
RECORD_METRICS: tuple[str, ...] = tuple(short for short, _unit in CLIENT_METRICS if short not in _tiered)


def tier_metric_specs(thresholds_cell: dict, tier: str) -> dict[str, dict]:
    """Return ``client.*`` threshold specs for one tier in a sweep cell."""
    names = RECORD_METRICS if tier == "record" else METRIC_TIERS.get(tier, ())
    specs = {}
    for short in names:
        full = f"client.{short}"
        spec = thresholds_cell.get(full)
        if spec is not None:
            specs[full] = spec
    return specs
