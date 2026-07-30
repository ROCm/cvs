'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Pure parsers for vLLM's engine-side Prometheus `/metrics` endpoint.

This module owns the *vocabulary and math* of the `prom.*` namespace -- the
mapping from two raw Prometheus text-exposition scrapes (one taken before a
sweep cell's client run, one taken after) to the namespaced metric dict that
downstream code (threshold files, the per-metric HTML rows, `evaluate_all`)
keys on. Deliberately free of I/O and orchestration, matching
`vllm_parsing.py`'s split: callers (`vllm_job.py`) fetch the scrape text,
this module turns it into numbers.

Namespacing contract: `prom.*` -- percentile metrics interpolated from
Prometheus Histograms scraped off the live vLLM server, distinct from
`client.*` (measured by the load generator) and `gpu.*` (amd-smi snapshots).
See VLLM_PROMETHEUS_METRICS_SPEC.md Sec 1 for why this is its own namespace
rather than joining either of those.

vLLM's histogram buckets are cumulative per scrape (each `le` bucket already
counts everything at or below it), but the *counters themselves* are
cumulative across the server process's lifetime, not per-request-run. Since a
server is reused across concurrency-only-differing sweep cells
(`server_signature()`), isolating one cell's observations requires diffing
two scrapes taken immediately before and after that cell's client run --
never a single scrape.
'''

from __future__ import annotations

import re

# Human-readable derived metrics exposed as HTML rows (one row per entry per
# cell), mirroring gpu.py's GPU_METRICS shape. These are the metrics this
# spec's Phase 1 closes a gap for -- see VLLM_PROMETHEUS_METRICS_SPEC.md Sec 5
# for the primary/secondary/out-of-scope breakdown.
PROM_METRICS: list[tuple[str, str]] = [
    ("queue_time_p50_ms", "ms"),
    ("queue_time_p95_ms", "ms"),
    ("prefill_time_p50_ms", "ms"),
    ("prefill_time_p95_ms", "ms"),
]
PROM_METRIC_UNITS: dict[str, str] = {k: u for k, u in PROM_METRICS}

# vLLM Prometheus histogram names this module reads, and the (short_name
# prefix, quantile) pairs each feeds into PROM_METRICS above.
_QUEUE_TIME_METRIC = "vllm:request_queue_time_seconds"
_PREFILL_TIME_METRIC = "vllm:request_prefill_time_seconds"

_QUANTILES: dict[str, float] = {
    "p50": 0.50,
    "p95": 0.95,
}

_BUCKET_LINE_RE = re.compile(
    r'^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)_bucket\{[^}]*le="(?P<le>[^"]+)"[^}]*\}\s+(?P<count>[0-9.eE+-]+)\s*$'
)
_SUM_LINE_RE = re.compile(r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)_sum(\{[^}]*\})?\s+(?P<value>[0-9.eE+-]+)\s*$")
_COUNT_LINE_RE = re.compile(r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)_count(\{[^}]*\})?\s+(?P<value>[0-9.eE+-]+)\s*$")
_GAUGE_LINE_RE = re.compile(r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(?P<value>[0-9.eE+-]+)\s*$")


def parse_prometheus_text(raw: "str | None") -> dict:
    """Hand-rolled Prometheus text-exposition-format parser.

    Returns {metric_name: {"buckets": {le: cumulative_count}, "sum": float,
    "count": float}} for every histogram found (le values are strings,
    including "+Inf"), plus {metric_name: float} for any bare gauge/counter
    line not part of a histogram. Ignores `# HELP`/`# TYPE` comment lines and
    any line it can't parse. Degrades to {} on None/empty/unparseable input
    -- never raises, matching gpu.py's `_try_parse` convention.
    """
    if not raw:
        return {}
    histograms: dict[str, dict] = {}
    gauges: dict[str, float] = {}
    try:
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _BUCKET_LINE_RE.match(line)
            if m:
                name = m.group("name")
                hist = histograms.setdefault(name, {"buckets": {}, "sum": None, "count": None})
                hist["buckets"][m.group("le")] = float(m.group("count"))
                continue
            m = _SUM_LINE_RE.match(line)
            if m:
                name = m.group("name")
                hist = histograms.setdefault(name, {"buckets": {}, "sum": None, "count": None})
                hist["sum"] = float(m.group("value"))
                continue
            m = _COUNT_LINE_RE.match(line)
            if m:
                name = m.group("name")
                hist = histograms.setdefault(name, {"buckets": {}, "sum": None, "count": None})
                hist["count"] = float(m.group("value"))
                continue
            m = _GAUGE_LINE_RE.match(line)
            if m:
                gauges[m.group("name")] = float(m.group("value"))
    except (ValueError, TypeError):
        return {}
    result: dict = dict(histograms)
    for name, val in gauges.items():
        if name not in result:
            result[name] = val
    return result


def diff_histogram(before: "dict | None", after: "dict | None") -> "dict[str, float] | None":
    """Per-bucket subtraction isolating one sweep cell's observations out of
    a server-process-lifetime-cumulative Prometheus histogram.

    before/after: histogram dicts as returned by parse_prometheus_text() for
    one metric name (i.e. {"buckets": {le: count}, "sum": ..., "count": ...}).
    Returns {le: after_count - before_count} for every `le` present in
    `after` (a bucket boundary can only appear once the server has been
    running long enough to register it; `before` may be missing a boundary
    `after` has if this is the server's first-ever scrape). Missing `before`
    buckets are treated as 0. Negative diffs (e.g. a server restart between
    scrapes resetting the counters) are clamped to 0 rather than propagated,
    per VLLM_PROMETHEUS_METRICS_SPEC.md Sec 3.2/8.3.

    Returns None if `after` is missing/empty (nothing to diff against).
    """
    if not after or not after.get("buckets"):
        return None
    before_buckets = (before or {}).get("buckets", {})
    after_buckets = after["buckets"]
    return {le: max(0.0, count - before_buckets.get(le, 0.0)) for le, count in after_buckets.items()}


def histogram_quantile(buckets: "dict[str, float] | None", q: float) -> "float | None":
    """Linear interpolation between cumulative bucket boundaries -- the same
    algorithm PromQL's `histogram_quantile()` uses.

    buckets: {le: cumulative_count}, `le` values are numeric strings or
    "+Inf". Returns None if buckets is empty/missing or total count (the
    "+Inf" bucket) is 0 -- mirrors vllm_parsing.py's `_safe_div` None-safe
    convention, never a ZeroDivisionError.

    PromQL parity for the +Inf bucket: linear interpolation is only valid
    between two finite boundaries. If the target quantile falls into the
    unbounded "+Inf" bucket, PromQL cannot interpolate past the highest
    finite boundary and clamps to it instead of extrapolating to infinity --
    without this, an overloaded server (some requests genuinely exceeding
    every finite bucket) would report `inf` ms instead of a finite p95/p99.
    The one exception is a "+Inf"-only histogram (no finite boundary exists
    to clamp to), where PromQL itself returns +Inf.
    """
    if not buckets:
        return None
    try:
        parsed = sorted(((float("inf") if le == "+Inf" else float(le), count) for le, count in buckets.items()))
    except (TypeError, ValueError):
        return None
    total = parsed[-1][1]
    if total <= 0:
        return None
    target = q * total
    prev_bound, prev_count = 0.0, 0.0
    for bound, count in parsed:
        if count >= target:
            if bound == float("inf"):
                return bound if len(parsed) == 1 else prev_bound
            if bound == prev_bound or count == prev_count:
                return bound
            frac = (target - prev_count) / (count - prev_count)
            return prev_bound + frac * (bound - prev_bound)
        prev_bound, prev_count = bound, count
    return prev_bound


def _quantile_ms(before_metrics: dict, after_metrics: dict, metric_name: str, q: float) -> "float | None":
    diffed = diff_histogram(before_metrics.get(metric_name), after_metrics.get(metric_name))
    seconds = histogram_quantile(diffed, q)
    return None if seconds is None else seconds * 1000.0


def to_prom_metrics(before_text: "str | None", after_text: "str | None") -> dict:
    """Composed entry point: two raw scrape texts -> the `prom.*` metric dict.

    Analogous to vllm_parsing.py's to_client_metrics(). Returns an all-None
    dict (never a partial one, never a raise) if either scrape is
    missing/unparseable -- see VLLM_PROMETHEUS_METRICS_SPEC.md Sec 3.4: a
    transport failure must degrade every prom.* key for the cell, not crash it.
    """
    all_none = {f"prom.{short}": None for short, _unit in PROM_METRICS}
    if not before_text or not after_text:
        return all_none

    before_metrics = parse_prometheus_text(before_text)
    after_metrics = parse_prometheus_text(after_text)
    if not before_metrics or not after_metrics:
        return all_none

    result = dict(all_none)
    for qname, q in _QUANTILES.items():
        result[f"prom.queue_time_{qname}_ms"] = _quantile_ms(before_metrics, after_metrics, _QUEUE_TIME_METRIC, q)
        result[f"prom.prefill_time_{qname}_ms"] = _quantile_ms(before_metrics, after_metrics, _PREFILL_TIME_METRIC, q)
    return result
