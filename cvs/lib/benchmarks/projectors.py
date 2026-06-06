"""Per-harness JSON → scalar projection.

A projector receives:
- the BenchmarkSpec it ran
- the parsed JSON payload from the harness

and returns a flat ``{scalar_key: float}`` dict. The runner merges these into
``ctx.result.scalars``, where ``threshold.json`` keys name them directly.

Convention:
- accuracy harnesses (lm-eval) project to the bare benchmark id, optionally
  plus ``<id>_stderr``. Kept for back-compat with Step-4 thresholds.
- perf harnesses (vllm-bench-serve) project to a dotted ``<id>.<field>``
  namespace so multiple invocations of the same harness don't collide and
  thresholds can name e.g. ``serve_synth_short.ttft_p95_ms``.
"""

from __future__ import annotations

from typing import Any, Callable

from cvs.lib.benchmarks.registry import BenchmarkSpec


# ---- lm-eval-harness ------------------------------------------------------


def _is_real_number(v: Any) -> bool:
    """True for int/float but NOT bool (bool is an int subclass in Python).

    Without this, a payload field that happens to be True/False would be
    coerced to 1.0/0.0 and treated as a real measurement by the threshold
    engine.
    """
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _project_lm_eval(spec: BenchmarkSpec, payload: dict[str, Any]) -> dict[str, float]:
    """Extract spec.score_metric from lm-eval JSON under the benchmark id key.

    lm-eval writes ``{"results": {task: {metric: value, metric_stderr: ...}}}``.
    Modern lm-eval suffixes the metric name with ``,<filter>`` (e.g.
    ``acc,none`` or ``exact_match,strict-match``). We try, in order:
      1) the spec's preferred filter (``score_filter``)
      2) the bare metric key (legacy)
      3) ``,none`` (default filter slug)
    Also surfaces ``<id>_stderr`` for diagnostics using the same lookup
    order.
    """
    out: dict[str, float] = {}
    results = payload.get("results") or {}
    task = spec.extra.get("task", spec.id)
    task_results = results.get(task, {})

    if not spec.score_metric:
        return out

    score = _pick_metric(task_results, spec.score_metric, spec.score_filter)
    if _is_real_number(score):
        out[spec.id] = float(score)

    stderr = _pick_metric(task_results, f"{spec.score_metric}_stderr", spec.score_filter)
    if _is_real_number(stderr):
        out[f"{spec.id}_stderr"] = float(stderr)

    return out


def _pick_metric(task_results: dict[str, Any], metric: str, preferred_filter: str | None) -> Any:
    """Walk filter-suffix variants for ``metric`` in priority order."""
    candidates: list[str] = []
    if preferred_filter:
        candidates.append(f"{metric},{preferred_filter}")
    candidates.append(metric)
    candidates.append(f"{metric},none")
    for key in candidates:
        if key in task_results:
            return task_results[key]
    return None


# ---- vllm bench serve ----------------------------------------------------


# Fields the vllm BenchmarkMetrics JSON exports as scalars (int/float).
# Kept as a tuple so we don't silently absorb future JSON additions: if a
# new field appears in vllm and we want it, list it here explicitly.
_VLLM_SERVE_SCALAR_FIELDS: tuple[str, ...] = (
    # Throughput / counters.
    "completed",
    "total_input",
    "total_output",
    "request_throughput",          # req/s (tracker row 41 component)
    "request_goodput",             # req/s meeting SLO (tracker row 51)
    "output_throughput",           # output tok/s (tracker row 41,42)
    "total_token_throughput",      # (in + out) tok/s
    # Latency means / medians / std.
    "mean_ttft_ms",   "median_ttft_ms",  "std_ttft_ms",
    "mean_tpot_ms",   "median_tpot_ms",  "std_tpot_ms",
    "mean_itl_ms",    "median_itl_ms",   "std_itl_ms",
    "mean_e2el_ms",   "median_e2el_ms",  "std_e2el_ms",
)

# Percentile arrays we explode into named scalars (P50/P90/P95/P99 typical).
_VLLM_SERVE_PCTL_FIELDS: tuple[str, ...] = (
    "percentiles_ttft_ms",
    "percentiles_tpot_ms",
    "percentiles_itl_ms",
    "percentiles_e2el_ms",
)


def _project_vllm_bench_serve(spec: BenchmarkSpec, payload: dict[str, Any]) -> dict[str, float]:
    """Project ``vllm bench serve --save-result`` JSON to scalars.

    The JSON top-level is a single flat object (vllm.benchmarks.serve.
    BenchmarkMetrics merged with --metadata). All exported scalars live
    under ``<spec.id>.<field>``; percentile lists are unrolled to
    ``<spec.id>.<metric>_p<NN>_ms``.
    """
    out: dict[str, float] = {}

    for field in _VLLM_SERVE_SCALAR_FIELDS:
        v = payload.get(field)
        if _is_real_number(v):
            out[f"{spec.id}.{field}"] = float(v)

    for pkey in _VLLM_SERVE_PCTL_FIELDS:
        # vllm writes percentiles as list[(percentile, value)].
        raw = payload.get(pkey)
        if not isinstance(raw, list):
            continue
        # Field naming: "percentiles_ttft_ms" -> metric "ttft"
        # ("percentiles_" prefix and "_ms" suffix stripped).
        metric = pkey[len("percentiles_"):-len("_ms")]
        for entry in raw:
            if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
                continue
            pct, val = entry
            if not _is_real_number(val):
                continue
            try:
                pct_int = int(round(float(pct)))
            except (TypeError, ValueError):
                continue
            out[f"{spec.id}.{metric}_p{pct_int:02d}_ms"] = float(val)

    return out


# ---- dispatch -------------------------------------------------------------


PROJECTORS: dict[str, Callable[[BenchmarkSpec, dict[str, Any]], dict[str, float]]] = {
    "lm-eval-harness": _project_lm_eval,
    "vllm-bench-serve": _project_vllm_bench_serve,
}


def project(spec: BenchmarkSpec, payload: dict[str, Any]) -> dict[str, float]:
    if spec.harness not in PROJECTORS:
        raise KeyError(
            f"benchmark {spec.id!r}: no projector for harness {spec.harness!r}; "
            f"known: {sorted(PROJECTORS)}"
        )
    return PROJECTORS[spec.harness](spec, payload)
