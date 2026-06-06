"""BENCHMARK_REGISTRY — accuracy-only in v1.

Per ``cvs-dtni-v1-spec.md`` §"Step 4: Accuracy benchmarks": MMLU + GSM8K via
``lm-eval-harness``. Other tracker accuracy benchmarks (MMLU-Pro, BBH,
HellaSwag, etc.) and all perf benchmarks land in v2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkSpec:
    """One benchmark.

    - ``id``: short slug used in workload.benchmarks.
    - ``harness``: HARNESS_INVOKERS key — picks the invocation builder.
    - ``dataset_id``: catalog dataset key the harness will fetch (or has cached).
    - ``score_metric``: key in the harness output JSON we extract as the
      pass/fail scalar. The threshold file references the spec id, not this.
    - ``score_filter``: lm-eval filter slug to pair with score_metric (newer
      lm-eval keys are ``"<metric>,<filter>"``). ``None`` falls back to the
      bare metric key and then to ``,none``.
    - ``shots``: in-context examples for few-shot eval (``--num_fewshot``).
    - ``extra``: harness-specific kwargs (e.g. ``{"task": "mmlu"}``).
    """

    id: str
    harness: str
    dataset_id: str
    score_metric: str
    score_filter: str | None = None
    shots: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


_ACCURACY: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("mmlu",  "lm-eval-harness", "mmlu",  "acc",         score_filter="none",         shots=5, extra={"task": "mmlu"}),
    # gsm8k under lm-eval >=0.4 uses a strict-match filter chain by default.
    BenchmarkSpec("gsm8k", "lm-eval-harness", "gsm8k", "exact_match", score_filter="strict-match", shots=5, extra={"task": "gsm8k"}),
)


BENCHMARK_REGISTRY: dict[str, BenchmarkSpec] = {b.id: b for b in _ACCURACY}


def lookup(benchmark_id: str) -> BenchmarkSpec:
    if benchmark_id not in BENCHMARK_REGISTRY:
        raise KeyError(
            f"unknown benchmark id {benchmark_id!r}; "
            f"known: {sorted(BENCHMARK_REGISTRY)}"
        )
    return BENCHMARK_REGISTRY[benchmark_id]


def list_benchmarks() -> tuple[str, ...]:
    return tuple(BENCHMARK_REGISTRY)
