"""BENCHMARK_REGISTRY — accuracy + perf in v1.

Per ``cvs-dtni-v1-spec.md`` and the DTNI Validation Tracker §"PERFORMANCE
BENCHMARK METRICS" / §"ACCURACY BENCHMARK METRICS":

- accuracy: MMLU + GSM8K via ``lm-eval-harness``
- perf: vLLM serving metrics via ``vllm-bench-serve`` (TTFT/TPOT/ITL/E2EL
  percentiles, throughputs, goodput). One harness run yields the whole
  multi-scalar family at once.

Other tracker rows (sampler-derived: VRAM, KV cache %, mem-BW%, MFU; sweep-
derived: latency-vs-load curve, scale efficiency, max-concurrent-at-SLA;
sidecar-derived: model load time) land in v2 — see
``jobs/ec849125/tmp/cvs-dtni-v2-harness-plan.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkSpec:
    """One benchmark invocation.

    - ``id``: short slug used in workload.benchmarks AND as the threshold
      scalar prefix (lm-eval) or full key (perf). For perf benchmarks,
      scalars land at ``<id>.<field>`` (e.g. ``serve_synth.ttft_p95_ms``)
      so multiple invocations of the same harness don't collide.
    - ``harness``: HARNESS_INVOKERS / PROJECTORS key — picks the invocation
      builder AND the JSON → scalar projector.
    - ``dataset_id``: catalog dataset key the harness will fetch (or has
      cached). ``""`` for synthetic-traffic harnesses.
    - ``score_metric`` / ``score_filter`` / ``shots``: lm-eval-specific
      (None / 0 for non-lm-eval harnesses).
    - ``extra``: harness-specific kwargs. For ``vllm-bench-serve``: keys
      include ``num_prompts``, ``random_input_len``, ``random_output_len``,
      ``max_concurrency``, ``request_rate``, ``dataset_name``,
      ``dataset_path``, ``percentiles``, ``goodput_slo`` (string passed
      through to ``--goodput``).
    """

    id: str
    harness: str
    dataset_id: str = ""
    score_metric: str | None = None
    score_filter: str | None = None
    shots: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


_ACCURACY: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("mmlu",  "lm-eval-harness", "mmlu",  "acc",         score_filter="none",         shots=5, extra={"task": "mmlu"}),
    # gsm8k under lm-eval >=0.4 uses a strict-match filter chain by default.
    BenchmarkSpec("gsm8k", "lm-eval-harness", "gsm8k", "exact_match", score_filter="strict-match", shots=5, extra={"task": "gsm8k"}),
)

# vLLM-serve perf benchmarks. Each invocation produces the full BenchmarkMetrics
# family from `vllm bench serve --save-result`. Workloads pick whichever shapes
# they want; the threshold file names the specific scalars (e.g.
# `serve_short.ttft_p95_ms`) it cares about.
_PERF: tuple[BenchmarkSpec, ...] = (
    # Synthetic random-token traffic, short context (tracker rows 33,34,35,
    # 36,37,39,41,42,43,51 — TTFT/TPOT/prefill/normalized/decode-tail/e2e/
    # global-throughput/decode-throughput/per-gpu/goodput).
    BenchmarkSpec(
        id="serve_synth_short",
        harness="vllm-bench-serve",
        extra={
            "dataset_name": "random",
            "num_prompts": 200,
            "random_input_len": 128,
            "random_output_len": 256,
            "max_concurrency": 32,
            "request_rate": "inf",  # offline burst
            "percentiles": "50,90,95,99",
            # SLOs target tracker row 51 (goodput) and 39 (P95/P99 e2e).
            "goodput_slo": "ttft:2000 tpot:200",
        },
    ),
    # Long-prefill stress (tracker row 23 + perf rows 35,36 — prefill latency,
    # nTTFT). Smaller batch keeps the run bounded.
    BenchmarkSpec(
        id="serve_synth_long",
        harness="vllm-bench-serve",
        extra={
            "dataset_name": "random",
            "num_prompts": 64,
            "random_input_len": 2048,
            "random_output_len": 256,
            "max_concurrency": 8,
            "request_rate": "inf",
            "percentiles": "50,90,95,99",
        },
    ),
    # ShareGPT realistic load (tracker row 22) is intentionally NOT in v1:
    # it needs a per-workload dataset_path the schema can't yet thread.
    # Lands in v2 once the workload schema gets per-benchmark `extras`.
)


BENCHMARK_REGISTRY: dict[str, BenchmarkSpec] = {b.id: b for b in _ACCURACY + _PERF}


def lookup(benchmark_id: str) -> BenchmarkSpec:
    if benchmark_id not in BENCHMARK_REGISTRY:
        raise KeyError(
            f"unknown benchmark id {benchmark_id!r}; "
            f"known: {sorted(BENCHMARK_REGISTRY)}"
        )
    return BENCHMARK_REGISTRY[benchmark_id]


def list_benchmarks() -> tuple[str, ...]:
    return tuple(BENCHMARK_REGISTRY)
