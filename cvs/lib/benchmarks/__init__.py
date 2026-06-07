"""Accuracy benchmark machinery for DTNI v1 (Step 4, accuracy-only).

One harness in v1: ``lm-eval-harness`` (covers MMLU + GSM8K + the LightEval
supplementary set the tracker references). Perf benchmarks, samplers, sweeps,
and the multi-lifecycle driver are deferred to v2 — see
``jobs/ec849125/tmp/cvs-dtni-v2-harness-plan.md``.
"""

from cvs.lib.benchmarks.registry import (  # noqa: F401
    BENCHMARK_REGISTRY,
    BenchmarkSpec,
    list_benchmarks,
    lookup,
)
