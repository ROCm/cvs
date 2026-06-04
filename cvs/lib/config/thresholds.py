"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import operator
import re
from typing import Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Annotated

# Comparison operators. Every threshold carries an explicit ``op`` so the
# comparison direction is *never* inferred from the metric name (the old
# substring-"ms" heuristic flipped the comparison for any metric whose name
# happened to contain those letters, e.g. ``latency_seconds``).
_OPS: Dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}

OpLiteral = Literal[">=", "<=", ">", "<", "==", "!="]

_WHERE_RE = re.compile(r"^\s*(>=|<=|==|!=|>|<)\s*([-+]?\d*\.?\d+)\s*$")


def _percentile(values: List[float], pct: float) -> Optional[float]:
    """Linear-interpolation percentile (pct in [0, 100]). None if no data."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


class ResultView:
    """Read-only view over a workload's parsed results, consumed by thresholds.

    Decouples threshold evaluation from how results are stored. An adapter's
    ``parse`` step populates one of these from framework-native telemetry:

    - ``scalars`` : derived single-value metrics (e.g. ``total_throughput``,
      ``elapsed_s``).
    - ``samples`` : long list of per-request/sample dicts (e.g. ttft_ms, tpot_ms).
    - ``trajectory`` : long-format rows ``{step, metric, value, role, host}``.
    """

    def __init__(
        self,
        scalars: Optional[Dict[str, float]] = None,
        samples: Optional[List[Dict[str, float]]] = None,
        trajectory: Optional[List[Dict[str, object]]] = None,
    ) -> None:
        self.scalars = scalars or {}
        self.samples = samples or []
        self.trajectory = trajectory or []

    def scalar(self, name: str) -> Optional[float]:
        value = self.scalars.get(name)
        return None if value is None else float(value)

    def sample_values(self, metric: str) -> List[float]:
        out = []
        for row in self.samples:
            if metric in row and row[metric] is not None:
                out.append(float(row[metric]))
        return out

    def series(self, metric: str, role: Optional[str] = None) -> List[float]:
        rows = [r for r in self.trajectory if r.get("metric") == metric]
        if role is not None:
            rows = [r for r in rows if r.get("role") == role]
        rows = sorted(rows, key=lambda r: r.get("step", 0))
        return [float(r["value"]) for r in rows if r.get("value") is not None]


class ThresholdVerdict(BaseModel):
    """Outcome of evaluating one threshold. Persisted verbatim into the manifest."""

    model_config = ConfigDict(extra="forbid")

    threshold_type: str
    metric: Optional[str] = None
    op: Optional[str] = None
    expected: Optional[float] = None
    actual: Optional[float] = None
    margin: Optional[float] = None
    passed: bool
    detail: Optional[str] = None


class Threshold(BaseModel):
    """Base for the typed threshold predicates."""

    model_config = ConfigDict(extra="forbid")

    def evaluate(self, view: ResultView) -> ThresholdVerdict:  # pragma: no cover - abstract
        raise NotImplementedError


class PercentileThreshold(Threshold):
    """Percentile of a per-sample metric, e.g. P99 TTFT <= 50 ms."""

    type: Literal["percentile"] = "percentile"
    metric: str
    # Bounded at load: an out-of-range percentile is an operator typo, not a
    # runtime condition. Without this, ``percentile > 100`` indexes past the end
    # of the sorted samples (IndexError mid-run) and ``percentile < 0`` silently
    # returns a value below the data -- a bogus pass/fail with no error.
    percentile: float = Field(default=99.0, ge=0, le=100)
    op: OpLiteral
    value: float
    unit: Optional[str] = None

    def evaluate(self, view: ResultView) -> ThresholdVerdict:
        """Evaluate the percentile threshold against samples, with a
        framework-scalar fallback.

        Preferred path: compute the percentile from per-request samples in
        ``view.samples`` (lets CVS ask any percentile from one run and gets
        cross-framework consistency since the computation is ours, not the
        framework's).

        Fallback path: when ``view.samples`` is empty for this metric --
        e.g. the new ``vllm bench serve`` CLI dropped per-request arrays
        from --save-result -- and a matching pre-computed percentile
        scalar exists under the conventional name ``p{int(percentile)}_{metric}``
        (which vLLM/sglang/TGI all populate), use it. The verdict's
        ``detail`` annotates which path produced the value so a manifest
        reader can tell at a glance whether percentile semantics are
        framework-native or CVS-computed.
        """
        samples = view.sample_values(self.metric)
        if samples:
            actual = _percentile(samples, self.percentile)
            detail = None
        else:
            scalar_name = f"p{int(self.percentile)}_{self.metric}"
            actual = view.scalar(scalar_name)
            detail = (
                f"from framework scalar '{scalar_name}' (no per-request samples)"
                if actual is not None
                else f"no samples for metric '{self.metric}' and no fallback scalar '{scalar_name}'"
            )
        passed = actual is not None and _OPS[self.op](actual, self.value)
        return ThresholdVerdict(
            threshold_type=self.type,
            metric=self.metric,
            op=self.op,
            expected=self.value,
            actual=actual,
            margin=None if actual is None else actual - self.value,
            passed=passed,
            detail=detail,
        )


class RateThreshold(Threshold):
    """Derived rate scalar, e.g. throughput >= 1200 tokens/sec."""

    type: Literal["rate"] = "rate"
    metric: str
    op: OpLiteral
    value: float
    unit: Optional[str] = None

    def evaluate(self, view: ResultView) -> ThresholdVerdict:
        actual = view.scalar(self.metric)
        passed = actual is not None and _OPS[self.op](actual, self.value)
        return ThresholdVerdict(
            threshold_type=self.type,
            metric=self.metric,
            op=self.op,
            expected=self.value,
            actual=actual,
            margin=None if actual is None else actual - self.value,
            passed=passed,
            detail=None if actual is not None else f"no scalar '{self.metric}'",
        )


class GoodputThreshold(Threshold):
    """MLPerf-shaped filtered rate: requests/sec meeting per-sample SLAs.

    ``where`` maps a sample metric to a constraint string like ``"<=200"``.
    Goodput = (# samples satisfying every constraint) / ``elapsed_s`` scalar.

    B2: ``where`` is validated at load (the field validator below) so a
    malformed constraint fails closed with a clear error instead of silently
    filtering every sample out and reporting goodput 0.
    """

    type: Literal["goodput"] = "goodput"
    op: OpLiteral
    value: float
    where: Dict[str, str] = Field(default_factory=dict)
    elapsed_scalar: str = "elapsed_s"
    unit: Optional[str] = None

    @field_validator("where")
    @classmethod
    def _validate_where(cls, value: Dict[str, str]) -> Dict[str, str]:
        for metric, constraint in value.items():
            if not _WHERE_RE.match(constraint):
                raise ValueError(
                    f"malformed goodput constraint for metric '{metric}': {constraint!r} "
                    f"(expected e.g. '<=200', '>= 1.5')"
                )
        return value

    def _passes_filters(self, row: Dict[str, float]) -> bool:
        for metric, constraint in self.where.items():
            match = _WHERE_RE.match(constraint)
            if not match:
                return False
            op_str, bound = match.group(1), float(match.group(2))
            sample = row.get(metric)
            if sample is None or not _OPS[op_str](float(sample), bound):
                return False
        return True

    def evaluate(self, view: ResultView) -> ThresholdVerdict:
        elapsed = view.scalar(self.elapsed_scalar)
        good = sum(1 for row in view.samples if self._passes_filters(row))
        actual = (good / elapsed) if elapsed else None
        passed = actual is not None and _OPS[self.op](actual, self.value)
        return ThresholdVerdict(
            threshold_type=self.type,
            metric="goodput",
            op=self.op,
            expected=self.value,
            actual=actual,
            margin=None if actual is None else actual - self.value,
            passed=passed,
            detail=None if actual is not None else f"missing '{self.elapsed_scalar}' scalar",
        )


class MonotonicityThreshold(Threshold):
    """Trajectory monotonicity over a trailing window, e.g. loss non-increasing.

    ``window`` is the trailing fraction (0, 1] of the series to inspect.
    ``tolerance`` permits small reversals (noise) before declaring a violation.

    B1: the not-enough-data guard is on the *windowed tail*, not the full
    series. A small ``window`` can carve a length-1 tail out of a long series;
    the pairwise loop would then never run and a vacuous pass would be reported.
    """

    type: Literal["monotonicity"] = "monotonicity"
    metric: str
    direction: Literal["non_increasing", "non_decreasing"] = "non_increasing"
    window: float = Field(default=0.25, gt=0, le=1)
    # Allowed reversal magnitude; must be non-negative. A negative tolerance makes
    # ``worst <= tolerance`` unsatisfiable even for a perfectly monotonic series
    # (worst is >= 0), so every run would fail with no error -- reject at load.
    tolerance: float = Field(default=0.0, ge=0)
    role: Optional[str] = None

    def evaluate(self, view: ResultView) -> ThresholdVerdict:
        series = view.series(self.metric, self.role)
        tail = series[max(0, int(len(series) * (1.0 - self.window))) :]
        if len(tail) < 2:
            return ThresholdVerdict(
                threshold_type=self.type,
                metric=self.metric,
                passed=False,
                detail=f"not enough trajectory points in window for '{self.metric}'",
            )
        worst = 0.0
        for prev, cur in zip(tail, tail[1:]):
            delta = cur - prev
            step_violation = delta if self.direction == "non_increasing" else -delta
            worst = max(worst, step_violation)
        passed = worst <= self.tolerance
        return ThresholdVerdict(
            threshold_type=self.type,
            metric=self.metric,
            op="<=",
            expected=self.tolerance,
            actual=worst,
            margin=worst - self.tolerance,
            passed=passed,
            detail=f"direction={self.direction} window={self.window}",
        )


class ConvergenceThreshold(Threshold):
    """Series reaches ``target`` +/- ``epsilon`` by ``by_step`` (if set)."""

    type: Literal["convergence"] = "convergence"
    metric: str
    target: float
    # Tolerance band around ``target``; must be non-negative. A negative epsilon
    # makes ``abs(v - target) <= epsilon`` unsatisfiable, so the series can never
    # converge regardless of the data -- a silent always-fail. Reject at load.
    epsilon: float = Field(ge=0)
    # Same class of bug as PercentileThreshold.percentile: a by_step < 1 is
    # meaningless and would slice the series from the wrong end (``series[:0]``
    # -> vacuous, ``series[:-k]`` -> silently drops the tail).
    by_step: Optional[int] = Field(default=None, ge=1)
    role: Optional[str] = None

    def evaluate(self, view: ResultView) -> ThresholdVerdict:
        series = view.series(self.metric, self.role)
        window = series if self.by_step is None else series[: self.by_step]
        converged = any(abs(v - self.target) <= self.epsilon for v in window)
        final = window[-1] if window else None
        return ThresholdVerdict(
            threshold_type=self.type,
            metric=self.metric,
            op="~=",
            expected=self.target,
            actual=final,
            margin=None if final is None else final - self.target,
            passed=converged,
            detail=f"epsilon={self.epsilon} by_step={self.by_step}",
        )


class StabilityThreshold(Threshold):
    """Rolling variance bound over a per-sample metric or trajectory series."""

    type: Literal["stability"] = "stability"
    metric: str
    # Variance is always >= 0, so a negative bound can never be satisfied and
    # would fail every run with no error. Reject at load (0 means "require a
    # perfectly constant series").
    max_variance: float = Field(ge=0)
    source: Literal["samples", "trajectory"] = "samples"
    role: Optional[str] = None

    def evaluate(self, view: ResultView) -> ThresholdVerdict:
        data = view.sample_values(self.metric) if self.source == "samples" else view.series(self.metric, self.role)
        if len(data) < 2:
            return ThresholdVerdict(
                threshold_type=self.type,
                metric=self.metric,
                passed=False,
                detail=f"not enough data for '{self.metric}'",
            )
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        passed = variance <= self.max_variance
        return ThresholdVerdict(
            threshold_type=self.type,
            metric=self.metric,
            op="<=",
            expected=self.max_variance,
            actual=variance,
            margin=variance - self.max_variance,
            passed=passed,
            detail=f"source={self.source}",
        )


# Discriminated union: the ``type`` field selects the concrete predicate, and
# extra="forbid" on each member turns a misspelled field into a load-time error.
ThresholdUnion = Annotated[
    Union[
        PercentileThreshold,
        RateThreshold,
        GoodputThreshold,
        MonotonicityThreshold,
        ConvergenceThreshold,
        StabilityThreshold,
    ],
    Field(discriminator="type"),
]
