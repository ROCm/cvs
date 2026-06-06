"""Threshold evaluation. Three kinds: min, max_ms, within.

Returns a list of dicts (serialized into verdict.json directly).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

VALID_KINDS = ("min", "max_ms", "within", "min_tok_s", "min_ratio")


@dataclass
class Verdict:
    metric: str
    actual: float | None
    threshold: float | tuple[float, float]
    kind: str
    passed: bool
    note: str | None = None


def evaluate_threshold(name: str, spec: dict, actual: float | None) -> Verdict:
    """spec = {"kind": ..., "value": ...} (or "lo"/"hi" for within)."""
    kind = spec.get("kind")
    if kind not in VALID_KINDS:
        raise ValueError(f"threshold {name!r}: unknown kind {kind!r} (valid: {VALID_KINDS})")

    if actual is None:
        return Verdict(
            metric=name, actual=None, threshold=spec.get("value"),
            kind=kind, passed=False, note="metric not produced by adapter",
        )

    if kind in ("min", "min_tok_s", "min_ratio"):
        thr = float(spec["value"])
        return Verdict(name, float(actual), thr, kind, float(actual) >= thr)

    if kind == "max_ms":
        thr = float(spec["value"])
        return Verdict(name, float(actual), thr, kind, float(actual) <= thr)

    if kind == "within":
        lo, hi = float(spec["lo"]), float(spec["hi"])
        return Verdict(name, float(actual), (lo, hi), kind, lo <= float(actual) <= hi)

    raise AssertionError(f"unreachable kind={kind}")  # validated above


def evaluate_all(thresholds: dict[str, dict], actuals: dict[str, float]) -> list[dict]:
    """Evaluate each named threshold; return list of dicts for JSON."""
    verdicts = []
    for name, spec in thresholds.items():
        v = evaluate_threshold(name, spec, actuals.get(name))
        d = asdict(v)
        if isinstance(d["threshold"], tuple):
            d["threshold"] = list(d["threshold"])
        verdicts.append(d)
    return verdicts


def all_passed(verdicts: list[dict]) -> bool:
    return all(v["passed"] for v in verdicts)
