'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''
from __future__ import annotations

from typing import Any, Dict, List, Optional


class ThresholdViolation(Exception):
    def __init__(self, violations):
        self.violations = list(violations)
        super().__init__("\n".join(self.violations))


def _to_float(x):
    return float(x)


def _check_one(metric, actual_raw, spec):
    kind = spec["kind"]
    actual = _to_float(actual_raw)
    if kind == "min":
        target = _to_float(spec["value"])
        if actual < target:
            return f"{metric}: actual {actual} < min {target}"
    elif kind == "max_ms":
        target = _to_float(spec["value"])
        if actual > target:
            return f"{metric}: actual {actual} ms > max {target} ms"
    elif kind == "within":
        target = _to_float(spec["value"])
        pct = _to_float(spec["tolerance_pct"])
        lo, hi = target * (1 - pct / 100.0), target * (1 + pct / 100.0)
        if not (lo <= actual <= hi):
            return f"{metric}: actual {actual} outside {target} ±{pct}%"
    elif kind == "min_tok_s":
        target = _to_float(spec["value"])
        if actual < target:
            return f"{metric}: actual {actual} tok/s < min {target} tok/s"
    elif kind == "min_ratio":
        ref_metric = spec["reference"]
        ratio = _to_float(spec["value"])
        actuals = spec.get("_actuals", {})
        if ref_metric not in actuals:
            return f"{metric}: reference metric '{ref_metric}' missing from actuals"
        ref_actual = _to_float(actuals[ref_metric])
        if ref_actual == 0:
            return f"{metric}: reference '{ref_metric}' is 0; cannot compute ratio"
        observed = actual / ref_actual
        if observed < ratio:
            return f"{metric}: observed ratio {observed:.3f} < min {ratio} (vs {ref_metric})"
    else:
        return f"{metric}: unknown threshold kind '{kind}'"
    return None


def evaluate_all(actuals, thresholds):
    violations = []
    for metric, spec in thresholds.items():
        if metric not in actuals:
            violations.append(f"{metric}: missing from actuals")
            continue
        spec_with_actuals = dict(spec)
        if spec.get("kind") == "min_ratio":
            spec_with_actuals["_actuals"] = actuals
        v = _check_one(metric, actuals[metric], spec_with_actuals)
        if v:
            violations.append(v)
    if violations:
        raise ThresholdViolation(violations)
