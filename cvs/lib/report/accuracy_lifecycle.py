'''Extract accuracy metrics recorded in pytest lifecycle rows.'''

from __future__ import annotations

import math
import os
from typing import Any, Dict, Mapping, Optional

SCALE_ACCURACY_REF_ENV = "CVS_ATOM_SCALE_ACCURACY_REF_JSON"


def _finite_accuracy_value(value):
    if type(value) not in (int, float):
        return None
    try:
        normalized = float(value)
    except (OverflowError, ValueError):
        return None
    return normalized if math.isfinite(normalized) else None


def resolve_scale_accuracy_ref_json_path(config_path: str = "") -> str:
    return (config_path or os.environ.get(SCALE_ACCURACY_REF_ENV, "")).strip()


def extract_accuracy_from_lifecycle(lifecycle_report: Mapping[str, list]) -> Dict[str, float]:
    """Flatten ``test_accuracy_eval`` lifecycle records into metric keys."""
    out: Dict[str, float] = {}
    for nodeid, rows in lifecycle_report.items():
        if "test_accuracy_eval" not in nodeid:
            continue
        for label, value, unit in rows:
            if unit == "s":
                continue
            if "." not in str(label):
                continue
            normalized = _finite_accuracy_value(value)
            if normalized is not None:
                out[str(label)] = normalized
    return out


def build_accuracy_prev_run_panel(
    current: Mapping[str, float],
    baseline_payload: Mapping[str, Any],
    *,
    metric_key: str = "gsm8k_flex.gsm8k.exact_match__flexible-extract",
    max_drop: float = 0.01,
) -> Optional[dict]:
    baseline = baseline_payload.get("accuracy") or {}
    if not isinstance(baseline, dict):
        return None
    current_val = _finite_accuracy_value(current.get(metric_key))
    baseline_val = _finite_accuracy_value(baseline.get(metric_key))
    if current_val is None or baseline_val is None:
        return None
    delta = current_val - baseline_val
    regression = delta < -max_drop
    return {
        "metric_key": metric_key,
        "max_drop": max_drop,
        "current": current_val,
        "baseline": baseline_val,
        "compare.prev_run.gsm8k_delta": delta,
        "regression": regression,
    }


def build_scale_accuracy_panel(
    current: Mapping[str, float],
    reference_payload: Mapping[str, Any],
    *,
    metric_keys: tuple[str, ...] = (
        "gsm8k_flex.gsm8k.exact_match__flexible-extract",
        "hellaswag.hellaswag.acc_norm__none",
        "mmlu_pro.mmlu_pro.exact_match__custom-extract",
    ),
    max_drop: float = 0.01,
) -> Optional[dict]:
    """Compare accuracy metrics across topologies (tracker #50 scaffold)."""
    baseline = reference_payload.get("accuracy") or {}
    if not isinstance(baseline, dict):
        return None
    rows = []
    any_regression = False
    for metric_key in metric_keys:
        current_val = _finite_accuracy_value(current.get(metric_key))
        baseline_val = _finite_accuracy_value(baseline.get(metric_key))
        if current_val is None or baseline_val is None:
            continue
        delta = current_val - baseline_val
        regression = delta < -max_drop
        any_regression = any_regression or regression
        rows.append(
            {
                "metric_key": metric_key,
                "current": current_val,
                "baseline": baseline_val,
                "delta": delta,
                "regression": regression,
            }
        )
    if not rows:
        return None
    return {
        "max_drop": max_drop,
        "rows": rows,
        "regression": any_regression,
        "compare.scale_accuracy.regression": any_regression,
    }
