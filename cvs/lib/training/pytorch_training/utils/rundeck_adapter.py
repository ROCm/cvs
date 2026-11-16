'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Adapt PyTorch Training per-sweep results into the shared training Run Deck.
'''

from __future__ import annotations

import math

from cvs.lib.training.pytorch_training.utils.metrics import METRIC_PREFIX, METRICS


def _sweep_record(variant_config, sweep_name):
    getter = getattr(variant_config, "sweep", None)
    if not callable(getter):
        return None
    try:
        return getter(sweep_name)
    except (KeyError, TypeError, ValueError):
        return None


def training_cell_dimensions(variant_config, sweep_name):
    """Deck cell dimensions read from the sweep record on the variant."""
    sweep = _sweep_record(variant_config, sweep_name)
    if sweep is None:
        return {"model": "", "precision": "", "seq": "", "mbs": "", "gbs": ""}
    batch = sweep.global_batch_size
    return {
        "model": str(sweep.model),
        "precision": str(sweep.precision),
        "seq": str(sweep.seq_len),
        "mbs": str(sweep.micro_batch_size),
        "gbs": str(batch) if batch is not None else f"GA{sweep.gradient_accumulation_steps}",
    }


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _flat_metrics(actuals):
    out = {}
    for short, _unit in METRICS:
        value = (actuals or {}).get(METRIC_PREFIX + short)
        if _finite(value):
            out[short] = [str(value)]
    return out


def _loss_curve(points):
    curve = []
    for point in points or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        step, loss = point[0], point[1]
        if _finite(step) and _finite(loss):
            curve.append([int(step), float(loss)])
    return curve


def flat_train_res_from_training(training_results, loss_series=None):
    """Build the flat ``train_res_dict`` the training Run Deck consumes.

    ``training_results[sweep_name]`` is ``{host: {training.<metric>: value}}``.
    Rank zero owns the artifact, so the first host is the deck cell. Loss
    samples from ``loss_series[sweep_name]`` become ``_loss_curve``.
    """
    flat = {}
    series = loss_series or {}
    for name, host_results in (training_results or {}).items():
        if not isinstance(host_results, dict) or not host_results:
            continue
        actuals = next(iter(host_results.values()))
        if not isinstance(actuals, dict):
            continue
        combo = _flat_metrics(actuals)
        curve = _loss_curve(series.get(name))
        if curve:
            combo["_loss_curve"] = curve
        flat[name] = combo
    return flat
