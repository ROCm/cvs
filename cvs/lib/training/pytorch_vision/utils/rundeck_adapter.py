'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Adapt PyTorch Vision per-sweep results into the shared training Run Deck.
'''

from __future__ import annotations

import math
import re

from cvs.lib.training.pytorch_vision.utils.metrics import METRIC_PREFIX, METRICS

_TOKEN_RE = re.compile(r"([A-Za-z_]+)=([^,]+)")


def _sweep_record(variant_config, sweep_name):
    getter = getattr(variant_config, "sweep", None)
    if not callable(getter):
        return None
    try:
        return getter(sweep_name)
    except (KeyError, TypeError, ValueError):
        return None


def _data_label(sweep):
    mode = str(getattr(sweep, "data_mode", "") or "")
    if mode != "rocal":
        return mode
    device = str(getattr(sweep, "rocal_device", "") or "gpu").upper()
    return f"rocAL-{device}"


def vision_cell_dimensions(variant_config, sweep_name):
    """Parse a vision sweep into deck cell dimensions.

    Tokenized sweep names carry the axes, for example
    ``MODEL=resnet50,PRECISION=BF16,MBS=128,GA=1,RES=224,DATA=rocAL-GPU``.
    Smoke and scorecard names such as ``W1-SMOKE-ROCAL-GPU-STANDARD`` do not,
    so missing tokens are filled from the sweep record on the variant.
    """
    tokens = {key.upper(): value.strip() for key, value in _TOKEN_RE.findall(str(sweep_name or ""))}
    dims = {
        "model": tokens.get("MODEL") or "",
        "precision": tokens.get("PRECISION") or "",
        "mbs": tokens.get("MBS") or "",
        "ga": tokens.get("GA") or "",
        "res": tokens.get("RES") or "",
        "data": tokens.get("DATA") or "",
    }
    sweep = _sweep_record(variant_config, sweep_name)
    if sweep is None:
        return dims
    fallback = {
        "model": getattr(sweep, "model", "") or "",
        "precision": getattr(sweep, "precision", "") or "",
        "mbs": getattr(sweep, "batch_size", "") or "",
        "ga": getattr(sweep, "gradient_accumulation_steps", "") or "",
        "res": getattr(sweep, "image_size", "") or "",
        "data": _data_label(sweep),
    }
    for key, value in fallback.items():
        if not dims[key] and value not in ("", None):
            dims[key] = str(value)
    return dims


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _flat_metrics(actuals):
    out = {}
    for short, _unit in METRICS:
        value = (actuals or {}).get(METRIC_PREFIX + short)
        if not _finite(value):
            continue
        out[short] = [str(value)]
    return out


def _loss_curve(points):
    curve = []
    for point in points or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        step, loss = point[0], point[1]
        if not _finite(step) or not _finite(loss):
            continue
        curve.append([int(step), float(loss)])
    return curve


def flat_train_res_from_vision(training_results, loss_series=None):
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
