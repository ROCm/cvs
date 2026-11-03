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


def vision_cell_dimensions(_variant_config, sweep_name):
    """Parse a vision sweep name into deck cell dimensions.

    Sweep names already carry the axes, for example
    ``MODEL=resnet50,PRECISION=BF16,MBS=128,GA=1,RES=224,DATA=rocAL-GPU``.
    Missing tokens yield empty strings.
    """
    tokens = {key.upper(): value.strip() for key, value in _TOKEN_RE.findall(str(sweep_name or ""))}
    return {
        "model": tokens.get("MODEL") or "",
        "precision": tokens.get("PRECISION") or "",
        "mbs": tokens.get("MBS") or "",
        "ga": tokens.get("GA") or "",
        "res": tokens.get("RES") or "",
        "data": tokens.get("DATA") or "",
    }


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
