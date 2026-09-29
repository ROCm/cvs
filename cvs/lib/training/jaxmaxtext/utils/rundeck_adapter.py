'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Adapt JAX MaxText nested per-sweep results into the shared training Run Deck
cell shape, and parse a sweep name into deck cell dimensions.

The shared training deck (cvs/lib/report/training_cells.py) consumes a flat
``train_res_dict``::

    {sweep_name: {"<metric>": [values], "_loss_curve": [[step, val], ...], ...}}

where the builder takes the last value of each ``<metric>`` list as the scalar
actual and attaches the ``_*_curve`` series to the cell. The JAX suite collects
a richer nested dict for its own gating and console tables; this module derives
the flat, deck-facing view from it without disturbing that bookkeeping.
'''

from __future__ import annotations

import math
import re

from cvs.lib.training.jaxmaxtext.utils.maxtext_parsing import METRIC_PREFIX, TRAINING_METRICS

# MaxText stdout step-line keys -> deck curve keys.
_STDOUT_CURVES = (
    ("loss", "_loss_curve"),
    ("perplexity", "_perplexity_curve"),
    ("TFLOP/s/device", "_throughput_curve"),
    ("Tokens/s/device", "_tokens_curve"),
)
# TensorBoard tag candidates -> deck curve keys (first tag present wins).
_TB_CURVES = (
    (("learning/current_learning_rate", "learning/learning_rate"), "_learning_rate_curve"),
    (("learning/grad_norm", "grad_norm"), "_grad_norm_curve"),
)

_TOKEN_RE = re.compile(r"([A-Za-z_]+)=([^,]+)")


def jax_cell_dimensions(_variant_config, sweep_name):
    """Parse a JAX sweep name into deck cell dimensions.

    Sweep names look like ``PRECISION=BF16,SEQLEN=4096,BATCH=3`` (any subset);
    the implicit single run is ``default``. Returns ``{bs, sl, precision}`` keyed
    to the deck's ``dimension_fields``; missing tokens yield empty strings.
    """
    tokens = {key.upper(): value.strip() for key, value in _TOKEN_RE.findall(str(sweep_name or ""))}
    return {
        "bs": tokens.get("BATCH", ""),
        "sl": tokens.get("SEQLEN", ""),
        "precision": tokens.get("PRECISION", ""),
    }


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _steps_curve(step_metrics, key):
    curve = []
    for row in step_metrics or []:
        step = row.get("step")
        value = row.get(key)
        if step is None or not _finite(value):
            continue
        curve.append([int(step), float(value)])
    return curve


def _tb_curve(tb_scalars, tags):
    for tag in tags:
        series = (tb_scalars or {}).get(tag)
        if series:
            return [[int(step), float(value)] for step, value in series if _finite(value)]
    return []


def _flat_metrics(results):
    """``{'training.<short>': value}`` -> ``{'<short>': [str(value)]}`` (deck shape)."""
    out = {}
    for short, _unit in TRAINING_METRICS:
        value = (results or {}).get(METRIC_PREFIX + short)
        if value is None:
            continue
        out[short] = [str(value)]
    return out


def flat_train_res_from_nested(training_res_dict):
    """Derive the flat deck ``train_res_dict`` from the JAX suite's nested results.

    Input ``training_res_dict["sweeps"][name]`` carries ``results``,
    ``step_metrics``, optional ``tb_scalars`` and ``planned_steps``. Output is the
    flat mapping consumed by ``cvs/lib/report/training_cells.py``.
    """
    flat = {}
    for name, rec in (training_res_dict.get("sweeps") or {}).items():
        if not isinstance(rec, dict):
            continue
        combo = _flat_metrics(rec.get("results") or {})
        step_metrics = rec.get("step_metrics") or []
        for key, dst in _STDOUT_CURVES:
            curve = _steps_curve(step_metrics, key)
            if curve:
                combo[dst] = curve
        tb_scalars = rec.get("tb_scalars") or {}
        for tags, dst in _TB_CURVES:
            curve = _tb_curve(tb_scalars, tags)
            if curve:
                combo[dst] = curve
        planned = rec.get("planned_steps")
        if isinstance(planned, (int, float)) and planned > 0:
            combo["_planned_steps"] = int(planned)
        flat[name] = combo
    return flat
