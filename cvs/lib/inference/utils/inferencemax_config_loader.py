'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

InferenceMax suite JSON loader (legacy ``config`` + ``benchmark_params`` shape).

Threshold sibling discovery matches :func:`cvs.lib.utils.config_loader.substitute_config`
glob semantics until InferenceMax migrates to the typed ``VariantConfig`` schema.
'''

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict


def _merge_inferencemax_threshold_into_benchmark_params(suite: Dict[str, Any], th: Dict[str, Any]) -> None:
    """Merge InferenceMax threshold JSON into ``suite[\"benchmark_params\"]`` (mutates ``suite``)."""
    bp = suite.get("benchmark_params")
    if not isinstance(bp, dict):
        return

    th_bp = th.get("benchmark_params")
    per_model: Dict[str, Dict[str, Any]] = {}
    if isinstance(th_bp, dict):
        for model, patch in th_bp.items():
            if isinstance(patch, dict) and "result_dict" in patch and isinstance(patch["result_dict"], dict):
                per_model[model] = copy.deepcopy(patch["result_dict"])

    default_rd = None
    if "result_dict" in th and isinstance(th["result_dict"], dict):
        default_rd = copy.deepcopy(th["result_dict"])

    for model, model_cfg in bp.items():
        if not isinstance(model_cfg, dict):
            continue
        if model in per_model:
            model_cfg["result_dict"] = per_model[model]
        elif default_rd is not None:
            model_cfg["result_dict"] = copy.deepcopy(default_rd)


def inferencemax_benchmark_model_name(suite: Dict[str, Any]) -> str:
    """Pick the ``benchmark_params`` sub-key the single-node InferenceMax suite runs."""
    bp = suite.get("benchmark_params")
    if not isinstance(bp, dict) or not bp:
        raise ValueError("InferenceMax suite JSON must contain a non-empty benchmark_params object")

    explicit = suite.get("benchmark_model")
    if explicit is not None and str(explicit).strip():
        name = str(explicit).strip()
        if name not in bp:
            raise ValueError(
                f'suite "benchmark_model" is {name!r} but that key is missing from benchmark_params '
                f"(available: {sorted(bp.keys())!r})"
            )
        if not isinstance(bp[name], dict):
            raise ValueError(f"benchmark_params[{name!r}] must be an object")
        return name

    keys = [k for k in bp if not str(k).startswith("_")]
    if not keys:
        raise ValueError("benchmark_params has no model keys (only _*-prefixed entries?)")
    if len(keys) > 1:
        raise ValueError(
            "benchmark_params has multiple model keys "
            f"{sorted(keys)!r}; set top-level \"benchmark_model\" to the key this suite should run."
        )
    if not isinstance(bp[keys[0]], dict):
        raise ValueError(f"benchmark_params[{keys[0]!r}] must be an object")
    return keys[0]


def load_inferencemax_suite_raw(config_path) -> Dict[str, Any]:
    """Load InferenceMax suite JSON and optionally merge sibling ``*threshold.json``."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"InferenceMax suite config not found: {config_path}")

    suite: Dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))

    threshold_candidates = sorted(config_path.parent.glob("*threshold.json"))
    if len(threshold_candidates) > 1:
        raise ValueError(f"multiple *threshold.json files next to config (ambiguous): {threshold_candidates}")
    if not threshold_candidates:
        return suite

    th = json.loads(threshold_candidates[0].read_text(encoding="utf-8"))
    th = {k: v for k, v in th.items() if not k.startswith("_")}
    _merge_inferencemax_threshold_into_benchmark_params(suite, th)
    return suite
