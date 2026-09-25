'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Normalize Megatron ``train_res_dict`` into Run Deck cell records.
'''

from dataclasses import replace

from cvs.lib.report.cell_build import CellRecordBuilder, metric_pass, bar_pct, margin_text
from cvs.lib.training.megatron.utils.training_config_loader import (
    DEFAULT_SWEEP_NAME,
    parse_sweep_cell_key,
)

_HOST = "cluster"
_SCALING_SUFFIX = "scaling_efficiency_pct"


def _training_nnodes(variant_config):
    if variant_config is None:
        return None
    for attr in ("nnodes", "num_nodes"):
        raw = getattr(variant_config, attr, None)
        if raw not in (None, ""):
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
    container = getattr(variant_config, "container", None)
    env = None
    if isinstance(container, dict):
        env = container.get("env")
    elif container is not None:
        env = getattr(container, "env", None)
    if isinstance(env, dict):
        raw = env.get("NNODES") or env.get("nnodes")
        if raw not in (None, ""):
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
    return None


def hide_training_scaling_efficiency(variant_config, lifecycle_report=None, suite_stem=None):
    """True for single-node Megatron / Primus (no scale-out to report)."""
    stem = str(suite_stem or "")
    if "megatron_single" in stem:
        return True
    if "megatron_distributed" in stem:
        return False
    nodeids = " ".join(str(k) for k in (lifecycle_report or {}))
    if "megatron_single.py" in nodeids:
        return True
    if "megatron_distributed.py" in nodeids:
        return False
    nnodes = _training_nnodes(variant_config)
    return nnodes is not None and nnodes <= 1


def without_scaling_efficiency(config):
    """Drop scaling-efficiency columns / highlights / charts from a training deck config."""

    def _is_scaling_key(key):
        if not key:
            return False
        return str(key) == config.full_metric(_SCALING_SUFFIX) or str(key).endswith(_SCALING_SUFFIX)

    return replace(
        config,
        results_columns=tuple((label, key) for label, key in config.results_columns if not _is_scaling_key(key)),
        cell_highlights=tuple(
            pair for pair in (config.cell_highlights or ()) if pair and pair[0] != _SCALING_SUFFIX
        ),
        chart_series=tuple(ch for ch in (config.chart_series or ()) if ch.metric_suffix != _SCALING_SUFFIX),
    )


def flatten_training_combo_actuals(raw):
    """Match ``test_metric``: last list value under ``training.<key>``; skip ``_`` keys."""
    if not isinstance(raw, dict):
        return {}
    actuals = {}
    for key, value in raw.items():
        if not key or key.startswith("_") or not value:
            continue
        sample = value[-1] if isinstance(value, (list, tuple)) else value
        try:
            actuals["training." + key] = float(sample)
        except (TypeError, ValueError, IndexError):
            continue
    return actuals


def _combo_dimensions(sweep_name, variant_config):
    sweep = getattr(variant_config, "sweep", None)
    combos = getattr(sweep, "combinations", None) or {}
    combo = combos.get(sweep_name) if isinstance(combos, dict) else None
    if combo is not None:
        if hasattr(combo, "micro_batch_size"):
            return {
                "micro_batch_size": str(combo.micro_batch_size),
                "global_batch_size": str(combo.global_batch_size),
                "precision": str(getattr(combo, "precision", "") or ""),
                "tensor_parallelism": str(
                    getattr(combo, "tensor_parallelism", None)
                    or (getattr(combo, "model_extra", None) or {}).get("tensor_parallelism")
                    or ""
                ),
                "pipeline_parallelism": str(
                    getattr(combo, "pipeline_parallelism", None)
                    or (getattr(combo, "model_extra", None) or {}).get("pipeline_parallelism")
                    or ""
                ),
            }
        if isinstance(combo, dict):
            return {
                "micro_batch_size": str(combo.get("micro_batch_size") or ""),
                "global_batch_size": str(combo.get("global_batch_size") or ""),
                "precision": str(combo.get("precision") or ""),
                "tensor_parallelism": str(combo.get("tensor_parallelism") or ""),
                "pipeline_parallelism": str(combo.get("pipeline_parallelism") or ""),
            }
    tp = getattr(variant_config, "train_params", None) or {}
    if sweep_name == DEFAULT_SWEEP_NAME or not sweep_name:
        return {
            "micro_batch_size": str(tp.get("micro_batch_size") or ""),
            "global_batch_size": str(tp.get("global_batch_size") or ""),
            "precision": str(tp.get("precision") or ""),
            "tensor_parallelism": str(tp.get("tensor_parallelism") or ""),
            "pipeline_parallelism": str(tp.get("pipeline_parallelism") or ""),
        }
    try:
        parsed = parse_sweep_cell_key(sweep_name)
    except ValueError:
        parsed = {
            "micro_batch_size": "",
            "global_batch_size": "",
            "precision": "",
        }
    parsed["tensor_parallelism"] = str(tp.get("tensor_parallelism") or "")
    parsed["pipeline_parallelism"] = str(tp.get("pipeline_parallelism") or "")
    return parsed


def _model_id(variant_config):
    tp = getattr(variant_config, "train_params", None) or {}
    return str(tp.get("tokenizer_model") or tp.get("model") or "—")


def _gpu_id(variant_config):
    return str(getattr(variant_config, "gpu_arch", None) or getattr(variant_config, "gpu_name", None) or "—")


def _train_params_parallel(variant_config, dims):
    tp = getattr(variant_config, "train_params", None) or {}
    tensor = dims.get("tensor_parallelism") or str(tp.get("tensor_parallelism") or "1")
    pipeline = dims.get("pipeline_parallelism") or str(tp.get("pipeline_parallelism") or "1")
    return tensor, pipeline


def _cell_id(variant_config, sweep_name):
    cell_key = getattr(variant_config, "cell_key", None)
    if callable(cell_key):
        try:
            return cell_key(sweep_name)
        except Exception:
            pass
    return sweep_name


def _lifecycle_for_cell(config, lifecycle_report, cell_id):
    out = {}
    for nodeid, rows in (lifecycle_report or {}).items():
        if cell_id and f"[{cell_id}" not in nodeid:
            continue
        if config.inference_test_substring and config.inference_test_substring not in nodeid:
            continue
        for label, value, unit in rows:
            if unit != "s" or label not in config.cell_lifecycle_labels:
                continue
            try:
                out[label] = float(value)
            except (TypeError, ValueError):
                continue
    return out


def _pytest_nodeids(config, lifecycle_report, cell_id):
    inference_nid = ""
    metrics_nid = ""
    for nodeid in lifecycle_report or {}:
        if cell_id and f"[{cell_id}" not in nodeid:
            continue
        if config.inference_test_substring and config.inference_test_substring in nodeid:
            inference_nid = inference_nid or nodeid
        test_name = nodeid.rsplit("::", 1)[-1].split("[", 1)[0]
        if test_name == "test_metric":
            metrics_nid = nodeid
    return {
        "pytest_inference_nodeid": inference_nid,
        "pytest_metrics_nodeid": metrics_nid,
    }


def build_training_cells(config, variant_config, train_res_dict, lifecycle_report):
    """Build deck cells from Megatron ``train_res_dict`` string sweep keys."""
    builder = CellRecordBuilder(config)
    enforce = bool(getattr(variant_config, "enforce_thresholds", False))
    thresholds_map = getattr(variant_config, "thresholds", None) or {}
    model = _model_id(variant_config)
    gpu = _gpu_id(variant_config)
    cells = []
    for sweep_name, raw in sorted((train_res_dict or {}).items(), key=lambda kv: str(kv[0])):
        if raw is None or not isinstance(raw, dict):
            continue
        actuals = flatten_training_combo_actuals(raw)
        dims = _combo_dimensions(sweep_name, variant_config)
        tensor, pipeline = _train_params_parallel(variant_config, dims)
        cell_id = _cell_id(variant_config, sweep_name)
        thresholds_cell = thresholds_map.get(cell_id) or {}
        mbs = dims.get("micro_batch_size") or ""
        gbs = dims.get("global_batch_size") or ""
        precision = dims.get("precision") or ""
        metrics = []
        for short, label in config.cell_highlights:
            full = config.full_metric(short)
            spec = thresholds_cell.get(full)
            actual = actuals.get(full)
            metrics.append(
                {
                    "label": label,
                    "metric": full,
                    "actual": actual,
                    "unit": config.metric_units.get(short) or config.metric_units.get(full, ""),
                    "spec": spec,
                    "status": (
                        metric_pass(full, actual, spec, evaluator=config.metric_verdict)
                        if enforce and spec
                        else "record"
                    ),
                    "bar_pct": (
                        bar_pct(float(actual), spec)
                        if spec is not None and actual is not None
                        else None
                    ),
                    "margin": margin_text(actual, spec) if spec else None,
                }
            )
        tiers = {
            tier: builder.tier_status(actuals, thresholds_cell, tier, enforce)
            for tier in config.metric_tier_order
        }
        subtitle = f"MBS={mbs} GBS={gbs} · {precision}" if precision else f"MBS={mbs} GBS={gbs}"
        cell = {
            "model": model,
            "gpu": gpu,
            "mbs": mbs,
            "gbs": gbs,
            "precision": precision,
            "tp": tensor,
            "pp": pipeline,
            "policy": cell_id,
            "concurrency": precision or "1",
            "host": _HOST,
            "show_host_in_label": False,
            "cell_id": cell_id,
            "label": cell_id,
            "subtitle": subtitle,
            "metrics": metrics,
            "tiers": tiers,
            "actuals": dict(actuals),
            "cell_lifecycle": _lifecycle_for_cell(config, lifecycle_report, cell_id),
            **_pytest_nodeids(config, lifecycle_report, cell_id),
        }
        for src, dst in (
            ("_loss_curve", "loss_curve"),
            ("_perplexity_curve", "perplexity_curve"),
            ("_learning_rate_curve", "learning_rate_curve"),
            ("_grad_norm_curve", "grad_norm_curve"),
            ("_throughput_curve", "throughput_curve"),
            ("_tokens_curve", "tokens_curve"),
        ):
            curve = raw.get(src)
            if isinstance(curve, list) and curve:
                cell[dst] = curve
        cells.append(cell)
    return cells


def build_training_results_table(config, cells):
    headers = [label for label, _key in config.results_columns]
    field_by_header = {
        "Model": "model",
        "GPU": "gpu",
        "MBS": "mbs",
        "GBS": "gbs",
        "Precision": "precision",
        "TP": "tp",
        "PP": "pp",
        "Host": "host",
    }
    rows = []
    for cell in cells:
        row = []
        for label, key in config.results_columns:
            if key is None:
                row.append(cell.get(field_by_header.get(label), "—") or "—")
            else:
                value = cell.get("actuals", {}).get(key)
                row.append(value if value is not None else "—")
        rows.append(row)
    return {"headers": headers, "rows": rows}


_CHART_X_DIMENSIONS = (("mbs", "MBS="), ("gbs", "GBS="), ("precision", ""))


def chart_x_labels(cells):
    """Tick labels holding only the sweep dimensions that vary; full key stays in the tooltip."""
    varying = [
        (field, prefix)
        for field, prefix in _CHART_X_DIMENSIONS
        if len({str(cell.get(field) or "") for cell in cells}) > 1
    ]
    if not varying:
        varying = [(field, prefix) for field, prefix in _CHART_X_DIMENSIONS if any(c.get(field) for c in cells)]
    labels = []
    for index, cell in enumerate(cells):
        parts = [f"{prefix}{cell.get(field)}" for field, prefix in varying if cell.get(field)]
        labels.append(" ".join(parts) or str(cell.get("cell_id") or index))
    return labels


def build_training_chart_series(config, cells):
    series = {}
    x_labels = chart_x_labels(cells)
    x_tips = [cell.get("cell_id") or x_labels[i] for i, cell in enumerate(cells)]
    for chart in config.chart_series:
        full = config.full_metric(chart.metric_suffix)
        points = []
        for index, cell in enumerate(cells):
            value = cell.get("actuals", {}).get(full)
            if value is None:
                continue
            try:
                points.append((index, float(value)))
            except (TypeError, ValueError):
                continue
        if len(points) < 2:
            continue
        series[chart.metric_suffix] = [
            {
                "isl": "all",
                "osl": "all",
                "label": "Megatron sweep",
                "points": points,
                "x_labels": [x_labels[i] for i, _val in points],
                "x_tips": [x_tips[i] for i, _val in points],
            }
        ]
    return series


def build_training_summaries(config, cells):
    best = None
    for cell in cells:
        tput = cell.get("actuals", {}).get(config.headline_metric)
        if tput is None:
            continue
        try:
            value = float(tput)
        except (TypeError, ValueError):
            continue
        if best is None or value > best[0]:
            best = (value, cell)
    if best is None:
        return []
    value, cell = best
    return [
        {
            "label": "Megatron sweep",
            "max_output_throughput": value,
            "conc_at_max_tput": cell.get("cell_id"),
            "headline_unit": config.metric_units.get("throughput_per_gpu")
            or config.metric_units.get(config.headline_metric)
            or "TFLOP/s/GPU",
            "meta": "Peak at " + str(cell.get("cell_id") or ""),
            "cell_count": len(cells),
        }
    ]
