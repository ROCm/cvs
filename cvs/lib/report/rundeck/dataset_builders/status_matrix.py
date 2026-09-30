'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

Status matrix dataset builder for categorical (pass/fail/na) suite results such
as ANC node checks. Unlike the numeric ``matrix`` (RCCL bus_bw) or ``sweep``
(inference metrics) builders, each cell is a node x group verdict with an
optional per-item breakdown and artifact links, rendered by the
``status_matrix`` card. The card and run card both bind from this builder's
output (``datasets.status_matrix``); nothing here overlays the shared payload.

Expected ``sources["results"]`` shape (the suite's results dict)::

    {
      "_meta":  {"cluster": "...", "version": "...", "suite": "...", "generated_at": "..."},
      "groups": {
        "<group>": {
          "nodes": {
            "<node_label>": {
              "status": "pass" | "fail" | "na",
              "items_summary": "Items: 12 Total | 10 PASSED, 2 FAILED",
              "items": [{"name": "...", "status": "fail", "message": "..."}],
              "errors_json_href": "<rel path>",
              "log_tarball_href": "<rel path>",
              "metrics": [{
                "name": "rtotal", "value": 412.5, "unit": "GB/s",
                "threshold": 400.0, "direction": "higher", "status": "pass",
                "group": "a2a"
              }],
              "series": [{
                "name": "gpu_power", "unit": "W", "group": "iet",
                "x_label": "GPU", "y_label": "Power",
                "points": [{"x": "GPU0", "y": 320.1}]
              }],
              "heatmaps": [{
                "name": "xgmi", "unit": "GB/s", "group": "pbqt",
                "row_label": "Src", "col_label": "Dst",
                "rows": ["GPU0", "GPU1"], "cols": ["GPU0", "GPU1"],
                "values": [[None, 48.2], [47.1, None]],
                "threshold": 40.0, "direction": "higher"
              }]
            }, ...
          }
        }, ...
      }
    }

``metrics``, ``series``, and ``heatmaps`` are optional. Missing keys and
malformed entries are dropped so verdict-only suites keep working.

``metrics`` objects: ``name`` and numeric ``value`` are required. Optional keys
are ``unit``, ``threshold``, ``direction`` (``higher`` or ``lower``), ``status``
(``pass``, ``fail``, or ``na``), and ``group`` (defaults to the parent group).
``series`` objects: ``name`` and ``points`` are required. A point is
``{"x": ..., "y": number}`` or ``[x, y]``. Optional keys are ``unit``, ``group``,
``x_label``, and ``y_label``. ``heatmaps`` objects: ``name``, ``rows``, ``cols``,
and ``values`` are required and row/column lengths must match. Optional keys are
``unit``, ``group``, ``threshold``, ``direction``, ``row_label``, and ``col_label``.

The builder also emits ``overview`` and ``metric_charts``:

* ``overview.pass_rate`` is passed / (passed + failed), or null when nothing was
  evaluated. n/a cells are excluded. ``overview.evaluated``, ``overview.total``,
  and ``overview.counts`` (``pass``, ``fail``, ``na``) accompany it.
  ``overview.failures_by_node`` rows are ``{node, pass, fail, na}``.
  ``overview.failures_by_group`` rows are ``{group, pass, fail, na}``.
  Both lists are ordered by fail count, then label.
* ``metric_charts.metrics`` rolls scalar metrics up by group and name. Each item
  is ``{name, group, unit, threshold, direction, points}`` and each point is
  ``{node, value, unit, threshold, direction, status, group}``.
* ``metric_charts.series`` copies each series and adds ``node``.
* ``metric_charts.heatmaps`` copies each heatmap and adds ``node``.

Grid cells keep the same normalized ``metrics``, ``series``, and ``heatmaps``
without repeating ``node`` (the grid key is the node).
'''

from __future__ import annotations

from typing import Any, Mapping

from cvs.lib.report.rundeck.dataset_builders.registry import register_dataset_builder

_STATUSES = ("pass", "fail", "na")
_DIRECTIONS = ("higher", "lower")


def _results(sources: Mapping[str, Any]) -> Mapping[str, Any]:
    return sources.get("results") or sources.get("cvs_results_dict") or {}


def _ordered_nodes(groups: Mapping[str, Any]) -> list[str]:
    seen: list[str] = []
    for group in groups.values():
        for node in group.get("nodes") or {}:
            if node not in seen:
                seen.append(node)
    return seen


def _cell_status(node_entry: Any) -> str:
    if not isinstance(node_entry, dict):
        return "na"
    status = str(node_entry.get("status") or "na").lower()
    return status if status in _STATUSES else "na"


def _as_float(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _direction(value):
    text = str(value or "").strip().lower()
    return text if text in _DIRECTIONS else ""


def _metric_status(value):
    text = str(value or "").strip().lower()
    return text if text in _STATUSES else ""


def _text(value):
    if value is None:
        return ""
    return str(value)


def _group_name(raw, parent_group):
    group = raw.get("group") if isinstance(raw, dict) else None
    if group is None or str(group).strip() == "":
        return str(parent_group or "")
    return str(group)


def _threshold(raw):
    if not isinstance(raw, dict) or "threshold" not in raw:
        return None
    if raw.get("threshold") in ("", None):
        return None
    return _as_float(raw.get("threshold"))


def _normalize_metric(raw, parent_group):
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    value = _as_float(raw.get("value"))
    if not name or value is None:
        return None
    return {
        "name": name,
        "value": value,
        "unit": _text(raw.get("unit")),
        "threshold": _threshold(raw),
        "direction": _direction(raw.get("direction")),
        "status": _metric_status(raw.get("status")),
        "group": _group_name(raw, parent_group),
    }


def _point_x(value):
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return value
    return str(value)


def _normalize_point(raw):
    if isinstance(raw, dict):
        y_val = _as_float(raw.get("y"))
        if y_val is None:
            return None
        return {"x": _point_x(raw.get("x")), "y": y_val}
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        y_val = _as_float(raw[1])
        if y_val is None:
            return None
        return {"x": _point_x(raw[0]), "y": y_val}
    return None


def _normalize_series(raw, parent_group):
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    points_raw = raw.get("points")
    if not name or not isinstance(points_raw, list):
        return None
    points = []
    for item in points_raw:
        point = _normalize_point(item)
        if point is not None:
            points.append(point)
    if not points:
        return None
    return {
        "name": name,
        "unit": _text(raw.get("unit")),
        "group": _group_name(raw, parent_group),
        "x_label": _text(raw.get("x_label")),
        "y_label": _text(raw.get("y_label")),
        "points": points,
    }


def _normalize_heatmap(raw, parent_group):
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    rows = raw.get("rows")
    cols = raw.get("cols")
    values = raw.get("values")
    if not name or not isinstance(rows, list) or not isinstance(cols, list) or not isinstance(values, list):
        return None
    if not rows or not cols or len(values) != len(rows):
        return None
    normalized = []
    for row in values:
        if not isinstance(row, list) or len(row) != len(cols):
            return None
        normalized.append([_as_float(item) for item in row])
    return {
        "name": name,
        "unit": _text(raw.get("unit")),
        "group": _group_name(raw, parent_group),
        "rows": [str(item) for item in rows],
        "cols": [str(item) for item in cols],
        "values": normalized,
        "threshold": _threshold(raw),
        "direction": _direction(raw.get("direction")),
        "row_label": _text(raw.get("row_label")),
        "col_label": _text(raw.get("col_label")),
    }


def _coerce_list(entry, key, normalize, parent_group):
    raw = entry.get(key) if isinstance(entry, dict) else None
    if not isinstance(raw, list):
        return []
    kept = []
    for item in raw:
        normalized = normalize(item, parent_group)
        if normalized is not None:
            kept.append(normalized)
    return kept


def _empty_counts():
    return {"pass": 0, "fail": 0, "na": 0}


def _bump(bucket, status):
    bucket[status] = bucket.get(status, 0) + 1


def _ranked(counts, label_key):
    rows = []
    for label, bucket in counts.items():
        rows.append(
            {
                label_key: label,
                "pass": bucket["pass"],
                "fail": bucket["fail"],
                "na": bucket["na"],
            }
        )
    rows.sort(key=lambda row: (-row["fail"], str(row[label_key])))
    return rows


def _add_metric_point(order, index, node, metric):
    key = (metric["group"], metric["name"])
    bucket = index.get(key)
    if bucket is None:
        bucket = {
            "name": metric["name"],
            "group": metric["group"],
            "unit": metric["unit"],
            "threshold": metric["threshold"],
            "direction": metric["direction"],
            "points": [],
        }
        index[key] = bucket
        order.append(bucket)
    else:
        if not bucket["unit"] and metric["unit"]:
            bucket["unit"] = metric["unit"]
        if bucket["threshold"] is None and metric["threshold"] is not None:
            bucket["threshold"] = metric["threshold"]
        if not bucket["direction"] and metric["direction"]:
            bucket["direction"] = metric["direction"]
    bucket["points"].append(
        {
            "node": node,
            "value": metric["value"],
            "unit": metric["unit"],
            "threshold": metric["threshold"],
            "direction": metric["direction"],
            "status": metric["status"],
            "group": metric["group"],
        }
    )


@register_dataset_builder("status_matrix")
def build_status_matrix_datasets(sources: dict, profile: dict) -> dict:
    results = _results(sources)
    groups = results.get("groups") or {}
    meta = results.get("_meta") or {}

    group_names = list(groups.keys())
    node_labels = _ordered_nodes(groups)

    # grid[node][group] -> cell record for the expandable full-results card.
    grid = {node: {} for node in node_labels}
    n_pass = n_fail = n_na = 0
    by_node = {node: _empty_counts() for node in node_labels}
    by_group = {group_name: _empty_counts() for group_name in group_names}
    metric_order = []
    metric_index = {}
    chart_series = []
    chart_heatmaps = []

    for node in node_labels:
        for group_name in group_names:
            raw = (groups.get(group_name, {}).get("nodes") or {}).get(node)
            entry = raw if isinstance(raw, dict) else {}
            status = _cell_status(raw)
            if status == "pass":
                n_pass += 1
            elif status == "fail":
                n_fail += 1
            else:
                n_na += 1
            _bump(by_node[node], status)
            _bump(by_group[group_name], status)
            metrics = _coerce_list(entry, "metrics", _normalize_metric, group_name)
            series = _coerce_list(entry, "series", _normalize_series, group_name)
            heatmaps = _coerce_list(entry, "heatmaps", _normalize_heatmap, group_name)
            for metric in metrics:
                _add_metric_point(metric_order, metric_index, node, metric)
            for item in series:
                chart_series.append({**item, "node": node})
            for item in heatmaps:
                chart_heatmaps.append({**item, "node": node})
            grid[node][group_name] = {
                "status": status,
                "items_summary": entry.get("items_summary", ""),
                "items": entry.get("items") or [],
                "errors_json_href": entry.get("errors_json_href", ""),
                "log_tarball_href": entry.get("log_tarball_href", ""),
                "metrics": metrics,
                "series": series,
                "heatmaps": heatmaps,
            }

    overall = "fail" if n_fail else ("pass" if n_pass else "na")

    # ANC was the first status-matrix suite. Other suites set version_label so
    # the run card is not labeled as ANC.
    version_label = str(meta.get("version_label") or "ANC version")
    run_card_display = [
        ("Cluster", str(meta.get("cluster") or "—"), False),
        (version_label, str(meta.get("version") or "—"), False),
        ("Suite", str(meta.get("suite") or "—"), False),
        ("Groups", str(len(group_names)), False),
        ("Nodes", str(len(node_labels)), False),
        ("Passed", f"{n_pass} node×group", False),
        ("Failed", f"{n_fail} node×group", False),
    ]

    # n/a means the group did not run on that node, so it is not a pass or a fail.
    evaluated = n_pass + n_fail
    pass_rate = (n_pass / evaluated) if evaluated else None

    return {
        "nodes": node_labels,
        "groups": group_names,
        "grid": grid,
        "overall_status": overall,
        "run_card_display": run_card_display,
        "counts": {"pass": n_pass, "fail": n_fail, "na": n_na},
        "meta": dict(meta),
        "overview": {
            "pass_rate": pass_rate,
            "evaluated": evaluated,
            "total": n_pass + n_fail + n_na,
            "counts": {"pass": n_pass, "fail": n_fail, "na": n_na},
            "failures_by_node": _ranked(by_node, "node"),
            "failures_by_group": _ranked(by_group, "group"),
        },
        "metric_charts": {
            "metrics": metric_order,
            "series": chart_series,
            "heatmaps": chart_heatmaps,
        },
    }
