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
            }, ...
          }
        }, ...
      }
    }
'''

from __future__ import annotations

from typing import Any, Mapping

from cvs.lib.report.rundeck.dataset_builders.registry import register_dataset_builder

_STATUSES = ("pass", "fail", "na")


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
            grid[node][group_name] = {
                "status": status,
                "items_summary": entry.get("items_summary", ""),
                "items": entry.get("items") or [],
                "errors_json_href": entry.get("errors_json_href", ""),
                "log_tarball_href": entry.get("log_tarball_href", ""),
            }

    overall = "fail" if n_fail else ("pass" if n_pass else "na")

    run_card_display = [
        ("Cluster", str(meta.get("cluster") or "—"), False),
        ("ANC version", str(meta.get("version") or "—"), False),
        ("Suite", str(meta.get("suite") or "—"), False),
        ("Groups", str(len(group_names)), False),
        ("Nodes", str(len(node_labels)), False),
        ("Passed", f"{n_pass} node×group", False),
        ("Failed", f"{n_fail} node×group", False),
    ]

    return {
        "nodes": node_labels,
        "groups": group_names,
        "grid": grid,
        "overall_status": overall,
        "run_card_display": run_card_display,
        "counts": {"pass": n_pass, "fail": n_fail, "na": n_na},
        "meta": dict(meta),
    }
