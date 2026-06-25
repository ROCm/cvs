'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Training parity panel — compare two training suite report JSON sidecars.

Independent of inference framework parity (M4) and of multi-node scaling panels.
'''

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, List, Optional


def _load_training_nodes(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = data.get("nodes") or {}
    if nodes:
        return nodes
    # Fallback: training report may store rows list
    rows = data.get("node_rows") or []
    out = {}
    for row in rows:
        node_id = row.get("node") or row.get("host")
        if node_id:
            out[str(node_id)] = row
    return out


def _fmt(value: Any) -> str:
    if value is None:
        return "\u2014"
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def build_training_parity_panel(
    *,
    reference_json_path: str,
    candidate_json_path: str,
    reference_label: str = "Reference",
    candidate_label: str = "Candidate",
    metric_key: str = "throughput_per_gpu",
) -> dict:
    """Merge two training report JSON files into a parity panel payload."""
    ref_nodes = _load_training_nodes(Path(reference_json_path))
    cand_nodes = _load_training_nodes(Path(candidate_json_path))
    all_nodes = sorted(set(ref_nodes) | set(cand_nodes))
    rows: List[dict] = []
    for node in all_nodes:
        ref_val = (ref_nodes.get(node) or {}).get(metric_key)
        cand_val = (cand_nodes.get(node) or {}).get(metric_key)
        ratio = None
        if ref_val is not None and cand_val is not None:
            try:
                ref_f = float(ref_val)
                if ref_f > 0:
                    ratio = float(cand_val) / ref_f
            except (TypeError, ValueError):
                ratio = None
        rows.append(
            {
                "node": node,
                "reference": ref_val,
                "candidate": cand_val,
                f"compare.prev_run.{metric_key}_ratio": ratio,
            }
        )
    return {
        "panel_id": "training_parity",
        "title": "Training parity",
        "reference_label": reference_label,
        "candidate_label": candidate_label,
        "metric_key": metric_key,
        "rows": rows,
    }


def render_training_parity_panel_html(panel: dict) -> str:
    ref_lbl = html.escape(panel.get("reference_label", "Reference"))
    cand_lbl = html.escape(panel.get("candidate_label", "Candidate"))
    metric = html.escape(panel.get("metric_key", "metric"))
    ratio_key = f"compare.prev_run.{panel.get('metric_key', 'metric')}_ratio"
    body = "".join(
        f"<tr><td>{html.escape(str(r['node']))}</td>"
        f"<td>{_fmt(r.get('reference'))}</td>"
        f"<td>{_fmt(r.get('candidate'))}</td>"
        f"<td>{_fmt(r.get(ratio_key))}</td></tr>"
        for r in panel.get("rows") or []
    )
    return (
        f"<table class='results-table'><tr><th>Node</th><th>{ref_lbl}</th>"
        f"<th>{cand_lbl}</th><th>Ratio</th></tr>{body}</table>"
        f"<p class='muted'>Metric: {metric}</p>"
    )
