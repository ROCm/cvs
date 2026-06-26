'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Training parity panel — compare two training suite report JSON sidecars.

Independent of inference framework parity (M4) and of multi-node scaling panels.
'''

from __future__ import annotations

import html
from pathlib import Path
from typing import List

from cvs.lib.report.compare import metric_ratio
from cvs.lib.report.formatting import fmt_num
from cvs.lib.report.json_io import load_training_nodes


def build_training_parity_panel(
    *,
    reference_json_path: str,
    candidate_json_path: str,
    reference_label: str = "Reference",
    candidate_label: str = "Candidate",
    metric_key: str = "throughput_per_gpu",
) -> dict:
    """Merge two training report JSON files into a parity panel payload."""
    ref_nodes = load_training_nodes(Path(reference_json_path))
    cand_nodes = load_training_nodes(Path(candidate_json_path))
    all_nodes = sorted(set(ref_nodes) | set(cand_nodes))
    rows: List[dict] = []
    for node in all_nodes:
        ref_val = (ref_nodes.get(node) or {}).get(metric_key)
        cand_val = (cand_nodes.get(node) or {}).get(metric_key)
        ratio = None
        if ref_val is not None and cand_val is not None:
            try:
                ratio = metric_ratio(float(cand_val), float(ref_val))
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
        f"<td>{fmt_num(r.get('reference'), digits=2)}</td>"
        f"<td>{fmt_num(r.get('candidate'), digits=2)}</td>"
        f"<td>{fmt_num(r.get(ratio_key), digits=2)}</td></tr>"
        for r in panel.get("rows") or []
    )
    return (
        f"<table class='results-table'><tr><th>Node</th><th>{ref_lbl}</th>"
        f"<th>{cand_lbl}</th><th>Ratio</th></tr>{body}</table>"
        f"<p class='muted'>Metric: {metric}</p>"
    )
