'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Training suite reports (Megatron / JAX). Independent of inference reports and
inference framework parity (M4).
'''

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from cvs.lib import globals
from cvs.lib.report.panels.training_parity import (
    build_training_parity_panel,
    render_training_parity_panel_html,
)
from cvs.lib.report.types import TrainingReportConfig

log = globals.log


def build_training_report_payload(
    *,
    config: TrainingReportConfig,
    training_res_dict: Mapping[str, dict],
    cvs_version: str = "unknown",
    pytest_html_path: str = "",
) -> dict:
    """Structured payload for training HTML/JSON export."""
    node_rows = []
    for node_id in sorted(training_res_dict):
        metrics = training_res_dict[node_id]
        row = {"node": node_id}
        for key, _label in config.node_metric_columns:
            row[key] = metrics.get(key)
        node_rows.append(row)

    panels = {}

    return {
        "schema_version": 1,
        "report_kind": "training",
        "report_kind": "training",
        "suite_id": config.suite_id,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "cvs_version": cvs_version,
        "nodes": dict(training_res_dict),
        "node_rows": node_rows,
        "node_metric_columns": [
            {"key": key, "label": label} for key, label in config.node_metric_columns
        ],
        "report": {
            "title": config.title,
            "subtitle": config.subtitle,
            "footer": config.footer,
        },
        "provenance": {"pytest_html_path": pytest_html_path},
        "panels": panels,
    }


def render_training_report_html(payload: dict) -> str:
    report = payload["report"]
    columns = payload.get("node_rows") or []
    col_specs = payload.get("node_metric_columns") or []
    headers = ["Node"] + [spec["label"] for spec in col_specs]
    keys = [spec["key"] for spec in col_specs]
    body = "".join(
        "<tr>"
        + f"<td>{html.escape(str(row.get('node', '')))}</td>"
        + "".join(f"<td>{html.escape(str(row.get(k, '')))}</td>" for k in keys)
        + "</tr>"
        for row in columns
    )
    parity_html = ""
    parity_panel = (payload.get("panels") or {}).get("training_parity")
    if parity_panel and parity_panel.get("rows"):
        parity_html = (
            f"<section class='panel'><h2>Training parity</h2>"
            f"{render_training_parity_panel_html(parity_panel)}</section>"
        )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>{html.escape(report['title'])}</title>
<style>body{{font-family:system-ui;margin:1.5rem}} table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ddd;padding:0.5rem}}</style></head><body>
<h1>{html.escape(report['title'])}</h1>
<p>{html.escape(report['subtitle'])}</p>
<table><tr>{''.join(f'<th>{html.escape(h)}</th>' for h in headers)}</tr>{body}</table>
{parity_html}
<footer><p>{html.escape(report['footer'])}</p></footer>
</body></html>"""


def write_training_report(
    path: Path,
    *,
    config: TrainingReportConfig,
    training_res_dict: Mapping[str, dict],
    cvs_version: str = "unknown",
    pytest_html_path: str = "",
) -> dict:
    payload = build_training_report_payload(
        config=config,
        training_res_dict=training_res_dict,
        cvs_version=cvs_version,
        pytest_html_path=pytest_html_path,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json_path = path.with_suffix(".json")
    export = {k: v for k, v in payload.items() if not k.startswith("_")}
    json_path.write_text(json.dumps(export, indent=2, default=str), encoding="utf-8")

    from cvs.lib.report.training_parity_session import attach_training_parity_panel

    attach_training_parity_panel(payload, config, candidate_json_path=json_path)

    path.write_text(render_training_report_html(payload), encoding="utf-8")
    export = {k: v for k, v in payload.items() if not k.startswith("_")}
    json_path.write_text(json.dumps(export, indent=2, default=str), encoding="utf-8")
    return {"html": path, "json": json_path, "payload": payload}


def publish_training_suite_report(
    config: TrainingReportConfig,
    *,
    training_res_dict: Mapping[str, dict],
    report_manager,
    pytest_config,
) -> Optional[dict]:
    import importlib.metadata

    try:
        cvs_version = importlib.metadata.version("cvs")
    except importlib.metadata.PackageNotFoundError:
        cvs_version = "dev"

    htmlpath = getattr(pytest_config.option, "htmlpath", None)
    if not htmlpath:
        return None

    html_path = Path(htmlpath).resolve()
    artifacts = write_training_report(
        html_path.parent / f"{config.report_basename}.html",
        config=config,
        training_res_dict=training_res_dict,
        cvs_version=cvs_version,
        pytest_html_path=str(html_path),
    )
    log.info(
        "Training suite report written (%s): %s",
        config.suite_id,
        artifacts["html"],
    )
    if report_manager and report_manager.is_enabled:
        report_manager.add_html_to_report(artifacts["html"], link_name=config.link_name)
        report_manager.add_html_to_report(
            artifacts["json"], link_name=f"{config.link_name} JSON"
        )
    return artifacts
