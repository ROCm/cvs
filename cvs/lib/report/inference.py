'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Test and offline helpers for sweep Run Deck payloads.

Production publish path: ``rundeck.generate_rundeck`` (session finish via
``report_plugins``). This module keeps ``build_inference_report_payload`` and
``write_report`` for unit tests until those call ``build_rundeck_payload`` directly.
'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from cvs.lib.report.artifacts import write_html_json_artifacts
from cvs.lib.report.ci_summary import write_inference_ci_summary
from cvs.lib.report.inference_payload import build_inference_report_payload
from cvs.lib.report.rundeck.payload import apply_summary_meta
from cvs.lib.report.rundeck.render import render_rundeck_html
from cvs.lib.report.types import InferenceReportConfig
from cvs.lib.report.viewer.scaffold import viewer_basename_for, write_interactive_viewer

__all__ = [
    "build_inference_report_payload",
    "render_report_html",
    "write_report",
]


def render_report_html(payload: dict) -> str:
    return render_rundeck_html(payload)


def write_report(
    path: Path,
    *,
    config: InferenceReportConfig,
    variant_config,
    inf_res_dict: Mapping[tuple, Any],
    lifecycle_report: Mapping[str, list],
    cvs_version: str = "unknown",
    pytest_html_path: str = "",
    log_file_path: str = "",
    provenance: Optional[Mapping[str, str]] = None,
) -> dict:
    """Build payload, render HTML + JSON sidecar. Used by report unit tests."""
    out_path = Path(path)
    payload = build_inference_report_payload(
        config=config,
        variant_config=variant_config,
        inf_res_dict=inf_res_dict,
        lifecycle_report=lifecycle_report,
        cvs_version=cvs_version,
        pytest_html_path=pytest_html_path,
        log_file_path=log_file_path,
        provenance=provenance,
        report_dir=out_path.parent,
    )
    payload = apply_summary_meta(payload, config)

    viewer_path = None
    html_path, json_path = write_html_json_artifacts(
        out_path,
        payload=payload,
        render_html=render_report_html,
    )

    if config.interactive_viewer:
        viewer_name = viewer_basename_for(config.report_basename)
        viewer_path = out_path.parent / viewer_name
        write_interactive_viewer(
            viewer_path,
            json_basename=f"{config.report_basename}.json",
            title=config.title,
            subtitle=config.subtitle,
            tier_order=config.metric_tier_order,
            embed_payload=payload,
        )
    summary_path = write_inference_ci_summary(payload, config, out_path.parent)
    result = {"html": html_path, "json": json_path, "payload": payload, "summary": summary_path}
    if viewer_path is not None:
        result["viewer"] = viewer_path
    return result
