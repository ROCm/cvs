'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Generic inference suite reports for DTNI.

Sits on top of the tabular ``print_results_table`` output: same ``inf_res_dict``
keys and column presets, plus optional HTML/JSON dashboard when pytest ``--html`` is set.

**Suite conftest** — wire via ``inference_wiring`` (see ``ADDING_A_SUITE.md``)::

    from cvs.lib.report.inference_wiring import (
        bind_inference_suite_report_session,
        configure_inference_suite_report,
    )
    from cvs.lib.report.presets.my_suite import MY_SUITE_REPORT_CONFIG

    def pytest_configure(config):
        configure_inference_suite_report(config, MY_SUITE_REPORT_CONFIG)

Demo preset: ``cvs.lib.report.presets.inferencex_atom.INFERENCEX_ATOM_REPORT_CONFIG``.
Session-end generation is automatic when ``--html`` is set; no lifecycle report test.
'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from cvs.lib import globals
from cvs.lib.report.artifacts import write_html_json_artifacts
from cvs.lib.report.inference_html import render_report_embed_html, render_report_html
from cvs.lib.report.inference_payload import build_inference_report_payload
from cvs.lib.report.provenance import build_inference_report_provenance
from cvs.lib.report.types import InferenceReportConfig
from cvs.lib.report.viewer.scaffold import viewer_basename_for, write_interactive_viewer

log = globals.log

# Re-export public API for existing imports.
__all__ = [
    "build_inference_report_payload",
    "make_report_test",
    "publish_inference_suite_report",
    "render_report_embed_html",
    "render_report_html",
    "write_report",
]


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
    """Build payload, render HTML + JSON sidecar. Returns artifact paths and payload."""
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
    payload["report"]["session_lifecycle_labels"] = config.session_lifecycle_labels
    payload["report"]["cell_lifecycle_labels"] = config.cell_lifecycle_labels

    path = out_path
    total_cells = len(payload["cells"])
    viewer_path = None
    summary_mode = config.interactive_viewer and total_cells > config.viewer_cell_threshold
    if summary_mode:
        payload["summary"] = {
            "mode": "truncated",
            "total_cells": total_cells,
            "inline_limit": config.viewer_cell_threshold,
            "viewer_html": viewer_basename_for(config.report_basename),
            "gated_tiers": list(config.gated_tiers),
        }
    else:
        payload["summary"] = {"mode": "full", "total_cells": total_cells}
        if config.interactive_viewer:
            payload["summary"]["viewer_html"] = viewer_basename_for(config.report_basename)

    html_path, json_path = write_html_json_artifacts(
        path,
        payload=payload,
        render_html=render_report_html,
    )

    if config.interactive_viewer:
        viewer_name = viewer_basename_for(config.report_basename)
        viewer_path = path.parent / viewer_name
        write_interactive_viewer(
            viewer_path,
            json_basename=f"{config.report_basename}.json",
            title=config.title,
            subtitle=config.subtitle,
            tier_order=config.metric_tier_order,
            embed_payload=payload,
        )
    result = {"html": html_path, "json": json_path, "payload": payload}
    if viewer_path is not None:
        result["viewer"] = viewer_path
    return result


def publish_inference_suite_report(
    config: InferenceReportConfig,
    *,
    variant_config,
    inf_res_dict: Mapping[tuple, Any],
    lifecycle_report: Mapping[str, list],
    report_manager,
    pytest_config,
) -> Optional[dict]:
    """Write suite report artifacts and register them with the pytest HTML bundle."""
    if variant_config is None:
        log.warning("Skipping suite report generation: variant_config not in session store")
        return None

    import importlib.metadata

    try:
        cvs_version = importlib.metadata.version("cvs")
    except importlib.metadata.PackageNotFoundError:
        cvs_version = "dev"

    htmlpath = getattr(pytest_config.option, "htmlpath", None)
    if not htmlpath:
        return None

    html_path = Path(htmlpath).resolve()
    log_file = getattr(pytest_config.option, "log_file", None)
    provenance = build_inference_report_provenance(
        pytest_config,
        cvs_version=cvs_version,
        pytest_html_path=str(html_path),
        log_file_path=str(Path(log_file).resolve()) if log_file else "",
    )
    artifacts = write_report(
        html_path.parent / f"{config.report_basename}.html",
        config=config,
        variant_config=variant_config,
        inf_res_dict=inf_res_dict,
        lifecycle_report=lifecycle_report,
        cvs_version=cvs_version,
        pytest_html_path=str(html_path),
        log_file_path=str(Path(log_file).resolve()) if log_file else "",
        provenance=provenance,
    )
    log.info(
        "Suite report written (%s): %s (json: %s)",
        config.suite_id,
        artifacts["html"],
        artifacts["json"],
    )

    if report_manager and report_manager.is_enabled:
        report_manager.add_html_to_report(artifacts["html"], link_name=config.link_name)
        report_manager.add_html_to_report(
            artifacts["json"], link_name=f"{config.link_name} JSON"
        )
        viewer = artifacts.get("viewer")
        if viewer is not None:
            report_manager.add_html_to_report(
                viewer, link_name=f"{config.link_name} viewer"
            )

    return artifacts


def make_report_test(config: InferenceReportConfig):
    """Return a lifecycle stage test that writes the suite report when pytest-html is enabled."""

    def test_report(inf_res_dict, variant_config, lifecycle, request):
        htmlpath = getattr(request.config.option, "htmlpath", None)
        if not htmlpath:
            return

        import pytest_html

        mgr = getattr(request.config, "_html_report_manager", None)
        artifacts = publish_inference_suite_report(
            config,
            variant_config=variant_config,
            inf_res_dict=inf_res_dict,
            lifecycle_report=lifecycle.report,
            report_manager=mgr,
            pytest_config=request.config,
        )
        if artifacts and mgr and getattr(request.config.option, "self_contained_html", False):
            embed = render_report_embed_html(artifacts["payload"])
            request.node.user_properties.append(
                ("pytest_html_extra", pytest_html.extras.html(embed))
            )

    test_report.__doc__ = (
        f"Write {config.title} HTML/JSON when pytest-html is enabled (render-only)."
    )
    return test_report
