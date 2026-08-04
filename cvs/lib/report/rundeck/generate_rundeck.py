'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Single publish entry point for CVS Run Deck.
'''

from __future__ import annotations

import importlib.metadata
import shutil
from pathlib import Path
from typing import Any, Optional

from cvs.lib import globals
from cvs.lib.report.artifacts import write_html_json_artifacts
from cvs.lib.report.ci_summary import write_inference_ci_summary
from cvs.lib.report.provenance import build_inference_report_provenance
from cvs.lib.report.registry import get_resolved_profile, get_session_results
from cvs.lib.report.rundeck.config_adapter import resolve_report_config
from cvs.lib.report.rundeck.payload import apply_summary_meta, build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html
from cvs.lib.report.types import InferenceReportConfig
from cvs.lib.report.viewer.scaffold import viewer_basename_for, write_interactive_viewer

log = globals.log


def _cvs_version() -> str:
    try:
        return importlib.metadata.version("cvs")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def _bundle_artifact_hrefs(*, html_path: Path, log_file_path: str, report_manager) -> tuple[Path, str, str]:
    out_dir = html_path.parent
    pytest_href = html_path.name
    log_href = ""
    if report_manager and report_manager.is_enabled:
        out_dir = report_manager.log_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        pytest_href = f"../{html_path.name}"
        if log_file_path:
            log_src = Path(log_file_path)
            log_href = log_src.name
            if log_src.is_file():
                shutil.copy2(log_src, out_dir / log_src.name)
    elif log_file_path:
        log_href = Path(log_file_path).name
    return out_dir, pytest_href, log_href


def generate_rundeck(session, report_manager) -> Optional[dict[str, Any]]:
    """Build and publish Run Deck artifacts at pytest session finish."""
    profile = get_resolved_profile(session.config)
    if profile is None:
        suite_name = getattr(session.config, "_suite_name", "unknown")
        log.info(
            "Skipping Run Deck generation: no deck profile registered for suite '%s'",
            suite_name,
        )
        return None

    store = get_session_results()
    results = store.get("cvs_results_dict") or store.get("inf_res_dict")
    if not results:
        log.info("Skipping Run Deck generation: no results in session store")
        return None

    variant_config = store.get("variant_config")
    builder_id = profile.get("dataset_builder") if isinstance(profile, dict) else "sweep"
    if variant_config is None and builder_id == "sweep":
        log.warning("Skipping Run Deck generation: variant_config not in session store")
        return None

    config = resolve_report_config(profile)
    cvs_version = _cvs_version()
    htmlpath = getattr(session.config.option, "htmlpath", None)
    if not htmlpath:
        return None

    html_path = Path(htmlpath).resolve()
    log_file = getattr(session.config.option, "log_file", None)
    log_file_path = str(Path(log_file).resolve()) if log_file else ""
    out_dir, pytest_href, log_href = _bundle_artifact_hrefs(
        html_path=html_path,
        log_file_path=log_file_path,
        report_manager=report_manager,
    )
    provenance = build_inference_report_provenance(
        session.config,
        cvs_version=cvs_version,
        pytest_html_path=str(html_path),
        log_file_path=log_file_path,
        pytest_html_href=pytest_href,
        log_file_href=log_href,
    )
    runtime = store.get("runtime_provenance") or {}
    if isinstance(runtime, dict):
        provenance.update({k: str(v) for k, v in runtime.items() if v})

    if variant_config is not None:
        from cvs.core.image_display import format_image_display

        if not provenance.get("image_display"):
            image_pin = getattr(getattr(variant_config, "run_card", None), "atom_image_pin", "") or ""
            if image_pin:
                provenance.setdefault("image_tag", image_pin)
                provenance["image_display"] = format_image_display(image_tag=image_pin)
        launch_builder = getattr(config, "launch_provenance_builder", None)
        if launch_builder:
            try:
                provenance.update({k: str(v) for k, v in launch_builder(variant_config).items() if v})
            except Exception as exc:
                log.warning("Could not build launch provenance: %s", exc)

    payload = build_rundeck_payload(
        profile=profile,
        store=store,
        provenance=provenance,
        cvs_version=cvs_version,
        pytest_html_path=str(html_path),
        log_file_path=log_file_path,
        report_dir=out_dir,
    )
    payload = apply_summary_meta(payload, config)

    out_path = out_dir / f"{config.report_basename}.html"
    html_path_written, json_path = write_html_json_artifacts(
        out_path,
        payload=payload,
        render_html=render_rundeck_html,
    )

    viewer_path = None
    if config.interactive_viewer and isinstance(profile, (dict, InferenceReportConfig)):
        if not isinstance(profile, dict) or profile.get("dataset_builder", "sweep") == "sweep":
            viewer_name = viewer_basename_for(config.report_basename)
            viewer_path = out_dir / viewer_name
            write_interactive_viewer(
                viewer_path,
                json_basename=f"{config.report_basename}.json",
                title=config.title,
                subtitle=config.subtitle,
                tier_order=config.metric_tier_order,
                embed_payload=payload,
            )

    summary_path = write_inference_ci_summary(payload, config, out_dir)
    artifacts = {
        "html": html_path_written,
        "json": json_path,
        "payload": payload,
        "summary": summary_path,
    }
    if viewer_path is not None:
        artifacts["viewer"] = viewer_path

    log.info(
        "Run Deck written (%s): %s (json: %s, summary: %s)",
        config.suite_id,
        artifacts["html"],
        artifacts["json"],
        artifacts["summary"],
    )

    if report_manager and report_manager.is_enabled:
        report_manager.add_html_to_report(artifacts["html"], link_name=config.link_name)
        report_manager.add_html_to_report(artifacts["json"], link_name=f"{config.link_name} JSON")
        report_manager.add_html_to_report(artifacts["summary"], link_name=f"{config.link_name} summary")
        viewer = artifacts.get("viewer")
        if viewer is not None:
            report_manager.add_html_to_report(viewer, link_name=f"{config.link_name} viewer")

    return artifacts
