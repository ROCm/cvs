'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

Single publish entry point for CVS Run Deck.
'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from cvs.lib import globals
from cvs.lib.report.artifacts import write_html_json_artifacts
from cvs.lib.report.ci_summary import write_inference_ci_summary
from cvs.lib.report.provenance import build_inference_report_provenance
from cvs.lib.report.registry import get_resolved_profile, get_session_results
from cvs.lib.report.rundeck.config_adapter import resolve_report_config
from cvs.lib.report.rundeck.payload import SummaryMetaApplier, build_rundeck_payload
from cvs.lib.report.rundeck.publish_helpers import bundle_artifact_hrefs, cvs_version, enrich_provenance
from cvs.lib.report.rundeck.render import render_rundeck_html
from cvs.lib.report.types import InferenceReportConfig
from cvs.lib.report.viewer.scaffold import viewer_basename_for, write_interactive_viewer

log = globals.log


class RundeckPublisher:
    """Build and publish Run Deck artifacts at pytest session finish."""

    def __init__(self, session, report_manager):
        self.session = session
        self.report_manager = report_manager
        self.config = session.config

    def publish(self) -> Optional[dict[str, Any]]:
        profile = get_resolved_profile(self.config)
        if profile is None:
            suite_name = getattr(self.config, "_suite_name", "unknown")
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
        htmlpath = getattr(self.config.option, "htmlpath", None)
        if not htmlpath:
            return None

        version = cvs_version()
        html_path = Path(htmlpath).resolve()
        log_file = getattr(self.config.option, "log_file", None)
        log_file_path = str(Path(log_file).resolve()) if log_file else ""
        out_dir, pytest_href, log_href = bundle_artifact_hrefs(
            html_path=html_path,
            log_file_path=log_file_path,
            report_manager=self.report_manager,
        )
        provenance = build_inference_report_provenance(
            self.config,
            cvs_version=version,
            pytest_html_path=str(html_path),
            log_file_path=log_file_path,
            pytest_html_href=pytest_href,
            log_file_href=log_href,
        )
        runtime = store.get("runtime_provenance") or {}
        provenance = enrich_provenance(
            provenance,
            config=config,
            variant_config=variant_config,
            runtime_provenance=runtime if isinstance(runtime, dict) else None,
        )

        payload = build_rundeck_payload(
            profile=profile,
            store=store,
            provenance=provenance,
            cvs_version=version,
            pytest_html_path=str(html_path),
            log_file_path=log_file_path,
            report_dir=out_dir,
        )
        payload = SummaryMetaApplier(config).apply(payload)

        out_path = out_dir / f"{config.report_basename}.html"
        html_path_written, json_path = write_html_json_artifacts(
            out_path,
            payload=payload,
            render_html=render_rundeck_html,
        )

        viewer_path = self._write_viewer(profile, config, out_dir, payload)
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
        self._register_artifacts(artifacts, config)
        return artifacts

    def _write_viewer(
        self,
        profile,
        config: InferenceReportConfig,
        out_dir: Path,
        payload: dict[str, Any],
    ) -> Optional[Path]:
        if not config.interactive_viewer or not isinstance(profile, (dict, InferenceReportConfig)):
            return None
        if isinstance(profile, dict) and profile.get("dataset_builder", "sweep") != "sweep":
            return None
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
        return viewer_path

    def _register_artifacts(self, artifacts: dict[str, Any], config: InferenceReportConfig) -> None:
        if not self.report_manager or not self.report_manager.is_enabled:
            return
        self.report_manager.add_html_to_report(artifacts["html"], link_name=config.link_name)
        self.report_manager.add_html_to_report(artifacts["json"], link_name=f"{config.link_name} JSON")
        self.report_manager.add_html_to_report(artifacts["summary"], link_name=f"{config.link_name} summary")
        viewer = artifacts.get("viewer")
        if viewer is not None:
            self.report_manager.add_html_to_report(viewer, link_name=f"{config.link_name} viewer")


def generate_rundeck(session, report_manager) -> Optional[dict[str, Any]]:
    """Build and publish Run Deck artifacts at pytest session finish."""
    return RundeckPublisher(session, report_manager).publish()
