'''Comparison and baseline panel assembly for inference report payloads.'''

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Optional

from cvs.lib.report.accuracy_lifecycle import (
    build_accuracy_prev_run_panel,
    build_scale_accuracy_panel,
    extract_accuracy_from_lifecycle,
    resolve_scale_accuracy_ref_json_path,
)
from cvs.lib.report.json_io import load_report_json
from cvs.lib.report.panels.framework_parity import (
    build_framework_parity_panel,
    resolve_parity_ref_json_path,
)
from cvs.lib.report.panels.prev_run import build_prev_run_panel, resolve_prev_run_json_path
from cvs.lib.report.types import InferenceReportConfig


class ComparisonPanelBuilder:
    """Build optional comparison panels (prev run, accuracy, parity, launch)."""

    def __init__(
        self,
        config: InferenceReportConfig,
        report_dir: Optional[Path] = None,
        *,
        provenance: Optional[Mapping[str, str]] = None,
    ):
        self.config = config
        self.report_dir = report_dir
        self.provenance = provenance or {}

    def build(
        self,
        cells: List[dict],
        *,
        lifecycle_report: Optional[Mapping[str, list]] = None,
        variant_config=None,
    ) -> dict:
        panels: dict = {}
        if self.provenance.get("launch_server_cmd"):
            panels["launch"] = {
                "example_cell": self.provenance.get("launch_example_cell", ""),
                "server_cmd": self.provenance.get("launch_server_cmd", ""),
                "bench_cmd": self.provenance.get("launch_bench_cmd", ""),
            }

        prev_run_path = resolve_prev_run_json_path(
            self.config.prev_run_json,
            report_basename=self.config.report_basename,
            report_dir=self.report_dir,
        )
        if prev_run_path:
            prev_run_panel = build_prev_run_panel(
                cells,
                Path(prev_run_path),
                headline_metric=self.config.headline_metric,
            )
            if prev_run_panel:
                panels["prev_run"] = prev_run_panel

            if lifecycle_report:
                current_accuracy = extract_accuracy_from_lifecycle(lifecycle_report)
                baseline_payload = load_report_json(Path(prev_run_path)) or {}
                accuracy_prev = build_accuracy_prev_run_panel(
                    current_accuracy,
                    baseline_payload,
                    metric_key=self.config.gsm8k_prev_run_metric,
                    max_drop=self.config.gsm8k_prev_run_max_drop,
                )
                if accuracy_prev:
                    panels["accuracy_prev_run"] = accuracy_prev

                scale_ref = resolve_scale_accuracy_ref_json_path(getattr(self.config, "scale_accuracy_ref_json", ""))
                if scale_ref:
                    scale_payload = load_report_json(Path(scale_ref)) or {}
                    scale_panel = build_scale_accuracy_panel(current_accuracy, scale_payload)
                    if scale_panel:
                        panels["scale_accuracy"] = scale_panel

        parity_path = resolve_parity_ref_json_path(self.config.framework_parity_ref_json)
        if parity_path:
            driver = "atom"
            if variant_config is not None:
                params = getattr(variant_config, "params", None)
                if params is not None:
                    driver = str(getattr(params, "driver", "atom"))
            parity_panel = build_framework_parity_panel(
                cells,
                Path(parity_path),
                driver=driver,
                headline_metric=self.config.headline_metric,
            )
            if parity_panel:
                panels["framework_parity"] = parity_panel
        return panels
