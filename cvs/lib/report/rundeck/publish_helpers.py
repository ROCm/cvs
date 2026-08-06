'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

Shared helpers for Run Deck artifact publishing (pytest session finish and tests).
'''

from __future__ import annotations

import importlib.metadata
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional

from cvs.lib import globals
from cvs.lib.report.types import InferenceReportConfig

log = globals.log


def cvs_version() -> str:
    try:
        return importlib.metadata.version("cvs")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def bundle_artifact_hrefs(
    *,
    html_path: Path,
    log_file_path: str,
    report_manager,
) -> tuple[Path, str, str]:
    """Return output dir and zip-safe hrefs for pytest HTML and run log."""
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


def enrich_provenance(
    provenance: Mapping[str, str],
    *,
    config: InferenceReportConfig,
    variant_config: Any,
    runtime_provenance: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Merge runtime fields and optional profile hooks into report provenance."""
    prov = dict(provenance)
    if isinstance(runtime_provenance, dict):
        prov.update({k: str(v) for k, v in runtime_provenance.items() if v})

    if variant_config is not None and not prov.get("image_display"):
        run_card = getattr(variant_config, "run_card", None)
        image_tag = (
            prov.get("image_tag") or getattr(run_card, "image_tag", None) or getattr(run_card, "image_pin", None) or ""
        )
        if image_tag:
            from cvs.core.image_display import format_image_display

            prov.setdefault("image_tag", str(image_tag))
            prov["image_display"] = format_image_display(image_tag=str(image_tag))

    launch_builder = getattr(config, "launch_provenance_builder", None)
    if launch_builder and variant_config is not None:
        try:
            prov.update({k: str(v) for k, v in launch_builder(variant_config).items() if v})
        except Exception as exc:
            log.warning("Could not build launch provenance: %s", exc)
    return prov
