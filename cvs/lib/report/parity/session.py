'''Session-end inference parity publishing (Phase E).'''

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Optional

from cvs.lib import globals
from cvs.lib.report.parity.inference import (
    build_session_parity_config,
    publish_inference_parity_report,
)
from cvs.lib.report.types import InferenceReportConfig

log = globals.log

_ENV_COMPARE = "CVS_INFERENCE_PARITY_COMPARE"


def resolve_parity_compare_jsons(
    report_config: InferenceReportConfig,
) -> tuple[tuple[str, str], ...]:
    """Merge preset ``parity_compare_jsons`` with optional ``CVS_INFERENCE_PARITY_COMPARE`` env.

    Env format (comma-separated)::

        vllm=/path/vllm_report.json,sglang=/path/sglang_report.json
    """
    merged: dict[str, str] = dict(report_config.parity_compare_jsons)
    raw = os.environ.get(_ENV_COMPARE, "").strip()
    if raw:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            fw_id, _, path = item.partition("=")
            fw_id = fw_id.strip()
            path = path.strip()
            if fw_id and path:
                merged[fw_id] = path
            else:
                log.warning("Ignoring invalid %s entry: %r", _ENV_COMPARE, item)
    return tuple(merged.items())


def publish_session_inference_parity(
    report_config: InferenceReportConfig,
    *,
    report_manager,
    pytest_config,
    reference_json_path: Optional[Path] = None,
) -> Optional[dict]:
    """Publish inference parity HTML/JSON when compare paths are configured."""
    compare = resolve_parity_compare_jsons(report_config)
    if not compare:
        return None

    htmlpath = getattr(pytest_config.option, "htmlpath", None)
    if not htmlpath:
        return None

    if reference_json_path is None:
        reference_json_path = Path(htmlpath).resolve().parent / f"{report_config.report_basename}.json"
    if not reference_json_path.is_file():
        log.warning("Skipping inference parity: reference JSON not found: %s", reference_json_path)
        return None

    missing = [path for _, path in compare if not Path(path).is_file()]
    if missing:
        log.warning("Skipping inference parity: comparator JSON not found: %s", missing)
        return None

    effective = replace(report_config, parity_compare_jsons=compare)
    parity_config = build_session_parity_config(effective, str(reference_json_path))
    if parity_config is None:
        return None

    if not report_config.parity_metrics and {fw for fw, _ in compare} <= {"vllm", "sglang"}:
        from cvs.lib.report.presets.inference_parity import W1_INFERENCE_PARITY_METRICS

        ref_id = report_config.parity_reference_framework_id or "atom"
        parity_config = replace(
            parity_config,
            metrics=W1_INFERENCE_PARITY_METRICS,
            reference_framework_id=ref_id,
        )

    artifacts = publish_inference_parity_report(
        parity_config,
        report_manager=report_manager,
        pytest_config=pytest_config,
    )
    if artifacts:
        log.info("Inference parity report written: %s", artifacts.get("html"))
    return artifacts
