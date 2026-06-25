'''Session-end training parity publishing (Phase F).'''

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from cvs.lib import globals
from cvs.lib.report.panels.training_parity import build_training_parity_panel
from cvs.lib.report.types import TrainingReportConfig

log = globals.log

_ENV_BASELINE = "CVS_TRAINING_PARITY_BASELINE_JSON"


def resolve_training_parity_baseline(report_config: TrainingReportConfig) -> str:
    return (os.environ.get(_ENV_BASELINE) or report_config.parity_baseline_json or "").strip()


def attach_training_parity_panel(
    payload: dict,
    report_config: TrainingReportConfig,
    *,
    candidate_json_path: Path,
) -> Optional[dict]:
    """Merge training parity panel into a training report payload when baseline is set."""
    baseline = resolve_training_parity_baseline(report_config)
    if not baseline:
        return None
    if not Path(baseline).is_file():
        log.warning("Skipping training parity: baseline JSON not found: %s", baseline)
        return None
    if not candidate_json_path.is_file():
        log.warning("Skipping training parity: candidate JSON not found: %s", candidate_json_path)
        return None

    panel = build_training_parity_panel(
        reference_json_path=baseline,
        candidate_json_path=str(candidate_json_path),
        reference_label=report_config.parity_reference_label,
        candidate_label=report_config.parity_candidate_label,
        metric_key=report_config.parity_metric_key,
    )
    payload.setdefault("panels", {})["training_parity"] = panel
    return panel
