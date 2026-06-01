"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

# Long-format sidecars: one row per (request|sample) and one row per
# (step, metric, role, host). New metrics add ROWS, never COLUMNS, so the
# Parquet schema is stable across frameworks and runs.
SAMPLE_COLUMNS = ["request_id", "ts", "metric", "value", "role", "host"]
TRAJECTORY_COLUMNS = ["step", "metric", "value", "role", "host"]


def _write_parquet(path, rows: List[Dict]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(p, index=False)
    return p


def write_samples(path, rows: List[Dict]) -> Path:
    """Write per-sample rows (wide dicts allowed) to ``samples.parquet``.

    Rows are stored as-is (wide), e.g. ``{request_id, ts, ttft_ms, tpot_ms,
    itl_ms, e2el_ms, output_tokens, role, host}``. Thresholds read named columns
    from this table.
    """
    return _write_parquet(path, rows)


def write_trajectory(path, rows: List[Dict]) -> Path:
    """Write long-format trajectory rows ``{step, metric, value, role, host}``."""
    return _write_parquet(path, rows)


def read_samples(path) -> pd.DataFrame:
    return pd.read_parquet(path)


def read_trajectory(path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_resolved_config(path, config_dump: Dict) -> Path:
    """Write the fully-resolved config as YAML.

    Recorded as-is: security/redaction was removed (W7), so the dump is written
    verbatim. Target clusters are closed/internal; do not claim redaction here.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(config_dump, sort_keys=False))
    return p
