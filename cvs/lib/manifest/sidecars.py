"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

# Long-format sidecars: one row per (request|sample) and one row per
# (step, metric, role, host). New metrics add ROWS, never COLUMNS, so the
# Parquet schema is stable across frameworks and runs.
SAMPLE_COLUMNS = ["request_id", "ts", "metric", "value", "role", "host"]
TRAJECTORY_COLUMNS = ["step", "metric", "value", "role", "host"]


def _write_parquet(path, rows: List[Dict], columns: Optional[List[str]] = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the declared schema when there are no rows: an empty DataFrame
    # otherwise carries zero columns, so a downstream named-column read raises
    # KeyError instead of returning an empty series.
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns or [])
    frame.to_parquet(p, index=False)
    return p


def write_samples(path, rows: List[Dict]) -> Path:
    """Write per-sample rows (wide dicts allowed) to ``samples.parquet``.

    Rows are stored as-is (wide), e.g. ``{request_id, ts, ttft_ms, tpot_ms,
    itl_ms, e2el_ms, output_tokens, role, host}``. Thresholds read named columns
    from this table.

    Caveats: (1) per-sample metric columns are wide/dynamic, so ``SAMPLE_COLUMNS``
    is only the identity skeleton used on EMPTY input -- a populated file carries
    whatever metric keys the rows had, and a consumer must not assume a fixed
    metric column exists. (2) A metric absent in some rows widens that column to
    ``float64`` with ``NaN`` (so an int key like ``request_id`` may read back as
    float).
    """
    return _write_parquet(path, rows, SAMPLE_COLUMNS)


def write_trajectory(path, rows: List[Dict]) -> Path:
    """Write long-format trajectory rows ``{step, metric, value, role, host}``.

    Enforces the long-format invariant: a new measurement is a new ROW, never a
    new column. Any key outside ``TRAJECTORY_COLUMNS`` is rejected so the Parquet
    schema stays stable across frameworks.
    """
    allowed = set(TRAJECTORY_COLUMNS)
    for row in rows:
        extra = set(row) - allowed
        if extra:
            raise ValueError(
                f"trajectory rows are long-format; unexpected column(s) {sorted(extra)}. "
                "Emit a new metric as a new ROW (metric/value), not a new column."
            )
    return _write_parquet(path, rows, TRAJECTORY_COLUMNS)


def write_resolved_config(path, config_dump: Dict) -> Path:
    """Write the fully-resolved config as YAML.

    Recorded as-is: security/redaction was removed (W7), so the dump is written
    verbatim. Target clusters are closed/internal; do not claim redaction here.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(config_dump, sort_keys=False))
    return p
