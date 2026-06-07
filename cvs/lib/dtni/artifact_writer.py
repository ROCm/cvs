"""Write parquet + log + verdict.json artifacts.

Filename convention: {arch}_{model_id}_{framework}_{hash}_{ts}.<ext>
- model_id is sanitized (no '/', '_' collapsed) so filename parsing is safe
- ts is UTC iso compact (no colons): 20260605T143000Z
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_SAFE_CHARS = re.compile(r"[^a-z0-9.\-]+")


def sanitize_for_filename(s: str) -> str:
    return _SAFE_CHARS.sub("-", s.lower()).strip("-")


def utc_compact_ts(now: datetime | None = None) -> str:
    n = now or datetime.now(timezone.utc)
    return n.strftime("%Y%m%dT%H%M%SZ")


def artifact_basename(*, arch: str, model_id: str, framework: str,
                      workload_hash: str, ts: str | None = None) -> str:
    ts = ts or utc_compact_ts()
    return "_".join([
        sanitize_for_filename(arch),
        sanitize_for_filename(model_id),
        sanitize_for_filename(framework),
        workload_hash,
        ts,
    ])


def write_artifacts(
    *,
    out_dir: Path,
    basename: str,
    samples_df: pd.DataFrame | None,
    log_text: str,
    verdict: dict[str, Any],
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if samples_df is not None and not samples_df.empty:
        p = out_dir / f"{basename}.parquet"
        samples_df.to_parquet(p, index=False)
        paths["parquet"] = p
    log_path = out_dir / f"{basename}.log"
    log_path.write_text(log_text or "", encoding="utf-8")
    paths["log"] = log_path
    verdict_path = out_dir / f"{basename}.verdict.json"
    verdict_path.write_text(
        json.dumps(verdict, indent=2, sort_keys=False, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["verdict"] = verdict_path
    return paths
