"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pydantic import ValidationError

from cvs.lib.manifest.layout import RunLayout
from cvs.lib.manifest.schema import Manifest

logger = logging.getLogger(__name__)

# Fixed identity/verdict columns of the fact table, in row order. Used so an
# empty (zero manifests / all-unreadable) export still yields a well-formed,
# column-bearing table instead of a schema-less Parquet that KeyErrors
# downstream. MUST stay in sync with the static keys built by
# _flatten_manifest (a parity unit test guards this); dynamic scalar_* columns
# are appended per run when rows exist.
FACT_COLUMNS = [
    "run_id",
    "test_id",
    "cell_id",
    "config_hash",
    "workload_hash",
    "verification_hash",
    "cvs_git_sha",
    "started_at",
    "finished_at",
    "model",
    "overall_status",
    "failure_category",
]


def _flatten_manifest(manifest: Manifest) -> Dict[str, object]:
    """One fact-table row per run: identity + verdict + flattened scalars."""
    ident = manifest.identity
    row: Dict[str, object] = {
        "run_id": ident.run_id,
        "test_id": ident.test_id,
        "cell_id": ident.cell_id,
        "config_hash": ident.config_hash,
        "workload_hash": ident.workload_hash,
        "verification_hash": ident.verification_hash,
        "cvs_git_sha": ident.cvs_git_sha,
        "started_at": ident.started_at,
        "finished_at": ident.finished_at,
        "model": manifest.config.model,
        "overall_status": manifest.verdicts.overall_status,
        "failure_category": manifest.verdicts.failure_category,
    }
    for name, value in manifest.verdicts.scalars.items():
        row[f"scalar_{name}"] = value
    return row


def _within_window(started_at: Optional[str], cutoff: datetime, run_id: str) -> bool:
    """True iff started_at parses and is >= cutoff. Logs a visible WARNING on skip.

    A None or unparseable started_at is a legitimate schema state (Identity.started_at
    is Optional[str]); skipping silently would mask data; aborting would let one bad
    timestamp poison a fleet-wide export. So: warn + drop.
    """
    if started_at is None:
        logger.warning("skipping run %s: started_at is null", run_id)
        return False
    try:
        ts = datetime.fromisoformat(started_at)
    except (ValueError, TypeError) as exc:
        logger.warning("skipping run %s: unparseable started_at %r (%s)", run_id, started_at, exc)
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts >= cutoff


def collect_manifests(root, since: Optional[timedelta] = None) -> List[Manifest]:
    """Walk a run-directory tree and load every ``manifest.json`` found.

    If `since` is provided, only manifests whose Identity.started_at is within
    `since` of now (UTC) are returned. Manifests with a null or unparseable
    started_at are dropped with a visible WARNING (matching the corrupt-manifest
    visible-skip discipline). Default `since=None` preserves prior behavior.
    """
    root_path = Path(root)
    cutoff = datetime.now(timezone.utc) - since if since is not None else None
    manifests: List[Manifest] = []
    for manifest_path in sorted(root_path.rglob(RunLayout.MANIFEST)):
        # Skip unreadable/partial/invalid manifests: OSError (I/O), ValueError
        # (covers json.JSONDecodeError and UnicodeDecodeError), and ValidationError
        # (schema mismatch). Make the skip VISIBLE -- a silent drop masks schema
        # regressions and fleet-wide data loss.
        try:
            manifest = Manifest.read(manifest_path)
        except (OSError, ValueError, ValidationError) as exc:
            logger.warning("skipping unreadable manifest %s: %s", manifest_path, exc)
            continue
        if cutoff is not None and not _within_window(manifest.identity.started_at, cutoff, manifest.identity.run_id):
            continue
        manifests.append(manifest)
    return manifests


def export_runs(root, out_path, since: Optional[timedelta] = None) -> Path:
    """Flatten N run directories into one Parquet fact table.

    The result opens directly in pandas / polars / duckdb -- no service. Joining
    sidecar rows is left to the analyst (the manifest carries sidecar pointers).

    `since` (optional timedelta) restricts the export to runs whose
    Identity.started_at is within the window; default None preserves prior behavior.
    """
    manifests = collect_manifests(root, since=since)
    rows = [_flatten_manifest(m) for m in manifests]
    # Empty input must still produce the fixed-schema fact table, not a
    # column-less Parquet (which KeyErrors any downstream column access).
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=FACT_COLUMNS)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    return out
