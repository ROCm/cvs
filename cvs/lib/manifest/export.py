"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

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


def collect_manifests(root) -> List[Manifest]:
    """Walk a run-directory tree and load every ``manifest.json`` found."""
    root_path = Path(root)
    manifests: List[Manifest] = []
    for manifest_path in sorted(root_path.rglob(RunLayout.MANIFEST)):
        # Skip unreadable/partial/invalid manifests: OSError (I/O), ValueError
        # (covers json.JSONDecodeError and UnicodeDecodeError), and ValidationError
        # (schema mismatch). Make the skip VISIBLE -- a silent drop masks schema
        # regressions and fleet-wide data loss.
        try:
            manifests.append(Manifest.read(manifest_path))
        except (OSError, ValueError, ValidationError) as exc:
            logger.warning("skipping unreadable manifest %s: %s", manifest_path, exc)
            continue
    return manifests


def export_runs(root, out_path) -> Path:
    """Flatten N run directories into one Parquet fact table.

    The result opens directly in pandas / polars / duckdb -- no service. Joining
    sidecar rows is left to the analyst (the manifest carries sidecar pointers).
    """
    manifests = collect_manifests(root)
    rows = [_flatten_manifest(m) for m in manifests]
    # Empty input must still produce the fixed-schema fact table, not a
    # column-less Parquet (which KeyErrors any downstream column access).
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=FACT_COLUMNS)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    return out
