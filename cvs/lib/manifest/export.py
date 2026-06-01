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

from cvs.lib.manifest.layout import RunLayout
from cvs.lib.manifest.schema import Manifest


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
        try:
            manifests.append(Manifest.read(manifest_path))
        except Exception:  # noqa: BLE001 - skip unreadable/partial manifests
            continue
    return manifests


def export_runs(root, out_path) -> Path:
    """Flatten N run directories into one Parquet fact table.

    The result opens directly in pandas / polars / duckdb -- no service. Joining
    sidecar rows is left to the analyst (the manifest carries sidecar pointers).
    """
    manifests = collect_manifests(root)
    rows = [_flatten_manifest(m) for m in manifests]
    frame = pd.DataFrame(rows)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    return out
