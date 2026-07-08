'''Run provenance helpers for suite reports (config paths, version, git).'''

from __future__ import annotations

import subprocess
from typing import Any, List, Mapping, Tuple


RunCardRow = Tuple[str, str, bool]


def git_commit_short() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def build_inference_report_provenance(
    pytest_config,
    *,
    cvs_version: str,
    pytest_html_path: str = "",
    log_file_path: str = "",
    pytest_html_href: str = "",
    log_file_href: str = "",
) -> dict[str, str]:
    cluster = getattr(pytest_config.option, "cluster_file", None) or ""
    config_file = getattr(pytest_config.option, "config_file", None) or ""
    provenance: dict[str, str] = {
        "cvs_version": cvs_version,
        "pytest_html_path": pytest_html_path,
        "log_file_path": log_file_path,
        "cluster_file": str(cluster) if cluster else "",
        "config_file": str(config_file) if config_file else "",
    }
    commit = git_commit_short()
    if commit:
        provenance["git_commit"] = commit
    if pytest_html_path or pytest_html_href:
        from pathlib import Path

        href = pytest_html_href or Path(pytest_html_path).name
        provenance["pytest_html_href"] = href
        provenance["pytest_html_basename"] = href
    if log_file_path or log_file_href:
        from pathlib import Path

        href = log_file_href or Path(log_file_path).name
        provenance["log_file_href"] = href
    return provenance


def provenance_run_card_rows(provenance: Mapping[str, Any]) -> List[RunCardRow]:
    rows: List[RunCardRow] = []
    if provenance.get("cvs_version"):
        rows.append(("CVS version", str(provenance["cvs_version"]), False))
    if provenance.get("git_commit"):
        rows.append(("Git commit", str(provenance["git_commit"]), False))
    if provenance.get("cluster_file"):
        rows.append(("Cluster file", str(provenance["cluster_file"]), False))
    if provenance.get("config_file"):
        rows.append(("Config file", str(provenance["config_file"]), False))
    return rows


def extend_run_card_display(
    rows: List[RunCardRow],
    provenance: Mapping[str, Any],
) -> List[RunCardRow]:
    seen = {label for label, _value, _link in rows}
    out = list(rows)
    for label, value, is_link in provenance_run_card_rows(provenance):
        if label not in seen:
            out.append((label, value, is_link))
            seen.add(label)
    return out
