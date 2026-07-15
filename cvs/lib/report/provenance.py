'''Run provenance helpers for suite reports (config paths, version, git).'''

from __future__ import annotations

import subprocess
from typing import Any, List, Mapping, Tuple


RunCardRow = Tuple[str, str, bool]


def _git_run(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_commit_short() -> str:
    return _git_run(["rev-parse", "--short", "HEAD"])


def git_branch_name() -> str:
    branch = _git_run(["rev-parse", "--abbrev-ref", "HEAD"])
    return "" if branch == "HEAD" else branch


def git_worktree_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def format_git_ref(*, commit: str = "", branch: str = "", dirty: bool = False) -> str:
    if not commit and not branch:
        return ""
    parts: list[str] = []
    if commit:
        parts.append(commit)
    if branch:
        parts.append(branch)
    ref = " @ ".join(parts) if len(parts) == 2 else parts[0]
    if dirty:
        ref = f"{ref} (dirty)"
    return ref


def format_image_display(*, image_tag: str = "", image_digest: str = "", image_id: str = "") -> str:
    digest = image_digest or image_id
    if digest:
        short = digest
        if "@" in short:
            short = short.split("@", 1)[1]
        if short.startswith("sha256:") and len(short) > 19:
            short = f"{short[:19]}\u2026"
        if image_tag:
            return f"{image_tag} @ {short}"
        return short
    return image_tag or "\u2014"


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
    branch = git_branch_name()
    dirty = git_worktree_dirty()
    if commit:
        provenance["git_commit"] = commit
    if branch:
        provenance["git_branch"] = branch
    if dirty:
        provenance["git_dirty"] = "true"
    git_ref = format_git_ref(commit=commit, branch=branch, dirty=dirty)
    if git_ref:
        provenance["git_ref"] = git_ref
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
    git_ref = provenance.get("git_ref") or provenance.get("git_commit")
    if git_ref:
        rows.append(("Git ref", str(git_ref), False))
    image_display = provenance.get("image_display")
    if not image_display:
        image_display = format_image_display(
            image_tag=str(provenance.get("image_tag") or ""),
            image_digest=str(provenance.get("image_digest") or ""),
            image_id=str(provenance.get("image_id") or ""),
        )
    if image_display and image_display != "\u2014":
        rows.append(("Image", str(image_display), False))
    if provenance.get("launch_summary"):
        rows.append(("Launch", str(provenance["launch_summary"]), False))
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
