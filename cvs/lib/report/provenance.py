'''Run provenance helpers for suite reports (config paths, version, git).'''

from __future__ import annotations

import subprocess
from typing import Any, List, Mapping, Tuple


RunCardRow = Tuple[str, str, bool]


class GitMetadata:
    """Read git commit, branch, and dirty state for report provenance."""

    @staticmethod
    def _run(args: list[str]) -> str:
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

    @classmethod
    def commit_short(cls) -> str:
        return cls._run(["rev-parse", "--short", "HEAD"])

    @classmethod
    def branch_name(cls) -> str:
        branch = cls._run(["rev-parse", "--abbrev-ref", "HEAD"])
        return "" if branch == "HEAD" else branch

    @classmethod
    def worktree_dirty(cls) -> bool:
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

    @staticmethod
    def format_ref(*, commit: str = "", branch: str = "", dirty: bool = False) -> str:
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


class ProvenanceCollector:
    """Collect pytest run metadata and format run-card provenance rows."""

    def __init__(
        self,
        pytest_config,
        *,
        cvs_version: str,
        pytest_html_path: str = "",
        log_file_path: str = "",
        pytest_html_href: str = "",
        log_file_href: str = "",
    ):
        self.pytest_config = pytest_config
        self.cvs_version = cvs_version
        self.pytest_html_path = pytest_html_path
        self.log_file_path = log_file_path
        self.pytest_html_href = pytest_html_href
        self.log_file_href = log_file_href

    def collect(self) -> dict[str, str]:
        cluster = getattr(self.pytest_config.option, "cluster_file", None) or ""
        config_file = getattr(self.pytest_config.option, "config_file", None) or ""
        provenance: dict[str, str] = {
            "cvs_version": self.cvs_version,
            "pytest_html_path": self.pytest_html_path,
            "log_file_path": self.log_file_path,
            "cluster_file": str(cluster) if cluster else "",
            "config_file": str(config_file) if config_file else "",
        }
        commit = GitMetadata.commit_short()
        branch = GitMetadata.branch_name()
        dirty = GitMetadata.worktree_dirty()
        if commit:
            provenance["git_commit"] = commit
        if branch:
            provenance["git_branch"] = branch
        if dirty:
            provenance["git_dirty"] = "true"
        git_ref = GitMetadata.format_ref(commit=commit, branch=branch, dirty=dirty)
        if git_ref:
            provenance["git_ref"] = git_ref
        if self.pytest_html_path or self.pytest_html_href:
            from pathlib import Path

            href = self.pytest_html_href or Path(self.pytest_html_path).name
            provenance["pytest_html_href"] = href
            provenance["pytest_html_basename"] = href
        if self.log_file_path or self.log_file_href:
            from pathlib import Path

            href = self.log_file_href or Path(self.log_file_path).name
            provenance["log_file_href"] = href
        return provenance

    @staticmethod
    def run_card_rows(provenance: Mapping[str, Any]) -> List[RunCardRow]:
        rows: List[RunCardRow] = []
        if provenance.get("cvs_version"):
            rows.append(("CVS version", str(provenance["cvs_version"]), False))
        git_ref = provenance.get("git_ref") or provenance.get("git_commit")
        if git_ref:
            rows.append(("Git ref", str(git_ref), False))
        image_display = provenance.get("image_display")
        if not image_display:
            from cvs.lib.image_display import format_image_display

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

    @staticmethod
    def extend_run_card_display(
        rows: List[RunCardRow],
        provenance: Mapping[str, Any],
    ) -> List[RunCardRow]:
        seen = {label for label, _value, _link in rows}
        out = list(rows)
        for label, value, is_link in ProvenanceCollector.run_card_rows(provenance):
            if label not in seen:
                out.append((label, value, is_link))
                seen.add(label)
        return out


def git_commit_short() -> str:
    return GitMetadata.commit_short()


def git_branch_name() -> str:
    return GitMetadata.branch_name()


def git_worktree_dirty() -> bool:
    return GitMetadata.worktree_dirty()


def format_git_ref(*, commit: str = "", branch: str = "", dirty: bool = False) -> str:
    return GitMetadata.format_ref(commit=commit, branch=branch, dirty=dirty)


def build_inference_report_provenance(
    pytest_config,
    *,
    cvs_version: str,
    pytest_html_path: str = "",
    log_file_path: str = "",
    pytest_html_href: str = "",
    log_file_href: str = "",
) -> dict[str, str]:
    return ProvenanceCollector(
        pytest_config,
        cvs_version=cvs_version,
        pytest_html_path=pytest_html_path,
        log_file_path=log_file_path,
        pytest_html_href=pytest_html_href,
        log_file_href=log_file_href,
    ).collect()


def provenance_run_card_rows(provenance: Mapping[str, Any]) -> List[RunCardRow]:
    return ProvenanceCollector.run_card_rows(provenance)


def extend_run_card_display(
    rows: List[RunCardRow],
    provenance: Mapping[str, Any],
) -> List[RunCardRow]:
    return ProvenanceCollector.extend_run_card_display(rows, provenance)
