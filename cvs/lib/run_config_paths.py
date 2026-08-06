"""
Resolve run_config paths that must work from any clone location or after pip install.

baseline_csv_folder:
  - Omit, null, or ""  -> comparisons use the same tree as the current run (legacy).
  - Absolute/relative path -> expanded; relative paths resolve from the process cwd.
  - "{use_internal_baseline_data_path}" -> cvs/baseline_data next to the installed
    or editable package (survives ``pip install`` into site-packages).
"""

from __future__ import annotations

import getpass
import os
from typing import Optional

# Token for run_config["baseline_csv_folder"] — resolves beside the cvs package.
CVS_INTERNAL_BASELINE_DATA_PATH = "{use_internal_baseline_data_path}"


def resolve_runner_results_base(run_config: dict) -> str:
    """Absolute path to run_config runner_log_folder (before nnodes/pyt subpaths)."""
    folder = run_config.get("runner_log_folder", "/tmp/cvs_results")
    if not isinstance(folder, str):
        folder = str(folder)
    if "~" in folder:
        folder = os.path.expanduser(folder)
    if "{headnode-user-id}" in folder:
        folder = folder.replace("{headnode-user-id}", getpass.getuser())
    return os.path.abspath(folder)


def _cvs_baseline_data_dir() -> str:
    import cvs

    root = os.path.dirname(os.path.abspath(cvs.__file__))
    return os.path.normpath(os.path.join(root, "baseline_data"))


def resolve_baseline_csv_folder(run_config: dict) -> Optional[str]:
    """
    Root directory that mirrors the layout under runner_log_folder (e.g. .../4N/pyt-*/...).

    Returns None when unset or blank — callers keep baseline artifacts in the run tree.
    """
    raw = run_config.get("baseline_csv_folder")
    if raw is None:
        return None
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    if s == CVS_INTERNAL_BASELINE_DATA_PATH:
        return _cvs_baseline_data_dir()
    if "~" in s:
        s = os.path.expanduser(s)
    if "{headnode-user-id}" in s:
        s = s.replace("{headnode-user-id}", getpass.getuser())
    return os.path.normpath(os.path.abspath(s))


def parallel_tree_path(runner_results_base: str, artifact_path: str, baseline_root: str) -> str:
    """
    Map a path under runner_results_base to the same relative path under baseline_root.
    The relative layout under artifact_path is copied under baseline_root.
    runner_results_base should match resolve_runner_results_base(run_config).
    """
    rb = os.path.abspath(runner_results_base)
    ap = os.path.abspath(artifact_path)
    try:
        common = os.path.commonpath([ap, rb])
    except ValueError as e:
        raise ValueError(
            f"artifact_path {artifact_path!r} is not under runner_results_base {runner_results_base!r}"
        ) from e
    if os.path.normpath(common) != os.path.normpath(rb):
        raise ValueError(f"artifact_path {artifact_path!r} is not under runner_results_base {runner_results_base!r}")
    rel = os.path.relpath(ap, rb)
    return os.path.normpath(os.path.join(os.path.abspath(baseline_root), rel))
