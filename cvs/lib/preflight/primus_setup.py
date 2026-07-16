"""
Automatic Primus repository and virtualenv setup for node_smoke preflight.

When ``node_smoke.auto_setup`` is enabled (default), clones or updates Primus
on each reachable node and ensures the configured venv has the minimal deps
for ``node_smoke`` (ROCm PyTorch).  Primus on ``dev/preflight-direct-test``
is run from the git checkout via ``runner/primus-cli`` — there is no
``setup.py`` / ``pyproject.toml`` to ``pip install -e .``.
"""

from __future__ import annotations

import os
import re
import shlex
from typing import Any, Dict, List, Optional

from cvs.lib.preflight.base import PreflightCheck
from cvs.lib.preflight.node_smoke import get_nested_config, _config_flag_enabled

_GIT_ERROR_RE = re.compile(r"(?:^|\n)(?:fatal:|error:\s+pathspec)", re.IGNORECASE)
_PIP_ERROR_RE = re.compile(r"(?:^|\n)ERROR:\s(?!.*pathspec)", re.IGNORECASE)
_SHELL_ERROR_RE = re.compile(
    r"(?:^|\n)(?:bash:\s|/bin/bash:\s|syntax error:|fatal: timed out waiting)",
    re.IGNORECASE,
)

# Primus preflight-direct docs: node_smoke needs torch (ROCm build) only.
_DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/rocm6.2"
_SETUP_OK_MARKER = "CVS_PRIMUS_SETUP_OK"


def _venv_root_from_activate(venv_activate: str) -> str:
    """Return venv root directory from ``.../.venv/bin/activate`` path."""
    parts = venv_activate.rstrip("/").split("/")
    if len(parts) >= 3 and parts[-1] == "activate" and parts[-2] == "bin":
        return "/".join(parts[:-2])
    if len(parts) >= 2 and parts[-2] == "bin":
        return "/".join(parts[:-2])
    return venv_activate.rstrip("/")


def _remove_broken_primus_dir(primus_q: str) -> str:
    """Remove a partial clone (directory exists but .git is missing or incomplete)."""
    return f"if [ -e {primus_q} ] && [ ! -d {primus_q}/.git ]; then rm -rf {primus_q}; fi"


def _setup_lock_path(primus_dir: str) -> str:
    """Lock file beside primus_dir (parent must exist before clone creates primus_dir)."""
    parent = os.path.dirname(primus_dir.rstrip("/")) or "."
    return os.path.join(parent, ".cvs_primus_setup.lock")


def _with_setup_lock(primus_dir: str, body: str) -> str:
    """Serialize clone/venv on shared storage (NFS home) via flock."""
    parent_q = shlex.quote(os.path.dirname(primus_dir.rstrip("/")) or ".")
    lock_q = shlex.quote(_setup_lock_path(primus_dir))
    return f"mkdir -p {parent_q} && ( flock -w 900 9 || exit 1; {body} ) 9>{lock_q}"


def _git_sync_existing_repo(primus_q: str, branch_q: str, recurse_submodules: bool) -> str:
    """Fetch, checkout, and optionally update submodules on an existing clone."""
    submodule_cmd = (
        "git submodule update --init --recursive"
        if recurse_submodules
        else "true"
    )
    return (
        f"git fetch origin {branch_q} && "
        f"(git checkout {branch_q} || git checkout -B {branch_q} origin/{branch_q}) && "
        f"git pull --ff-only origin {branch_q} 2>/dev/null || true && "
        f"{submodule_cmd}"
    )


def build_primus_clone_or_update_command(
    *,
    primus_dir: str,
    git_url: str,
    git_branch: str,
    recurse_submodules: bool = False,
    force_reclone: bool = False,
) -> str:
    """Shell snippet: clone Primus or update an existing checkout."""
    primus_q = shlex.quote(primus_dir)
    url_q = shlex.quote(git_url)
    branch_q = shlex.quote(git_branch)
    parent_q = shlex.quote(os.path.dirname(primus_dir.rstrip("/")) or ".")
    recurse_flag = "--recurse-submodules" if recurse_submodules else ""
    clone_flags = f"--branch {branch_q} --single-branch {recurse_flag}".strip()
    sync_existing = _git_sync_existing_repo(primus_q, branch_q, recurse_submodules)

    cleanup = _remove_broken_primus_dir(primus_q)

    if force_reclone:
        return (
            f"rm -rf {primus_q} && "
            f"mkdir -p {parent_q} && "
            f"git clone {clone_flags} {url_q} {primus_q}"
        )

    return (
        f"if [ -d {primus_q}/.git ]; then "
        f"bash -c 'cd {primus_q} && {sync_existing}'; "
        f"else "
        f"{cleanup} && mkdir -p {parent_q} && "
        f"git clone {clone_flags} {url_q} {primus_q}; "
        f"fi"
    )


def build_primus_verify_command(*, primus_dir: str, venv_activate: str) -> str:
    """Shell snippet: verify Primus checkout and torch in venv."""
    primus_q = shlex.quote(primus_dir)
    activate_q = shlex.quote(venv_activate)
    return (
        f"test -f {primus_q}/runner/primus-cli && test -f {activate_q} && "
        f"bash -c 'source {activate_q} && cd {primus_q} && python -c \"import torch\"'"
    )


def _finish_setup_command(cmd: str) -> str:
    """Append a stdout marker so silent successful exits (e.g. follower wait) parse as PASS."""
    return f"{cmd} && echo {_SETUP_OK_MARKER}"


def build_wait_for_shared_primus_command(
    *,
    primus_dir: str,
    venv_activate: str,
    poll_interval: int = 5,
    max_wait: int = 900,
) -> str:
    """Wait until another node finishes clone/venv on shared NFS home."""
    primus_q = shlex.quote(primus_dir)
    activate_q = shlex.quote(venv_activate)
    attempts = max(1, max_wait // max(1, poll_interval))
    # Flat shell (no nested bash -c) so SSH quoting stays intact across nodes.
    return (
        f"i=0; while [ $i -lt {attempts} ]; do "
        f"test -f {primus_q}/runner/primus-cli && test -f {activate_q} && "
        f". {activate_q} && python -c \"import torch\" 2>/dev/null && "
        f"echo {_SETUP_OK_MARKER} && exit 0; "
        f"i=$((i+1)); sleep {poll_interval}; done; "
        f"echo \"fatal: timed out waiting for shared Primus install\"; exit 1"
    )


def build_primus_venv_install_command(
    *,
    primus_dir: str,
    venv_activate: str,
    pip_install_mode: str = "minimal",
    torch_pip_index_url: str = _DEFAULT_TORCH_INDEX,
) -> str:
    """Shell snippet: create venv and install deps for node_smoke."""
    primus_q = shlex.quote(primus_dir)
    activate_q = shlex.quote(venv_activate)
    venv_root_q = shlex.quote(_venv_root_from_activate(venv_activate))
    venv_parent_q = shlex.quote(os.path.dirname(_venv_root_from_activate(venv_activate)) or ".")
    index_q = shlex.quote(torch_pip_index_url)

    create_venv = (
        f"if [ ! -f {activate_q} ]; then "
        f"mkdir -p {venv_parent_q} && python3 -m venv {venv_root_q}; "
        f"fi"
    )

    mode = (pip_install_mode or "minimal").strip().lower()
    if mode == "skip":
        install = "true"
    elif mode == "requirements":
        install = (
            f"bash -c 'source {activate_q} && cd {primus_q} && "
            f"pip install -r requirements.txt --no-cache-dir'"
        )
    else:
        # minimal: ROCm torch only (Primus node_smoke). No pip install -e .
        install = (
            f"bash -c 'source {activate_q} && "
            f"if ! python -c \"import torch\" 2>/dev/null; then "
            f"pip install torch --index-url {index_q} --no-cache-dir; "
            f"fi'"
        )

    verify = build_primus_verify_command(primus_dir=primus_dir, venv_activate=venv_activate)

    return f"{create_venv} && {install} && {verify}"


def parse_setup_output(output: str) -> Dict[str, Any]:
    """Classify setup command output as PASS/FAIL."""
    text = (output or "").strip()
    if "ABORT: Host Unreachable Error" in text:
        return {"status": "FAIL", "errors": ["SSH unreachable during setup"]}
    if _SHELL_ERROR_RE.search(text):
        return {"status": "FAIL", "errors": ["shell error during Primus setup"]}
    if "fatal: timed out waiting for shared Primus install" in text:
        return {"status": "FAIL", "errors": ["timed out waiting for shared Primus install"]}
    if _GIT_ERROR_RE.search(text):
        if "could not lock config file" in text.lower():
            return {
                "status": "FAIL",
                "errors": [
                    "git clone conflict on shared storage — enable shared_install "
                    "(default) so only one node clones Primus"
                ],
            }
        return {"status": "FAIL", "errors": ["git error during Primus setup"]}
    if _PIP_ERROR_RE.search(text) or re.search(r"\bpip\b.*\berror\b", text, re.IGNORECASE):
        return {"status": "FAIL", "errors": ["pip install failed during Primus setup"]}
    if "No module named 'torch'" in text or "ModuleNotFoundError" in text:
        return {"status": "FAIL", "errors": ["torch not available in venv after setup"]}
    if _SETUP_OK_MARKER in text:
        return {"status": "PASS", "errors": []}
    if not text:
        return {"status": "FAIL", "errors": ["empty setup output"]}
    return {"status": "FAIL", "errors": ["setup did not report success"]}


class PrimusSetup(PreflightCheck):
    """Clone/update Primus and prepare the preflight venv on cluster nodes."""

    def __init__(self, phdl, node_list: List[str], config_dict=None):
        super().__init__(phdl, config_dict)
        self.node_list = list(node_list)
        self._load_settings()

    def _load_settings(self):
        cfg = self.config_dict or {}
        self.primus_dir = get_nested_config(cfg, "node_smoke", "primus_dir", "")
        self.venv_activate = get_nested_config(cfg, "node_smoke", "venv_activate", "")
        self.git_url = get_nested_config(
            cfg, "node_smoke", "primus_git_url", "https://github.com/AMD-AIG-AIMA/Primus.git"
        )
        self.git_branch = get_nested_config(cfg, "node_smoke", "primus_git_branch", "dev/preflight-direct-test")
        self.recurse_submodules = _config_flag_enabled(
            get_nested_config(cfg, "node_smoke", "primus_git_recurse_submodules", False), default=False
        )
        self.force_reclone = _config_flag_enabled(get_nested_config(cfg, "node_smoke", "force_reclone", False))
        self.pip_install_mode = get_nested_config(cfg, "node_smoke", "pip_install_mode", "minimal")
        self.torch_pip_index_url = get_nested_config(
            cfg, "node_smoke", "torch_pip_index_url", _DEFAULT_TORCH_INDEX
        )
        self.setup_timeout = int(get_nested_config(cfg, "node_smoke", "setup_timeout", 600))
        self.shared_install = _config_flag_enabled(
            get_nested_config(cfg, "node_smoke", "shared_install", True), default=True
        )

    def _validate(self) -> Optional[str]:
        if not self.primus_dir:
            return "node_smoke.primus_dir is required for auto_setup"
        if not self.venv_activate:
            return "node_smoke.venv_activate is required for auto_setup"
        if not self.git_url:
            return "node_smoke.primus_git_url is required for auto_setup"
        if not self.git_branch:
            return "node_smoke.primus_git_branch is required for auto_setup"
        if not self.node_list:
            return "no reachable nodes for Primus auto_setup"
        return None

    def run(self) -> Dict[str, Any]:
        err = self._validate()
        if err:
            return {"status": "FAIL", "skipped": True, "message": err, "node_results": {}}

        hosts = [h for h in self.phdl.reachable_hosts if h in self.node_list]
        if not hosts:
            return {
                "status": "FAIL",
                "skipped": True,
                "message": "no reachable hosts for Primus auto_setup",
                "node_results": {},
            }

        clone_cmd = build_primus_clone_or_update_command(
            primus_dir=self.primus_dir,
            git_url=self.git_url,
            git_branch=self.git_branch,
            recurse_submodules=self.recurse_submodules,
            force_reclone=self.force_reclone,
        )
        venv_cmd = build_primus_venv_install_command(
            primus_dir=self.primus_dir,
            venv_activate=self.venv_activate,
            pip_install_mode=self.pip_install_mode,
            torch_pip_index_url=self.torch_pip_index_url,
        )
        setup_body = _finish_setup_command(f"{clone_cmd} && {venv_cmd}")
        per_node_setup = _with_setup_lock(self.primus_dir, setup_body)
        follower_setup = build_wait_for_shared_primus_command(
            primus_dir=self.primus_dir,
            venv_activate=self.venv_activate,
            max_wait=self.setup_timeout,
        )

        use_shared = self.shared_install and len(hosts) > 1
        leader = hosts[0]
        if use_shared:
            setup_mode = f"shared (leader={leader})"
        else:
            setup_mode = "per-node"

        commands: List[str] = []
        for h in self.phdl.reachable_hosts:
            if h not in hosts:
                commands.append("true")
            elif use_shared and h != leader:
                commands.append(follower_setup)
            else:
                commands.append(per_node_setup)

        self.log_info(
            f"Primus auto_setup on {len(hosts)} node(s) [{setup_mode}]: "
            f"dir={self.primus_dir}, branch={self.git_branch}, "
            f"venv={self.venv_activate}, pip_mode={self.pip_install_mode}"
        )

        out_dict = self.phdl.exec_cmd_list(commands, timeout=self.setup_timeout)

        hosts_set = set(hosts)
        node_results: Dict[str, Any] = {}
        failed_nodes: List[str] = []
        for host, output in out_dict.items():
            if host not in hosts_set:
                continue
            parsed = parse_setup_output(output)
            node_results[host] = {
                "status": parsed["status"],
                "errors": parsed["errors"],
            }
            if parsed["status"] == "FAIL":
                failed_nodes.append(host)
                for e in parsed["errors"]:
                    self.log_error(f"Node {host} Primus setup: {e}")
                snippet = (output or "").strip()[-800:]
                if snippet:
                    self.log_error(f"Node {host} setup output (tail): {snippet}")

        status = "FAIL" if failed_nodes else "PASS"
        self.results = {
            "status": status,
            "skipped": False,
            "primus_dir": self.primus_dir,
            "git_branch": self.git_branch,
            "venv_activate": self.venv_activate,
            "pip_install_mode": self.pip_install_mode,
            "shared_install": self.shared_install,
            "setup_leader": leader if use_shared else None,
            "total_nodes": len(node_results),
            "failed_nodes": failed_nodes,
            "node_results": node_results,
        }
        return self.results
