'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import re
import shlex
import time

from cvs.lib.inference.base import InferenceBaseJob, _GIT_CLONE_FAIL_RE
from cvs.lib.verify_lib import fail_test


def _clone_dirname_from_git_url(repo_url: str) -> str:
    """Last path segment of a git URL without ``.git`` (git's default clone directory)."""
    part = (repo_url or "").rstrip("/").rsplit("/", 1)[-1]
    return part.removesuffix(".git") if part.endswith(".git") else part or "InferenceX"


class InferenceMaxJob(InferenceBaseJob):
    """InferenceMAX-specific implementation."""

    # Runner preflight: this CVS build supports opt-in host-mounted server scripts
    # (``use_host_mounted_server_script`` + ``benchmark_server_script_path``).
    uses_local_server_scripts = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Upstream InferenceMAX org repo is a stub; real tree lives at SemiAnalysisAI/InferenceX (see that README).
        self.if_dict.setdefault(
            'inferencemax_repo',
            'https://github.com/SemiAnalysisAI/InferenceX.git',
        )

    def _using_host_server_scripts(self) -> bool:
        """Host-only server launch: opt-in so configs may keep ``benchmark_server_script_path`` for docs or mounts while still cloning InferenceX (e.g. ``cvs run`` without a prior script deploy)."""
        if not self.if_dict.get('use_host_mounted_server_script'):
            return False
        p = self.if_dict.get('benchmark_server_script_path')
        return bool(p and str(p).strip())

    def _host_mount_relative_script(self) -> str:
        """Path to pass to ``bash`` under ``benchmark_server_script_path`` (strip InferenceX-style prefixes if present)."""
        ss = self.server_script.replace('\\', '/')
        base = 'single_node' if int(self.nnodes) == 1 else 'multi_node'
        for prefix in (f'benchmarks/{base}/', 'benchmarks/single_node/', 'benchmarks/multi_node/'):
            if ss.startswith(prefix):
                return ss[len(prefix) :]
        return ss

    def get_server_script_directory(self):
        """Script directory: host mount (``benchmark_server_script_path``) or cloned InferenceX tree."""
        if self._using_host_server_scripts():
            return str(self.if_dict['benchmark_server_script_path']).rstrip('/')
        name = _clone_dirname_from_git_url(self.if_dict.get('inferencemax_repo', ''))
        return f'/app/{name}'

    def get_server_script_path(self):
        """Script path under mount (host mode) or ``benchmarks/<node_layout>/`` under the clone."""
        if self._using_host_server_scripts():
            return self._host_mount_relative_script()
        base = "single_node" if int(self.nnodes) == 1 else "multi_node"
        return f'benchmarks/{base}/{self.server_script}'

    def get_result_filename(self):
        """InferenceMAX result filename."""
        return 'inferencemax_test_result.json'

    def get_completion_pattern(self):
        """InferenceMAX completion pattern."""
        return re.compile('Serving Benchmark Result', re.I)

    def get_log_subdir(self):
        """InferenceMAX uses 'inference-max' log subdirectory."""
        return 'inference-max'

    def clone_inferencemax_repo(self):
        """Clone InferenceX/InferenceMAX repo into ``/app`` inside the container (matches ``get_server_script_directory``)."""
        if self._using_host_server_scripts():
            # Server runs from host-mounted scripts only; skip git clone.
            self.s_phdl.exec(
                f"docker exec {shlex.quote(self.container_name)} /bin/bash -c {shlex.quote('mkdir -p /workspace')}"
            )
            return
        repo = self.if_dict["inferencemax_repo"]
        clone_name = _clone_dirname_from_git_url(repo)
        # Remove legacy stub checkout name; clone under /app so paths match get_server_script_directory().
        inner = (
            "set -e; mkdir -p /app; cd /app; "
            "rm -rf InferenceMAX; "
            f"if [ ! -d {shlex.quote(clone_name)} ]; then git clone {shlex.quote(repo)}; fi"
        )
        cmd = f"docker exec {shlex.quote(self.container_name)} /bin/bash -c {shlex.quote(inner)}"
        out_dict = self.s_phdl.exec(cmd)
        for node in out_dict.keys():
            out = out_dict[node] or ""
            if re.search("already exists", out, re.I):
                continue
            if _GIT_CLONE_FAIL_RE.search(out):
                fail_test("Errors or failures seen in pulling InferenceMAX repo from Github, pls check")
        time.sleep(3)
        ls_inner = f"ls -ld /app/{clone_name}"
        self.s_phdl.exec(
            f"docker exec {shlex.quote(self.container_name)} /bin/bash -c {shlex.quote(ls_inner)}"
        )
        # InferenceX benchmark scripts expect /workspace for server.log and gpu_metrics.csv
        # (see benchmark_lib.sh); CVS host-docker runs do not mount it by default.
        self.s_phdl.exec(
            f"docker exec {shlex.quote(self.container_name)} /bin/bash -c {shlex.quote('mkdir -p /workspace')}"
        )

    def start_inference_server_job(self):
        """Start InferenceMAX server - clone repo, then call base implementation."""
        self.clone_inferencemax_repo()
        super().start_inference_server_job()
