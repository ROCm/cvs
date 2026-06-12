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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Upstream InferenceMAX org repo is a stub; real tree lives at SemiAnalysisAI/InferenceX (see that README).
        self.if_dict.setdefault(
            'inferencemax_repo',
            'https://github.com/SemiAnalysisAI/InferenceX.git',
        )

    def get_server_script_directory(self):
        """InferenceMAX / InferenceX scripts live under ``/app/<clone-dir>/`` inside the container."""
        name = _clone_dirname_from_git_url(self.if_dict.get('inferencemax_repo', ''))
        return f'/app/{name}'

    def get_server_script_path(self):
        """InferenceMAX scripts are in the cloned repo."""
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

    def start_inference_server_job(self):
        """Start InferenceMAX server - clone repo, then call base implementation."""
        self.clone_inferencemax_repo()
        super().start_inference_server_job()
