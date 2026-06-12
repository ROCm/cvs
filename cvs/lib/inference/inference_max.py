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


class InferenceMaxJob(InferenceBaseJob):
    """InferenceMAX-specific implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.if_dict.setdefault('inferencemax_repo', 'https://github.com/InferenceMAX/InferenceMAX.git')

    def get_server_script_directory(self):
        """InferenceMAX scripts are in the cloned repo."""
        return '/app/InferenceX'

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
        """Clone InferenceMAX repository."""
        repo = self.if_dict["inferencemax_repo"]
        inner = f"git clone {shlex.quote(repo)}"
        cmd = f"docker exec {shlex.quote(self.container_name)} /bin/bash -c {shlex.quote(inner)}"
        out_dict = self.s_phdl.exec(cmd)
        for node in out_dict.keys():
            out = out_dict[node] or ""
            if re.search("already exists", out, re.I):
                continue
            if _GIT_CLONE_FAIL_RE.search(out):
                fail_test("Errors or failures seen in pulling InferenceMAX repo from Github, pls check")
        time.sleep(3)
        ls_inner = "ls -ld /app/InferenceX"
        self.s_phdl.exec(
            f"docker exec {shlex.quote(self.container_name)} /bin/bash -c {shlex.quote(ls_inner)}"
        )

    def start_inference_server_job(self):
        """Start InferenceMAX server - clone repo, then call base implementation."""
        self.clone_inferencemax_repo()
        super().start_inference_server_job()
