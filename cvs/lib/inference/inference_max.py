'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import re
import time

from cvs.lib.inference.base import InferenceBaseJob
from cvs.lib.verify_lib import fail_test


class InferenceMaxJob(InferenceBaseJob):
    """InferenceMAX-specific implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.if_dict.setdefault('inferencemax_repo', 'https://github.com/SemiAnalysisAI/InferenceX.git')
        self.if_dict.setdefault('inferencemax_commit_id', 'c5bbe050ccef019615db0ceb4dd983cc2faa6335')

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
        cmd = f'''docker exec {self.container_name} /bin/bash -c "git clone {self.if_dict['inferencemax_repo']} && cd InferenceX && git checkout {self.if_dict['inferencemax_commit_id']}" '''
        out_dict = self.s_phdl.exec(cmd)
        for node in out_dict.keys():
            if re.search('error|fail', out_dict[node], re.I):
                fail_test('Errors or failures seen in pulling InferenceMAX repo from Github, pls check')
        time.sleep(3)
        self.s_phdl.exec(f'''docker exec {self.container_name} /bin/bash -c "ls -ld /app/InferenceX" ''')

    def start_inference_server_job(self):
        """Start InferenceMAX server - clone repo, then call base implementation."""
        self.clone_inferencemax_repo()
        super().start_inference_server_job()
