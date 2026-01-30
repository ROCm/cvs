'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import re

from cvs.lib.inference.base import InferenceBaseJob


class VllmJob(InferenceBaseJob):
    """vLLM-specific implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.if_dict.setdefault('benchmark_server_script_path', '/host_scripts')

    def get_server_script_directory(self):
        """vLLM scripts are mounted from host."""
        return self.if_dict['benchmark_server_script_path']

    def get_result_filename(self):
        """vLLM result filename."""
        return 'vllm_test_result.json'

    def get_completion_pattern(self):
        """vLLM completion pattern."""
        return re.compile('End-to-end Latency', re.I)

    def get_log_subdir(self):
        """vLLM uses 'vllm' log subdirectory."""
        return 'vllm'
