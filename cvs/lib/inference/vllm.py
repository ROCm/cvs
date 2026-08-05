'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

DEPRECATED -- scheduled for removal on 2026-11-11.

Host+docker vLLM job used by the legacy single-node suites under
``cvs/tests/inference/vllm_legacy/``. Removed in #223 without a notice period
and restored here to serve the 3-month OSS deprecation window (notice given
2026-08-11).

Replacement: ``cvs.lib.inference.vllm_job.VllmJob``, driven by the unified
topology-parametrized suite ``cvs.tests.inference.vllm``, which covers both
single-node and PP-distributed runs from one ``schema_version: 1`` config.

Bound to the frozen ``base_legacy`` rather than the live ``base`` -- see that
module's header for the behaviour differences that make the distinction load
bearing.
'''

import datetime
import re
import time
import warnings

from cvs.lib import globals
from cvs.lib.inference.base_legacy import InferenceBaseJob

# Notice served 2026-08-11; 3-month OSS deprecation window.
DEPRECATION_REMOVAL_DATE = datetime.date(2026, 11, 11)

warnings.warn(
    "cvs.lib.inference.vllm and the cvs/tests/inference/vllm_legacy suites are "
    "deprecated and will be removed on 2026-11-11. Migrate to the unified suite "
    "cvs.tests.inference.vllm (cvs.lib.inference.vllm_job.VllmJob), which covers "
    "single-node and distributed runs from one schema_version: 1 config.",
    DeprecationWarning,
    stacklevel=2,
)


class VllmJob(InferenceBaseJob):
    """vLLM-specific implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.if_dict.setdefault('benchmark_server_script_path', '/host_scripts')

    def get_server_script_path(self):
        """vLLM scripts are mounted from host."""
        return self.server_script

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

    def stop_server(self):
        """Stop the vLLM server process."""
        log = globals.log
        log.info("Stopping vLLM server")
        self.s_phdl.exec(f'docker exec {self.container_name} pkill -f "vllm serve"')
        time.sleep(5)  # Wait for graceful shutdown

    def restart_server(self):
        """Restart the vLLM server with updated parameters."""
        log = globals.log
        log.info("Restarting vLLM server with updated parameters")
        self.stop_server()
        self.build_server_inference_job_cmd()
        self.start_inference_server_job()
