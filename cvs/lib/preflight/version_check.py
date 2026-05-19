"""
ROCm Version Checking Module

This module provides functionality for checking ROCm version consistency across cluster nodes.
"""

import json

from cvs.lib.preflight.base import PreflightCheck


class RocmVersionCheck(PreflightCheck):
    """Check ROCm version consistency across cluster nodes."""

    def __init__(self, phdl, expected_version, config_dict=None):
        """
        Initialize ROCm version check.

        Args:
            phdl: Parallel SSH handle for cluster nodes
            expected_version: Expected ROCm version string (e.g., "6.2.0")
            config_dict: Optional configuration dictionary
        """
        super().__init__(phdl, config_dict)
        self.expected_version = expected_version

    @staticmethod
    def _extract_rocm_version(output):
        """Parse ROCm version from amd-smi JSON output."""
        raw = output.strip()
        if not raw or raw == 'NOT_FOUND':
            return 'NOT_FOUND'

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return 'NOT_FOUND'

        version = None
        if isinstance(payload, list) and payload:
            version = payload[0].get('rocm_version')
        elif isinstance(payload, dict):
            version = payload.get('rocm_version')

        if isinstance(version, str) and version.strip():
            return version.strip()
        return 'NOT_FOUND'

    def run(self):
        """
        Execute ROCm version check across all cluster nodes.

        Returns:
            dict: Results with per-node ROCm version status
        """
        self.log_info(f"Checking ROCm version consistency (expected: {self.expected_version})")

        cmd = """
        if command -v amd-smi >/dev/null 2>&1; then
            amd-smi version --json 2>/dev/null || echo 'NOT_FOUND'
        else
            echo 'NOT_FOUND'
        fi
        """
        self.results = {}
        out_dict = self.phdl.exec(cmd)

        for node, output in out_dict.items():
            version = self._extract_rocm_version(output)
            self.results[node] = {
                'detected_version': version,
                'expected_version': self.expected_version,
                'status': 'PASS' if version == self.expected_version else 'FAIL',
                'errors': [],
            }

            if version == 'NOT_FOUND':
                self.results[node]['errors'].append("ROCm version not found via amd-smi JSON output")
            elif version != self.expected_version:
                self.results[node]['errors'].append(
                    f"Version mismatch: expected {self.expected_version}, found {version}"
                )

        return self.results
