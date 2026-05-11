"""
ROCm Version Checking Module

This module provides functionality for checking ROCm version consistency across cluster nodes.
"""

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

    def _rocm_version_remote_command(self):
        """Shell snippet to detect ROCm version on a node."""
        return """
        # Method 1: amd-smi (most reliable for newer ROCm)
        if command -v amd-smi >/dev/null 2>&1; then
            amd-smi version 2>/dev/null | sed -n 's/.*ROCm version: \\([0-9.]*\\).*/\\1/p'
        fi | head -1 | grep -v '^$' || 
        # Method 2: ROCm info files (fallback)
        (cat /opt/rocm/.info/version 2>/dev/null || cat /opt/rocm*/share/doc/rocm/version 2>/dev/null) | head -1 | grep -v '^$' ||
        echo 'NOT_FOUND'
        """

    def run(self):
        """
        Execute ROCm version check across all cluster nodes.

        Returns:
            dict: Results with per-node ROCm version status
        """
        self.log_info(f"Checking ROCm version consistency (expected: {self.expected_version})")

        cmd = self._rocm_version_remote_command()
        self.results = {}
        out_dict = self.phdl.exec(cmd)

        for node, output in out_dict.items():
            version = output.strip()
            self.results[node] = {
                'detected_version': version,
                'expected_version': self.expected_version,
                'status': 'PASS' if version == self.expected_version else 'FAIL',
                'errors': [],
            }

            if version == 'NOT_FOUND':
                self.results[node]['errors'].append(
                    "ROCm version not found - neither amd-smi nor ROCm info files available"
                )
            elif version != self.expected_version:
                self.results[node]['errors'].append(
                    f"Version mismatch: expected {self.expected_version}, found {version}"
                )

        return self.results
