"""
Contract tests for LogsCollector.
"""

import unittest

from app.unittests.testing import FakeSshManager
from app.collectors.logs_collector import LogsCollector


# Distinctive substrings of the three log commands collect_all_logs issues:
#  - AMD logs grep includes "XGMI"
#  - userspace grep includes "out of memory"
#  - plain dmesg uses level "...crit,err 2>" (the others use "crit,err,warn")
AMD_NEEDLE = "XGMI"
USERSPACE_NEEDLE = "out of memory"
DMESG_NEEDLE = "crit,err 2>"


class TestLogsCollector(unittest.IsolatedAsyncioTestCase):
    async def test_collect_all_logs_returns_stripped_strings(self):
        ssh = FakeSshManager(
            ["node1", "node2"],
            command_map={
                AMD_NEEDLE: {"node1": "amdgpu xgmi link error\n", "node2": "  "},
                USERSPACE_NEEDLE: {"node1": "oom killed process 123\n", "node2": ""},
                DMESG_NEEDLE: {"node1": "critical err here\n", "node2": ""},
            },
        )

        result = await LogsCollector().collect_all_logs(ssh)

        self.assertIn("timestamp", result)
        self.assertEqual(result["amd_logs"]["node1"], "amdgpu xgmi link error")
        self.assertEqual(result["userspace_errors"]["node1"], "oom killed process 123")
        self.assertEqual(result["dmesg_errors"]["node1"], "critical err here")
        # Clean nodes are stored as empty strings (not errors).
        self.assertEqual(result["amd_logs"]["node2"], "")
        self.assertEqual(result["dmesg_errors"]["node2"], "")

    async def test_collect_all_logs_error_markers(self):
        ssh = FakeSshManager(
            ["node1"],
            command_map={
                AMD_NEEDLE: {"node1": "ABORT: Host Unreachable Error"},
                USERSPACE_NEEDLE: {"node1": "ERROR: boom"},
                DMESG_NEEDLE: {"node1": "ABORT: Host Unreachable Error"},
            },
        )

        result = await LogsCollector().collect_all_logs(ssh)

        self.assertEqual(result["amd_logs"]["node1"], {"error": "Failed to collect AMD logs"})
        self.assertEqual(result["userspace_errors"]["node1"], {"error": "Failed to collect logs"})
        self.assertEqual(result["dmesg_errors"]["node1"], {"error": "Failed to collect logs"})


if __name__ == "__main__":
    unittest.main()
