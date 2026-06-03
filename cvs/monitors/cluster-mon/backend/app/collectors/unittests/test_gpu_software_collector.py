"""
Contract tests for GPUSoftwareCollector.

Pins the exec_async return-shape -> parsed-output behavior the migration must
preserve. Uses FakeSshManager; no real SSH.
"""

import json
import unittest

from app.unittests.testing import FakeSshManager
from app.collectors.gpu_software_collector import GPUSoftwareCollector


class TestGPUSoftwareCollector(unittest.IsolatedAsyncioTestCase):
    async def test_collect_all_software_info_parses_version_and_firmware(self):
        version_json = json.dumps(
            [
                {
                    "tool": "AMDSMI Tool",
                    "version": "26.2.0+021c61fc",
                    "amdsmi_library_version": "26.2.0",
                    "rocm_version": "7.0.2",
                    "amdgpu_version": "6.16.6",
                    "amd_hsmp_driver_version": "N/A",
                }
            ]
        )
        firmware_json = json.dumps({"gpu": 0, "fw_list": [{"fw_name": "VBIOS", "fw_version": "022.040.000"}]})

        ssh = FakeSshManager(
            ["node1"],
            command_map={
                "version --json": {"node1": version_json},
                "firmware --json": {"node1": firmware_json},
            },
        )

        result = await GPUSoftwareCollector().collect_all_software_info(ssh)

        self.assertIn("timestamp", result)
        self.assertEqual(result["rocm_version"]["node1"]["rocm_version"], "7.0.2")
        self.assertEqual(result["rocm_version"]["node1"]["amdgpu_version"], "6.16.6")
        self.assertEqual(result["rocm_version"]["node1"]["amdsmi_tool"], "26.2.0+021c61fc")
        self.assertEqual(result["gpu_firmware"]["node1"], json.loads(firmware_json))

    async def test_error_and_abort_markers_yield_na_and_error(self):
        ssh = FakeSshManager(
            ["bad1", "bad2"],
            command_map={
                "version --json": {"bad1": "ABORT: Host Unreachable Error", "bad2": "ERROR: boom"},
                "firmware --json": {"bad1": "ABORT: Host Unreachable Error", "bad2": "ERROR: boom"},
            },
        )

        result = await GPUSoftwareCollector().collect_all_software_info(ssh)

        self.assertEqual(result["rocm_version"]["bad1"], {"rocm_version": "N/A", "amdgpu_version": "N/A"})
        self.assertEqual(result["rocm_version"]["bad2"], {"rocm_version": "N/A", "amdgpu_version": "N/A"})
        self.assertEqual(result["gpu_firmware"]["bad1"], {"error": "ABORT: Host Unreachable Error"})
        self.assertEqual(result["gpu_firmware"]["bad2"], {"error": "ERROR: boom"})

    async def test_collect_rocm_version_parses_file_and_driver(self):
        ssh = FakeSshManager(
            ["node1"],
            command_map={
                "/opt/rocm*/.info/version": {"node1": "7.0.2\n"},
                "showdriverversion": {"node1": "Driver version: 6.16.6\n"},
            },
        )

        result = await GPUSoftwareCollector().collect_rocm_version(ssh)

        self.assertEqual(result["node1"]["rocm_version"], "7.0.2")
        self.assertEqual(result["node1"]["driver_version"], "6.16.6")

    async def test_parse_json_output_handles_malformed_json(self):
        parsed = GPUSoftwareCollector.parse_json_output({"node1": "not-json", "node2": "ABORT: Host Unreachable Error"})
        self.assertIn("error", parsed["node1"])
        self.assertEqual(parsed["node2"], {"error": "ABORT: Host Unreachable Error"})


if __name__ == "__main__":
    unittest.main()
