"""
Contract tests for GPUMetricsCollector (synchronous ssh_manager.exec path).
"""

import json
import unittest

from app.unittests.testing import FakeSshManager
from app.collectors.gpu_collector import GPUMetricsCollector


class TestGPUMetricsCollector(unittest.IsolatedAsyncioTestCase):
    async def test_collect_gpu_utilization_parses_json(self):
        amd_smi = json.dumps([{"gpu": 0, "usage": {"gfx_activity": {"value": 42}}}])
        ssh = FakeSshManager(["node1"], command_map={"amd-smi metric --json": {"node1": amd_smi}})

        result = await GPUMetricsCollector().collect_gpu_utilization(ssh)

        self.assertEqual(result["node1"], json.loads(amd_smi))

    async def test_collect_gpu_utilization_error_marker(self):
        ssh = FakeSshManager(
            ["node1"], command_map={"amd-smi metric --json": {"node1": "ABORT: Host Unreachable Error"}}
        )
        result = await GPUMetricsCollector().collect_gpu_utilization(ssh)
        self.assertEqual(result["node1"], {"error": "ABORT: Host Unreachable Error"})

    def test_parse_json_output_strips_warning_prefix(self):
        out = 'WARNING: not in render group\n[{"gpu": 0}]'
        parsed = GPUMetricsCollector.parse_json_output({"node1": out})
        self.assertEqual(parsed["node1"], [{"gpu": 0}])

    def test_parse_json_output_malformed(self):
        parsed = GPUMetricsCollector.parse_json_output({"node1": "not-json"})
        self.assertIn("error", parsed["node1"])


if __name__ == "__main__":
    unittest.main()
