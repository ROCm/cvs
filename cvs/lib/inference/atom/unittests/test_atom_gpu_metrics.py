'''Unit tests for atom_gpu_metrics helpers.'''

import unittest

from cvs.lib.inference.atom.atom_gpu_metrics import gpu_results_from_poll, merge_gpu_into_results


class TestAtomGpuMetrics(unittest.TestCase):
    def test_gpu_results_from_poll(self):
        readings = [{"gpu.used_vram": 1000}, {"gpu.used_vram": 2000}]
        out = gpu_results_from_poll(readings, load_s=12.5, load_mb=512)
        self.assertEqual(out["gpu.peak_gpu_memory_mb"], 2000)
        self.assertEqual(out["gpu.model_load_s"], 12.5)
        self.assertEqual(out["gpu.model_load_memory_mb"], 512)

    def test_merge_gpu_into_results(self):
        results = {"host1": {"client.output_throughput": 100.0}}
        merge_gpu_into_results(results, {"gpu.peak_gpu_memory_mb": 999.0})
        self.assertEqual(results["host1"]["gpu.peak_gpu_memory_mb"], 999.0)


if __name__ == "__main__":
    unittest.main()
