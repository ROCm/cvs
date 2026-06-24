import unittest

from cvs.lib.dtni.vllm_benchmark_scripts import (
    bash_export_bench_script_from_vllm_install,
    clamped_bench_random_range_ratio_str,
    validated_bench_script_basename,
)


class TestVllmBenchmarkScripts(unittest.TestCase):
    def test_validated_bench_script_basename(self):
        self.assertEqual(validated_bench_script_basename("benchmark_serving.py"), "benchmark_serving.py")
        self.assertEqual(validated_bench_script_basename("subdir/benchmark_serving.py"), "benchmark_serving.py")
        with self.assertRaises(ValueError):
            validated_bench_script_basename("not-a-script")

    def test_clamped_bench_random_range_ratio_str_no_clamp(self):
        out, was = clamped_bench_random_range_ratio_str("0.8", 100, 100, 8192)
        self.assertEqual(out, "0.8")
        self.assertFalse(was)

    def test_clamped_bench_random_range_ratio_str_clamps(self):
        out, was = clamped_bench_random_range_ratio_str("0.8", 4000, 1000, 5000)
        self.assertTrue(was)
        self.assertEqual(out, "0")

    def test_bash_export_defines_cvs_run_bench(self):
        frag = bash_export_bench_script_from_vllm_install("benchmark_serving.py")
        self.assertIn("_cvs_run_bench", frag)
        self.assertIn("/app/bench_serving", frag)


if __name__ == "__main__":
    unittest.main()
