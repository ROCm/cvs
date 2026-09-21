'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs/lib/training/jaxmaxtext/utils/gpu_peak_tflops.py.
'''

import unittest

from cvs.lib.training.jaxmaxtext.utils.gpu_peak_tflops import compute_mfu, peak_tflops


class PeakTflopsTests(unittest.TestCase):
    def test_default_table_substring_and_precision(self):
        self.assertAlmostEqual(peak_tflops("AMD Instinct MI325X", "BF16"), 1307.4)
        self.assertAlmostEqual(peak_tflops("MI325X", "FP8"), 2614.9)

    def test_precision_aliases(self):
        self.assertAlmostEqual(peak_tflops("MI300X", "bfloat16"), 1307.4)
        self.assertAlmostEqual(peak_tflops("MI300X", "e4m3"), 2614.9)

    def test_unknown_gpu_or_precision_returns_none(self):
        self.assertIsNone(peak_tflops("MI355X", "BF16"))  # not in default table
        self.assertIsNone(peak_tflops("MI325X", "INT4"))
        self.assertIsNone(peak_tflops("", "BF16"))

    def test_flat_override_wins(self):
        self.assertAlmostEqual(peak_tflops("MI355X", "BF16", overrides={"BF16": 2500.0}), 2500.0)

    def test_nested_override_by_gpu(self):
        ov = {"MI355X": {"BF16": 2500.0, "FP8": 5000.0}}
        self.assertAlmostEqual(peak_tflops("AMD Instinct MI355X", "FP8", overrides=ov), 5000.0)


class ComputeMfuTests(unittest.TestCase):
    def test_percentage(self):
        self.assertAlmostEqual(compute_mfu(653.7, 1307.4), 50.0)

    def test_missing_or_nonpositive_returns_none(self):
        self.assertIsNone(compute_mfu(None, 1307.4))
        self.assertIsNone(compute_mfu(100.0, None))
        self.assertIsNone(compute_mfu(100.0, 0))


if __name__ == "__main__":
    unittest.main()
