'''Unit tests for accuracy lifecycle extraction and prev-run panel.'''

import math
import unittest

from cvs.lib.report.accuracy_lifecycle import (
    build_accuracy_prev_run_panel,
    build_scale_accuracy_panel,
    extract_accuracy_from_lifecycle,
)


class TestAccuracyLifecycle(unittest.TestCase):
    def test_extract_accuracy_from_lifecycle(self):
        lifecycle = {
            "pkg/atom.py::test_accuracy_eval[gsm8k_flex]": [
                ("accuracy_eval", 12.3, "s"),
                ("gsm8k_flex.gsm8k.exact_match__flexible-extract", 0.95, ""),
            ]
        }
        out = extract_accuracy_from_lifecycle(lifecycle)
        self.assertEqual(out["gsm8k_flex.gsm8k.exact_match__flexible-extract"], 0.95)
        self.assertNotIn("accuracy_eval", out)

    def test_extract_accuracy_accepts_only_finite_builtin_numbers(self):
        lifecycle = {
            "pkg/vllm.py::test_accuracy_eval[types]": [
                ("task.zero", 0, ""),
                ("task.one", 1.0, ""),
                ("task.string", "0.5", ""),
                ("task.bool", True, ""),
                ("task.none", None, ""),
                ("task.list", [0.5], ""),
                ("task.object", {"value": 0.5}, ""),
                ("task.nan", math.nan, ""),
                ("task.inf", math.inf, ""),
                ("task.neg_inf", -math.inf, ""),
            ]
        }

        self.assertEqual(
            extract_accuracy_from_lifecycle(lifecycle),
            {"task.zero": 0.0, "task.one": 1.0},
        )

    def test_accuracy_prev_run_panel_regression(self):
        current = {"gsm8k_flex.gsm8k.exact_match__flexible-extract": 0.92}
        baseline = {"accuracy": {"gsm8k_flex.gsm8k.exact_match__flexible-extract": 0.94}}
        panel = build_accuracy_prev_run_panel(current, baseline, max_drop=0.01)
        self.assertIsNotNone(panel)
        self.assertTrue(panel["regression"])
        self.assertAlmostEqual(panel["compare.prev_run.gsm8k_delta"], -0.02)

    def test_accuracy_prev_run_rejects_invalid_current_and_baseline_values(self):
        metric = "gsm8k_flex.gsm8k.exact_match__flexible-extract"
        invalid = ("0.5", True, None, [0.5], {"value": 0.5}, math.nan, math.inf, -math.inf)
        for value in invalid:
            with self.subTest(location="current", value=value):
                self.assertIsNone(
                    build_accuracy_prev_run_panel(
                        {metric: value},
                        {"accuracy": {metric: 0.5}},
                    )
                )
            with self.subTest(location="baseline", value=value):
                self.assertIsNone(
                    build_accuracy_prev_run_panel(
                        {metric: 0.5},
                        {"accuracy": {metric: value}},
                    )
                )

    def test_accuracy_prev_run_normalizes_valid_int_float_boundaries(self):
        metric = "gsm8k_flex.gsm8k.exact_match__flexible-extract"
        panel = build_accuracy_prev_run_panel(
            {metric: 0},
            {"accuracy": {metric: 1.0}},
        )

        self.assertEqual(panel["current"], 0.0)
        self.assertEqual(panel["baseline"], 1.0)
        self.assertEqual(panel["compare.prev_run.gsm8k_delta"], -1.0)

    def test_scale_accuracy_panel(self):
        current = {
            "gsm8k_flex.gsm8k.exact_match__flexible-extract": 0.93,
            "hellaswag.hellaswag.acc_norm__none": 0.80,
        }
        reference = {
            "accuracy": {
                "gsm8k_flex.gsm8k.exact_match__flexible-extract": 0.94,
                "hellaswag.hellaswag.acc_norm__none": 0.81,
            }
        }
        panel = build_scale_accuracy_panel(current, reference, max_drop=0.01)
        self.assertIsNotNone(panel)
        self.assertTrue(panel["regression"])
        self.assertEqual(len(panel["rows"]), 2)

    def test_scale_accuracy_ignores_invalid_values(self):
        current = {
            "gsm8k_flex.gsm8k.exact_match__flexible-extract": "0.93",
            "hellaswag.hellaswag.acc_norm__none": 1,
        }
        reference = {
            "accuracy": {
                "gsm8k_flex.gsm8k.exact_match__flexible-extract": 0.94,
                "hellaswag.hellaswag.acc_norm__none": 0.5,
            }
        }

        panel = build_scale_accuracy_panel(current, reference)

        self.assertEqual(len(panel["rows"]), 1)
        self.assertEqual(panel["rows"][0]["current"], 1.0)


if __name__ == "__main__":
    unittest.main()
