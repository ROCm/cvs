'''Unit tests for accuracy lifecycle extraction and prev-run panel.'''

import unittest

from cvs.lib.report.accuracy_lifecycle import (
    build_accuracy_prev_run_panel,
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

    def test_accuracy_prev_run_panel_regression(self):
        current = {"gsm8k_flex.gsm8k.exact_match__flexible-extract": 0.92}
        baseline = {"accuracy": {"gsm8k_flex.gsm8k.exact_match__flexible-extract": 0.94}}
        panel = build_accuracy_prev_run_panel(current, baseline, max_drop=0.01)
        self.assertIsNotNone(panel)
        self.assertTrue(panel["regression"])
        self.assertAlmostEqual(panel["compare.prev_run.gsm8k_delta"], -0.02)


if __name__ == "__main__":
    unittest.main()
