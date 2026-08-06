'''Tests for inference report preset builder.'''

import unittest

from cvs.lib.report.rundeck.config_builder import make_inference_report_config


class TestConfigBuilder(unittest.TestCase):
    def test_make_inference_report_config_defaults(self):
        cfg = make_inference_report_config(
            suite_id="demo_suite",
            results_columns=(
                ("Model", None),
                ("Output tok/s", "client.output_throughput"),
                ("Mean TTFT (ms)", "client.mean_ttft_ms"),
            ),
            metric_units={"output_throughput": "tok/s", "mean_ttft_ms": "ms"},
            tier_metric_specs=lambda _c, _t: {},
        )
        self.assertEqual(cfg.suite_id, "demo_suite")
        self.assertEqual(cfg.report_basename, "demo_suite_report")
        self.assertEqual(cfg.inference_test_substring, "test_demo_suite")
        self.assertTrue(cfg.cell_highlights)
        self.assertTrue(cfg.chart_series)
        self.assertTrue(cfg.interactive_viewer)

    def test_make_inference_report_config_overrides(self):
        cfg = make_inference_report_config(
            suite_id="x",
            results_columns=(),
            metric_units={},
            tier_metric_specs=lambda _c, _t: {},
            inference_test_substring="test_custom_inference",
            report_basename="custom_report",
        )
        self.assertEqual(cfg.inference_test_substring, "test_custom_inference")
        self.assertEqual(cfg.report_basename, "custom_report")


if __name__ == "__main__":
    unittest.main()
