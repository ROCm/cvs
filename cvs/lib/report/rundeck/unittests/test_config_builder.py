'''Tests for inference report preset builder.'''

import unittest
from dataclasses import asdict, fields, replace

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
        self.assertIsNone(cfg.metric_verdict)
        self.assertIsNone(cfg.metric_contract)

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

    def test_metric_contract_fields_use_normal_dataclass_paths(self):
        def metric_verdict(_metric, _actual, _spec):
            return "pass", ""

        contract = {"id": "demo", "version": 1}
        cfg = make_inference_report_config(
            suite_id="x",
            results_columns=(),
            metric_units={},
            tier_metric_specs=lambda _c, _t: {},
            metric_verdict=metric_verdict,
            metric_contract=contract,
        )
        contract["version"] = 2

        self.assertIs(cfg.metric_verdict, metric_verdict)
        self.assertEqual(cfg.metric_contract, {"id": "demo", "version": 1})
        self.assertGreaterEqual({field.name for field in fields(cfg)}, {"metric_verdict", "metric_contract"})
        self.assertEqual(asdict(cfg)["metric_contract"], {"id": "demo", "version": 1})
        replaced = replace(cfg, title="Replacement")
        self.assertIs(replaced.metric_verdict, metric_verdict)
        self.assertEqual(replaced.metric_contract, {"id": "demo", "version": 1})


if __name__ == "__main__":
    unittest.main()
