'''Unit tests for prev-run comparison panel.'''

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from cvs.lib.report.panels import panel_builder
from cvs.lib.report.panels.panel_builder import ComparisonPanelBuilder
from cvs.lib.report.panels.prev_run import build_prev_run_panel, resolve_prev_run_json_path
from cvs.lib.report.testing.fixtures import generic_inference_report_config

ACCURACY_METRIC = "gsm8k_flex.gsm8k.exact_match__flexible-extract"


class TestPrevRun(unittest.TestCase):
    def build_accuracy_panels(self, baseline_payload):
        cell = {
            "cell_id": "ISL=128,OSL=128,TP=1,PP=1,CONC=1",
            "host": "head",
            "actuals": {"output_throughput": 100.0},
        }
        baseline_payload = {**baseline_payload, "cells": [cell]}
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "baseline.json"
            baseline.write_text(json.dumps(baseline_payload), encoding="utf-8")
            config = replace(
                generic_inference_report_config(),
                suite_id="vllm",
                prev_run_json=str(baseline),
                metric_contract={"id": "vllm-bare", "version": 1},
            )
            lifecycle = {
                "pkg/vllm.py::test_accuracy_eval[gsm8k_flex]": [
                    (ACCURACY_METRIC, 0.92, ""),
                ]
            }
            with mock.patch(
                "cvs.lib.report.panels.panel_builder.load_report_json",
                wraps=panel_builder.load_report_json,
            ) as loader:
                panels = ComparisonPanelBuilder(config).build([cell], lifecycle_report=lifecycle)
        return panels, loader.call_count

    def test_prev_run_panel_flags_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline = tmp_path / "baseline.json"
            baseline.write_text(
                json.dumps(
                    {
                        "cells": [
                            {
                                "cell_id": "ISL=1024,OSL=1024,TP=8,CONC=128",
                                "host": "10.0.0.1",
                                "concurrency": 128,
                                "actuals": {"client.output_throughput": 4000.0},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            current_cells = [
                {
                    "cell_id": "ISL=1024,OSL=1024,TP=8,CONC=128",
                    "host": "10.0.0.1",
                    "concurrency": 128,
                    "actuals": {"client.output_throughput": 3600.0},
                }
            ]
            panel = build_prev_run_panel(current_cells, baseline, threshold_pct=5.0)
            self.assertIsNotNone(panel)
            row = panel["rows"][0]
            self.assertTrue(row["regression"])
            self.assertEqual(row["compare.prev_run.throughput_delta_pct"], -10.0)

    def test_resolve_prev_run_json_path_sibling(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sibling = tmp_path / "inferencex_atom_report_prev.json"
            sibling.write_text("{}", encoding="utf-8")
            resolved = resolve_prev_run_json_path(
                "",
                report_basename="inferencex_atom_report",
                report_dir=tmp_path,
            )
            self.assertEqual(resolved, str(sibling))

    def test_accuracy_comparison_survives_performance_contract_mismatch(self):
        panels, load_count = self.build_accuracy_panels(
            {
                "schema_version": 1,
                "suite_id": "vllm",
                "accuracy": {ACCURACY_METRIC: 0.94},
            }
        )

        self.assertEqual(load_count, 1)
        self.assertFalse(panels["prev_run"]["compatible"])
        self.assertIn("metric_contract", panels["prev_run"]["incompatibility"])
        self.assertAlmostEqual(
            panels["accuracy_prev_run"]["compare.prev_run.gsm8k_delta"],
            -0.02,
        )

    def test_accuracy_comparison_rejects_wrong_suite(self):
        panels, load_count = self.build_accuracy_panels(
            {
                "schema_version": 1,
                "suite_id": "other",
                "accuracy": {ACCURACY_METRIC: 0.94},
            }
        )

        self.assertEqual(load_count, 1)
        self.assertIn("suite_id", panels["prev_run"]["incompatibility"])
        self.assertNotIn("accuracy_prev_run", panels)

    def test_accuracy_comparison_rejects_wrong_schema(self):
        panels, load_count = self.build_accuracy_panels(
            {
                "schema_version": 2,
                "suite_id": "vllm",
                "accuracy": {ACCURACY_METRIC: 0.94},
            }
        )

        self.assertEqual(load_count, 1)
        self.assertIn("schema_version", panels["prev_run"]["incompatibility"])
        self.assertNotIn("accuracy_prev_run", panels)

    def test_accuracy_comparison_accepts_fully_compatible_baseline(self):
        panels, load_count = self.build_accuracy_panels(
            {
                "schema_version": 1,
                "suite_id": "vllm",
                "metric_contract": {"id": "vllm-bare", "version": 1},
                "accuracy": {ACCURACY_METRIC: 0.94},
            }
        )

        self.assertEqual(load_count, 1)
        self.assertTrue(panels["prev_run"]["compatible"])
        self.assertIn("accuracy_prev_run", panels)

    def test_non_object_baseline_is_handled_safely(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "invalid-shape.json"
            baseline.write_text("[1, 2, 3]", encoding="utf-8")
            config = replace(
                generic_inference_report_config(),
                suite_id="vllm",
                prev_run_json=str(baseline),
                metric_contract={"id": "vllm-bare", "version": 1},
            )

            panels = ComparisonPanelBuilder(config).build(
                [],
                lifecycle_report={
                    "pkg/vllm.py::test_accuracy_eval[gsm8k_flex]": [
                        ("gsm8k_flex.gsm8k.exact_match__flexible-extract", 0.92, ""),
                    ]
                },
            )

        self.assertFalse(panels["prev_run"]["compatible"])
        self.assertNotIn("accuracy_prev_run", panels)


if __name__ == "__main__":
    unittest.main()
