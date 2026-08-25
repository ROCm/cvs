'''Unit tests for framework parity panel.'''

import json
import tempfile
import unittest
from pathlib import Path

from cvs.lib.report.compare import build_framework_parity_row
from cvs.lib.report.panels.framework_parity import build_framework_parity_panel


class TestFrameworkParity(unittest.TestCase):
    def test_framework_parity_row_ratio(self):
        cell = {
            "cell_id": "ISL=128,OSL=32,TP=8,CONC=1",
            "host": "10.0.0.1",
            "concurrency": 1,
            "actuals": {"client.output_throughput": 2000.0, "client.mean_ttft_ms": 100.0},
        }
        ref = {
            "cell_id": "ISL=128,OSL=32,TP=8,CONC=1",
            "host": "10.0.0.1",
            "concurrency": 1,
            "actuals": {"client.output_throughput": 1000.0, "client.mean_ttft_ms": 50.0},
        }
        row = build_framework_parity_row(
            cell,
            ref,
            headline_metric="client.output_throughput",
            ratio_metric_key="compare.vllm.output_throughput_ratio",
            ttft_metric_key="compare.vllm.mean_ttft_ms_ratio",
        )
        self.assertEqual(row["compare.vllm.output_throughput_ratio"], 2.0)
        self.assertEqual(row["compare.vllm.mean_ttft_ms_ratio"], 2.0)

    def test_framework_parity_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "ref.json"
            ref.write_text(
                json.dumps(
                    {
                        "cells": [
                            {
                                "cell_id": "ISL=128,OSL=32,TP=8,CONC=1",
                                "host": "10.0.0.1",
                                "concurrency": 1,
                                "actuals": {"client.output_throughput": 1000.0},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            current = [
                {
                    "cell_id": "ISL=128,OSL=32,TP=8,CONC=1",
                    "host": "10.0.0.1",
                    "concurrency": 1,
                    "actuals": {"client.output_throughput": 900.0},
                }
            ]
            panel = build_framework_parity_panel(current, ref, driver="vllm_atom")
            self.assertIsNotNone(panel)
            self.assertEqual(panel["rows"][0]["compare.vllm.output_throughput_ratio"], 0.9)


if __name__ == "__main__":
    unittest.main()
