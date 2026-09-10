'''Integration tests for the vLLM Run Deck metric contract.'''

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cvs.lib.report.ci_summary import render_ci_summary_html
from cvs.lib.report.inference import build_inference_report_payload
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.config_adapter import build_inference_config_from_profile
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html
from cvs.lib.report.rundeck.viewer_config import build_viewer_config
from cvs.lib.report.viewer.scaffold import write_interactive_viewer


CELL = "ISL=128,OSL=128,TP=1,PP=1,CONC=1"
KEY = ("model", "", "128", "128", CELL, 1)


def _config():
    return build_inference_config_from_profile(load_json_profile("vllm"))


def _variant(actual_thresholds=None, enforce=True):
    return SimpleNamespace(
        model_id="model",
        enforce_thresholds=enforce,
        thresholds={CELL: actual_thresholds or {}},
        server_params=SimpleNamespace(model="model", tensor_parallel_size=1),
        cell_key=lambda _isl, _osl, _conc: CELL,
    )


class TestVllmReportContract(unittest.TestCase):
    def test_strict_verdict_hook_fails_missing_actual(self):
        payload = build_inference_report_payload(
            config=_config(),
            variant_config=_variant(
                {"output_throughput": {"kind": "min", "value": 1}}
            ),
            inf_res_dict={KEY: {"head": {}}},
            lifecycle_report={},
        )

        self.assertEqual(payload["cells"][0]["tiers"]["throughput"], "fail")
        self.assertEqual(payload["overall_status"], "fail")

    def test_both_payload_builders_include_contract_and_bare_viewer_columns(self):
        variant = _variant({"output_throughput": {"kind": "min", "value": 1}})
        results = {
            KEY: {
                "head": {
                    "output_throughput": 2.0,
                    "mean_ttft_ms": 4.0,
                    "gpu_compute_util_pct": 50.0,
                }
            }
        }
        profile = load_json_profile("vllm")
        inference_payload = build_inference_report_payload(
            config=_config(),
            variant_config=variant,
            inf_res_dict=results,
            lifecycle_report={},
        )
        rundeck_payload = build_rundeck_payload(
            profile=profile,
            store={
                "inf_res_dict": results,
                "variant_config": variant,
                "lifecycle_report": {},
            },
        )

        for payload in (inference_payload, rundeck_payload):
            with self.subTest(builder=payload.keys()):
                self.assertEqual(
                    payload["metric_contract"],
                    {"id": "vllm-bare", "version": 1},
                )
                self.assertEqual(
                    payload["viewer_config"]["metric_contract"],
                    {"id": "vllm-bare", "version": 1},
                )
                columns = payload["viewer_config"]["table_columns"]
                metric_columns = {column["metric"] for column in columns if "metric" in column}
                self.assertIn("total_token_throughput", metric_columns)
                self.assertIn("output_throughput", payload["viewer_config"]["metrics"])
                self.assertFalse(any(metric.startswith("client.") for metric in metric_columns))
                self.assertIn("output_throughput", payload["cells"][0]["actuals"])

    def test_config_only_viewer_config_retains_contract(self):
        viewer = build_viewer_config({}, _config())
        self.assertEqual(viewer["metric_contract"], {"id": "vllm-bare", "version": 1})
        self.assertIn("output_throughput", viewer["metrics"])

    def test_legacy_previous_run_is_explicitly_incompatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "legacy.json"
            baseline.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "suite_id": "vllm",
                        "cells": [
                            {
                                "cell_id": CELL,
                                "host": "head",
                                "actuals": {"client.output_throughput": 100},
                            }
                        ],
                    }
                )
            )
            config = _config()
            object.__setattr__(config, "prev_run_json", str(baseline))
            payload = build_inference_report_payload(
                config=config,
                variant_config=_variant(),
                inf_res_dict={KEY: {"head": {"output_throughput": 90}}},
                lifecycle_report={},
            )

            panel = payload["panels"]["prev_run"]
            self.assertFalse(panel["compatible"])
            self.assertEqual(panel["rows"], [])
            self.assertIn("metric_contract", panel["incompatibility"])
            deck = render_rundeck_html(payload)
            summary = render_ci_summary_html(
                payload,
                config,
                full_report_basename="vllm_run_deck",
            )
            self.assertIn("Baseline comparison incompatible", deck)
            self.assertIn("comparisons suppressed", summary)
            self.assertNotIn("none flagged", summary)

    def test_manual_viewer_validates_contract_before_indexing_baseline_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "viewer.html"
            write_interactive_viewer(
                path,
                json_basename="vllm.json",
                title="vLLM",
                embed_payload={
                    "schema_version": 1,
                    "suite_id": "vllm",
                    "metric_contract": {"id": "vllm-bare", "version": 1},
                    "cells": [],
                    "viewer_config": build_viewer_config({}, _config()),
                },
            )
            document = path.read_text()

        validator = document.index("baselineIncompatibilityReason(data)")
        indexing = document.index("(data.cells || []).forEach")
        self.assertLess(validator, indexing)
        self.assertIn("Baseline incompatible:", document)
        self.assertIn("comparisons suppressed", document)


if __name__ == "__main__":
    unittest.main()
