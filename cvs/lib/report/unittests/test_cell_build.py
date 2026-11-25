'''Unit tests for cell record building.'''

import unittest
from dataclasses import replace
import math
from types import SimpleNamespace

from cvs.lib.inference.utils.vllm_metrics import metric_verdict as vllm_metric_verdict
from cvs.lib.report.cell_build import (
    CellRecordBuilder,
    bar_pct,
    build_all_cells,
    build_cell_record,
    is_gated,
    metric_pass,
    resolve_pytest_nodeids_for_cell,
    tier_status,
)
from cvs.lib.report.formatting import pytest_row_href
from cvs.lib.report.testing.fixtures import generic_inference_report_config, generic_variant


class TestCellBuild(unittest.TestCase):
    def test_is_gated(self):
        cases = {
            "info": ({"kind": "info", "value": 0}, False),
            "min": ({"kind": "min", "value": 5}, True),
            "max": ({"kind": "max", "value": 5}, True),
            "no kind": ({"value": 5}, True),
            "none": (None, False),
            "not a dict": ("min", False),
        }
        for name, (spec, gated) in cases.items():
            with self.subTest(name=name):
                self.assertIs(is_gated(spec), gated)

    def test_metric_pass_leaves_info_to_the_caller(self):
        info = {"kind": "info", "value": 0}
        self.assertEqual(metric_pass("output_throughput", 1.0, info), "fail")
        self.assertEqual(metric_pass("output_throughput", 1.0, {"kind": "min", "value": 5}), "fail")
        self.assertEqual(metric_pass("output_throughput", 9.0, {"kind": "min", "value": 5}), "pass")

    def test_tier_with_only_info_specs_records(self):
        info = {"kind": "info", "value": 0}
        tiers = {
            "info_only": {"client.output_throughput": info},
            "mixed": {"client.output_throughput": info, "client.failed": {"kind": "max", "value": 0}},
            "empty": {},
        }
        config = replace(generic_inference_report_config(), tier_metric_specs=lambda _cell, tier: tiers[tier])
        builder = CellRecordBuilder(config)
        actuals = {"client.output_throughput": 1.0, "client.failed": 3.0}
        self.assertEqual(builder.tier_status(actuals, {}, "info_only", True, record_info=True), "record")
        self.assertEqual(builder.tier_status(actuals, {}, "mixed", True, record_info=True), "fail")
        passing = {**actuals, "client.failed": 0.0}
        self.assertEqual(builder.tier_status(passing, {}, "mixed", True, record_info=True), "pass")
        self.assertEqual(builder.tier_status(actuals, {}, "empty", True, record_info=True), "na")
        self.assertEqual(builder.tier_status(actuals, {}, "info_only", True), "fail")
        self.assertEqual(builder.tier_status(passing, {}, "mixed", True), "fail")

    def test_module_tier_status_records_info(self):
        info = {"kind": "info", "value": 0}
        config = replace(generic_inference_report_config(), tier_metric_specs=lambda cell, _tier: cell)
        self.assertEqual(
            tier_status(config, {"client.output_throughput": 1.0}, {"client.output_throughput": info}, "t", True),
            "record",
        )

    def test_margin_shown_when_record_only_and_spec_present(self):
        variant = SimpleNamespace(
            enforce_thresholds=False,
            thresholds={
                "ISL=1024,OSL=1024,TP=8,CONC=128": {
                    "client.output_throughput": {"kind": "min_tok_s", "value": 1000.0},
                },
            },
            cell_key=lambda isl, osl, conc: f"ISL={isl},OSL={osl},TP=8,CONC={conc}",
        )
        key = ("org/example-model", "mi300x", "1024", "1024", "default", 128)
        cell = build_cell_record(
            generic_inference_report_config(),
            key=key,
            host="10.0.0.1",
            actuals={"client.output_throughput": 1200.0},
            variant_config=variant,
            lifecycle_report={},
            multi_host=False,
        )
        throughput = next(m for m in cell["metrics"] if m["metric"] == "client.output_throughput")
        self.assertEqual(throughput["status"], "record")
        self.assertIsNotNone(throughput["margin"])
        self.assertIn("above gate", throughput["margin"])

    def test_info_highlight_draws_no_gate(self):
        variant = SimpleNamespace(
            enforce_thresholds=True,
            thresholds={
                "ISL=1024,OSL=1024,TP=8,CONC=128": {"client.output_throughput": {"kind": "info", "value": 0}},
            },
            cell_key=lambda isl, osl, conc: f"ISL={isl},OSL={osl},TP=8,CONC={conc}",
        )
        config = replace(
            generic_inference_report_config(),
            tier_metric_specs=lambda cell, tier: cell if tier == "throughput" else {},
        )
        cell = build_cell_record(
            config,
            key=("org/example-model", "mi300x", "1024", "1024", "default", 128),
            host="10.0.0.1",
            actuals={"client.output_throughput": 1200.0},
            variant_config=variant,
            lifecycle_report={},
            multi_host=False,
        )
        throughput = next(m for m in cell["metrics"] if m["metric"] == "client.output_throughput")
        self.assertEqual((throughput["status"], throughput["bar_pct"], throughput["margin"]), ("record", None, None))
        self.assertIs(throughput["gated"], False)
        self.assertEqual(cell["tiers"]["throughput"], "record")

    def test_extra_curves_are_lifted_off_actuals(self):
        variant = SimpleNamespace(
            enforce_thresholds=False,
            thresholds={},
            cell_key=lambda isl, osl, conc: f"ISL={isl},OSL={osl},TP=8,CONC={conc}",
        )
        key = ("org/example-model", "mi300x", "1024", "1024", "default", 128)
        series = {
            "step time (ms)": [[1, 12.0], [10, 11.0]],
            "FID": {"points": [[10.0, 40.0], [50.0, 12.0]], "default_on": True, "x_title": "NFE"},
        }
        cell = build_cell_record(
            generic_inference_report_config(),
            key=key,
            host="10.0.0.1",
            actuals={"client.output_throughput": 1200.0, "_extra_curves": series},
            variant_config=variant,
            lifecycle_report={},
            multi_host=False,
        )
        self.assertEqual(cell["extra_curves"], series)
        self.assertNotIn("_extra_curves", cell["actuals"])

    def test_resolve_pytest_nodeids_for_cell(self):
        config = generic_inference_report_config()
        lifecycle = {
            "cvs/tests/x.py::test_inference[combo-128]": [],
            "cvs/tests/x.py::test_cell_metrics[tier0-128]": [],
        }
        ids = resolve_pytest_nodeids_for_cell(config, lifecycle, 128)
        self.assertIn("test_inference", ids["pytest_inference_nodeid"])
        self.assertIn("test_cell_metrics", ids["pytest_metrics_nodeid"])

    def test_resolve_pytest_nodeids_prefers_vllm_verification_parent(self):
        config = generic_inference_report_config()
        parent = "cvs/tests/inference/vllm/vllm_single.py::test_verify_cell_metrics[combo-128]"
        lifecycle = {
            "cvs/tests/inference/vllm/vllm_single.py::test_vllm_inference[combo-128]": [],
            "cvs/tests/inference/vllm/vllm_single.py::test_metric[output-128]": [],
            parent: [],
        }

        ids = resolve_pytest_nodeids_for_cell(config, lifecycle, 128)

        self.assertEqual(ids["pytest_metrics_nodeid"], parent)

    def test_pytest_row_href_encodes_nodeid(self):
        href = pytest_row_href("run.html", "cvs/tests/x.py::test_metric[a-128]")
        self.assertTrue(href.startswith("run.html#"))
        self.assertIn("test_metric", href)

    def test_build_cell_record_marks_enforced_failures(self):
        def _tier_specs(thresholds_cell, tier):
            if tier != "throughput":
                return {}
            return dict(thresholds_cell)

        cfg = replace(generic_inference_report_config(), tier_metric_specs=_tier_specs)
        variant = SimpleNamespace(
            enforce_thresholds=True,
            thresholds={
                "ISL=1024,OSL=1024,TP=8,CONC=128": {
                    "client.output_throughput": {"kind": "min_tok_s", "value": 5000.0},
                },
            },
            cell_key=lambda isl, osl, conc: f"ISL={isl},OSL={osl},TP=8,CONC={conc}",
        )
        key = ("org/example-model", "mi300x", "1024", "1024", "default", 128)
        cell = build_cell_record(
            cfg,
            key=key,
            host="10.0.0.1",
            actuals={"client.output_throughput": 1200.0},
            variant_config=variant,
            lifecycle_report={},
            multi_host=False,
        )
        throughput = next(m for m in cell["metrics"] if m["metric"] == "client.output_throughput")
        self.assertEqual(throughput["status"], "fail")
        self.assertEqual(cell["tiers"]["throughput"], "fail")

    def test_bar_pct_branches(self):
        self.assertEqual(bar_pct(50.0, {"kind": "min_tok_s", "value": 100.0}), 50.0)
        self.assertEqual(bar_pct(10.0, {"kind": "max_ms", "value": 100.0}), 100.0)
        self.assertEqual(bar_pct(0.0, {"kind": "max_ms", "value": 100.0}), 100.0)
        self.assertEqual(bar_pct(1.0, {"kind": "within", "value": 1.0}), 100.0)
        self.assertEqual(bar_pct(1.0, {"kind": "unknown", "value": 1.0}), 50.0)
        self.assertEqual(bar_pct(1.0, {"kind": "min", "value": 0.0}), 0.0)

    def test_default_evaluator_preserves_legacy_bar_coercion(self):
        variant = generic_variant()
        variant.enforce_thresholds = False
        key = ("org/example-model", "mi300x", "1024", "1024", "default", 128)
        for actual, expected in (("500", 50.0), (True, 0.1)):
            with self.subTest(actual=actual):
                cell = build_cell_record(
                    generic_inference_report_config(),
                    key=key,
                    host="10.0.0.1",
                    actuals={"client.output_throughput": actual},
                    variant_config=variant,
                    lifecycle_report={},
                    multi_host=False,
                )
                self.assertEqual(cell["metrics"][0]["bar_pct"], expected)

    def test_vllm_evaluator_rejects_invalid_bar_values(self):
        config = replace(
            generic_inference_report_config(),
            metric_prefix="",
            cell_highlights=(("output_throughput", "Output tok/s"),),
            metric_units={"output_throughput": "tok/s"},
            metric_verdict=vllm_metric_verdict,
        )
        variant = generic_variant()
        cell_id = variant.cell_key("1024", "1024", 128)
        variant.thresholds = {cell_id: {"output_throughput": {"kind": "min", "value": 1000.0}}}
        key = ("org/example-model", "mi300x", "1024", "1024", "default", 128)

        for actual in (True, math.nan, math.inf, "500", "not-a-number"):
            with self.subTest(actual=actual):
                cell = build_cell_record(
                    config,
                    key=key,
                    host="10.0.0.1",
                    actuals={"output_throughput": actual},
                    variant_config=variant,
                    lifecycle_report={},
                    multi_host=False,
                )
                metric = cell["metrics"][0]
                self.assertEqual(metric["status"], "fail")
                self.assertIsNone(metric["bar_pct"])

    def test_build_all_cells_orders_by_shape_then_concurrency(self):
        variant = generic_variant()
        variant.enforce_thresholds = False
        inf_res = {}
        for isl, osl, conc in (("8192", "1024", 4), ("128", "32", 16), ("128", "32", 1), ("1024", "128", 2)):
            inf_res[("org/example-model", "mi300x", isl, osl, "default", conc)] = {
                "10.0.0.1": {"client.output_throughput": 1.0}
            }
        cells = build_all_cells(
            generic_inference_report_config(),
            variant_config=variant,
            inf_res_dict=inf_res,
            lifecycle_report={},
        )
        self.assertEqual(
            [(c["isl"], c["osl"], c["concurrency"]) for c in cells],
            [("128", "32", 1), ("128", "32", 16), ("1024", "128", 2), ("8192", "1024", 4)],
        )


if __name__ == "__main__":
    unittest.main()
