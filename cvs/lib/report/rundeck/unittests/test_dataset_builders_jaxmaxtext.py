'''Unit tests for the JAX MaxText Run Deck sweep layout.'''

import unittest
from types import SimpleNamespace

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.dataset_builders.jaxmaxtext import (
    build_jaxmaxtext_datasets,
    parse_sweep_key,
    tier_metric_specs,
)
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html


def _variant(enforce=True):
    thresholds = {
        "BS=2,PRECISION=BF16,SL=8192": {
            "training.tflops_per_sec_per_gpu": {"kind": "min", "value": 80},
            "training.tokens_per_sec_per_gpu": {"kind": "min", "value": 100},
            "training.tokens_per_sec_total": {"kind": "info", "value": 0},
            "training.final_loss": {"kind": "max", "value": 3},
            "training.loss_decreased": {"kind": "min", "value": 1},
        },
        "BS=3,PRECISION=FP8,SL=8192": {
            "training.tflops_per_sec_per_gpu": {"kind": "min", "value": 110},
            "training.tokens_per_sec_per_gpu": {"kind": "min", "value": 100},
            "training.final_loss": {"kind": "max", "value": 3},
            "training.loss_decreased": {"kind": "min", "value": 1},
        },
    }
    return SimpleNamespace(
        model=SimpleNamespace(id="llama3.1-70b"),
        gpu_arch="mi325x",
        nnodes=4,
        thresholds=thresholds,
        enforce_thresholds=enforce,
        framework="jaxmaxtext",
    )


def _results():
    return {
        "mode": "distributed",
        "sweeps": {
            "BS=2,PRECISION=BF16,SL=8192": {
                "results": {
                    "training.tflops_per_sec_per_gpu": 90,
                    "training.tokens_per_sec_per_gpu": 120,
                    "training.tokens_per_sec_total": 3840,
                    "training.final_loss": 2.5,
                    "training.loss_decreased": 1,
                },
                "num_nodes": 4,
            },
            "BS=3,PRECISION=FP8,SL=8192": {
                "results": {
                    "training.tflops_per_sec_per_gpu": 120,
                    "training.tokens_per_sec_per_gpu": 70,
                    "training.tokens_per_sec_total": 2240,
                    "training.final_loss": 2.2,
                    "training.loss_decreased": 1,
                },
                "num_nodes": 4,
            },
        },
    }


class TestJaxMaxTextDatasetBuilder(unittest.TestCase):
    def setUp(self):
        self.profile = load_json_profile("jaxmaxtext")

    def test_parse_sweep_key_uses_training_dimensions(self):
        self.assertEqual(
            parse_sweep_key("BS=3,PRECISION=FP8,SL=8192"),
            {"batch_size": "3", "precision": "FP8", "sequence_length": "8192"},
        )

    def test_builds_throughput_chart_gate_matrix_and_full_table(self):
        datasets = build_jaxmaxtext_datasets(
            {"results": _results(), "variant": _variant()},
            self.profile,
        )

        throughput = datasets["charts"]["throughput"]
        self.assertEqual(len(throughput["training.tokens_per_sec_per_gpu"][0]["points"]), 2)
        self.assertEqual(datasets["gate_matrix"][0]["tiers"], {"throughput": "pass", "quality": "pass"})
        self.assertEqual(datasets["gate_matrix"][1]["tiers"], {"throughput": "fail", "quality": "pass"})
        self.assertEqual(datasets["overall_status"], "fail")
        self.assertIn("nnodes", datasets["results_table"]["headers"])
        self.assertIn("step time p95 ms", datasets["results_table"]["headers"])
        self.assertEqual(len(datasets["results_table"]["rows"]), 2)
        self.assertEqual(datasets["cells"], [])

    def test_info_thresholds_are_not_treated_as_gates(self):
        specs = tier_metric_specs(
            {
                "training.tokens_per_sec_per_gpu": {"kind": "min", "value": 100},
                "training.tokens_per_sec_total": {"kind": "info", "value": 0},
            },
            "throughput",
        )
        self.assertEqual(list(specs), ["training.tokens_per_sec_per_gpu"])

    def test_record_only_run_keeps_threshold_matrix_render_only(self):
        datasets = build_jaxmaxtext_datasets(
            {"results": _results(), "variant": _variant(enforce=False)},
            self.profile,
        )
        self.assertEqual(datasets["overall_status"], "record")
        self.assertTrue(
            all(status == "record" for row in datasets["gate_matrix"] for status in row["tiers"].values())
        )

    def test_payload_renders_training_surfaces_without_inference_dimensions(self):
        payload = build_rundeck_payload(
            profile=self.profile,
            store={
                "training_res_dict": _results(),
                "cvs_results_dict": _results(),
                "variant_config": _variant(),
                "lifecycle_report": {},
            },
            cvs_version="test",
        )
        doc = render_rundeck_html(payload)

        self.assertIn("Training throughput vs sweep cells", doc)
        self.assertIn("Gate matrix vs thresholds", doc)
        self.assertIn("Full results", doc)
        self.assertIn("BS=2 \u00b7 BF16 \u00b7 SL=8192", doc)
        self.assertNotIn("ISL=", doc)
        self.assertNotIn("OSL=", doc)
        self.assertNotIn("C=BS=", doc)
        self.assertFalse(self.profile["interactive_viewer"])


if __name__ == "__main__":
    unittest.main()
