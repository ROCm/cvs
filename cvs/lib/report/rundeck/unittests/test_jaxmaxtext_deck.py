'''Unit tests for the JAX MaxText Run Deck profile on the shared training builder.'''

import unittest
from types import SimpleNamespace

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.config_adapter import build_inference_config_from_profile
from cvs.lib.report.rundeck.dataset_builders.registry import build_datasets
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html


def _variant():
    return SimpleNamespace(
        gpu_arch="MI300X",
        gpu_name="MI300X",
        enforce_thresholds=True,
        model=SimpleNamespace(id="meta-llama/Llama-3.1-8B"),
        training=SimpleNamespace(distributed=True),
        container=SimpleNamespace(image="rocm/jax-maxtext:latest"),
        cluster={"node_dict": {"n0": {}, "n1": {}}},
        thresholds={
            "BS=4,PRECISION=BF16,SL=8192": {
                "training.tflops_per_sec_per_gpu": {"kind": "min", "value": 100},
            },
            "BS=8,PRECISION=BF16,SL=8192": {
                "training.tflops_per_sec_per_gpu": {"kind": "min", "value": 100},
            },
        },
    )


def _flat():
    return {
        "BS=4,PRECISION=BF16,SL=8192": {
            "tflops_per_sec_per_gpu": ["180", "185.4"],
            "tokens_per_sec_per_gpu": ["3000"],
            "final_loss": ["2.0123456"],
            "step_time_p50_ms": ["671.2"],
            "step_time_p95_ms": ["708.0"],
            "_loss_curve": [[0, 2.5], [10, 2.0]],
            "_learning_rate_curve": [[0, 1e-4], [10, 9e-5]],
            "_planned_steps": 20,
        },
        "BS=8,PRECISION=BF16,SL=8192": {
            "tflops_per_sec_per_gpu": ["150"],
            "final_loss": ["2.20"],
        },
    }


class TestJaxDeckDatasets(unittest.TestCase):
    def test_cells_use_bs_sl_precision_dimensions(self):
        datasets = build_datasets(
            "training_sweep",
            {"results": _flat(), "variant": _variant(), "lifecycle_report": {}},
            load_json_profile("jaxmaxtext"),
        )
        cells = datasets["cells"]
        self.assertEqual(len(cells), 2)
        b4 = next(c for c in cells if c["bs"] == "4")
        self.assertEqual(b4["sl"], "8192")
        self.assertEqual(b4["precision"], "BF16")
        self.assertEqual(b4["subtitle"], "BS=4 SL=8192 \u00b7 BF16")
        self.assertEqual(b4["cell_id"], "BS=4,PRECISION=BF16,SL=8192")
        self.assertEqual(b4["actuals"]["training.tflops_per_sec_per_gpu"], 185.4)
        self.assertEqual(b4["loss_curve"], [[0, 2.5], [10, 2.0]])
        self.assertEqual(b4["learning_rate_curve"], [[0, 1e-4], [10, 9e-5]])
        self.assertEqual(b4["planned_steps"], 20)
        self.assertNotIn("mbs", b4)

    def test_results_table_and_chart_use_jax_columns(self):
        datasets = build_datasets(
            "training_sweep",
            {"results": _flat(), "variant": _variant(), "lifecycle_report": {}},
            load_json_profile("jaxmaxtext"),
        )
        headers = datasets["results_table"]["headers"]
        self.assertIn("BS", headers)
        self.assertIn("SL", headers)
        self.assertNotIn("MBS", headers)
        self.assertIn("tflops_per_sec_per_gpu", datasets["chart_series"])
        entry = datasets["chart_series"]["tflops_per_sec_per_gpu"][0]
        self.assertEqual(entry["label"], "JAX MaxText sweep")
        self.assertEqual(entry["x_labels"], ["BS=4", "BS=8"])

    def test_results_table_includes_all_metrics_rounded(self):
        datasets = build_datasets(
            "training_sweep",
            {"results": _flat(), "variant": _variant(), "lifecycle_report": {}},
            load_json_profile("jaxmaxtext"),
        )
        headers = datasets["results_table"]["headers"]
        for expected in [
            "Model",
            "GPU",
            "BS",
            "SL",
            "Precision",
            "tflops_per_sec_per_gpu",
            "final_loss",
            "eval_loss",
            "steps_to_target",
            "time_to_target_seconds",
        ]:
            self.assertIn(expected, headers)
        loss_idx = headers.index("final_loss")
        loss_vals = [row[loss_idx] for row in datasets["results_table"]["rows"]]
        # 2.0123456 rounds to at most four decimals.
        self.assertIn(2.0123, loss_vals)

    def test_subtitle_reflects_run_mode(self):
        payload = build_rundeck_payload(
            profile=load_json_profile("jaxmaxtext"),
            store={"cvs_results_dict": _flat(), "variant_config": _variant(), "lifecycle_report": {}},
            cvs_version="0.2.0",
        )
        self.assertEqual(payload["report"]["subtitle"], "JAX MaxText \u00b7 distributed training summary")

    def test_viewer_uses_resolved_subtitle(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        from cvs.lib.report.rundeck.config_adapter import build_inference_config_from_profile
        from cvs.lib.report.rundeck.generate_rundeck import RundeckPublisher

        profile = load_json_profile("jaxmaxtext")
        config = build_inference_config_from_profile(profile)
        payload = build_rundeck_payload(
            profile=profile,
            store={"cvs_results_dict": _flat(), "variant_config": _variant(), "lifecycle_report": {}},
            cvs_version="0.2.0",
        )
        publisher = RundeckPublisher(SimpleNamespace(config=SimpleNamespace()), None)
        with tempfile.TemporaryDirectory() as tmp:
            path = publisher._write_viewer(profile, config, Path(tmp), payload)
            text = path.read_text(encoding="utf-8")
            # The rendered subtitle element resolves {mode}; the raw profile is
            # only embedded as (non-displayed) JSON data.
            self.assertIn('<p class="subtitle">JAX MaxText \u00b7 distributed training summary</p>', text)

    def test_gate_matrix_and_summary(self):
        datasets = build_datasets(
            "training_sweep",
            {"results": _flat(), "variant": _variant(), "lifecycle_report": {}},
            load_json_profile("jaxmaxtext"),
        )
        self.assertTrue(datasets["gate_matrix"])
        summary = datasets["sweep_summaries"][0]
        self.assertEqual(summary["label"], "JAX MaxText sweep")
        self.assertEqual(summary["meta"], "Peak at BS=4,PRECISION=BF16,SL=8192")


class TestJaxProfileResolves(unittest.TestCase):
    def test_prefix_headline_and_dimension_fields(self):
        config = build_inference_config_from_profile(load_json_profile("jaxmaxtext"))
        self.assertEqual(config.metric_prefix, "training.")
        self.assertEqual(config.headline_metric, "training.tflops_per_sec_per_gpu")
        self.assertEqual(config.sweep_series_label, "JAX MaxText sweep")
        self.assertEqual([f[0] for f in config.dimension_fields], ["bs", "sl", "precision"])
        self.assertTrue(callable(config.cell_dimensions))


class TestJaxPayloadAndHtml(unittest.TestCase):
    def test_payload_renders_with_run_card_and_viewer_group_by(self):
        profile = load_json_profile("jaxmaxtext")
        payload = build_rundeck_payload(
            profile=profile,
            store={
                "cvs_results_dict": _flat(),
                "variant_config": _variant(),
                "lifecycle_report": {},
            },
            cvs_version="0.2.0",
        )
        self.assertEqual(payload["suite_id"], "jaxmaxtext")
        self.assertEqual(len(payload["cells"]), 2)
        self.assertEqual(payload["viewer_config"]["group_by"], ["precision"])
        html = render_rundeck_html(payload)
        self.assertIn("JAX MaxText Run Deck", html)
        self.assertIn("JAX MaxText", html)
        self.assertIn("BS=4,PRECISION=BF16,SL=8192", html)


if __name__ == "__main__":
    unittest.main()
