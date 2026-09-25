'''Unit tests for Megatron training-sweep Run Deck builder.'''

import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.config_adapter import build_inference_config_from_profile
from cvs.lib.report.rundeck.dataset_builders.registry import build_datasets
from cvs.lib.report.rundeck.generate_rundeck import RundeckPublisher
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html
from cvs.lib.report.rundeck.runtime.cards import DeckCardRenderer
from cvs.lib.report.rundeck.runtime.sweep_charts import SweepChartRenderer
from cvs.lib.report.training_cells import chart_x_labels, flatten_training_combo_actuals
from cvs.lib.report.viewer.scaffold import viewer_basename_for
from cvs.lib.training.megatron.utils.megatron_metrics import tier_metric_specs


def _variant():
    return SimpleNamespace(
        gpu_arch="MI300X",
        gpu_name="MI300X",
        enforce_thresholds=True,
        train_params={
            "tokenizer_model": "meta-llama/Llama-3.1-8B",
            "tensor_parallelism": "1",
            "pipeline_parallelism": "1",
        },
        container=SimpleNamespace(image="rocm/primus:latest", env={"NNODES": "2"}),
        thresholds={
            "MBS=4,GBS=128,PRECISION=FP8": {
                "training.throughput_per_gpu": {"kind": "min", "value": 100},
            },
            "MBS=4,GBS=128,PRECISION=BF16": {
                "training.throughput_per_gpu": {"kind": "min", "value": 100},
            },
        },
        cell_key=lambda name: name,
    )


def _train_res():
    return {
        "MBS=4,GBS=128,PRECISION=FP8": {
            "throughput_per_gpu": ["90", "200"],
            "elapsed_time_per_iteration": ["1.5"],
            "step_time_p50_ms": ["671.2"],
            "step_time_p95_ms": ["708.0"],
            "tokens_per_gpu": ["3000"],
            "_log_tail": "ignored",
            "_loss_curve": [[0, 2.5], [10, 2.1], [20, 1.8]],
            "_perplexity_curve": [[0, 12.182], [10, 8.166], [20, 6.05]],
            "_learning_rate_curve": [[0, 1e-4], [10, 9e-5]],
            "_grad_norm_curve": [[0, 3.0], [10, 2.0]],
            "_throughput_curve": [[0, 100.0], [10, 140.0]],
            "_tokens_curve": [[0, 20000.0], [10, 24000.0]],
        },
        "MBS=4,GBS=128,PRECISION=BF16": {
            "throughput_per_gpu": ["150"],
            "elapsed_time_per_iteration": ["2.0"],
            "step_time_p50_ms": ["1532.9"],
            "step_time_p95_ms": ["1554.2"],
            "tokens_per_gpu": ["2500"],
        },
        "MBS=4,GBS=128,PRECISION=MXFP4": None,
    }


def _profile():
    profile = load_json_profile("megatron")
    hooks = dict(profile.get("hooks") or {})
    hooks.pop("run_card_display", None)
    profile = dict(profile)
    profile["hooks"] = hooks
    return profile


class TestFlattenTrainingComboActuals(unittest.TestCase):
    def test_last_list_value_and_skips_private_keys(self):
        actuals = flatten_training_combo_actuals(
            {
                "throughput_per_gpu": ["1", "2.5"],
                "_log_tail": "nope",
                "_loss_curve": [[1, 2.0]],
                "empty": [],
            }
        )
        self.assertEqual(actuals["training.throughput_per_gpu"], 2.5)
        self.assertNotIn("training._log_tail", actuals)
        self.assertNotIn("training._loss_curve", actuals)
        self.assertNotIn("training.empty", actuals)


class TestTrainingSweepBuilder(unittest.TestCase):
    def test_cells_use_mbs_gbs_precision_not_isl(self):
        profile = _profile()
        datasets = build_datasets(
            "training_sweep",
            {
                "results": _train_res(),
                "variant": _variant(),
                "lifecycle_report": {
                    "tests/training/megatron/megatron_distributed.py::test_training[MBS=4,GBS=128,PRECISION=FP8]": [
                        ("training", 12.0, "s"),
                    ],
                },
            },
            profile,
        )
        cells = datasets["cells"]
        self.assertEqual(len(cells), 2)
        fp8 = next(c for c in cells if c["precision"] == "FP8")
        self.assertEqual(fp8["mbs"], "4")
        self.assertEqual(fp8["gbs"], "128")
        self.assertEqual(fp8["cell_id"], "MBS=4,GBS=128,PRECISION=FP8")
        self.assertEqual(fp8["actuals"]["training.throughput_per_gpu"], 200.0)
        self.assertNotIn("training._loss_curve", fp8["actuals"])
        self.assertEqual(fp8["loss_curve"], [[0, 2.5], [10, 2.1], [20, 1.8]])
        self.assertEqual(fp8["perplexity_curve"], [[0, 12.182], [10, 8.166], [20, 6.05]])
        self.assertEqual(fp8["learning_rate_curve"], [[0, 1e-4], [10, 9e-5]])
        self.assertEqual(fp8["grad_norm_curve"], [[0, 3.0], [10, 2.0]])
        self.assertEqual(fp8["throughput_curve"], [[0, 100.0], [10, 140.0]])
        self.assertEqual(fp8["tokens_curve"], [[0, 20000.0], [10, 24000.0]])
        self.assertEqual(fp8["tiers"]["throughput"], "pass")
        self.assertEqual(fp8["cell_lifecycle"].get("training"), 12.0)
        self.assertNotIn("isl", fp8)
        bf16 = next(c for c in cells if c["precision"] == "BF16")
        self.assertNotIn("loss_curve", bf16)
        headers = datasets["results_table"]["headers"]
        self.assertIn("MBS", headers)
        self.assertNotIn("ISL", headers)
        self.assertNotIn("Mean step (ms)", headers)
        self.assertIn("P50 step (ms)", headers)
        self.assertIn("P95 step (ms)", headers)
        self.assertIn("throughput_per_gpu", datasets["chart_series"])
        self.assertIn("step_time_p50_ms", datasets["chart_series"])
        self.assertIn("step_time_p95_ms", datasets["chart_series"])
        summary = datasets["sweep_summaries"][0]
        self.assertEqual(summary["headline_unit"], "TFLOP/s/GPU")
        self.assertEqual(summary["meta"], "Peak at MBS=4,GBS=128,PRECISION=FP8")
        self.assertNotIn("ttft_at_max_tput", summary)
        self.assertEqual(datasets["gate_matrix"][0]["label"], "MBS=4,GBS=128,PRECISION=BF16")
        self.assertIn("Scaling eff. (%)", headers)

    def test_single_node_omits_scaling_efficiency(self):
        profile = _profile()
        variant = _variant()
        variant.container.env["NNODES"] = "1"
        datasets = build_datasets(
            "training_sweep",
            {
                "results": _train_res(),
                "variant": variant,
                "suite_stem": "megatron_single",
                "lifecycle_report": {},
            },
            profile,
        )
        headers = datasets["results_table"]["headers"]
        self.assertNotIn("Scaling eff. (%)", headers)
        self.assertNotIn("scaling_efficiency_pct", datasets["chart_series"])
        fp8 = next(c for c in datasets["cells"] if c["precision"] == "FP8")
        self.assertFalse(any(m.get("metric") == "training.scaling_efficiency_pct" for m in fp8["metrics"]))

        payload = build_rundeck_payload(
            profile=load_json_profile("megatron"),
            store={
                "cvs_results_dict": _train_res(),
                "variant_config": variant,
                "suite_stem": "megatron_single",
                "lifecycle_report": {},
            },
            cvs_version="0.2.0",
        )
        viewer_cols = payload["viewer_config"]["table_columns"]
        self.assertFalse(any(c.get("metric") == "training.scaling_efficiency_pct" for c in viewer_cols))
        self.assertNotIn("training.scaling_efficiency_pct", payload["viewer_config"]["metrics"])
        html = render_rundeck_html(payload)
        self.assertNotIn("Scaling eff", html)

    def test_payload_and_html(self):
        profile = load_json_profile("megatron")
        payload = build_rundeck_payload(
            profile=profile,
            store={
                "cvs_results_dict": _train_res(),
                "variant_config": _variant(),
                "lifecycle_report": {},
            },
            cvs_version="0.2.0",
        )
        self.assertEqual(payload["suite_id"], "megatron")
        self.assertEqual(len(payload["cells"]), 2)
        vc = payload["viewer_config"]
        self.assertEqual(vc["group_by"], ["mbs", "gbs"])
        self.assertFalse(vc["interactivity"]["enabled"])
        self.assertFalse(vc["sweep_charts"]["enabled"])
        fp8 = next(c for c in payload["cells"] if c["precision"] == "FP8")
        self.assertEqual(fp8["loss_curve"][0], [0, 2.5])
        html = render_rundeck_html(payload)
        self.assertIn("Megatron Run Deck", html)
        self.assertIn("P50 step time", html)
        self.assertIn("P95 step time", html)
        self.assertIn("MBS=4,GBS=128,PRECISION=FP8", html)
        self.assertIn("training.throughput_per_gpu", str(payload["cells"][0]["metrics"]))
        self.assertNotIn("ISL=4", html)
        self.assertIn("Primus", html)

    def test_tier_metric_specs_reads_training_keys(self):
        specs = tier_metric_specs(
            {"training.throughput_per_gpu": {"kind": "min", "value": 1}},
            "throughput",
        )
        self.assertIn("training.throughput_per_gpu", specs)

    def test_profile_resolves_training_prefix(self):
        profile = _profile()
        config = build_inference_config_from_profile(profile)
        self.assertEqual(config.metric_prefix, "training.")
        self.assertEqual(config.full_metric("throughput_per_gpu"), "training.throughput_per_gpu")
        self.assertEqual(config.headline_metric, "training.throughput_per_gpu")
        self.assertTrue(config.interactive_viewer)

    def test_default_sweep_reads_train_params(self):
        from cvs.lib.report.training_cells import build_training_cells
        from cvs.lib.training.megatron.utils.training_config_loader import DEFAULT_SWEEP_NAME

        profile = _profile()
        config = build_inference_config_from_profile(profile)
        variant = _variant()
        variant.train_params.update(
            {
                "micro_batch_size": "4",
                "global_batch_size": "128",
                "precision": "BF16",
            }
        )
        cells = build_training_cells(
            config,
            variant,
            {DEFAULT_SWEEP_NAME: {"throughput_per_gpu": ["90"]}},
            {},
        )
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]["mbs"], "4")
        self.assertEqual(cells[0]["gbs"], "128")
        self.assertEqual(cells[0]["precision"], "BF16")
        self.assertEqual(cells[0]["subtitle"], "MBS=4 GBS=128 · BF16")

    def test_publisher_writes_training_sweep_viewer(self):
        profile = _profile()
        config = build_inference_config_from_profile(profile)
        payload = build_rundeck_payload(
            profile=profile,
            store={
                "cvs_results_dict": _train_res(),
                "variant_config": _variant(),
                "lifecycle_report": {},
            },
            cvs_version="0.2.0",
        )
        self.assertEqual(viewer_basename_for(config.report_basename), "megatron_run_deck_viewer.html")
        publisher = RundeckPublisher(SimpleNamespace(config=SimpleNamespace()), None)
        with tempfile.TemporaryDirectory() as tmp:
            path = publisher._write_viewer(profile, config, Path(tmp), payload)
            self.assertIsNotNone(path)
            self.assertEqual(path.name, "megatron_run_deck_viewer.html")
            text = path.read_text(encoding="utf-8")
            self.assertIn("loss-panel", text)
            self.assertIn("lr-panel", text)
            self.assertIn("grad-norm-panel", text)
            self.assertIn("tflops-panel", text)
            self.assertIn("tokens-panel", text)
            self.assertIn("buildLossChart", text)
            self.assertIn("buildStepCharts", text)
            self.assertIn("loss_curve", text)


class TestTrainingChartLabels(unittest.TestCase):
    _CELLS = [
        {"mbs": "2", "gbs": "128", "precision": "MXFP8", "cell_id": "MBS=2,GBS=128,PRECISION=MXFP8"},
        {"mbs": "4", "gbs": "128", "precision": "MXFP8", "cell_id": "MBS=4,GBS=128,PRECISION=MXFP8"},
        {"mbs": "4", "gbs": "32", "precision": "BF16", "cell_id": "MBS=4,GBS=32,PRECISION=BF16"},
    ]

    def test_labels_keep_only_varying_dimensions(self):
        cells = [dict(c, mbs="4", gbs="128") for c in self._CELLS]
        self.assertEqual(chart_x_labels(cells), ["MXFP8", "MXFP8", "BF16"])

    def test_labels_drop_commas_when_all_dimensions_vary(self):
        self.assertEqual(
            chart_x_labels(self._CELLS),
            ["MBS=2 GBS=128 MXFP8", "MBS=4 GBS=128 MXFP8", "MBS=4 GBS=32 BF16"],
        )

    def test_bar_chart_ticks_are_short_and_tooltip_keeps_full_key(self):
        chart_html = SweepChartRenderer().render_bar_chart(
            "TFLOP/s/GPU",
            [(0, 1270.5), (1, 1456.8)],
            "TFLOP/s/GPU",
            x_labels=["MBS=2 GBS=128 MXFP8", "MBS=4 GBS=128 MXFP8"],
            x_tips=["MBS=2,GBS=128,PRECISION=MXFP8", "MBS=4,GBS=128,PRECISION=MXFP8"],
        )
        self.assertIn(
            "<span class='chart-xlbl'>"
            "<span class='chart-xlbl-line'>MBS=2</span>"
            "<span class='chart-xlbl-line'>GBS=128</span>"
            "<span class='chart-xlbl-line'>MXFP8</span></span>",
            chart_html,
        )
        self.assertNotIn("MBS=2,GBS=128,PRECISION=MXFP8</span>", chart_html)
        self.assertIn("MBS=4,GBS=128,PRECISION=MXFP8: 1,456.8 TFLOP/s/GPU", chart_html)

    def test_bar_chart_falls_back_to_concurrency_ticks(self):
        chart_html = SweepChartRenderer().render_bar_chart("TTFT", [(1, 10.0), (8, 20.0)], "ms")
        self.assertIn("<span class='chart-xlbl-line'>C=1</span>", chart_html)
        self.assertIn("C=8: 20.0 ms", chart_html)

    def test_series_carries_labels_and_tips(self):
        datasets = build_datasets(
            "training_sweep",
            {"results": _train_res(), "variant": _variant(), "lifecycle_report": {}},
            _profile(),
        )
        entry = datasets["chart_series"]["throughput_per_gpu"][0]
        self.assertEqual(entry["x_labels"], ["BF16", "FP8"])
        self.assertEqual(
            entry["x_tips"],
            ["MBS=4,GBS=128,PRECISION=BF16", "MBS=4,GBS=128,PRECISION=FP8"],
        )


class TestLifecycleTimeline(unittest.TestCase):
    @staticmethod
    def _payload(expand):
        return {
            "report": {
                "session_lifecycle_labels": ("container_launch", "training", "teardown"),
                "expand_lifecycle_labels": expand,
            },
            "lifecycle": {"container_launch": 30.0, "training": 400.0, "teardown": 10.0},
            "cells": [
                {"subtitle": "MBS=4 GBS=128 FP8", "cell_lifecycle": {"training": 400.0}},
                {"subtitle": "MBS=4 GBS=128 BF16", "cell_lifecycle": {"training": 373.6}},
            ],
        }

    def test_expanded_stage_totals_cells_and_tones_each(self):
        markup = DeckCardRenderer().render_lifecycle(self._payload(("training",)), {}, None)
        self.assertIn("773.6s", markup)
        self.assertIn("MBS=4 GBS=128 FP8", markup)
        self.assertIn("MBS=4 GBS=128 BF16", markup)

    def test_no_tone_repeats_between_stages_and_cells(self):
        markup = DeckCardRenderer().render_lifecycle(self._payload(("training",)), {}, None)
        tones = re.findall(r"tl-(tone\d)", markup)
        self.assertEqual(tones, ["tone1", "tone2", "tone3", "tone4", "tone5"])

    def test_blocks_are_filled_with_white_text(self):
        from cvs.lib.report.rundeck.runtime.theme import report_css

        # The cell-card CSS ships its own .tl-val and is appended after the deck rules.
        css = report_css()
        self.assertIn(".tl-seg, .tl-cell, .tl-group-head { background: var(--tl-c", css)
        self.assertIn(".tl-val { font-size: 0.8rem; font-weight: 700; color: #fff; }", css)
        self.assertIn(".cell-mini-seg .tl-val", css)

    def test_unexpanded_stage_stays_one_segment(self):
        markup = DeckCardRenderer().render_lifecycle(self._payload(()), {}, None)
        self.assertIn("400.0s", markup)
        self.assertNotIn("tl-group", markup)
        self.assertNotIn("BF16", markup)

    def test_profile_expands_training_stage(self):
        config = build_inference_config_from_profile(_profile())
        self.assertEqual(config.expand_lifecycle_labels, ("training",))
        self.assertEqual(
            config.session_lifecycle_labels,
            ("container_launch", "smoke", "training", "metrics", "teardown"),
        )


if __name__ == "__main__":
    unittest.main()
