'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for the training_sweep dataset builder + the jaxmaxtext deck profile:
the training_res_dict shape must flow through resolve/build into cells,
gate_matrix, and results_table without the inference 6-tuple cell key.
'''

import unittest
from types import SimpleNamespace

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.profiles.hooks.jaxmaxtext_run_card import jaxmaxtext_run_card_display
from cvs.lib.report.rundeck.dataset_builders.training_sweep import build_training_datasets
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.runtime.cards import render_card

_SWEEP = "NN4_BF16_B1_SL2048"


def _variant(*, enforce=True, final_loss_gate=8.0):
    return SimpleNamespace(
        model=SimpleNamespace(id="deepseek-v4-284b"),
        gpu_arch="MI325X",
        framework="jaxmaxtext",
        enforce_thresholds=enforce,
        training=SimpleNamespace(distributed=True, steps=30),
        thresholds={
            _SWEEP: {
                "training.tokens_per_sec_per_gpu": {"kind": "min", "value": 100.0},
                "training.tflops_per_sec_per_gpu": {"kind": "min", "value": 50.0},
                "training.final_loss": {"kind": "max", "value": final_loss_gate},
                "training.loss_decreased": {"kind": "min", "value": 1},
            }
        },
        enabled_sweeps=lambda: [SimpleNamespace(name=_SWEEP)],
    )


def _training_res_dict(final_loss=6.5):
    return {
        "mode": "distributed",
        "sweeps": {
            _SWEEP: {
                "results": {
                    "training.tokens_per_sec_per_gpu": 200.0,
                    "training.tflops_per_sec_per_gpu": 185.0,
                    "training.tokens_per_sec_total": 6400.0,
                    "training.scaling_efficiency_pct": 92.0,
                    "training.step_time_p50_ms": 9800.0,
                    "training.final_loss": final_loss,
                    "training.loss_decreased": 1.0,
                },
                "num_nodes": 4,
                "step_metrics": [{"step": 0, "loss": 12.0}, {"step": 29, "loss": final_loss}],
                "eval_metrics": [],
                "tb_scalars": {
                    "learning/loss": [[0, 12.0], [1, 11.0], [5, 10.0], [6, 9.5]],
                    "learning/grad_norm": [[0, 1.5], [5, 1.2]],
                    "perf/per_device_tflops_per_sec": [[0, 180.0], [5, 185.0]],
                    "perf/step_time_seconds": [[0, 110.0], [1, 90.0], [2, 10.0], [3, 9.8], [4, 10.1], [5, 9.9]],
                },
            }
        },
    }


def _profile():
    return load_json_profile("jaxmaxtext_single")


class ProfileLoadTests(unittest.TestCase):
    def test_extends_merges_base(self):
        prof = _profile()
        self.assertEqual(prof["dataset_builder"], "training_sweep")
        self.assertEqual(prof["sources"]["results"], "training_res_dict")
        self.assertEqual(prof["sweep"]["metric_prefix"], "training.")
        self.assertIn("single-node", prof["subtitle"])  # child override wins
        self.assertTrue(prof["interactive_viewer"])
        card_types = [c["type"] for c in prof["cards"]]
        self.assertIn("gate_matrix", card_types)  # from base
        self.assertIn("metric_bars", card_types)  # req 1: bars after lifecycle
        # metric_bars comes right after the lifecycle timeline
        self.assertEqual(card_types[card_types.index("lifecycle_timeline") + 1], "metric_bars")
        self.assertNotIn("sweep_charts", card_types)  # req 2: removed from deck page
        self.assertNotIn("image_gallery", card_types)


class BuildTrainingDatasetsTests(unittest.TestCase):
    def _build(self, **kw):
        sources = {"results": _training_res_dict(**kw), "variant": _variant()}
        return build_training_datasets(sources, _profile())

    def test_gate_matrix_row_per_sweep_with_training_label(self):
        ds = self._build()
        self.assertEqual(len(ds["gate_matrix"]), 1)
        row = ds["gate_matrix"][0]
        self.assertEqual(row["label"], _SWEEP)  # sweep name, not "policy · C="
        self.assertEqual(row["tiers"]["throughput"], "pass")
        self.assertEqual(row["tiers"]["convergence"], "pass")
        self.assertEqual(row["tiers"]["record"], "record")

    def test_results_table_headers_and_row(self):
        ds = self._build()
        headers = ds["results_table"]["headers"]
        self.assertEqual(headers[:3], ["Sweep", "Model", "Nodes"])
        self.assertIn("Final loss", headers)
        row = ds["results_table"]["rows"][0]
        self.assertEqual(row[0], _SWEEP)
        self.assertEqual(row[1], "deepseek-v4-284b")
        self.assertEqual(row[2], 4)
        self.assertIn(6.5, row)  # final loss value present

    def test_overall_status_pass_then_fail(self):
        self.assertEqual(self._build()["overall_status"], "pass")
        # tighten the final-loss gate below the achieved loss -> convergence fails
        sources = {"results": _training_res_dict(final_loss=9.9), "variant": _variant(final_loss_gate=8.0)}
        self.assertEqual(build_training_datasets(sources, _profile())["overall_status"], "fail")

    def test_record_only_when_not_enforced(self):
        sources = {"results": _training_res_dict(), "variant": _variant(enforce=False)}
        ds = build_training_datasets(sources, _profile())
        self.assertEqual(ds["gate_matrix"][0]["tiers"]["throughput"], "record")
        self.assertEqual(ds["overall_status"], "record")

    def test_empty_results_are_safe(self):
        ds = build_training_datasets({"results": {}, "variant": _variant()}, _profile())
        self.assertEqual(ds["gate_matrix"], [])
        self.assertEqual(ds["overall_status"], "na")

    def test_training_series_is_json_safe_and_present(self):
        ds = self._build()
        series = ds["training_series"][_SWEEP]
        # tags preserved; points are [step, value] lists (JSON-safe), not tuples
        self.assertIn("learning/loss", series)
        self.assertIn("perf/per_device_tflops_per_sec", series)
        self.assertEqual(series["learning/loss"][0], [0, 12.0])
        self.assertNotIn("charts", ds["cells"][0])  # no PNG embedding anymore

    def test_metric_bars_across_sweeps_excludes_bools(self):
        bars = {b["metric"]: b for b in self._build()["metric_bars"]}
        self.assertIn("training.tokens_per_sec_per_gpu", bars)
        self.assertEqual(bars["training.tokens_per_sec_per_gpu"]["values"][_SWEEP], 200.0)
        # loss_decreased is a bool metric -> not charted as a bar
        self.assertNotIn("training.loss_decreased", bars)

    def test_viewer_settings_from_profile(self):
        vs = self._build()["viewer_settings"]
        self.assertEqual(vs["skip_initial_steps"], 5)
        self.assertIn("learning/loss", vs["default_series"])
        self.assertIn("perf/per_device_tokens_per_sec", vs["default_series"])

    def test_step_time_dist_computed_from_series(self):
        dist = self._build()["step_time_dist"]
        self.assertEqual(len(dist), 1)
        d = dist[0]
        self.assertEqual(d["sweep"], _SWEEP)
        self.assertEqual(sum(d["counts"]), 4)  # 6 pts minus 2 rampup outliers
        self.assertLess(d["p50"], 11.0)  # steady-state ~10s, rampup (110/90s) excluded
        self.assertTrue(d["labels"])

    def test_step_time_dist_absent_without_series(self):
        res = _training_res_dict()
        del res["sweeps"][_SWEEP]["tb_scalars"]["perf/step_time_seconds"]
        ds = build_training_datasets({"results": res, "variant": _variant()}, _profile())
        self.assertEqual(ds["step_time_dist"], [])

    def test_info_specs_do_not_gate_tiers(self):
        # kind:"info" is record-only: it must not gate. eval_loss(info, None value)
        # must NOT force convergence to na; an all-info tier (stability) is na.
        variant = SimpleNamespace(
            model=SimpleNamespace(id="m"),
            gpu_arch="MI325X",
            framework="jaxmaxtext",
            enforce_thresholds=True,
            training=SimpleNamespace(distributed=True, steps=30),
            thresholds={
                _SWEEP: {
                    "training.tflops_per_sec_per_gpu": {"kind": "min", "value": 100.0},
                    "training.final_loss": {"kind": "max", "value": 15.0},
                    "training.loss_decreased": {"kind": "min", "value": 1},
                    "training.eval_loss": {"kind": "info", "value": 100.0},
                    "training.step_time_p50_ms": {"kind": "info", "value": 3600000.0},
                }
            },
            enabled_sweeps=lambda: [SimpleNamespace(name=_SWEEP)],
        )
        res = {
            "mode": "distributed",
            "sweeps": {
                _SWEEP: {
                    "results": {
                        "training.tflops_per_sec_per_gpu": 200.0,
                        "training.final_loss": 6.5,
                        "training.loss_decreased": 1.0,
                        "training.eval_loss": None,
                        "training.step_time_p50_ms": 9800.0,
                    },
                    "num_nodes": 4,
                }
            },
        }
        tiers = build_training_datasets({"results": res, "variant": variant}, _profile())["gate_matrix"][0]["tiers"]
        self.assertEqual(tiers["throughput"], "pass")
        self.assertEqual(tiers["convergence"], "pass")  # eval_loss(info/None) excluded
        self.assertEqual(tiers["stability"], "na")  # all-info -> no gating specs
        self.assertEqual(tiers["record"], "record")


class MetricBarsCardTests(unittest.TestCase):
    def test_renders_bars_and_step_time_dist(self):
        payload = {
            "datasets": {
                "training_sweep": {
                    "metric_bars": [
                        {
                            "metric": "training.tokens_per_sec_per_gpu",
                            "label": "tok/s/GPU",
                            "unit": "tok/s",
                            "values": {"NN4_BF16": 200.0, "NN4_FP8": 260.0},
                        }
                    ],
                    "step_time_dist": [
                        {"sweep": "NN4_BF16", "labels": [9.8, 10.1], "counts": [3, 1], "p50": 9.9, "p95": 10.1}
                    ],
                }
            }
        }
        card = {
            "type": "metric_bars",
            "id": "metric-bars",
            "title": "Results metrics",
            "bind": "datasets.training_sweep",
        }
        _sid, html_body, _nav = render_card(payload, card)
        self.assertIn("<canvas", html_body)
        self.assertIn("chart.js", html_body)  # self-contained Chart.js
        self.assertIn("tok/s/GPU", html_body)
        self.assertIn("step-time dist", html_body)  # step-time distribution included

    def test_hidden_when_no_data(self):
        card = {"type": "metric_bars", "bind": "datasets.training_sweep", "when_empty": "hide"}
        payload = {"datasets": {"training_sweep": {"metric_bars": [], "step_time_dist": []}}}
        _sid, html_body, in_nav = render_card(payload, card)
        self.assertEqual(html_body, "")
        self.assertFalse(in_nav)


class PayloadEndToEndTests(unittest.TestCase):
    def test_payload_from_training_store(self):
        store = {"cvs_results_dict": _training_res_dict(), "variant_config": _variant(), "lifecycle_report": {}}
        payload = build_rundeck_payload(profile=_profile(), store=store)
        self.assertEqual(payload["suite_id"], "jaxmaxtext")
        self.assertEqual(payload["overall_status"], "pass")
        self.assertEqual(len(payload["gate_matrix"]), 1)
        self.assertEqual(payload["results_table"]["rows"][0][0], _SWEEP)
        labels = {label for label, _v, _link in payload["run_card_display"]}
        self.assertIn("Model", labels)
        self.assertIn("GPU", labels)


class RunCardTests(unittest.TestCase):
    def test_rows_include_topology_and_thresholds(self):
        rows = jaxmaxtext_run_card_display(_variant(), {})
        as_dict = {label: value for label, value, _link in rows}
        self.assertEqual(as_dict["Model"], "deepseek-v4-284b")
        self.assertEqual(as_dict["GPU"], "MI325X")
        self.assertEqual(as_dict["Mode"], "distributed")
        self.assertEqual(as_dict["Thresholds"], "enforced")

    def test_none_variant_is_safe(self):
        self.assertEqual(jaxmaxtext_run_card_display(None, {}), [])


if __name__ == "__main__":
    unittest.main()
