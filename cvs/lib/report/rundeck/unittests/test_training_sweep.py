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
        self.assertTrue(any(c["type"] == "gate_matrix" for c in prof["cards"]))  # from base


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
