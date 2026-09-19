'''Unit tests for TorchTitan Run Deck profile hooks.'''

import unittest
from types import SimpleNamespace

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.profiles.hooks.torchtitan import (
    TorchTitanRundeckVariant,
    torchtitan_metric_verdict,
    torchtitan_run_card_display,
    torchtitan_tier_metric_specs,
    update_torchtitan_rundeck_results,
)
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html


def _variant():
    cell = "MBS=2,GBS=32,PRECISION=BF16"
    inner = SimpleNamespace(
        gpu_arch="MI355X",
        enforce_thresholds=True,
        model_params={
            "hf_model_name": "org/model",
            "tensor_parallel_degree": "2",
            "pipeline_parallel_degree": "1",
            "context_parallel_degree": "1",
            "expert_parallel_degree": "1",
            "data_parallel_shard_degree": "8",
        },
        sweep=SimpleNamespace(
            combinations={
                cell: SimpleNamespace(
                    micro_batch_size="2",
                    global_batch_size="32",
                    precision="BF16",
                )
            }
        ),
        thresholds={
            cell: {
                "training.tokens_per_sec": {"kind": "min", "value": 1000},
            }
        },
    )
    return cell, TorchTitanRundeckVariant(inner, 2)


class TestTorchTitanHooks(unittest.TestCase):
    def test_flattens_cell_keyed_metrics_for_sweep_builder(self):
        cell, variant = _variant()
        target = {}

        update_torchtitan_rundeck_results(
            target,
            {
                cell: {
                    "tokens_per_sec": ["900", "1200"],
                    "loss": ["3.5"],
                    "_log_tail": "not report data",
                }
            },
            variant,
        )

        key = ("org/model", "MI355X", "2", "32", cell, "BF16")
        self.assertEqual(target[key]["all nodes"]["training.tokens_per_sec"], "1200")
        self.assertEqual(target[key]["all nodes"]["training.loss"], "3.5")
        self.assertEqual(target[key]["all nodes"]["_rundeck_cell_id"], cell)
        self.assertNotIn("training._log_tail", target[key]["all nodes"])

    def test_run_card_has_training_topology(self):
        _cell, variant = _variant()

        rows = torchtitan_run_card_display(variant, {})
        display = {label: value for label, value, _is_link in rows}

        self.assertEqual(display["Model"], "org/model")
        self.assertEqual(display["GPU"], "MI355X")
        self.assertEqual(display["nnodes"], "2")
        self.assertIn("TP=2", display["Parallelism"])
        self.assertEqual(display["Thresholds"], "enforced")

    def test_gate_specs_exclude_record_only_info_metrics(self):
        specs = {
            "training.tokens_per_sec": {"kind": "min", "value": 1000},
            "training.scaling_efficiency_pct": {"kind": "info", "value": 85},
        }

        self.assertEqual(
            torchtitan_tier_metric_specs(specs, "gates"),
            {"training.tokens_per_sec": {"kind": "min", "value": 1000}},
        )
        self.assertEqual(torchtitan_tier_metric_specs(specs, "record"), {})

    def test_metric_verdict_fails_missing_and_below_minimum(self):
        spec = {"kind": "min", "value": 1000}

        self.assertEqual(torchtitan_metric_verdict("training.tokens_per_sec", 1200, spec)[0], "pass")
        self.assertEqual(torchtitan_metric_verdict("training.tokens_per_sec", 900, spec)[0], "fail")
        self.assertEqual(torchtitan_metric_verdict("training.tokens_per_sec", None, spec)[0], "fail")

    def test_profile_builds_training_graph_gates_and_table(self):
        cell, variant = _variant()
        results = {}
        update_torchtitan_rundeck_results(
            results,
            {cell: {"tokens_per_sec": ["1200"], "loss": ["3.5"]}},
            variant,
        )

        profile = load_json_profile("torchtitan")
        payload = build_rundeck_payload(
            profile=profile,
            store={
                "cvs_results_dict": results,
                "variant_config": variant,
                "lifecycle_report": {},
            },
        )
        rendered = render_rundeck_html(payload)

        self.assertEqual(payload["chart_series"]["tokens_per_sec"][0]["points"], [(cell, 1200.0)])
        self.assertEqual(payload["gate_matrix"][0]["tiers"]["gates"], "pass")
        self.assertIn("Parallelism", payload["results_table"]["headers"])
        self.assertIn("Training throughput vs cells", rendered)
        self.assertIn("Gate matrix vs thresholds", rendered)
        self.assertIn("Full results", rendered)


if __name__ == "__main__":
    unittest.main()
