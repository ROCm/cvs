'''Unit tests for Megatron training-sweep Run Deck builder.'''

import unittest
from types import SimpleNamespace

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.config_adapter import build_inference_config_from_profile
from cvs.lib.report.rundeck.dataset_builders.registry import build_datasets
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html
from cvs.lib.report.training_cells import flatten_training_combo_actuals
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
            "tokens_per_gpu": ["3000"],
            "_log_tail": "ignored",
        },
        "MBS=4,GBS=128,PRECISION=BF16": {
            "throughput_per_gpu": ["150"],
            "elapsed_time_per_iteration": ["2.0"],
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
                "empty": [],
            }
        )
        self.assertEqual(actuals["training.throughput_per_gpu"], 2.5)
        self.assertNotIn("training._log_tail", actuals)
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
                    "tests/training/megatron/megatron_single.py::test_training[MBS=4,GBS=128,PRECISION=FP8]": [
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
        self.assertEqual(fp8["tiers"]["throughput"], "pass")
        self.assertEqual(fp8["cell_lifecycle"].get("training"), 12.0)
        self.assertNotIn("isl", fp8)
        headers = datasets["results_table"]["headers"]
        self.assertIn("MBS", headers)
        self.assertNotIn("ISL", headers)
        self.assertIn("throughput_per_gpu", datasets["chart_series"])
        self.assertEqual(datasets["gate_matrix"][0]["label"], "MBS=4,GBS=128,PRECISION=BF16")

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
        self.assertNotIn("viewer_config", payload)
        html = render_rundeck_html(payload)
        self.assertIn("Megatron Run Deck", html)
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
        self.assertFalse(config.interactive_viewer)

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


if __name__ == "__main__":
    unittest.main()
