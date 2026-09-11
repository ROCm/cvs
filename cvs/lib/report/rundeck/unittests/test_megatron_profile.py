'''Unit tests for the Megatron named-cell Run Deck profile.'''

import unittest
from types import SimpleNamespace

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.dataset_builders.sweep import build_sweep_datasets
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html


def _variant():
    cells = (
        "MBS=4,GBS=128,PRECISION=FP8",
        "MBS=4,GBS=128,PRECISION=BF16",
    )
    combinations = {
        cell: SimpleNamespace(micro_batch_size="4", global_batch_size="128", precision=cell.rsplit("=", 1)[-1])
        for cell in cells
    }
    thresholds = {
        cell: {
            "training.throughput_per_gpu": {"kind": "min", "value": 100},
            "training.tokens_per_gpu": {"kind": "min", "value": 1000},
            "training.mem_usage": {"kind": "max", "value": 0.85, "optional": True},
        }
        for cell in cells
    }
    return SimpleNamespace(
        train_params={
            "model_name": "llama3.1_8B",
            "tensor_parallelism": "1",
            "pipeline_parallelism": "2",
            "fsdp": "0",
        },
        gpu_arch="MI300X",
        enforce_thresholds=True,
        thresholds=thresholds,
        sweep=SimpleNamespace(combinations=combinations, runs=list(cells)),
        container=SimpleNamespace(env={"NNODES": "2"}),
    )


def _results():
    return {
        "MBS=4,GBS=128,PRECISION=FP8": {
            "throughput_per_gpu": ["120.5"],
            "tokens_per_gpu": ["1400"],
            "mem_usage": [],
        },
        "MBS=4,GBS=128,PRECISION=BF16": {
            "throughput_per_gpu": ["90"],
            "tokens_per_gpu": ["1100"],
            "mem_usage": [],
        },
    }


class TestMegatronProfile(unittest.TestCase):
    def test_all_megatron_stems_share_profile(self):
        stems = (
            "megatron_single",
            "megatron_distributed",
            "megatron_llama3_1_8b_single",
            "megatron_llama3_1_8b_distributed",
            "megatron_llama3_1_70b_single",
            "megatron_llama3_1_70b_distributed",
        )
        for stem in stems:
            with self.subTest(stem=stem):
                profile = load_json_profile(stem)
                self.assertEqual(profile["suite_id"], "megatron")
                self.assertEqual(profile["dataset_builder"], "sweep")

    def test_named_sweep_preserves_training_dimensions_and_gates(self):
        profile = load_json_profile("megatron")
        datasets = build_sweep_datasets(
            {"results": _results(), "variant": _variant(), "lifecycle_report": {}},
            profile,
        )

        self.assertEqual(len(datasets["cells"]), 2)
        self.assertNotIn("isl", datasets["cells"][0])
        self.assertNotIn("osl", datasets["cells"][0])
        self.assertEqual(datasets["cells"][0]["micro_batch_size"], "4")
        self.assertEqual(datasets["cells"][0]["global_batch_size"], "128")
        self.assertEqual(
            [cell["tiers"]["thresholds"] for cell in datasets["cells"]],
            ["pass", "fail"],
        )
        self.assertEqual(datasets["overall_status"], "fail")
        self.assertIn("MBS", datasets["results_table"]["headers"])
        self.assertIn("Throughput/GPU", datasets["results_table"]["headers"])

        points = datasets["charts"]["throughput_per_gpu"]["cells"][0]["points"]
        self.assertEqual(points[0][0], "MBS=4 · GBS=128 · PRECISION=FP8")
        self.assertEqual(points[0][1], 120.5)

    def test_payload_renders_training_graphs_and_run_card(self):
        profile = load_json_profile("megatron")
        payload = build_rundeck_payload(
            profile=profile,
            store={
                "cvs_results_dict": _results(),
                "variant_config": _variant(),
                "lifecycle_report": {},
            },
            cvs_version="test",
        )
        run_card = {label: value for label, value, _link in payload["run_card_display"]}
        self.assertEqual(run_card["Model"], "llama3.1_8B")
        self.assertEqual(run_card["GPU"], "MI300X")
        self.assertEqual(run_card["Nodes"], "2")
        self.assertEqual(run_card["Parallelism"], "TP=1, PP=2, FSDP=0")
        self.assertFalse(payload["viewer_config"]["interactivity"]["enabled"])

        document = render_rundeck_html(payload)
        self.assertIn("Throughput per GPU vs cells", document)
        self.assertIn("Token throughput per GPU vs cells", document)
        self.assertIn("Gate matrix", document)
        self.assertIn("Full results", document)
        self.assertNotIn("ISL=", document)
        self.assertNotIn("OSL=", document)

    def test_legacy_lookup_mismatch_returns_none(self):
        from cvs.lib.report.profiles.hooks.megatron_run_card import build_legacy_rundeck_variant

        self.assertIsNone(
            build_legacy_rundeck_variant(
                {"config": {}},
                "megatron_llama3_1_8b_single",
                {},
                {"single_node": {"llama3_1_8b": {"mi300x": {}}}},
                "mi355",
                object(),
            )
        )
        self.assertIs(
            build_legacy_rundeck_variant({"pydantic": True}, "megatron_single", {}, {}, None, "keep"),
            "keep",
        )

    def test_crashed_combo_and_non_numeric_spec_do_not_raise(self):
        profile = load_json_profile("megatron")
        variant = _variant()
        cell = "MBS=4,GBS=128,PRECISION=FP8"
        variant.thresholds[cell]["training.throughput_per_gpu"] = {"kind": "info"}
        datasets = build_sweep_datasets(
            {
                "results": {
                    cell: {"throughput_per_gpu": "n/a", "tokens_per_gpu": ["1400"]},
                    "MBS=4,GBS=128,PRECISION=BF16": None,
                },
                "variant": variant,
            },
            profile,
        )
        self.assertEqual(len(datasets["cells"]), 2)
        self.assertEqual(datasets["cells"][1]["tiers"]["thresholds"], "na")
        thru = next(m for m in datasets["cells"][0]["metrics"] if m["metric"].endswith("throughput_per_gpu"))
        self.assertIsNone(thru["bar_pct"])

    def test_min_gate_uses_worst_sample(self):
        profile = load_json_profile("megatron")
        results = {
            "MBS=4,GBS=128,PRECISION=FP8": {"throughput_per_gpu": ["200", "90"], "tokens_per_gpu": ["1400"]},
            "MBS=4,GBS=128,PRECISION=BF16": {"throughput_per_gpu": ["120"], "tokens_per_gpu": ["1100"]},
        }
        datasets = build_sweep_datasets({"results": results, "variant": _variant()}, profile)
        self.assertEqual(datasets["cells"][0]["actuals"]["training.throughput_per_gpu"], 90.0)
        self.assertEqual(datasets["cells"][0]["tiers"]["thresholds"], "fail")

    def test_single_cell_renders_chart(self):
        profile = load_json_profile("megatron")
        cell = "MBS=4,GBS=128,PRECISION=FP8"
        payload = build_rundeck_payload(
            profile=profile,
            store={
                "cvs_results_dict": {cell: {"throughput_per_gpu": ["120.5"], "tokens_per_gpu": ["1400"]}},
                "variant_config": _variant(),
            },
            cvs_version="test",
        )
        document = render_rundeck_html(payload)
        self.assertIn("chart-bar", document)
        self.assertNotIn("No series data.", document)


if __name__ == "__main__":
    unittest.main()
