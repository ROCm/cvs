'''Unit tests for the TorchTitan training-sweep Run Deck builder.'''

import unittest
from types import SimpleNamespace

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.dataset_builders.registry import build_datasets
from cvs.lib.report.training_cells import hide_training_scaling_efficiency


def _variant():
    return SimpleNamespace(
        gpu_arch="MI355X",
        enforce_thresholds=False,
        train_params={
            "model_name": "llama3_1_8b",
            "tokenizer_model": "meta-llama/Llama-3.1-8B",
            "tensor_parallel_degree": "1",
            "pipeline_parallel_degree": "1",
        },
        container=SimpleNamespace(image="rocm/torchtitan:latest", env={"NNODES": "1"}),
        thresholds={},
        cell_key=lambda name: name,
    )


def _train_res():
    return {
        "MBS=1,GBS=8,PRECISION=bf16": {
            "tokens_per_sec": ["12000"],
            "tflops": ["180"],
            "loss": ["1.2"],
            "mem_usage_gb": ["40"],
            "step_time_p50_ms": ["11.5"],
            "step_time_p95_ms": ["19.15"],
            "scaling_efficiency_pct": ["90"],
            "_loss_curve": [[1, 2.0], [10, 1.2]],
            "_tps_curve": [[1, 10000.0], [10, 12000.0]],
            "_tflops_curve": [[1, 150.0], [10, 180.0]],
            "_grad_norm_curve": [[1, 1.0], [10, 0.5]],
        },
        "MBS=2,GBS=16,PRECISION=bf16": {
            "tokens_per_sec": ["20000"],
            "tflops": ["240"],
            "loss": ["1.1"],
            "_tps_curve": [[1, 18000.0], [10, 20000.0]],
            "_tflops_curve": [[1, 200.0], [10, 240.0]],
        },
    }


class TestTorchTitanTrainingSweep(unittest.TestCase):
    def test_cells_map_tps_and_tflops_curves(self):
        datasets = build_datasets(
            "training_sweep",
            {"results": _train_res(), "variant": _variant(), "lifecycle_report": {}},
            load_json_profile("torchtitan"),
        )
        cells = datasets["cells"]
        self.assertEqual(len(cells), 2)
        small = next(c for c in cells if c["mbs"] == "1")
        self.assertEqual(small["model"], "meta-llama/Llama-3.1-8B")
        self.assertEqual(small["tp"], "1")
        self.assertEqual(small["actuals"]["training.tokens_per_sec"], 12000.0)
        self.assertEqual(small["tokens_curve"], [[1, 10000.0], [10, 12000.0]])
        self.assertEqual(small["throughput_curve"], [[1, 150.0], [10, 180.0]])
        self.assertEqual(small["loss_curve"], [[1, 2.0], [10, 1.2]])
        self.assertEqual(datasets["sweep_summaries"][0]["label"], "TorchTitan sweep")
        self.assertEqual(datasets["sweep_summaries"][0]["headline_unit"], "tok/s/device")
        self.assertEqual(datasets["sweep_summaries"][0]["max_output_throughput"], 20000.0)
        self.assertIn("tokens_per_sec", datasets["chart_series"])
        self.assertEqual(datasets["chart_series"]["tokens_per_sec"][0]["label"], "TorchTitan sweep")

    def test_single_node_hides_scaling_efficiency(self):
        datasets = build_datasets(
            "training_sweep",
            {
                "results": _train_res(),
                "variant": _variant(),
                "lifecycle_report": {},
                "suite_stem": "torchtitan_single",
            },
            load_json_profile("torchtitan"),
        )
        headers = datasets["results_table"]["headers"]
        self.assertNotIn("Scaling eff. (%)", headers)
        self.assertIn("Tok/s/device", headers)
        suffixes = [ch.metric_suffix for ch in datasets["config"].chart_series]
        self.assertNotIn("scaling_efficiency_pct", suffixes)

    def test_distributed_keeps_scaling_efficiency(self):
        self.assertFalse(
            hide_training_scaling_efficiency(
                _variant(),
                {"tests/training/torchtitan/torchtitan_distributed.py::test_training": []},
                "torchtitan_distributed",
            )
        )
        self.assertTrue(
            hide_training_scaling_efficiency(
                None,
                {"tests/training/torchtitan/torchtitan_single.py::test_training": []},
                None,
            )
        )


if __name__ == "__main__":
    unittest.main()
