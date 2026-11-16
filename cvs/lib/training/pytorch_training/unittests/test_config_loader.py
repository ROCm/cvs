import copy
import json
import unittest
import warnings
from pathlib import Path

from cvs.lib.training.pytorch_training.utils.config_loader import (
    TrainingSweep,
    TrainingVariantConfig,
    resolve_node_thresholds,
)
from cvs.lib.training.pytorch_training.utils.metrics import GATED_METRICS


SAMPLES = Path(__file__).resolve().parents[4] / "input" / "config_file" / "training" / "pytorch_training"


def _sweep(**overrides):
    sweep = {
        "name": "W2-LLAMA-BF16-SL8192-MBS4-GBS128",
        "label": "W2-Llama-BF16",
        "model": "llama-3.1-8b",
        "model_path": "/models/Llama-3.1-8B",
        "precision": "BF16",
        "seq_len": 8192,
        "micro_batch_size": 4,
        "global_batch_size": 128,
    }
    sweep.update(overrides)
    return sweep


def _raw(**training_overrides):
    sweep = _sweep()
    training = {
        "workload": "W2",
        "phase": "performance",
        "run_mode": "perf",
        "peak_tflops_per_gpu": {"BF16": 1307.4, "FP8": 2614.9},
        "data": {"source": "/data/wikitext", "cache_dir": "/data/tokens"},
        "sweeps": [sweep],
    }
    training.update(training_overrides)
    return {
        "schema_version": 1,
        "framework": "pytorch_training",
        "gpu_name": "mi325x",
        "paths": {"shared_fs": "/home/u", "models_dir": "/m", "log_dir": "/home/u/LOGS", "hf_token_file": "/home/u/t"},
        "model": {"id": "meta-llama/Llama-3.1-8B", "remote": 0},
        "container": {"image": "img", "name": "c", "runtime": {"name": "docker", "args": {}}},
        "thresholds": {
            sweep["name"]: {f"training.{name}": {"kind": "info"} for name in GATED_METRICS},
        },
        "training": training,
    }


class TestSweepBatch(unittest.TestCase):
    def test_global_batch_derives_accumulation(self):
        sweep = TrainingSweep(**_sweep())
        self.assertEqual(sweep.accumulation_steps(8), 4)
        self.assertEqual(sweep.accumulation_steps(32), 1)
        self.assertEqual(sweep.global_batch(16), 128)

    def test_global_batch_must_divide_world(self):
        sweep = TrainingSweep(**_sweep(global_batch_size=96, micro_batch_size=3))
        with self.assertRaisesRegex(ValueError, "not a multiple"):
            sweep.accumulation_steps(64)

    def test_fixed_accumulation_ignores_world(self):
        sweep = TrainingSweep(**_sweep(global_batch_size=None, gradient_accumulation_steps=2))
        self.assertEqual(sweep.accumulation_steps(32), 2)
        self.assertEqual(sweep.global_batch(32), 256)

    def test_exactly_one_batch_field(self):
        with self.assertRaises(ValueError):
            TrainingSweep(**_sweep(gradient_accumulation_steps=2))
        with self.assertRaises(ValueError):
            TrainingSweep(**_sweep(global_batch_size=None))


class TestVariantValidation(unittest.TestCase):
    def test_valid_config_loads(self):
        variant = TrainingVariantConfig(**_raw())
        self.assertEqual(variant.gpu_name, "MI325X")
        self.assertEqual(variant.cell_key("W2-Llama-BF16"), "W2-LLAMA-BF16-SL8192-MBS4-GBS128")

    def test_phase_must_match_run_mode(self):
        with self.assertRaisesRegex(ValueError, "requires training.phase"):
            TrainingVariantConfig(**_raw(run_mode="smoke"))

    def test_accuracy_modes_need_eval(self):
        with self.assertRaisesRegex(ValueError, "eval_enabled"):
            TrainingVariantConfig(**_raw(run_mode="finetune", phase="accuracy", warmup_steps=0))

    def test_accuracy_modes_forbid_warmup(self):
        with self.assertRaisesRegex(ValueError, "warmup_steps=0"):
            TrainingVariantConfig(**_raw(run_mode="finetune", phase="accuracy", eval_enabled=True, warmup_steps=3))

    def test_soak_is_protected(self):
        with self.assertRaisesRegex(ValueError, "protected long run"):
            TrainingVariantConfig(**_raw(run_mode="soak", phase="soak", max_duration_seconds=60, timeout_s=600))

    def test_missing_peak_for_precision(self):
        with self.assertRaisesRegex(ValueError, "peak_tflops_per_gpu"):
            TrainingVariantConfig(**_raw(peak_tflops_per_gpu={"FP8": 2614.9}))

    def test_token_sweeps_need_data(self):
        with self.assertRaisesRegex(ValueError, "training.data"):
            TrainingVariantConfig(**_raw(data={}))

    def test_mx_precisions_rejected_off_mi355x(self):
        raw = _raw(peak_tflops_per_gpu={"MXFP8": 5033.2})
        raw["training"]["sweeps"][0]["precision"] = "MXFP8"
        with self.assertRaisesRegex(ValueError, "MI355X only"):
            TrainingVariantConfig(**raw)
        raw["gpu_name"] = "MI355X"
        TrainingVariantConfig(**raw)

    def test_threshold_coverage_enforced(self):
        raw = _raw()
        cell = raw["thresholds"]["W2-LLAMA-BF16-SL8192-MBS4-GBS128"]
        del cell["training.tokens_per_sec"]
        with self.assertRaisesRegex(ValueError, "missing gated-metric"):
            TrainingVariantConfig(**raw)
        raw["enforce_thresholds"] = False
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            TrainingVariantConfig(**raw)

    def test_sweep_baseline_overrides_workload_baseline(self):
        raw = _raw(scaling_baseline={"tokens_per_sec_total": 1000.0})
        raw["training"]["sweeps"][0]["scaling_baseline"] = {"tokens_per_sec_total": 2000.0, "eval_loss": 3.0}
        variant = TrainingVariantConfig(**raw)
        baseline = variant.scaling_baseline_for("W2-Llama-BF16")
        self.assertEqual(baseline.tokens_per_sec_total, 2000.0)
        self.assertEqual(baseline.eval_loss, 3.0)

    def test_unknown_benchmark_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown entries"):
            TrainingVariantConfig(**_raw(benchmarks={"enabled": True, "metrics": ["arc_pct"]}))


class TestNodeThresholds(unittest.TestCase):
    def test_by_nodes_selects_topology(self):
        thresholds = {"cell": {"m": {"kind": "min", "by_nodes": {"2": {"value": 5}}}}}
        self.assertEqual(resolve_node_thresholds(thresholds, 2)["cell"]["m"], {"kind": "min", "value": 5})
        self.assertEqual(resolve_node_thresholds(thresholds, 4)["cell"]["m"], {"kind": "info"})


class TestSampleConfigs(unittest.TestCase):
    def test_every_sample_validates(self):
        configs = sorted(SAMPLES.glob("*_config.json"))
        self.assertTrue(configs)
        for path in configs:
            with self.subTest(config=path.name):
                raw = json.loads(path.read_text())
                threshold_path = SAMPLES / raw["threshold_json"]
                self.assertTrue(threshold_path.exists(), threshold_path)
                text = json.dumps(raw)
                for placeholder in ("<changeme-models-host-path>", "<changeme-training-data-host-path>"):
                    text = text.replace(placeholder, "/x")
                text = text.replace("<changeme-local-scratch-host-path>", "/scratch").replace(
                    "<changeme-oob-interface>", "eth0"
                )
                raw = json.loads(text.replace("{user-id}", "u").replace("{shared_fs}", "/home/u"))
                raw["thresholds"] = copy.deepcopy(json.loads(threshold_path.read_text()))
                TrainingVariantConfig(**raw)


if __name__ == "__main__":
    unittest.main()
