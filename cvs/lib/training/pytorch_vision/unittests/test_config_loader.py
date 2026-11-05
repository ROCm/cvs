import json
import tempfile
import unittest
import warnings
from pathlib import Path

from cvs.lib.training.pytorch_vision.utils.config_loader import (
    load_vision_variant,
    validate_sweep_selector,
)
from cvs.lib.training.pytorch_vision.utils.metrics import GATED_METRICS


def _config():
    return {
        "schema_version": 1,
        "framework": "pytorch_vision_training",
        "gpu_name": "MI325X",
        "enforce_thresholds": True,
        "threshold_json": "w1_threshold.json",
        "paths": {
            "shared_fs": "/home/{user-id}",
            "models_dir": "{shared_fs}/models",
            "log_dir": "{shared_fs}/LOGS",
            "hf_token_file": "{shared_fs}/.hf_token",
        },
        "model": {"id": "torchvision/resnet50", "remote": 0},
        "container": {
            "lifetime": "per_run",
            "name": "vision_w1",
            "image": "rocm/pytorch:test",
            "runtime": {"name": "docker", "args": {}},
        },
        "training": {
            "distributed": False,
            "gpus_per_node": 8,
            "warmup_steps": 2,
            "steps": 4,
            "peak_tflops_per_gpu": 1307.4,
            "env_vars": {},
            "error_patterns": {},
            "sweeps": [
                {
                    "name": "cell-w1",
                    "label": "W1-BF16-R224-B128",
                    "model": "resnet50",
                    "precision": "BF16",
                    "batch_size": 128,
                    "image_size": 224,
                    "gradient_accumulation_steps": 1,
                    "training_flops_per_image": 24600000000,
                }
            ],
            "enabled_sweep_list": ["cell-w1"],
        },
    }


def _thresholds():
    return {"cell-w1": {f"training.{metric}": {"kind": "info"} for metric in GATED_METRICS}}


def _target_accuracy_config(enabled=True):
    config = _config()
    config["training"].update(
        {
            "enabled": enabled,
            "run_mode": "train_to_accuracy",
            "phase": "accuracy",
            "max_epochs": 90,
            "eval_enabled": True,
            "eval_every_epochs": 1,
            "eval_sample_count": 50_000,
            "warmup_steps": 0,
            "learning_rate": 0.8,
            "momentum": 0.9,
            "weight_decay": 0.0001,
            "lr_schedule": {
                "name": "multistep",
                "warmup_epochs": 5,
                "milestones_epochs": [30, 60, 80],
                "gamma": 0.1,
            },
            "accuracy": {"target_top1_pct": 75.5, "target_top5_pct": 92.5},
            "convergence": {
                "enabled": True,
                "stop_when_reached": True,
                "target_top1_pct": 75.5,
            },
        }
    )
    config["training"]["sweeps"][0].update(
        {
            "model": "resnet50",
            "precision": "BF16",
            "batch_size": 256,
            "image_size": 224,
            "gradient_accumulation_steps": 1,
            "data_mode": "rocal",
            "dataset_path": "/datasets/imagenet",
            "rocal_device": "gpu",
            "augmentation": "standard",
        }
    )
    return config


class TestVisionConfigLoader(unittest.TestCase):
    def _load(self, config=None, thresholds=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "w1_config.json").write_text(json.dumps(config or _config()))
        (root / "w1_threshold.json").write_text(json.dumps(thresholds or _thresholds()))
        return load_vision_variant(root / "w1_config.json", {"username": "tester"})

    def test_gpu_name_is_uppercased_and_exposed_as_gpu_arch(self):
        """gpu_name matches Megatron/TorchTitan; gpu_arch stays an alias for the
        shared report layer."""
        config = _config()
        config["gpu_name"] = "mi325x"
        variant = self._load(config)
        self.assertEqual(variant.gpu_name, "MI325X")
        self.assertEqual(variant.gpu_arch, "MI325X")

    def test_unknown_gpu_name_is_rejected(self):
        config = _config()
        config["gpu_name"] = "MI999X"
        with self.assertRaisesRegex(ValueError, "gpu_name must be one of"):
            self._load(config)

    def test_inline_documentation_keys_are_accepted(self):
        """The shipped configs carry _*_comment keys like the other training
        suites; they must not trip schema validation."""
        config = _config()
        config["_format_note"] = "doc"
        config["_sweeps_comment"] = "doc"
        variant = self._load(config)
        self.assertEqual(variant.training.workload, "W1")

    def test_loads_and_resolves_paths(self):
        variant = self._load()
        self.assertEqual(variant.paths.log_dir, "/home/tester/LOGS")
        self.assertEqual(variant.cell_key("cell-w1"), "cell-w1")
        self.assertEqual(variant.cell_key("W1-BF16-R224-B128", 224, 128), "cell-w1")

    def test_rejects_unknown_run(self):
        config = _config()
        config["training"]["enabled_sweep_list"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "unknown sweeps"):
            self._load(config=config)

    def test_rejects_invalid_error_pattern(self):
        config = _config()
        config["training"]["error_patterns"] = {"bad": "["}
        with self.assertRaisesRegex(ValueError, "is invalid"):
            self._load(config=config)

    def test_rejects_sweep_name_label_collision(self):
        config = _config()
        config["training"]["sweeps"][0]["label"] = "cell-w1"
        with self.assertRaisesRegex(ValueError, "names and labels must be distinct"):
            self._load(config=config)

    def test_rejects_ga_sweep_without_enabled_baseline(self):
        config = _config()
        sweep = config["training"]["sweeps"][0]
        sweep.update(
            {
                "name": "cell-ga4",
                "label": "W1-GA4",
                "batch_size": 32,
                "gradient_accumulation_steps": 4,
            }
        )
        config["training"]["enabled_sweep_list"] = ["cell-ga4"]
        with self.assertRaisesRegex(ValueError, "requires exactly one enabled GA=1"):
            self._load(config=config)

    def test_zero_init_residual_is_limited_to_resnets(self):
        config = _config()
        config["training"]["zero_init_residual"] = True
        self.assertTrue(self._load(config).training.zero_init_residual)
        config["training"]["sweeps"][0]["model"] = "vit_b_16"
        with self.assertRaisesRegex(ValueError, "zero_init_residual applies to torchvision ResNets"):
            self._load(config=config)

    def test_label_smoothing_is_limited_to_the_native_harness(self):
        config = _config()
        config["training"]["label_smoothing"] = 0.1
        self.assertEqual(self._load(config).training.label_smoothing, 0.1)
        config["training"]["harness"] = "hf_trainer"
        config["training"]["sweeps"][0]["model_id"] = "google/vit-base-patch16-224"
        with self.assertRaisesRegex(ValueError, "label_smoothing applies to the native harness"):
            self._load(config=config)

    def test_rejects_disabled_checkpoint_validation(self):
        config = _config()
        config["training"]["checkpoint_enabled"] = False
        with self.assertRaisesRegex(ValueError, "checkpoint_enabled"):
            self._load(config=config)

    def test_loads_explicit_phase_and_iterator_modes(self):
        config = _config()
        config["training"].update(
            {
                "phase": "smoke",
                "run_mode": "smoke",
                "eval_enabled": True,
            }
        )
        config["training"]["sweeps"][0].update(
            {
                "data_mode": "rocal",
                "dataset_path": "/datasets/imagenet",
                "rocal_device": "cpu",
                "augmentation": "heavy",
            }
        )
        variant = self._load(config=config)
        self.assertEqual(variant.training.run_mode, "smoke")
        self.assertEqual(variant.training.enabled_sweeps()[0].rocal_device, "cpu")
        self.assertEqual(variant.training.enabled_sweeps()[0].augmentation, "heavy")

    def test_protects_long_profiles(self):
        config = _target_accuracy_config()
        with self.assertRaisesRegex(ValueError, "protected long run"):
            self._load(config=config)

    def test_target_accuracy_requires_early_stop(self):
        config = _target_accuracy_config(enabled=False)
        config["training"]["convergence"]["stop_when_reached"] = False
        with self.assertRaisesRegex(ValueError, "stop_when_reached"):
            self._load(config=config)

    def test_accepts_target_accuracy_with_safety_cap(self):
        config = _target_accuracy_config(enabled=False)
        config["enforce_thresholds"] = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            variant = self._load(config=config)
        self.assertEqual(variant.training.lr_schedule.milestones_epochs, [30, 60, 80])
        self.assertEqual(variant.training.warmup_steps, 0)
        self.assertTrue(variant.training.convergence.stop_when_reached)

    def test_target_accuracy_recipe_is_configurable(self):
        config = _target_accuracy_config(enabled=False)
        config["enforce_thresholds"] = False
        config["training"]["learning_rate"] = 0.4
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            variant = self._load(config=config)
        self.assertEqual(variant.training.learning_rate, 0.4)

    def test_accuracy_profile_rejects_uncounted_optimizer_warmup(self):
        config = _config()
        config["training"].update(
            {
                "run_mode": "train_5k",
                "phase": "accuracy",
                "steps": 5000,
                "eval_enabled": True,
                "warmup_steps": 5,
            }
        )
        with self.assertRaisesRegex(ValueError, "uncounted optimizer updates"):
            self._load(config=config)

    def test_rejects_missing_gated_threshold_when_enforced(self):
        thresholds = _thresholds()
        next(iter(thresholds.values())).pop("training.images_per_sec")
        with self.assertRaisesRegex(ValueError, "missing gated-metric"):
            self._load(thresholds=thresholds)

    def test_warns_for_missing_threshold_when_record_only(self):
        config = _config()
        config["enforce_thresholds"] = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._load(config=config, thresholds={"unused": {}})
        self.assertTrue(any("record-only" in str(item.message) for item in caught))

    def test_rejects_unfilled_placeholder(self):
        config = _config()
        config["container"]["image"] = "<changeme>"
        with self.assertRaisesRegex(ValueError, "container.image"):
            self._load(config=config)

    def _task_config(self, task="detection"):
        config = _config()
        config["training"].update({"harness": "task", "task": task})
        config["training"]["sweeps"][0].update(
            {
                "model_id": "facebook/detr-resnet-50",
                "backend": "transformers",
                "data_mode": "dataset",
                "dataset_path": "/datasets/coco",
            }
        )
        return config

    def test_loads_task_harness(self):
        variant = self._load(config=self._task_config())
        self.assertEqual(variant.training.task, "detection")
        self.assertEqual(variant.training.enabled_sweeps()[0].data_mode, "dataset")

    def test_task_harness_requires_transformers_backend(self):
        config = self._task_config()
        config["training"]["sweeps"][0]["backend"] = "torchvision"
        with self.assertRaisesRegex(ValueError, "requires backend 'transformers'"):
            self._load(config=config)

    def test_non_classification_task_requires_task_harness(self):
        config = self._task_config()
        config["training"]["harness"] = "native"
        with self.assertRaisesRegex(ValueError, "requires harness=task"):
            self._load(config=config)

    def test_dataset_data_mode_is_task_harness_only(self):
        config = _config()
        config["training"]["sweeps"][0].update({"data_mode": "dataset", "dataset_path": "/datasets/coco"})
        with self.assertRaisesRegex(ValueError, "data_mode 'dataset'"):
            self._load(config=config)

    def test_native_vision_mamba_requires_mamba_backend(self):
        config = _config()
        config["training"]["sweeps"][0]["model"] = "vim_small"
        with self.assertRaisesRegex(ValueError, "requires backend 'mamba'"):
            self._load(config=config)
        config["training"]["sweeps"][0]["backend"] = "mamba"
        self.assertEqual(self._load(config=config).training.enabled_sweeps()[0].backend, "mamba")

    def test_finetune_runs_in_accuracy_phase(self):
        config = _config()
        config["training"].update({"run_mode": "finetune", "eval_enabled": True, "warmup_steps": 0})
        with self.assertRaisesRegex(ValueError, "finetune requires training.phase='accuracy'"):
            self._load(config=config)
        config["training"]["phase"] = "accuracy"
        self.assertEqual(self._load(config=config).training.run_mode, "finetune")

    def test_scale_parity_needs_metric_and_value_together(self):
        config = _config()
        config["training"]["scaling_baseline"] = {"images_per_sec_total": 1000.0, "accuracy_value": 80.0}
        with self.assertRaisesRegex(ValueError, "must be set together"):
            self._load(config=config)
        config["training"]["scaling_baseline"]["accuracy_metric"] = "images_per_sec"
        with self.assertRaisesRegex(ValueError, "accuracy_metric must be one of"):
            self._load(config=config)
        config["training"]["scaling_baseline"]["accuracy_metric"] = "top1_accuracy_pct"
        self.assertEqual(self._load(config=config).training.scaling_baseline.accuracy_value, 80.0)

    def test_rejects_unknown_top_level_field(self):
        config = _config()
        config["gpu_arhc"] = "typo"
        with self.assertRaisesRegex(ValueError, "gpu_arhc"):
            self._load(config=config)


class TestSweepSelector(unittest.TestCase):
    def test_rejects_duplicates(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_sweep_selector(["w1", "w1"], ["w1"])

    def test_accepts_selected_subset(self):
        validate_sweep_selector(["w1", "w2"], ["w1"])


if __name__ == "__main__":
    unittest.main()
