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
        "gpu_arch": "MI325X",
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
        "params": {"nproc_per_node": 8, "warmup_steps": 2, "measure_steps": 4},
        "env": {},
        "sweep": {
            "combinations": {
                "w1": {
                    "name": "resnet50_bf16",
                    "model": "resnet50",
                    "precision": "BF16",
                    "batch_size": 128,
                    "image_size": 224,
                }
            },
            "runs": ["w1"],
        },
    }


def _thresholds():
    cell = "MODEL=resnet50,PRECISION=BF16,RES=224,BS=128,GPUS=8"
    return {cell: {f"training.{metric}": {"kind": "info"} for metric in GATED_METRICS}}


class TestVisionConfigLoader(unittest.TestCase):
    def _load(self, config=None, thresholds=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "w1_config.json").write_text(json.dumps(config or _config()))
        (root / "w1_threshold.json").write_text(json.dumps(thresholds or _thresholds()))
        return load_vision_variant(root / "w1_config.json", {"username": "tester"})

    def test_loads_and_resolves_paths(self):
        variant = self._load()
        self.assertEqual(variant.paths.log_dir, "/home/tester/LOGS")
        self.assertEqual(
            variant.cell_key("w1"),
            "MODEL=resnet50,PRECISION=BF16,RES=224,BS=128,GPUS=8",
        )

    def test_rejects_unknown_run(self):
        config = _config()
        config["sweep"]["runs"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "unknown combinations"):
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
