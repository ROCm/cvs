"""Unit tests for TorchTitan training variant schema (training/torchtitan/variant.py)."""

import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from cvs.schema.config_file.training.torchtitan.variant import (
    TorchTitanVariantConfig,
    validate_sweep_selector,
)

_PACKAGE_ROOT = Path(__file__).resolve().parents[5]


def _minimal_variant(**overrides):
    payload = {
        "schema_version": 1,
        "framework": "torchtitan_single",
        "gpu_arch": "MI355X",
        "enforce_thresholds": False,
        "config": {"training_iterations": "10"},
        "model_params": {"model_name": "test"},
        "container": {
            "name": "torchtitan_test",
            "image": "rocm/torchtitan:latest",
            "runtime": {"name": "docker", "args": {}},
        },
        "sweep": {
            "combinations": {
                "cell_a": {
                    "name": "cell_a",
                    "micro_batch_size": "2",
                    "global_batch_size": "16",
                    "precision": "BF16",
                }
            },
            "runs": ["cell_a"],
        },
    }
    payload.update(overrides)
    return payload


class TestTorchTitanSweepSelector(unittest.TestCase):
    def test_duplicate_combination_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate sweep.combinations keys"):
            validate_sweep_selector(["x", "x"], ["x"])


class TestTorchTitanVariantConfig(unittest.TestCase):
    def test_minimal_payload_validates(self):
        config = TorchTitanVariantConfig.model_validate(_minimal_variant())
        self.assertEqual(config.framework, "torchtitan_single")
        self.assertEqual(config.cell_key("cell_a"), "MBS=2,GBS=16,PRECISION=BF16")

    def test_enforce_thresholds_requires_matching_threshold_cells(self):
        with self.assertRaisesRegex(ValidationError, "threshold.json does not match"):
            TorchTitanVariantConfig.model_validate(
                _minimal_variant(enforce_thresholds=True, thresholds={}),
            )

    def test_all_committed_variant_samples_validate(self):
        config_dir = _PACKAGE_ROOT / "input" / "config_file" / "training" / "torchtitan"
        for path in sorted(config_dir.glob("*.json")):
            if path.name.endswith("_threshold.json"):
                continue
            with self.subTest(sample=path.name):
                raw = json.loads(path.read_text())
                if raw.get("schema_version") != 1:
                    self.skipTest("legacy config without schema_version")
                known = {k: v for k, v in raw.items() if k in TorchTitanVariantConfig.model_fields}
                known["enforce_thresholds"] = False
                known["thresholds"] = {}
                TorchTitanVariantConfig.model_validate(known)


if __name__ == "__main__":
    unittest.main()
