"""Unit tests for Megatron training variant schema (training/megatron/variant.py)."""

import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from cvs.schema.config_file.training.megatron.variant import (
    MegatronVariantConfig,
    validate_sweep_selector,
)

_PACKAGE_ROOT = Path(__file__).resolve().parents[5]
_SAMPLE_CONFIG = (
    _PACKAGE_ROOT / "input" / "config_file" / "training" / "megatron" / "mi300x_megatron_llama-3.1-8b_single.json"
)


def _minimal_variant(**overrides):
    payload = {
        "schema_version": 1,
        "framework": "megatron_single",
        "gpu_arch": "MI300X",
        "enforce_thresholds": False,
        "config": {"training_iterations": "10"},
        "model_params": {"model_name": "test"},
        "container": {
            "name": "megatron_test",
            "image": "rocm/megatron:latest",
            "runtime": {"name": "docker", "args": {}},
        },
        "sweep": {
            "combinations": {
                "cell_a": {
                    "name": "cell_a",
                    "micro_batch_size": "1",
                    "global_batch_size": "8",
                    "precision": "BF16",
                }
            },
            "runs": ["cell_a"],
        },
    }
    payload.update(overrides)
    return payload


class TestMegatronSweepSelector(unittest.TestCase):
    def test_duplicate_combination_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate sweep.combinations keys"):
            validate_sweep_selector(["a", "a"], ["a"])

    def test_unknown_run_reference_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown combinations"):
            validate_sweep_selector(["a"], ["b"])


class TestMegatronVariantConfig(unittest.TestCase):
    def test_minimal_payload_validates(self):
        config = MegatronVariantConfig.model_validate(_minimal_variant())
        self.assertEqual(config.framework, "megatron_single")
        self.assertEqual(config.cell_key("cell_a"), "MBS=1,GBS=8,PRECISION=BF16")

    def test_sample_json_validates_without_thresholds_when_not_enforced(self):
        raw = json.loads(_SAMPLE_CONFIG.read_text())
        known = {k: v for k, v in raw.items() if k in MegatronVariantConfig.model_fields}
        known["enforce_thresholds"] = False
        known["thresholds"] = {}
        config = MegatronVariantConfig.model_validate(known)
        self.assertEqual(config.gpu_arch, "MI300X")

    def test_enforce_thresholds_requires_matching_threshold_cells(self):
        with self.assertRaisesRegex(ValidationError, "threshold.json does not match"):
            MegatronVariantConfig.model_validate(
                _minimal_variant(enforce_thresholds=True, thresholds={}),
            )

    def test_all_committed_variant_samples_validate(self):
        config_dir = _PACKAGE_ROOT / "input" / "config_file" / "training" / "megatron"
        for path in sorted(config_dir.glob("*.json")):
            if path.name.endswith("_threshold.json"):
                continue
            with self.subTest(sample=path.name):
                raw = json.loads(path.read_text())
                if raw.get("schema_version") != 1:
                    self.skipTest("legacy config without schema_version")
                known = {k: v for k, v in raw.items() if k in MegatronVariantConfig.model_fields}
                known["enforce_thresholds"] = False
                known["thresholds"] = {}
                MegatronVariantConfig.model_validate(known)


if __name__ == "__main__":
    unittest.main()
