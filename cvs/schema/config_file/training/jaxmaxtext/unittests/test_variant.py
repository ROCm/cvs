"""Unit tests for JAX MaxText training variant schema."""

import json
import unittest
import warnings
from pathlib import Path

from pydantic import ValidationError

from cvs.schema.config_file.training.jaxmaxtext.variant import (
    CheckpointResume,
    Convergence,
    LossCurve,
    NcclConfig,
    ScalingBaseline,
    SmokeTest,
    TrainingVariantConfig,
    validate_thresholds_cover_training,
)

_PACKAGE_ROOT = Path(__file__).resolve().parents[5]


class TestJaxMaxTextSchemaDefaults(unittest.TestCase):
    def test_scaling_baseline_defaults(self):
        sb = ScalingBaseline()
        self.assertEqual(sb.tokens_per_sec_total, 0.0)
        self.assertEqual(sb.num_nodes, 1)

    def test_convergence_defaults(self):
        c = Convergence()
        self.assertEqual(c.target_metric, "auto")
        self.assertEqual(c.target_value, 0.0)

    def test_loss_curve_defaults(self):
        lc = LossCurve()
        self.assertEqual(lc.sample_every, 10)
        self.assertEqual(lc.milestone_steps, [100, 500, 1000, 5000])
        self.assertTrue(lc.enforce)

    def test_smoke_defaults(self):
        s = SmokeTest()
        self.assertTrue(s.enabled)
        self.assertEqual(s.steps, 5)

    def test_checkpoint_resume_defaults(self):
        cr = CheckpointResume()
        self.assertFalse(cr.enabled)


class TestNcclConfig(unittest.TestCase):
    def test_ib_gid_index_changeme_rejected(self):
        with self.assertRaises(ValidationError):
            NcclConfig(ib_gid_index="<changeme>")


class TestValidateThresholdsCoverTraining(unittest.TestCase):
    _GATED = {
        "training.tflops_per_sec_per_gpu": {"kind": "min", "value": 1},
        "training.tokens_per_sec_per_gpu": {"kind": "min", "value": 1},
        "training.final_loss": {"kind": "max", "value": 15},
        "training.loss_decreased": {"kind": "min", "value": 1},
    }

    def test_missing_cell_raises_when_enforced(self):
        with self.assertRaises(ValueError):
            validate_thresholds_cover_training(
                expected_cells=["CELL_A"],
                thresholds={},
                enforce_thresholds=True,
                gated_metrics={"final_loss"},
            )

    def test_full_coverage_passes(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_thresholds_cover_training(
                expected_cells=["CELL_A"],
                thresholds={"CELL_A": dict(self._GATED)},
                enforce_thresholds=True,
                gated_metrics={"final_loss", "loss_decreased", "tflops_per_sec_per_gpu", "tokens_per_sec_per_gpu"},
            )


class TestJaxMaxTextVariantSamples(unittest.TestCase):
    def test_all_committed_variant_samples_validate(self):
        config_dir = _PACKAGE_ROOT / "input" / "config_file" / "training" / "jaxmaxtext"
        for path in sorted(config_dir.glob("*.json")):
            if path.name.endswith("_threshold.json"):
                continue
            with self.subTest(sample=path.name):
                raw = json.loads(path.read_text())
                nccl = (raw.get("training") or {}).get("nccl") or {}
                if any(v == "<changeme>" for v in nccl.values() if isinstance(v, str)):
                    self.skipTest("distributed template with cluster-specific nccl placeholders")
                known = {k: v for k, v in raw.items() if k in TrainingVariantConfig.model_fields}
                known["enforce_thresholds"] = False
                known["thresholds"] = {}
                TrainingVariantConfig.model_validate(known)


if __name__ == "__main__":
    unittest.main()
