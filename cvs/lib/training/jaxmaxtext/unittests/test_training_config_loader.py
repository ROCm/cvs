'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs/lib/training/jaxmaxtext/utils/training_config_loader.py: schema
defaults for the metric add-ons (scaling_baseline / convergence / loss_curve),
the expected_cells (sweep-name) contract, the threshold-coverage validator, and
a round-trip load of a real jaxmaxtext config file.
'''

import unittest
import warnings
from pathlib import Path

from cvs.lib.training.jaxmaxtext.utils.training_config_loader import (
    CheckpointResume,
    Convergence,
    LossCurve,
    ScalingBaseline,
    load_training_variant,
    validate_thresholds_cover_training,
)

# Repo package root (the inner `cvs/` dir that holds `input/`): the test lives at
# cvs/lib/training/jaxmaxtext/unittests/, so parents[4] is that package root.
_PKG_ROOT = Path(__file__).resolve().parents[4]
_SINGLE_CONFIG = _PKG_ROOT / "input/config_file/training/jaxmaxtext/mi300x_jaxmaxtext_llama-3.3-70b_single.json"


class SchemaDefaultsTests(unittest.TestCase):
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
        self.assertEqual(lc.max_slope, 0.0)
        self.assertTrue(lc.enforce)

    def test_checkpoint_resume_defaults(self):
        cr = CheckpointResume()
        self.assertFalse(cr.enabled)  # opt-in: off by default
        self.assertEqual(cr.sweep, "")
        self.assertEqual(cr.steps_before_ckpt, 6)
        self.assertEqual(cr.steps_after_resume, 6)
        self.assertEqual(cr.checkpoint_period, 5)
        self.assertEqual(cr.loss_tolerance, 0.1)
        self.assertEqual(cr.max_save_seconds, 0.0)
        self.assertEqual(cr.max_load_seconds, 0.0)
        self.assertEqual(cr.smoke_model_overrides, {})


class ValidateThresholdsCoverTrainingTests(unittest.TestCase):
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
            )

    def test_missing_cell_warns_when_not_enforced(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_thresholds_cover_training(
                expected_cells=["CELL_A"],
                thresholds={},
                enforce_thresholds=False,
            )
        self.assertTrue(any("does not match" in str(w.message) for w in caught))

    def test_gated_metric_gap_raises_when_enforced(self):
        # Cell present but missing the gated-metric specs -> coverage failure.
        with self.assertRaises(ValueError):
            validate_thresholds_cover_training(
                expected_cells=["CELL_A"],
                thresholds={"CELL_A": {}},
                enforce_thresholds=True,
            )

    def test_full_coverage_passes(self):
        # No exception, no warning when every cell + gated metric is covered.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_thresholds_cover_training(
                expected_cells=["CELL_A"],
                thresholds={"CELL_A": dict(self._GATED)},
                enforce_thresholds=True,
            )


class RealConfigRoundTripTests(unittest.TestCase):
    def setUp(self):
        if not _SINGLE_CONFIG.is_file():
            self.skipTest(f"config fixture missing: {_SINGLE_CONFIG}")
        # Empty cluster dict -> {user-id} resolves to the local OS user.
        self.cfg = load_training_variant(str(_SINGLE_CONFIG), {})

    def test_metric_addon_blocks_present(self):
        t = self.cfg.training
        self.assertIsInstance(t.scaling_baseline, ScalingBaseline)
        self.assertIsInstance(t.convergence, Convergence)
        self.assertIsInstance(t.loss_curve, LossCurve)

    def test_expected_cells_are_declared_sweep_names(self):
        # expected_cells() returns the declared sweep names verbatim -- the same
        # keys used in the threshold file and looked up at runtime by metric().
        expected = self.cfg.expected_cells()
        declared = [s.name for s in self.cfg.training.sweeps]
        self.assertEqual(expected, declared)
        # And every expected cell has a matching threshold entry (coverage).
        for cell in expected:
            self.assertIn(cell, self.cfg.thresholds)

    def test_eval_not_overridden(self):
        # After aligning with the MAD scripts, the config no longer sets
        # eval_interval / eval_steps explicitly -- eval is left to MaxText's
        # default (disabled). Assert the keys are absent rather than pinned.
        mc = self.cfg.training.maxtext_config
        self.assertNotIn("eval_interval", mc)
        self.assertNotIn("eval_steps", mc)


if __name__ == "__main__":
    unittest.main()
