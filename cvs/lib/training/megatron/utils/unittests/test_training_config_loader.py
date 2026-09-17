'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for train_params MBS/GBS/precision defaults and combo overlays in
cvs/lib/training/megatron/utils/training_config_loader.py.
'''

import unittest

from pydantic import ValidationError

from cvs.lib.training.megatron.utils.training_config_loader import (
    DEFAULT_SWEEP_NAME,
    MegatronSweep,
    _implicit_default_sweep,
)


_TP = {
    "micro_batch_size": "4",
    "global_batch_size": "128",
    "precision": "BF16",
}


class TestTrainParamsBatchPrecision(unittest.TestCase):
    def test_implicit_default_uses_train_params(self):
        data = _implicit_default_sweep({"train_params": _TP})
        combo = data["sweep"]["combinations"][DEFAULT_SWEEP_NAME]
        self.assertEqual(combo["micro_batch_size"], "4")
        self.assertEqual(combo["global_batch_size"], "128")
        self.assertEqual(combo["precision"], "BF16")
        self.assertEqual(data["sweep"]["runs"], [DEFAULT_SWEEP_NAME])

    def test_missing_train_params_fields_raise(self):
        with self.assertRaises(ValueError) as ctx:
            _implicit_default_sweep({"train_params": {"model_name": "llama3.1_8B"}})
        msg = str(ctx.exception)
        self.assertIn("train_params.micro_batch_size", msg)
        self.assertIn("train_params.global_batch_size", msg)
        self.assertIn("train_params.precision", msg)

    def test_declared_sweep_inherits_train_params(self):
        data = _implicit_default_sweep(
            {
                "train_params": _TP,
                "sweep": {
                    "combinations": {
                        "MBS=4,GBS=128,PRECISION=BF16": {"training_iterations": "20"}
                    },
                    "runs": ["MBS=4,GBS=128,PRECISION=BF16"],
                },
            }
        )
        combo = data["sweep"]["combinations"]["MBS=4,GBS=128,PRECISION=BF16"]
        self.assertEqual(combo["micro_batch_size"], "4")
        self.assertEqual(combo["global_batch_size"], "128")
        self.assertEqual(combo["precision"], "BF16")
        self.assertEqual(combo["training_iterations"], "20")

    def test_combo_body_overrides_precision(self):
        data = _implicit_default_sweep(
            {
                "train_params": _TP,
                "sweep": {
                    "combinations": {
                        "MBS=4,GBS=128,PRECISION=FP8": {
                            "training_iterations": "20",
                            "precision": "FP8",
                        }
                    },
                    "runs": ["MBS=4,GBS=128,PRECISION=FP8"],
                },
            }
        )
        combo = data["sweep"]["combinations"]["MBS=4,GBS=128,PRECISION=FP8"]
        self.assertEqual(combo["micro_batch_size"], "4")
        self.assertEqual(combo["global_batch_size"], "128")
        self.assertEqual(combo["precision"], "FP8")

    def test_combo_body_overrides_mbs_and_gbs(self):
        data = _implicit_default_sweep(
            {
                "train_params": _TP,
                "sweep": {
                    "combinations": {
                        "MBS=2,GBS=32,PRECISION=BF16": {
                            "micro_batch_size": "2",
                            "global_batch_size": "32",
                        }
                    },
                    "runs": ["MBS=2,GBS=32,PRECISION=BF16"],
                },
            }
        )
        combo = data["sweep"]["combinations"]["MBS=2,GBS=32,PRECISION=BF16"]
        self.assertEqual(combo["micro_batch_size"], "2")
        self.assertEqual(combo["global_batch_size"], "32")
        self.assertEqual(combo["precision"], "BF16")

    def test_combo_key_supplies_mbs_gbs_precision_when_body_omits_them(self):
        data = _implicit_default_sweep(
            {
                "train_params": _TP,
                "sweep": {
                    "combinations": {
                        "MBS=2,GBS=16,PRECISION=FP8": {
                            "training_iterations": "20",
                        },
                        "MBS=4,GBS=32,PRECISION=BF16": {
                            "training_iterations": "15",
                        },
                    },
                    "runs": ["MBS=4,GBS=32,PRECISION=BF16"],
                },
            }
        )
        fp8 = data["sweep"]["combinations"]["MBS=2,GBS=16,PRECISION=FP8"]
        self.assertEqual(fp8["micro_batch_size"], "2")
        self.assertEqual(fp8["global_batch_size"], "16")
        self.assertEqual(fp8["precision"], "FP8")
        bf16 = data["sweep"]["combinations"]["MBS=4,GBS=32,PRECISION=BF16"]
        self.assertEqual(bf16["micro_batch_size"], "4")
        self.assertEqual(bf16["global_batch_size"], "32")
        self.assertEqual(bf16["precision"], "BF16")
        self.assertEqual(bf16["training_iterations"], "15")


class TestMegatronSweepDefaultCombo(unittest.TestCase):
    def test_default_combo_does_not_invent_mbs_gbs_precision(self):
        with self.assertRaises(ValidationError):
            MegatronSweep.model_validate(
                {
                    "combinations": {DEFAULT_SWEEP_NAME: {}},
                    "runs": [DEFAULT_SWEEP_NAME],
                }
            )

    def test_key_must_match_effective_combo(self):
        with self.assertRaises(ValidationError):
            MegatronSweep.model_validate(
                {
                    "combinations": {
                        "MBS=4,GBS=128,PRECISION=FP8": {
                            "micro_batch_size": "4",
                            "global_batch_size": "128",
                            "precision": "BF16",
                        }
                    },
                    "runs": ["MBS=4,GBS=128,PRECISION=FP8"],
                }
            )

    def test_empty_runs_with_declared_combinations_raises(self):
        with self.assertRaises(ValidationError) as ctx:
            MegatronSweep.model_validate(
                {
                    "combinations": {
                        "MBS=4,GBS=32,PRECISION=BF16": {"training_iterations": "15"}
                    },
                    "runs": [],
                }
            )
        self.assertIn("sweep.runs is empty", str(ctx.exception))

    def test_key_fills_combo_when_body_omits_mbs_gbs_precision(self):
        sweep = MegatronSweep.model_validate(
            {
                "combinations": {
                    "MBS=4,GBS=32,PRECISION=BF16": {"training_iterations": "15"}
                },
                "runs": ["MBS=4,GBS=32,PRECISION=BF16"],
            }
        )
        combo = sweep.combinations["MBS=4,GBS=32,PRECISION=BF16"]
        self.assertEqual(combo.micro_batch_size, "4")
        self.assertEqual(combo.global_batch_size, "32")
        self.assertEqual(combo.precision, "BF16")


if __name__ == "__main__":
    unittest.main()
