'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for the `accuracy` field wired onto vllm's VariantConfig (step 5 of
the accuracy-harness plan). AccuracyConfig itself is fully covered by
test_accuracy_config.py; these tests only pin the wiring: default/opt-in
behavior and pass-through of an explicit accuracy block.
'''

import unittest

from cvs.lib.inference.utils.accuracy_config import AccuracyConfig
from cvs.lib.inference.utils.vllm_config_loader import VariantConfig


def _base_kwargs(**overrides):
    kwargs = dict(
        schema_version=1,
        framework="vllm",
        gpu_arch="mi300x",
        enforce_thresholds=False,
        paths={
            "shared_fs": "/home/x",
            "models_dir": "/home/x/models",
            "log_dir": "/home/x/LOGS",
            "hf_token_file": "/home/x/.hf",
        },
        model={"id": "/models/test-model", "remote": 0},
        sweep={
            "sequence_combinations": [{"name": "a", "isl": "1024", "osl": "1024"}],
            "runs": [{"combo": "a", "concurrency": 16}],
        },
        thresholds={},
    )
    kwargs.update(overrides)
    return kwargs


class TestVariantConfigAccuracyField(unittest.TestCase):
    def test_defaults_to_empty_accuracy_config_when_omitted(self):
        vc = VariantConfig(**_base_kwargs())
        self.assertIsInstance(vc.accuracy, AccuracyConfig)
        self.assertEqual(vc.accuracy.tasks, [])

    def test_accepts_explicit_accuracy_tasks(self):
        vc = VariantConfig(**_base_kwargs(accuracy={"tasks": [{"id": "mmlu", "task": "mmlu", "num_fewshot": 5}]}))
        self.assertEqual(len(vc.accuracy.tasks), 1)
        self.assertEqual(vc.accuracy.tasks[0].id, "mmlu")
        self.assertEqual(vc.accuracy.tasks[0].num_fewshot, 5)

    def test_duplicate_task_ids_rejected_through_variant_config(self):
        with self.assertRaises(ValueError):
            VariantConfig(
                **_base_kwargs(
                    accuracy={
                        "tasks": [
                            {"id": "mmlu", "task": "mmlu"},
                            {"id": "mmlu", "task": "mmlu"},
                        ]
                    }
                )
            )


class TestAccuracyThresholdKeyDoesNotTripSweepCoverage(unittest.TestCase):
    """The top-level "accuracy" threshold key must not be flagged as an
    unrecognized sweep-cell key by _check_thresholds_cover_sweep, now that it
    delegates to the shared validate_thresholds_cover_sweep."""

    _CELL = "ISL=1024,OSL=1024,TP=8,CONC=16"

    def _full_gated_specs(self):
        from cvs.lib.inference.utils.vllm_parsing import GATED_METRICS

        out = {}
        for m in GATED_METRICS:
            kind = "max_ms" if m.endswith("_ms") else "max" if m == "failed" else "min"
            out[f"client.{m}"] = {"kind": kind, "value": 0 if kind == "min" else 1e12}
        return out

    def test_accuracy_key_alongside_full_sweep_coverage_constructs(self):
        vc = VariantConfig(
            **_base_kwargs(
                enforce_thresholds=True,
                thresholds={
                    self._CELL: self._full_gated_specs(),
                    "accuracy": {"mmlu": {"mmlu.acc__none": {"kind": "min", "value": 0.5}}},
                },
            )
        )
        self.assertIn("accuracy", vc.thresholds)

    def test_typo_key_alongside_accuracy_still_raises(self):
        with self.assertRaises(ValueError) as ctx:
            VariantConfig(
                **_base_kwargs(
                    enforce_thresholds=True,
                    thresholds={
                        self._CELL: self._full_gated_specs(),
                        "accuracy": {"mmlu": {"mmlu.acc__none": {"kind": "min", "value": 0.5}}},
                        "acuracy": {},
                    },
                )
            )
        self.assertIn("threshold keys matching no sweep cell", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
