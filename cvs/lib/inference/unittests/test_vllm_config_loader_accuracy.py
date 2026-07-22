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
        vc = VariantConfig(
            **_base_kwargs(accuracy={"tasks": [{"id": "mmlu", "task": "mmlu", "num_fewshot": 5}]})
        )
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


if __name__ == "__main__":
    unittest.main()
