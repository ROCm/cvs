'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.config_validation.schemas.thresholds.
'''

import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from cvs.config_validation.schemas.base import problem_lines
from cvs.config_validation.schemas.thresholds import ThresholdFile
from cvs.unittests.support import FAKE_CRED

VLLM_CONFIG_DIR = Path(__file__).resolve().parents[4] / "input" / "config_file" / "inference" / "vllm"
CELL = "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16"
KIND_MESSAGE = "Input should be 'min', 'max' or 'info'"

# An old kind cvs.lib.conf_lib.config_blocks accepts beyond min, max and info, in the shape it takes there.
OLD_SPEC = {"kind": "min_ratio", "value": 0.9, "reference": FAKE_CRED}


def _problems(model, data):
    try:
        model.model_validate(data)
    except ValidationError as exc:
        return problem_lines(exc)
    raise AssertionError(f"{model.__name__} accepted {data!r}")


def _as_written(specs):
    return {metric: (spec["kind"], spec["value"]) for metric, spec in specs.items() if not metric.startswith("_")}


def _as_loaded(specs):
    return {metric: (spec.kind, spec.value) for metric, spec in specs.items()}


class TestThreshold(unittest.TestCase):
    def test_an_old_kind_is_refused_naming_the_field_path_and_never_the_kind(self):
        problems = _problems(ThresholdFile, {CELL: {"output_throughput": OLD_SPEC}})
        self.assertIn(f"{CELL}.output_throughput.kind: {KIND_MESSAGE}", problems)
        self.assertNotIn("min_ratio", "\n".join(problems))
        self.assertNotIn(FAKE_CRED, "\n".join(problems))


class TestThresholdFile(unittest.TestCase):
    def test_cells_load_as_extras_and_accuracy_as_sections_by_task_id(self):
        loaded = ThresholdFile.model_validate(
            {
                CELL: {
                    "output_throughput": {"kind": "min", "value": 900},
                    "mean_ttft_ms": {"kind": "max", "value": 80},
                },
                "accuracy": {"gsm8k_flex": {"exact_match": {"kind": "min", "value": 0.5}}, "gsm8k_strict": {}},
            }
        )
        self.assertEqual(list(loaded.model_extra), [CELL])
        self.assertEqual(
            _as_loaded(loaded.model_extra[CELL]), {"output_throughput": ("min", 900.0), "mean_ttft_ms": ("max", 80.0)}
        )
        self.assertEqual(list(loaded.accuracy), ["gsm8k_flex", "gsm8k_strict"])
        self.assertEqual(_as_loaded(loaded.accuracy["gsm8k_flex"]), {"exact_match": ("min", 0.5)})
        self.assertEqual(loaded.accuracy["gsm8k_strict"], {})

    def test_comment_keys_are_dropped_at_every_level(self):
        # A comment left in would be read as a cell, a metric, a task or a spec field.
        loaded = ThresholdFile.model_validate(
            {
                "_comment": "notes",
                CELL: {"_comment": "x", "output_throughput": {"kind": "min", "value": 1, "_note": "y"}},
                "accuracy": {
                    "_comment_tasks": "z",
                    "gsm8k_flex": {"_comment": "w", "exact_match": {"kind": "min", "value": 0}},
                },
            }
        )
        self.assertEqual(
            loaded.model_dump(),
            {
                "accuracy": {"gsm8k_flex": {"exact_match": {"kind": "min", "value": 0.0, "optional": False}}},
                CELL: {"output_throughput": {"kind": "min", "value": 1.0, "optional": False}},
            },
        )

    def test_every_shipped_vllm_threshold_file_validates_as_loaded_from_json(self):
        files = sorted(VLLM_CONFIG_DIR.glob("*_threshold.json"))
        self.assertTrue(files, f"no threshold files under {VLLM_CONFIG_DIR}")
        for path in files:
            with self.subTest(file=path.name):
                raw = json.loads(path.read_text())
                loaded = ThresholdFile.model_validate(raw)
                cells = {key: specs for key, specs in raw.items() if key != "accuracy" and not key.startswith("_")}
                self.assertEqual(list(loaded.model_extra), list(cells))
                for key, specs in cells.items():
                    self.assertEqual(_as_loaded(loaded.model_extra[key]), _as_written(specs))
                self.assertEqual(
                    {task: _as_loaded(specs) for task, specs in loaded.accuracy.items()},
                    {
                        task: _as_written(specs)
                        for task, specs in raw.get("accuracy", {}).items()
                        if not task.startswith("_")
                    },
                )


if __name__ == "__main__":
    unittest.main()
