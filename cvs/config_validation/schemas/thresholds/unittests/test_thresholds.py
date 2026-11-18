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
from cvs.config_validation.schemas.thresholds import ACCURACY_SECTIONS, ThresholdFile
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
    def test_cells_load_as_extras_and_accuracy_as_its_four_sections(self):
        loaded = ThresholdFile.model_validate(
            {
                CELL: {
                    "output_throughput": {"kind": "min", "value": 900},
                    "mean_ttft_ms": {"kind": "max", "value": 80},
                },
                "accuracy": {
                    "lm_eval": {"gsm8k_flex": {"exact_match": {"kind": "min", "value": 0.5}}, "gsm8k_strict": {}},
                    "custom_eval": {"niah_128k": {"pass_rate": {"kind": "min", "value": 0.8}}},
                    "quant_parity": {"quant_parity.probe_match": {"kind": "info", "value": 1}},
                },
            }
        )
        self.assertEqual(list(loaded.model_extra), [CELL])
        self.assertEqual(
            _as_loaded(loaded.model_extra[CELL]), {"output_throughput": ("min", 900.0), "mean_ttft_ms": ("max", 80.0)}
        )
        accuracy = loaded.accuracy
        self.assertEqual(list(accuracy.lm_eval), ["gsm8k_flex", "gsm8k_strict"])
        self.assertEqual(_as_loaded(accuracy.lm_eval["gsm8k_flex"]), {"exact_match": ("min", 0.5)})
        self.assertEqual(accuracy.lm_eval["gsm8k_strict"], {})
        self.assertEqual(_as_loaded(accuracy.custom_eval["niah_128k"]), {"pass_rate": ("min", 0.8)})
        self.assertEqual(_as_loaded(accuracy.quant_parity), {"quant_parity.probe_match": ("info", 1.0)})
        self.assertEqual(accuracy.mtp_quality, {})

    def test_an_accuracy_key_outside_the_four_sections_is_refused(self):
        # The old layout keyed accuracy by task id directly; a stray id would otherwise gate nothing.
        problems = _problems(ThresholdFile, {"accuracy": {"gsm8k_flex": {"exact_match": {"kind": "min", "value": 0}}}})
        self.assertEqual(problems, ["accuracy.gsm8k_flex: Extra inputs are not permitted"])

    def test_comment_keys_are_dropped_at_every_level(self):
        # A comment left in would be read as a cell, a metric, a task or a spec field.
        loaded = ThresholdFile.model_validate(
            {
                "_comment": "notes",
                CELL: {"_comment": "x", "output_throughput": {"kind": "min", "value": 1, "_note": "y"}},
                "accuracy": {
                    "_comment_sections": "v",
                    "lm_eval": {
                        "_comment_tasks": "z",
                        "gsm8k_flex": {"_comment": "w", "exact_match": {"kind": "min", "value": 0}},
                    },
                },
            }
        )
        self.assertEqual(
            loaded.model_dump(),
            {
                "accuracy": {
                    "lm_eval": {"gsm8k_flex": {"exact_match": {"kind": "min", "value": 0.0, "optional": False}}},
                    "custom_eval": {},
                    "quant_parity": {},
                    "mtp_quality": {},
                },
                CELL: {"output_throughput": {"kind": "min", "value": 1.0, "optional": False}},
            },
        )

    def test_every_shipped_vllm_threshold_file_validates_as_loaded_from_json(self):
        files = sorted(VLLM_CONFIG_DIR.glob("*_threshold.json"))
        self.assertTrue(files, f"no threshold files under {VLLM_CONFIG_DIR}")
        for path in files:
            with self.subTest(file=path.name):
                raw = json.loads(path.read_text())
                accuracy = {key: value for key, value in raw.get("accuracy", {}).items() if not key.startswith("_")}
                lm_eval = accuracy.get("lm_eval", {}) if set(accuracy) <= set(ACCURACY_SECTIONS) else accuracy
                loaded = ThresholdFile.model_validate({**raw, "accuracy": {"lm_eval": lm_eval}})
                cells = {key: specs for key, specs in raw.items() if key != "accuracy" and not key.startswith("_")}
                self.assertEqual(list(loaded.model_extra), list(cells))
                for key, specs in cells.items():
                    self.assertEqual(_as_loaded(loaded.model_extra[key]), _as_written(specs))
                self.assertEqual(
                    {task: _as_loaded(specs) for task, specs in loaded.accuracy.lm_eval.items()},
                    {task: _as_written(specs) for task, specs in lm_eval.items() if not task.startswith("_")},
                )


if __name__ == "__main__":
    unittest.main()
