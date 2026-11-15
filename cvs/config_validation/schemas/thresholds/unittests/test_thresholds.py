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
from cvs.config_validation.schemas.thresholds import Threshold, ThresholdFile, Thresholds
from cvs.unittests.support import FAKE_CRED

VLLM_CONFIG_DIR = Path(__file__).resolve().parents[4] / "input" / "config_file" / "inference" / "vllm"
CELL = "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16"
KIND_MESSAGE = "Input should be 'min', 'max' or 'info'"

# The kinds cvs.lib.conf_lib.config_blocks accepts beyond min, max and info, each in the shape it takes there.
OLD_SPECS = {
    "max_ms": {"kind": "max_ms", "value": 250},
    "min_tok_s": {"kind": "min_tok_s", "value": 1200},
    "within": {"kind": "within", "value": 0.8, "tolerance_pct": 5},
    "min_ratio": {"kind": "min_ratio", "value": 0.9, "reference": FAKE_CRED},
}


def _errors(model, data):
    try:
        model.model_validate(data)
    except ValidationError as exc:
        return sorted((error["loc"], error["type"]) for error in exc.errors())
    raise AssertionError(f"{model.__name__} accepted {data!r}")


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
    def test_each_kind_loads_and_optional_defaults_to_false_and_unset(self):
        for kind in ("min", "max", "info"):
            with self.subTest(kind=kind):
                spec = Threshold.model_validate({"kind": kind, "value": 5})
                self.assertEqual((spec.kind, spec.value, spec.optional), (kind, 5.0, False))
                # An engine that refuses optional tells a written false from the default by this set.
                self.assertNotIn("optional", spec.model_fields_set)
        self.assertTrue(Threshold.model_validate({"kind": "min", "value": 5, "optional": True}).optional)

    def test_old_kinds_are_refused_naming_the_field_path_and_never_the_kind(self):
        for kind, spec in OLD_SPECS.items():
            with self.subTest(kind=kind):
                problems = _problems(ThresholdFile, {CELL: {"output_throughput": spec}})
                self.assertIn(f"{CELL}.output_throughput.kind: {KIND_MESSAGE}", problems)
                self.assertNotIn(kind, "\n".join(problems))
                self.assertNotIn(FAKE_CRED, "\n".join(problems))

    def test_old_kinds_are_refused_in_accuracy_sections_too(self):
        for kind, spec in OLD_SPECS.items():
            with self.subTest(kind=kind):
                problems = _problems(ThresholdFile, {"accuracy": {"gsm8k_flex": {"exact_match": spec}}})
                self.assertIn(f"accuracy.gsm8k_flex.exact_match.kind: {KIND_MESSAGE}", problems)
                self.assertNotIn(kind, "\n".join(problems))

    def test_value_takes_json_ints_and_finite_floats(self):
        for value in (0, -3, 2.5, 10**12):
            with self.subTest(value=value):
                loaded = Threshold.model_validate({"kind": "max", "value": value}).value
                self.assertEqual((loaded, type(loaded)), (value, float))

    def test_value_refuses_quoted_numbers_booleans_and_non_finite_floats(self):
        for value, error in (
            ("100", "float_type"),
            (True, "float_type"),
            (None, "float_type"),
            ([1], "float_type"),
            (float("inf"), "finite_number"),
            (float("-inf"), "finite_number"),
            (float("nan"), "finite_number"),
        ):
            with self.subTest(value=value):
                self.assertEqual(_errors(Threshold, {"kind": "min", "value": value}), [(("value",), error)])

    def test_missing_fields_are_named(self):
        self.assertEqual(_errors(Threshold, {}), [(("kind",), "missing"), (("value",), "missing")])
        self.assertEqual(_errors(Threshold, {"kind": "min"}), [(("value",), "missing")])

    def test_wrong_types_are_rejected(self):
        self.assertEqual(_errors(Threshold, {"kind": 5, "value": 1}), [(("kind",), "literal_error")])
        self.assertEqual(_errors(Threshold, {"kind": "MIN", "value": 1}), [(("kind",), "literal_error")])
        self.assertEqual(
            _errors(Threshold, {"kind": "min", "value": 1, "optional": "sometimes"}), [(("optional",), "bool_parsing")]
        )

    def test_extra_key_is_rejected_and_comment_keys_dropped(self):
        self.assertEqual(
            _errors(Threshold, {"kind": "min", "value": 1, "unit": "ms"}), [(("unit",), "extra_forbidden")]
        )
        spec = Threshold.model_validate({"kind": "min", "value": 1, "_comment": "x"})
        self.assertEqual(spec.model_dump(), {"kind": "min", "value": 1.0, "optional": False})


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

    def test_an_empty_file_loads_with_no_cells_and_no_accuracy(self):
        loaded = ThresholdFile.model_validate({})
        self.assertEqual((loaded.model_extra, loaded.accuracy), ({}, {}))

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

    def test_wrong_shapes_are_named_by_path(self):
        for data, expected in (
            ({CELL: [1]}, [((CELL,), "dict_type")]),
            ({CELL: {"output_throughput": 900}}, [((CELL, "output_throughput"), "model_type")]),
            ({CELL: {"output_throughput": {"kind": "min"}}}, [((CELL, "output_throughput", "value"), "missing")]),
            ({"accuracy": [1]}, [(("accuracy",), "dict_type")]),
            ({"accuracy": {"gsm8k_flex": 0.5}}, [(("accuracy", "gsm8k_flex"), "dict_type")]),
        ):
            with self.subTest(data=data):
                self.assertEqual(_errors(ThresholdFile, data), expected)

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


class TestThresholds(unittest.TestCase):
    def test_defaults_record_nothing_named_and_enforce(self):
        loaded = Thresholds.model_validate({})
        self.assertEqual((loaded.threshold_json, loaded.enforce_thresholds), ("", True))

    def test_valid_input_loads(self):
        loaded = Thresholds.model_validate({"threshold_json": "mi325x_threshold.json", "enforce_thresholds": False})
        self.assertEqual((loaded.threshold_json, loaded.enforce_thresholds), ("mi325x_threshold.json", False))

    def test_wrong_types_are_rejected(self):
        self.assertEqual(_errors(Thresholds, {"threshold_json": 5}), [(("threshold_json",), "string_type")])
        self.assertEqual(
            _errors(Thresholds, {"enforce_thresholds": "sometimes"}), [(("enforce_thresholds",), "bool_parsing")]
        )

    def test_extra_key_is_rejected_and_comment_keys_dropped(self):
        self.assertEqual(_errors(Thresholds, {"threshold_file": "t.json"}), [(("threshold_file",), "extra_forbidden")])
        self.assertEqual(
            Thresholds.model_validate({"_comment_validation": "x"}).model_dump(), Thresholds().model_dump()
        )


if __name__ == "__main__":
    unittest.main()
