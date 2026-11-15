'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.config_validation.schemas.sweeps.
'''

import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from cvs.config_validation.schemas.base import problem_lines
from cvs.config_validation.schemas.sweeps import Axis, SweepsAndRuns
from cvs.config_validation.schemas.thresholds import Thresholds
from cvs.lib.conf_lib.schemas.vllm import RunCell

VLLM_CONFIG_DIR = Path(__file__).resolve().parents[4] / "input" / "config_file" / "inference" / "vllm"
VLLM_AXES = (Axis("ISL", "isl"), Axis("OSL", "osl"), Axis("TP", "tp"), Axis("PP", "pp"), Axis("CONC", "concurrency"))
KEY = "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16"
OTHER_KEY = "ISL=8192,OSL=1024,TP=8,PP=1,CONC=64"
EXPECTED = "expected ISL=<n>,OSL=<n>,TP=<n>,PP=<n>,CONC=<n>"

# Each bad key with the problem its error must name.
BAD_KEYS = (
    ("ISL=1024,OSL=1024,TP=8,PP=1,CONCURRENCY=16", "has unknown axis 'CONCURRENCY'"),
    ("isl=1024,OSL=1024,TP=8,PP=1,CONC=16", "has unknown axis 'isl'"),
    ("ISL=1024,OSL=1024,TP=8,PP=1,CONC=16,EP=8", "has unknown axis 'EP'"),
    ("ISL=1024, OSL=1024,TP=8,PP=1,CONC=16", "has unknown axis ' OSL'"),
    ("ISL=1024,OSL=1024,TP=8,PP=1,CONC=16,", "has unknown axis ''"),
    ("ISL=1024;OSL=1024;TP=8;PP=1;CONC=16", "needs a positive integer for ISL"),
    ("", "has unknown axis ''"),
    ("ISL=1024,OSL=1024,TP=8,CONC=16", "is missing axis PP"),
    ("ISL=1024,OSL=1024", "is missing axis TP, PP, CONC"),
    ("ISL=1024,ISL=1024,OSL=1024,TP=8,PP=1,CONC=16", "repeats axis ISL"),
    ("ISL=1024,OSL=1024,TP=8,PP=1,CONC=16,CONC=32", "repeats axis CONC"),
    ("ISL=1k,OSL=1024,TP=8,PP=1,CONC=16", "needs a positive integer for ISL"),
    ("ISL=1024,OSL=1024,TP=8.0,PP=1,CONC=16", "needs a positive integer for TP"),
    ("ISL=1024,OSL=1024,TP=8,PP=0,CONC=16", "needs a positive integer for PP"),
    ("ISL=1024,OSL=1024,TP=8,PP=-1,CONC=16", "needs a positive integer for PP"),
    ("ISL=01024,OSL=1024,TP=8,PP=1,CONC=16", "needs a positive integer for ISL"),
    ("ISL=1024,OSL=1024,TP=8,PP=1,CONC=", "needs a positive integer for CONC"),
    ("ISL=1024,OSL=1024,TP=8,PP=1,CONC", "needs a positive integer for CONC"),
    ("ISL=1024,OSL=1024,TP=8,PP=1,CONC=16=32", "needs a positive integer for CONC"),
    ("ISL=1024,OSL= 1024,TP=8,PP=1,CONC=16", "needs a positive integer for OSL"),
    ("OSL=1024,ISL=1024,TP=8,PP=1,CONC=16", "has its axes out of order"),
    ("ISL=1024,OSL=1024,TP=8,CONC=16,PP=1", "has its axes out of order"),
)


class _Serving(SweepsAndRuns):
    axes = VLLM_AXES


class _Training(SweepsAndRuns):
    axes = (Axis("NODES", "nodes"), Axis("GBS", "global_batch_size"))


class _GatedServing(Thresholds, SweepsAndRuns):
    axes = VLLM_AXES


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


def _axis_values(cell):
    return (cell.isl, cell.osl, cell.tp, cell.pp, cell.concurrency)


def _shipped_vllm_cell_keys():
    keys = set()
    for path in VLLM_CONFIG_DIR.glob("*.json"):
        if path.name.endswith("_threshold.json"):
            continue
        config = json.loads(path.read_text())
        keys.update(key for key in config["sweeps"] if not key.startswith("_"))
        keys.update(config["runs"])
    return sorted(keys)


class TestSweepsAndRuns(unittest.TestCase):
    def test_valid_input_parses_each_key_into_its_axis_fields_in_sweeps_order(self):
        config = _Serving.model_validate({"sweeps": {OTHER_KEY: {}, KEY: {"temperature": 0}}, "runs": [KEY]})
        cells = config.cells()
        self.assertEqual(list(cells), [OTHER_KEY, KEY])
        self.assertEqual((cells[KEY].key, *_axis_values(cells[KEY])), (KEY, 1024, 1024, 8, 1, 16))
        self.assertEqual(_axis_values(cells[OTHER_KEY]), (8192, 1024, 8, 1, 64))
        self.assertEqual((config.sweeps[KEY], config.runs), ({"temperature": 0}, [KEY]))

    def test_runs_default_to_empty_and_empty_sweeps_have_no_cells(self):
        config = _Serving.model_validate({"sweeps": {}})
        self.assertEqual((config.runs, config.cells()), ([], {}))

    def test_missing_sweeps_is_named(self):
        self.assertEqual(_errors(_Serving, {"runs": [KEY]}), [(("sweeps",), "missing")])

    def test_wrong_types_are_rejected(self):
        for data, expected in (
            ({"sweeps": [KEY]}, [(("sweeps",), "dict_type")]),
            ({"sweeps": {KEY: [1]}}, [(("sweeps", KEY), "dict_type")]),
            ({"sweeps": {KEY: {}}, "runs": KEY}, [(("runs",), "list_type")]),
            ({"sweeps": {KEY: {}}, "runs": [16]}, [(("runs", 0), "string_type")]),
        ):
            with self.subTest(data=data):
                self.assertEqual(_errors(_Serving, data), expected)

    def test_extra_key_is_rejected(self):
        self.assertEqual(_errors(_Serving, {"sweeps": {}, "cells": [KEY]}), [(("cells",), "extra_forbidden")])

    def test_comment_keys_are_dropped_from_both_levels_of_sweeps(self):
        # A comment beside the cells would otherwise fail as an unknown axis, or reach an engine as an override.
        config = _Serving.model_validate(
            {"_comment_sweeps": "x", "sweeps": {"_comment": "y", KEY: {"_note": "z", "temperature": 0}}}
        )
        self.assertEqual(config.sweeps, {KEY: {"temperature": 0}})
        self.assertEqual(list(config.cells()), [KEY])

    def test_each_bad_key_is_refused_at_load_naming_the_key(self):
        for key, problem in BAD_KEYS:
            with self.subTest(key=key):
                self.assertEqual(
                    _problems(_Serving, {"sweeps": {KEY: {}, key: {}}}),
                    [f"sweeps: cell key {key!r} {problem}; {EXPECTED}"],
                )

    def test_cells_refuses_a_bad_key_added_after_load(self):
        config = _Serving.model_validate({"sweeps": {KEY: {}}})
        config.sweeps["ISL=1024"] = {}
        with self.assertRaisesRegex(ValueError, r"^cell key 'ISL=1024' is missing axis OSL, TP, PP, CONC; expected "):
            config.cells()

    def test_shipped_vllm_keys_and_bad_keys_parse_as_the_vllm_schema_parses_them(self):
        shipped = _shipped_vllm_cell_keys()
        self.assertTrue(shipped, f"no vLLM configs under {VLLM_CONFIG_DIR}")
        config = _Serving.model_validate({"sweeps": {key: {} for key in shipped}})
        for key, cell in config.cells().items():
            with self.subTest(key=key):
                self.assertEqual(_axis_values(cell), _axis_values(RunCell.parse(key)))
        for key, _ in BAD_KEYS:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    RunCell.parse(key)

    def test_keys_parse_by_the_axes_the_config_class_declares(self):
        config = _Training.model_validate({"sweeps": {"NODES=2,GBS=256": {}}})
        cell = config.cells()["NODES=2,GBS=256"]
        self.assertEqual((cell.nodes, cell.global_batch_size), (2, 256))
        self.assertFalse(hasattr(cell, "isl"))
        self.assertEqual(
            _problems(_Training, {"sweeps": {KEY: {}}}),
            [f"sweeps: cell key {KEY!r} has unknown axis 'ISL'; expected NODES=<n>,GBS=<n>"],
        )

    def test_a_class_without_axes_refuses_every_key(self):
        self.assertEqual(SweepsAndRuns.model_validate({"sweeps": {}}).cells(), {})
        self.assertEqual(_errors(SweepsAndRuns, {"sweeps": {KEY: {}}}), [(("sweeps",), "value_error")])

    def test_axes_hold_when_combined_with_the_thresholds_mixin(self):
        config = _GatedServing.model_validate({"threshold_json": "t.json", "sweeps": {KEY: {}}, "runs": [KEY]})
        self.assertEqual((config.threshold_json, _axis_values(config.cells()[KEY])), ("t.json", (1024, 1024, 8, 1, 16)))
        self.assertEqual(
            _errors(_GatedServing, {"sweeps": {"NODES=2,GBS=256": {}}, "extra": 1}),
            [(("extra",), "extra_forbidden"), (("sweeps",), "value_error")],
        )

    def test_parsed_cells_are_immutable(self):
        cell = _Serving.model_validate({"sweeps": {KEY: {}}}).cells()[KEY]
        with self.assertRaises(AttributeError):
            cell.tp = 4


if __name__ == "__main__":
    unittest.main()
