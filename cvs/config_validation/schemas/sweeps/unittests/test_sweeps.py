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
from cvs.config_validation.schemas.sweeps import Axis, Cells, SweepsAndRuns
from cvs.lib.conf_lib.schemas.vllm import RunCell

VLLM_CONFIG_DIR = Path(__file__).resolve().parents[4] / "input" / "config_file" / "inference" / "vllm"
VLLM_AXES = (Axis("ISL", "isl"), Axis("OSL", "osl"), Axis("TP", "tp"), Axis("PP", "pp"), Axis("CONC", "concurrency"))
KEY = "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16"
OTHER_KEY = "ISL=8192,OSL=1024,TP=8,PP=1,CONC=64"
EXPECTED = "expected ISL=<n>,OSL=<n>,TP=<n>,PP=<n>,CONC=<n>"

# One bad key for each parse problem, with the problem its error must name.
BAD_KEYS = (
    ("ISL=1024,OSL=1024,TP=8,PP=1,CONCURRENCY=16", "has unknown axis 'CONCURRENCY'"),
    ("ISL=1024,OSL=1024,TP=8,CONC=16", "is missing axis PP"),
    ("ISL=1024,ISL=1024,OSL=1024,TP=8,PP=1,CONC=16", "repeats axis ISL"),
    ("ISL=01024,OSL=1024,TP=8,PP=1,CONC=16", "needs a positive integer for ISL"),
    ("OSL=1024,ISL=1024,TP=8,PP=1,CONC=16", "has its axes out of order"),
)


class _Serving(SweepsAndRuns):
    axes = VLLM_AXES


class _Training(SweepsAndRuns):
    axes = (Axis("NODES", "nodes"), Axis("GBS", "global_batch_size"))


class _Cells(Cells):
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

    def test_comment_keys_are_dropped_from_both_levels_of_sweeps(self):
        # A comment beside the cells would otherwise fail as an unknown axis, or reach an engine as an override.
        config = _Serving.model_validate(
            {"_comment_sweeps": "x", "sweeps": {"_comment": "y", KEY: {"_note": "z", "temperature": 0}}, "runs": [KEY]}
        )
        self.assertEqual(config.sweeps, {KEY: {"temperature": 0}})
        self.assertEqual(list(config.cells()), [KEY])

    def test_each_bad_key_is_refused_at_load_naming_the_key(self):
        for key, problem in BAD_KEYS:
            with self.subTest(key=key):
                self.assertEqual(
                    _problems(_Serving, {"sweeps": {KEY: {}, key: {}}, "runs": [KEY]}),
                    [f"sweeps: cell key {key!r} {problem}; {EXPECTED}"],
                )

    def test_shipped_vllm_keys_and_bad_keys_parse_as_the_vllm_schema_parses_them(self):
        shipped = _shipped_vllm_cell_keys()
        self.assertTrue(shipped, f"no vLLM configs under {VLLM_CONFIG_DIR}")
        config = _Serving.model_validate({"sweeps": {key: {} for key in shipped}, "runs": shipped})
        for key, cell in config.cells().items():
            with self.subTest(key=key):
                self.assertEqual(_axis_values(cell), _axis_values(RunCell.parse(key)))

    def test_keys_parse_by_the_axes_the_config_class_declares(self):
        config = _Training.model_validate({"sweeps": {"NODES=2,GBS=256": {}}, "runs": ["NODES=2,GBS=256"]})
        cell = config.cells()["NODES=2,GBS=256"]
        self.assertEqual((cell.nodes, cell.global_batch_size), (2, 256))
        self.assertFalse(hasattr(cell, "isl"))
        self.assertEqual(
            _problems(_Training, {"sweeps": {KEY: {}}, "runs": [KEY]}),
            [f"sweeps: cell key {KEY!r} has unknown axis 'ISL'; expected NODES=<n>,GBS=<n>"],
        )

    def test_runs_must_name_each_cell_to_run_once_from_sweeps(self):
        sweeps = {KEY: {}, OTHER_KEY: {}}
        self.assertEqual(_problems(_Serving, {"sweeps": sweeps}), ["runs: Field required"])
        self.assertEqual(
            _problems(_Serving, {"sweeps": sweeps, "runs": []}), ["runs: needs at least one sweeps key to run"]
        )
        self.assertEqual(
            _problems(_Serving, {"sweeps": sweeps, "runs": [KEY, OTHER_KEY, KEY]}),
            [f"runs: cell keys listed more than once: [{KEY!r}]"],
        )
        missing = "ISL=2048,OSL=1024,TP=8,PP=1,CONC=16"
        self.assertEqual(
            _problems(_Serving, {"sweeps": sweeps, "runs": [KEY, missing]}),
            [f"(config): runs names cell keys that sweeps does not define: [{missing!r}]"],
        )
        self.assertEqual(_Serving.model_validate({"sweeps": sweeps, "runs": [OTHER_KEY]}).runs, [OTHER_KEY])


class TestCells(unittest.TestCase):
    def test_valid_input_parses_each_key_into_its_axis_fields_in_list_order(self):
        config = _Cells.model_validate({"_comment_cells": "x", "cells": [OTHER_KEY, KEY]})
        cells = config.parsed_cells()
        self.assertEqual((config.cells, list(cells)), ([OTHER_KEY, KEY], [OTHER_KEY, KEY]))
        self.assertEqual((cells[KEY].key, *_axis_values(cells[KEY])), (KEY, 1024, 1024, 8, 1, 16))
        self.assertEqual(_axis_values(cells[OTHER_KEY]), (8192, 1024, 8, 1, 64))

    def test_missing_field_wrong_types_and_extra_key_are_named(self):
        for data, expected in (
            ({}, [(("cells",), "missing")]),
            ({"cells": KEY}, [(("cells",), "list_type")]),
            ({"cells": [KEY, 16]}, [(("cells", 1), "string_type")]),
            ({"cells": [KEY], "runs": [KEY]}, [(("runs",), "extra_forbidden")]),
        ):
            with self.subTest(data=data):
                self.assertEqual(_errors(_Cells, data), expected)

    def test_an_empty_list_and_a_repeated_key_are_refused(self):
        self.assertEqual(_problems(_Cells, {"cells": []}), ["cells: needs at least one cell key"])
        self.assertEqual(
            _problems(_Cells, {"cells": [KEY, OTHER_KEY, KEY]}), [f"cells: cell key {KEY!r} is listed more than once"]
        )

    def test_each_bad_key_is_refused_with_the_same_problem_as_in_sweeps(self):
        for key, problem in BAD_KEYS:
            with self.subTest(key=key):
                self.assertEqual(
                    _problems(_Cells, {"cells": [KEY, key]}), [f"cells: cell key {key!r} {problem}; {EXPECTED}"]
                )

    def test_a_class_without_axes_refuses_every_key(self):
        self.assertEqual(_errors(Cells, {"cells": [KEY]}), [(("cells",), "value_error")])


if __name__ == "__main__":
    unittest.main()
