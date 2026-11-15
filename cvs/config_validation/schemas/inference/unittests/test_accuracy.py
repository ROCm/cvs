'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.config_validation.schemas.inference.accuracy.
'''

import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from cvs.config_validation.schemas.base import problem_lines
from cvs.config_validation.schemas.inference.accuracy import (
    Accuracy,
    AccuracyConfig,
    AccuracyTask,
    LongContextAccuracy,
    LongContextCell,
)
from cvs.lib.inference.utils import accuracy_config, long_context_accuracy_config

INFERENCE_CONFIG_DIR = Path(__file__).resolve().parents[4] / "input" / "config_file" / "inference"

# Each new block beside the block it copies.
PAIRS = (
    (AccuracyTask, accuracy_config.AccuracyTask),
    (AccuracyConfig, accuracy_config.AccuracyConfig),
    (LongContextCell, long_context_accuracy_config.LongContextAccCell),
    (LongContextAccuracy, long_context_accuracy_config.LongContextAccuracyConfig),
)

# Valid and invalid inputs that reach every field type and validator of each pair.
INPUTS = {
    AccuracyTask: (
        {"id": "a", "task": "gsm8k"},
        {"id": "a", "tasks": "gsm8k, hellaswag"},
        {"id": "a", "tasks": ["gsm8k", "hellaswag"], "task": "gsm8k,hellaswag"},
        {"id": "a"},
        {"id": "a", "tasks": ["gsm8k"], "task": "hellaswag"},
        {"id": "a", "tasks": []},
        {"id": "a", "tasks": " , "},
        {"id": "a", "task": "gsm8k", "seed": 7},
        {"id": "a", "task": "gsm8k", "seed": "1,None,3"},
        {"id": "a", "task": "gsm8k", "seed": [5]},
        {"id": "a", "task": "gsm8k", "seed": [1, 2]},
        {"id": "a", "task": "gsm8k", "apply_chat_template": "TRUE"},
        {"id": "a", "task": "gsm8k", "apply_chat_template": "1"},
        {"id": "a", "task": "gsm8k", "apply_chat_template": "0"},
        {"id": "a", "task": "gsm8k", "apply_chat_template": "chatml"},
        {"id": "a", "task": "gsm8k", "lm_eval_model": "local-chat-completions"},
        {"id": "a", "task": "gsm8k", "lm_eval_model": "openai"},
        {"id": "a", "task": "gsm8k", "cache_requests": "yes"},
        {"id": "a", "task": "gsm8k", "num_concurrent": "4", "limit": "100", "exec_timeout_sec": "7200"},
        {"id": "a", "task": "gsm8k", "num_fewshot": 1.5},
        {"id": "a", "task": "gsm8k", "samples": {"gsm8k": [0, 1]}, "gen_kwargs": {"temperature": 0}},
        {"id": "a", "task": "gsm8k", "metadata": "strict-match"},
        {"id": "a", "task": "gsm8k", "_comment": "x", "metadata": {"_keep": 1}},
        {"id": "a", "task": "gsm8k", "model_args": "x"},
        {"task": "gsm8k"},
        {"id": 5, "task": "gsm8k"},
    ),
    AccuracyConfig: (
        {},
        {"tasks": [{"id": "b", "task": "gsm8k"}, {"id": "a", "task": "gsm8k"}]},
        {"tasks": [{"id": "b", "task": "x"}, {"id": "a", "task": "x"}, {"id": "b", "task": "y"}, {"id": "a"}]},
        {
            "tasks": [
                {"id": "b", "task": "x"},
                {"id": "a", "task": "x"},
                {"id": "b", "task": "y"},
                {"id": "a", "task": "y"},
            ]
        },
        {"tasks": {"id": "a", "task": "gsm8k"}},
        {"tasks": [], "harness": "lm-eval"},
    ),
    LongContextCell: (
        {"id": "niah_8k", "isl": 8192},
        {"id": "niah_8k", "isl": "8192", "osl": 16, "num_prompts": 4, "seed": 7},
        {"id": "niah_8k"},
        {"id": "niah_8k", "isl": 8192.5},
        {"id": "niah_8k", "isl": 8192, "depth": 0.5, "_comment": "x"},
    ),
    LongContextAccuracy: (
        {},
        {"cells": [{"id": "niah_8k", "isl": 8192}, {"id": "niah_32k", "isl": 32768}]},
        {"cells": [{"id": "niah_8k", "isl": 8192}, {"id": "niah_8k", "isl": 32768}]},
        {"cells": "niah_8k"},
    ),
}


def _outcome(model, data):
    try:
        return model.model_validate(data).model_dump()
    except ValidationError as exc:
        return [(error["loc"], error["type"], error["msg"]) for error in exc.errors(include_input=False)]


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


def _fields(model):
    return [
        (name, "required" if field.is_required() else field.get_default(call_default_factory=True))
        for name, field in model.model_fields.items()
    ]


def _shipped_blocks(name):
    """Every ``name`` block in the shipped vLLM and ATOM configs, at any depth, since ATOM nests some in profiles."""
    found = []

    def walk(node, where):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == name:
                    found.append((where, value))
                else:
                    walk(value, where)

    for suite in ("vllm", "atom"):
        for path in sorted((INFERENCE_CONFIG_DIR / suite).glob("*.json")):
            if not path.name.endswith("threshold.json"):
                walk(json.loads(path.read_text()), path.name)
    return found


class TestMatchesInferenceUtils(unittest.TestCase):
    def test_field_names_order_and_defaults_match_the_blocks_they_copy(self):
        for new, old in PAIRS:
            with self.subTest(block=new.__name__):
                self.assertEqual(_fields(new), _fields(old))

    def test_field_types_match_the_blocks_they_copy(self):
        for new, old in (
            (AccuracyTask, accuracy_config.AccuracyTask),
            (LongContextCell, long_context_accuracy_config.LongContextAccCell),
        ):
            with self.subTest(block=new.__name__):
                self.assertEqual(
                    {name: field.annotation for name, field in new.model_fields.items()},
                    {name: field.annotation for name, field in old.model_fields.items()},
                )

    def test_each_input_loads_or_fails_the_same_in_both(self):
        for new, old in PAIRS:
            for data in INPUTS[new]:
                with self.subTest(block=new.__name__, data=data):
                    self.assertEqual(_outcome(new, data), _outcome(old, data))

    def test_task_selection_reads_the_same_in_both(self):
        for data in INPUTS[AccuracyTask]:
            try:
                old = accuracy_config.AccuracyTask.model_validate(data)
            except ValidationError:
                continue
            new = AccuracyTask.model_validate(data)
            with self.subTest(data=data):
                self.assertEqual(
                    (new.task_names(), new.resolved_lm_eval_model), (old.task_names(), old.resolved_lm_eval_model)
                )

    def test_every_shipped_vllm_and_atom_block_validates_the_same_in_both(self):
        for name, new, old in (
            ("accuracy", AccuracyConfig, accuracy_config.AccuracyConfig),
            ("long_context_accuracy", LongContextAccuracy, long_context_accuracy_config.LongContextAccuracyConfig),
        ):
            blocks = _shipped_blocks(name)
            self.assertTrue(blocks, f"no {name} blocks under {INFERENCE_CONFIG_DIR}")
            for where, block in blocks:
                with self.subTest(config=where, block=name):
                    self.assertEqual(new.model_validate(block).model_dump(), old.model_validate(block).model_dump())


class TestAccuracyTask(unittest.TestCase):
    def test_a_minimal_task_takes_the_lm_eval_defaults(self):
        task = AccuracyTask.model_validate({"id": "gsm8k_flex", "task": "gsm8k"})
        self.assertEqual(
            (task.num_concurrent, task.batch_size, task.seed, task.metadata, task.exec_timeout_sec),
            (8, "1", [0, 1234, 1234, 1234], {}, 4 * 60 * 60),
        )
        self.assertEqual(
            (task.num_fewshot, task.apply_chat_template, task.resolved_lm_eval_model),
            (None, False, "local-completions"),
        )

    def test_task_or_tasks_resolves_to_one_task_list(self):
        for data, expected in (
            ({"task": "gsm8k"}, ("gsm8k", "gsm8k", ["gsm8k"])),
            ({"tasks": "gsm8k, hellaswag"}, ("gsm8k, hellaswag", "gsm8k,hellaswag", ["gsm8k", "hellaswag"])),
            ({"tasks": ["gsm8k", "bbh"], "task": "gsm8k, bbh"}, (["gsm8k", "bbh"], "gsm8k,bbh", ["gsm8k", "bbh"])),
        ):
            with self.subTest(data=data):
                task = AccuracyTask.model_validate({"id": "a", **data})
                self.assertEqual((task.tasks, task.task, task.task_names()), expected)

    def test_task_selection_problems_are_named(self):
        for data, expected in (
            ({}, ["(config): one of tasks or task is required"]),
            ({"tasks": ["gsm8k"], "task": "hellaswag"}, ["(config): task and tasks disagree"]),
            ({"tasks": []}, ["tasks: tasks must not be empty"]),
            ({"tasks": " , "}, ["(config): tasks must contain at least one task name"]),
        ):
            with self.subTest(data=data):
                self.assertEqual(_problems(AccuracyTask, {"id": "a", **data}), expected)

    def test_seed_takes_lm_evals_one_three_and_four_value_forms(self):
        for seed, expected in (
            (7, [7] * 4),
            ([5], [5] * 4),
            ("1,None,3", [1, None, 3, 1234]),
            ([1, None, 3, 4], [1, None, 3, 4]),
        ):
            with self.subTest(seed=seed):
                self.assertEqual(AccuracyTask.model_validate({"id": "a", "task": "x", "seed": seed}).seed, expected)
        self.assertEqual(
            _problems(AccuracyTask, {"id": "a", "task": "x", "seed": [1, 2]}),
            ["seed: seed must contain exactly four values: python,numpy,torch,fewshot"],
        )

    def test_chat_template_strings_and_the_lm_eval_model_they_select(self):
        for template, expected in (
            ("TRUE", (True, "local-chat-completions")),
            ("1", (True, "local-chat-completions")),
            ("0", (False, "local-completions")),
            ("chatml", ("chatml", "local-chat-completions")),
        ):
            with self.subTest(template=template):
                task = AccuracyTask.model_validate({"id": "a", "task": "x", "apply_chat_template": template})
                self.assertEqual((task.apply_chat_template, task.resolved_lm_eval_model), expected)
        task = AccuracyTask.model_validate(
            {"id": "a", "task": "x", "apply_chat_template": True, "lm_eval_model": "local-completions"}
        )
        self.assertEqual(task.resolved_lm_eval_model, "local-completions")

    def test_missing_id_wrong_types_and_extra_key_are_named(self):
        for data, expected in (
            ({"task": "gsm8k"}, [(("id",), "missing")]),
            ({"id": "a", "task": "gsm8k", "num_concurrent": "four"}, [(("num_concurrent",), "int_parsing")]),
            ({"id": "a", "task": "gsm8k", "lm_eval_model": "openai"}, [(("lm_eval_model",), "literal_error")]),
            ({"id": "a", "task": "gsm8k", "metadata": "strict-match"}, [(("metadata",), "dict_type")]),
            ({"id": "a", "task": "gsm8k", "model_args": "x"}, [(("model_args",), "extra_forbidden")]),
        ):
            with self.subTest(data=data):
                self.assertEqual(_errors(AccuracyTask, data), expected)

    def test_comment_keys_are_dropped_but_kept_inside_metadata(self):
        task = AccuracyTask.model_validate({"_comment": "x", "id": "a", "task": "gsm8k", "metadata": {"_keep": 1}})
        self.assertEqual(task.metadata, {"_keep": 1})


class TestAccuracyConfig(unittest.TestCase):
    def test_tasks_default_to_empty_and_load_in_order(self):
        self.assertEqual(AccuracyConfig.model_validate({}).tasks, [])
        config = AccuracyConfig.model_validate({"tasks": [{"id": "b", "task": "x"}, {"id": "a", "task": "x"}]})
        self.assertEqual([(type(task), task.id) for task in config.tasks], [(AccuracyTask, "b"), (AccuracyTask, "a")])

    def test_duplicate_ids_are_listed_once_each_in_sorted_order(self):
        tasks = [
            {"id": "b", "task": "x"},
            {"id": "a", "task": "x"},
            {"id": "c", "task": "x"},
            {"id": "b", "task": "y"},
            {"id": "a", "task": "y"},
        ]
        self.assertEqual(_problems(AccuracyConfig, {"tasks": tasks}), ["(config): duplicate task id(s): 'a', 'b'"])

    def test_a_task_problem_is_located_by_its_index(self):
        self.assertEqual(
            _problems(AccuracyConfig, {"tasks": [{"id": "a", "task": "x"}, {"id": "b"}]}),
            ["tasks.1: one of tasks or task is required"],
        )

    def test_wrong_type_and_extra_key_are_rejected(self):
        self.assertEqual(_errors(AccuracyConfig, {"tasks": {"id": "a", "task": "x"}}), [(("tasks",), "list_type")])
        self.assertEqual(_errors(AccuracyConfig, {"tasks": ["gsm8k"]}), [(("tasks", 0), "model_type")])
        self.assertEqual(
            _errors(AccuracyConfig, {"tasks": [], "harness": "lm-eval"}), [(("harness",), "extra_forbidden")]
        )


class TestAccuracy(unittest.TestCase):
    def test_no_block_means_no_tasks(self):
        self.assertEqual(Accuracy.model_validate({}).accuracy.tasks, [])
        self.assertEqual(Accuracy.model_validate({"_comment_accuracy": "x"}).model_dump(), {"accuracy": {"tasks": []}})

    def test_block_loads_as_an_accuracy_config(self):
        loaded = Accuracy.model_validate({"accuracy": {"_comment": "x", "tasks": [{"id": "a", "task": "gsm8k"}]}})
        self.assertIsInstance(loaded.accuracy, AccuracyConfig)
        self.assertEqual(loaded.accuracy.tasks[0].task_names(), ["gsm8k"])

    def test_wrong_type_and_extra_keys_are_named_at_each_level(self):
        self.assertEqual(_errors(Accuracy, {"accuracy": [{"id": "a", "task": "x"}]}), [(("accuracy",), "model_type")])
        self.assertEqual(_errors(Accuracy, {"accuracy": {"task": "x"}}), [(("accuracy", "task"), "extra_forbidden")])
        self.assertEqual(_errors(Accuracy, {"tasks": []}), [(("tasks",), "extra_forbidden")])


class TestLongContextCell(unittest.TestCase):
    def test_valid_cell_takes_the_defaults(self):
        cell = LongContextCell.model_validate({"id": "niah_8k", "isl": 8192})
        self.assertEqual(cell.model_dump(), {"id": "niah_8k", "isl": 8192, "osl": 32, "num_prompts": 8, "seed": 42})

    def test_missing_fields_wrong_type_and_extra_key_are_named(self):
        self.assertEqual(_errors(LongContextCell, {}), [(("id",), "missing"), (("isl",), "missing")])
        self.assertEqual(_errors(LongContextCell, {"id": "a", "isl": 8192.5}), [(("isl",), "int_from_float")])
        self.assertEqual(
            _errors(LongContextCell, {"id": "a", "isl": 8192, "depth": 0.5}), [(("depth",), "extra_forbidden")]
        )


class TestLongContextAccuracy(unittest.TestCase):
    def test_cells_default_to_empty_and_load_in_order(self):
        self.assertEqual(LongContextAccuracy.model_validate({}).cells, [])
        loaded = LongContextAccuracy.model_validate(
            {"cells": [{"id": "niah_32k", "isl": 32768}, {"id": "niah_8k", "isl": 8192}]}
        )
        self.assertEqual([cell.id for cell in loaded.cells], ["niah_32k", "niah_8k"])

    def test_duplicate_cell_ids_are_refused(self):
        cells = [{"id": "niah_8k", "isl": 8192}, {"id": "niah_8k", "isl": 32768}]
        self.assertEqual(
            _problems(LongContextAccuracy, {"cells": cells}), ["(config): duplicate long-context cell id(s): 'niah_8k'"]
        )

    def test_wrong_type_and_extra_key_are_rejected(self):
        self.assertEqual(_errors(LongContextAccuracy, {"cells": "niah_8k"}), [(("cells",), "list_type")])
        self.assertEqual(_errors(LongContextAccuracy, {"cells": [], "seed": 1}), [(("seed",), "extra_forbidden")])


if __name__ == "__main__":
    unittest.main()
