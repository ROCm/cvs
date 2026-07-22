'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.inference.utils.accuracy_config (AccuracyTask,
AccuracyConfig). These pin the construction-time validation contract of the two
pydantic models: field defaults/typing/coercion, extra="forbid" (inherited from
_Forbid), and the model_validator(mode="after") that rejects duplicate task ids.
No hardware.

Authored black-box from the behavioral spec; the implementation was not read.
Classification: both models are validation *subsystems* -- the only operation is
construct-time schema validation (taxonomy #4 Schema Boundary Strictness / #5
Cross-Field Relational Invariants). They expose no state-transition methods, so
no *Lifecycle transition table applies (see StructuredOutput justification).
'''

import unittest

from pydantic import ValidationError

from cvs.lib.inference.utils.accuracy_config import AccuracyConfig, AccuracyTask


def _task(**over):
    """A valid AccuracyTask with fields overridable per case."""
    base = {"id": "a", "task": "mmlu"}
    base.update(over)
    return AccuracyTask(**base)


def _dupes_message(exc):
    """Isolate the duplicate-id validator's own message from pydantic's wrapper.

    str(ValidationError) appends '[type=..., input_value=..., input_type=...]',
    and input_value repeats each task (and thus each id). To assert on the
    validator's rendered sorted-dupes list (count/order/quoting) without the
    wrapper's echoes, slice from the documented prefix up to pydantic's
    '[type=' metadata marker.
    """
    msg = str(exc)
    prefix = "duplicate task id(s):"
    i = msg.find(prefix)
    if i == -1:
        return None
    tail = msg[i:]
    j = tail.find("[type=")
    if j != -1:
        tail = tail[:j]
    return tail


class TestAccuracyTaskDefaults(unittest.TestCase):
    """AC12, AC23, AC24: defaults, empty-string id, include_path (no I/O)."""

    def test_defaults_on_minimal_task(self):
        t = AccuracyTask(id="a", task="mmlu")
        self.assertEqual(t.id, "a")
        self.assertEqual(t.task, "mmlu")
        self.assertEqual(t.num_fewshot, 0)
        self.assertEqual(t.metadata, {})
        self.assertEqual(t.include_path, "")
        self.assertEqual(t.num_concurrent, 8)
        self.assertIs(t.apply_chat_template, False)
        self.assertEqual(t.gen_kwargs, {})

    def test_empty_string_id_is_valid(self):
        # AC23: "" is a valid id, not treated as missing.
        t = AccuracyTask(id="", task="mmlu")
        self.assertEqual(t.id, "")

    def test_include_path_no_filesystem_check(self):
        # AC24: nonexistent path accepted as plain string, no I/O.
        t = AccuracyTask(id="a", task="mmlu", include_path="/some/nonexistent/path")
        self.assertEqual(t.include_path, "/some/nonexistent/path")

    def test_default_dicts_are_isolated_per_instance(self):
        # Mutable-default isolation for metadata/gen_kwargs on AccuracyTask.
        t1 = AccuracyTask(id="a", task="mmlu")
        t2 = AccuracyTask(id="b", task="mmlu")
        self.assertIsNot(t1.metadata, t2.metadata)
        self.assertIsNot(t1.gen_kwargs, t2.gen_kwargs)


class TestAccuracyTaskRequiredFields(unittest.TestCase):
    """AC13: id and task are required."""

    def test_missing_required_field_raises(self):
        for missing in ("id", "task"):
            with self.subTest(missing=missing):
                kwargs = {"id": "a", "task": "mmlu"}
                del kwargs[missing]
                with self.assertRaises(ValidationError):
                    AccuracyTask(**kwargs)


class TestAccuracyTaskExplicitNone(unittest.TestCase):
    """Explicit None is a distinct equivalence class from omission: no field is
    Optional, so None is rejected for every field (required str fields AND the
    non-None-defaulted int/dict/bool/str fields). Mirror of AC28 for the config's
    tasks field. Guards against a mutated schema (e.g. id: Optional[str], or
    metadata: Optional[Dict] = {}) silently accepting None while still passing
    the omission-only required-field test."""

    def test_explicit_none_per_field_raises(self):
        # (field, None value passed via a valid base task)
        for field in (
            "id",
            "task",
            "num_fewshot",
            "metadata",
            "include_path",
            "num_concurrent",
            "apply_chat_template",
            "gen_kwargs",
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValidationError):
                    _task(**{field: None})


class TestAccuracyTaskIntCoercion(unittest.TestCase):
    """AC14, AC15, AC16: int fields coerce numeric strings; no range constraint."""

    def test_int_coercion_success(self):
        # (field, input, expected int)
        cases = [
            ("num_fewshot", "5", 5),
            ("num_fewshot", 5, 5),
            ("num_fewshot", 5.0, 5),      # whole-number float coerces (vs 1.9 which rejects)
            ("num_fewshot", -1, -1),      # AC16: negative allowed, no ge
            ("num_fewshot", 0, 0),
            ("num_fewshot", True, 1),     # pydantic lax int: bool coerces to 1/0
            ("num_concurrent", "3", 3),
            ("num_concurrent", 3, 3),
            ("num_concurrent", 4.0, 4),   # whole-number float coerces (vs 2.5 which rejects)
            ("num_concurrent", 0, 0),     # AC16: zero allowed, no gt
            ("num_concurrent", -1, -1),
            ("num_concurrent", False, 0), # pydantic lax int: bool coerces to 1/0
        ]
        for field, value, expected in cases:
            with self.subTest(field=field, value=value):
                t = _task(**{field: value})
                got = getattr(t, field)
                self.assertEqual(got, expected)
                self.assertIsInstance(got, int)

    def test_int_coercion_failure_raises(self):
        cases = [
            ("num_fewshot", "not-an-int"),
            ("num_fewshot", 1.9),          # float with fractional part
            ("num_concurrent", "bad"),
            ("num_concurrent", 2.5),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValidationError):
                    _task(**{field: value})


class TestAccuracyTaskDictCoercion(unittest.TestCase):
    """AC17, AC18, AC19: dict fields accept mappings only."""

    def test_dict_success(self):
        cases = [
            ("metadata", {"k": "v"}),
            ("metadata", {}),
            ("gen_kwargs", {"a": 1}),
            ("gen_kwargs", {}),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                t = _task(**{field: value})
                self.assertEqual(getattr(t, field), value)

    def test_non_mapping_raises(self):
        cases = [
            ("metadata", "not-a-dict"),
            ("metadata", [1, 2]),
            ("metadata", 123),
            ("gen_kwargs", 123),
            ("gen_kwargs", "not-a-dict"),
            ("gen_kwargs", [1, 2]),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValidationError):
                    _task(**{field: value})


class TestAccuracyTaskBoolCoercion(unittest.TestCase):
    """AC20, AC21: bool field; 'maybe' is the guaranteed-fail non-bool word."""

    def test_bool_true_accepted(self):
        t = _task(apply_chat_template=True)
        self.assertIs(t.apply_chat_template, True)

    def test_bool_false_accepted(self):
        t = _task(apply_chat_template=False)
        self.assertIs(t.apply_chat_template, False)

    def test_non_bool_word_raises(self):
        with self.assertRaises(ValidationError):
            _task(apply_chat_template="maybe")


class TestAccuracyTaskStringTyping(unittest.TestCase):
    """AC22: id/task accept only str; non-str scalars are not auto-coerced."""

    def test_non_str_id_or_task_raises(self):
        cases = [
            {"id": 123, "task": "mmlu"},
            {"id": "a", "task": 123},
            {"id": 1.5, "task": "mmlu"},
            {"id": True, "task": "mmlu"},   # bool is a non-str scalar; not coerced to str
            {"id": "a", "task": False},     # bool is a non-str scalar; not coerced to str
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValidationError):
                    AccuracyTask(**kwargs)


class TestAccuracyTaskExtraForbid(unittest.TestCase):
    """AC10: unknown fields rejected (extra='forbid' from _Forbid)."""

    def test_unknown_field_raises(self):
        with self.assertRaises(ValidationError):
            AccuracyTask(id="a", task="mmlu", extra_field=1)


class TestAccuracyConfigConstruction(unittest.TestCase):
    """AC1, AC2, AC3, AC25, AC26, AC29: happy-path construction + element typing."""

    def test_empty_config_has_empty_tasks(self):
        # AC1 + edge case: zero tasks constructs, tasks == [].
        cfg = AccuracyConfig()
        self.assertEqual(cfg.tasks, [])

    def test_single_task_constructs(self):
        # AC2 + edge case: exactly one task is trivially unique.
        cfg = AccuracyConfig(tasks=[AccuracyTask(id="a", task="mmlu")])
        self.assertEqual(len(cfg.tasks), 1)
        self.assertEqual(cfg.tasks[0].id, "a")

    def test_three_distinct_ids_construct(self):
        # AC3.
        cfg = AccuracyConfig(
            tasks=[
                AccuracyTask(id="a", task="mmlu"),
                AccuracyTask(id="b", task="mmlu"),
                AccuracyTask(id="c", task="mmlu"),
            ]
        )
        self.assertEqual([t.id for t in cfg.tasks], ["a", "b", "c"])

    def test_list_of_dicts_becomes_tasks(self):
        # AC25.
        cfg = AccuracyConfig(tasks=[{"id": "a", "task": "mmlu"}])
        self.assertIsInstance(cfg.tasks[0], AccuracyTask)
        self.assertEqual(cfg.tasks[0].id, "a")

    def test_mixed_dicts_and_instances(self):
        # AC26: every element ends up an AccuracyTask.
        cfg = AccuracyConfig(
            tasks=[AccuracyTask(id="a", task="mmlu"), {"id": "b", "task": "gsm8k"}]
        )
        self.assertTrue(all(isinstance(t, AccuracyTask) for t in cfg.tasks))
        self.assertEqual([t.id for t in cfg.tasks], ["a", "b"])

    def test_order_and_length_preserved(self):
        # AC29.
        cfg = AccuracyConfig(
            tasks=[
                AccuracyTask(id="a", task="m"),
                AccuracyTask(id="b", task="m"),
                AccuracyTask(id="c", task="m"),
            ]
        )
        self.assertEqual([t.id for t in cfg.tasks], ["a", "b", "c"])


class TestAccuracyConfigTasksField(unittest.TestCase):
    """AC11, AC27, AC28: extra forbid + tasks element/None handling."""

    def test_unknown_field_raises(self):
        # AC11.
        with self.assertRaises(ValidationError):
            AccuracyConfig(tasks=[], extra_field=1)

    def test_non_dict_non_instance_element_raises(self):
        # AC27.
        with self.assertRaises(ValidationError):
            AccuracyConfig(tasks=["not-a-task"])

    def test_tasks_none_raises(self):
        # AC28: field is not Optional; only omission yields the [] default.
        with self.assertRaises(ValidationError):
            AccuracyConfig(tasks=None)


class TestAccuracyConfigDuplicateIds(unittest.TestCase):
    """AC4-AC9: the model_validator(mode='after') duplicate-id contract."""

    def test_single_duplicate_group(self):
        # AC4: prefix + the offending id present.
        with self.assertRaises(ValidationError) as ctx:
            AccuracyConfig(
                tasks=[
                    AccuracyTask(id="dup-mmlu", task="mmlu"),
                    AccuracyTask(id="dup-mmlu", task="mmlu"),
                ]
            )
        dupes = _dupes_message(ctx.exception)
        self.assertIsNotNone(dupes, "expected 'duplicate task id(s):' prefix")
        self.assertIn("dup-mmlu", dupes)

    def test_two_groups_sorted_ascending(self):
        # AC5: both ids present; 'dup-gsm8k' before 'dup-mmlu' (sorted, not
        # encounter order -- input intentionally lists mmlu first).
        with self.assertRaises(ValidationError) as ctx:
            AccuracyConfig(
                tasks=[
                    AccuracyTask(id="dup-mmlu", task="mmlu"),
                    AccuracyTask(id="dup-gsm8k", task="gsm8k"),
                    AccuracyTask(id="dup-mmlu", task="mmlu"),
                    AccuracyTask(id="dup-gsm8k", task="gsm8k"),
                ]
            )
        dupes = _dupes_message(ctx.exception)
        self.assertIsNotNone(dupes)
        i_gsm = dupes.find("dup-gsm8k")
        i_mmlu = dupes.find("dup-mmlu")
        self.assertNotEqual(i_gsm, -1)
        self.assertNotEqual(i_mmlu, -1)
        self.assertLess(i_gsm, i_mmlu, "ids must be sorted ascending in the message")

    def test_mixed_case_dupes_sorted_case_sensitively(self):
        # AC5 (sort discriminator): the sort must be case-SENSITIVE lexicographic,
        # distinct from AC8's case-sensitive equality. All-lowercase fixtures
        # (dup-gsm8k/dup-mmlu) cannot tell a correct sorted(dupes) from a
        # spec-violating sorted(dupes, key=str.lower). Use ids that differ in
        # leading case: case-sensitive sort orders uppercase before lowercase
        # ('Dup-Zebra' < 'dup-apple'), whereas a case-insensitive key flips them
        # ('dup-apple' < 'Dup-Zebra' since 'a' < 'z').
        with self.assertRaises(ValidationError) as ctx:
            AccuracyConfig(
                tasks=[
                    AccuracyTask(id="dup-apple", task="mmlu"),
                    AccuracyTask(id="Dup-Zebra", task="gsm8k"),
                    AccuracyTask(id="dup-apple", task="mmlu"),
                    AccuracyTask(id="Dup-Zebra", task="gsm8k"),
                ]
            )
        dupes = _dupes_message(ctx.exception)
        self.assertIsNotNone(dupes)
        i_zebra = dupes.find("Dup-Zebra")
        i_apple = dupes.find("dup-apple")
        self.assertNotEqual(i_zebra, -1)
        self.assertNotEqual(i_apple, -1)
        self.assertLess(
            i_zebra,
            i_apple,
            "dupes must be sorted case-sensitively: uppercase-leading 'Dup-Zebra' "
            "precedes 'dup-apple' (a case-insensitive sort key would reverse this)",
        )

    def test_triple_duplicate_id_listed_once(self):
        # AC6: id repeated 3x appears exactly once in the sorted-dupes list.
        with self.assertRaises(ValidationError) as ctx:
            AccuracyConfig(
                tasks=[
                    AccuracyTask(id="dup-mmlu", task="mmlu"),
                    AccuracyTask(id="dup-mmlu", task="mmlu"),
                    AccuracyTask(id="dup-mmlu", task="mmlu"),
                ]
            )
        dupes = _dupes_message(ctx.exception)
        self.assertIsNotNone(dupes)
        self.assertEqual(dupes.count("dup-mmlu"), 1)

    def test_duplicate_by_id_only_ignores_other_fields(self):
        # AC7: same id, different task -> still duplicates. The raise itself is
        # the discriminator: full-object comparison would construct successfully.
        # Use a distinctive id ("dup-x") that cannot be a substring of the fixed
        # "duplicate task id(s):" prefix, so assertIn actually probes the
        # validator's rendered dupes list rather than the constant prefix text.
        with self.assertRaises(ValidationError) as ctx:
            AccuracyConfig(
                tasks=[
                    AccuracyTask(id="dup-x", task="mmlu"),
                    AccuracyTask(id="dup-x", task="gsm8k"),
                ]
            )
        dupes = _dupes_message(ctx.exception)
        self.assertIsNotNone(dupes, "must be the duplicate-id validator, not another error")
        self.assertIn("dup-x", dupes)

    def test_only_repeated_ids_listed_not_unique_ones(self):
        # AC4 (message contents): the dupes list must contain ONLY ids that
        # actually repeat, not every distinct id in the config. Mix a duplicated
        # id with an id that appears exactly once and assert the unique one is
        # absent -- this fails a validator that reports sorted(set(all_ids)).
        with self.assertRaises(ValidationError) as ctx:
            AccuracyConfig(
                tasks=[
                    AccuracyTask(id="dup-repeated", task="mmlu"),
                    AccuracyTask(id="dup-repeated", task="gsm8k"),
                    AccuracyTask(id="only-once", task="mmlu"),
                ]
            )
        dupes = _dupes_message(ctx.exception)
        self.assertIsNotNone(dupes)
        self.assertIn("dup-repeated", dupes)
        self.assertNotIn("only-once", dupes)

    def test_case_sensitive_ids_not_duplicates(self):
        # AC8: 'MMLU' vs 'mmlu' differ only by case -> NOT duplicates.
        cfg = AccuracyConfig(
            tasks=[
                AccuracyTask(id="MMLU", task="mmlu"),
                AccuracyTask(id="mmlu", task="mmlu"),
            ]
        )
        self.assertEqual([t.id for t in cfg.tasks], ["MMLU", "mmlu"])

    def test_empty_string_duplicates_render_as_quotes(self):
        # AC9: two id="" -> duplicate; renders as '' in the sorted list.
        with self.assertRaises(ValidationError) as ctx:
            AccuracyConfig(
                tasks=[
                    AccuracyTask(id="", task="a"),
                    AccuracyTask(id="", task="b"),
                ]
            )
        dupes = _dupes_message(ctx.exception)
        self.assertIsNotNone(dupes)
        self.assertIn("''", dupes)


class TestAccuracyConfigNonMutation(unittest.TestCase):
    """AC30, AC31: input list not mutated; default tasks list not shared."""

    def test_caller_dict_list_not_mutated(self):
        # AC30.
        lst = [{"id": "a", "task": "m"}]
        AccuracyConfig(tasks=lst)
        self.assertEqual(len(lst), 1)
        self.assertIsInstance(lst[0], dict)
        self.assertEqual(lst[0], {"id": "a", "task": "m"})

    def test_default_tasks_not_shared_between_instances(self):
        # AC31.
        a = AccuracyConfig()
        b = AccuracyConfig()
        self.assertIsNot(a.tasks, b.tasks)
        a.tasks.append(AccuracyTask(id="x", task="m"))
        self.assertEqual(b.tasks, [])


class TestValidationPrecedence(unittest.TestCase):
    """AC32: per-element field error surfaces before the duplicate-id validator."""

    def test_field_error_preempts_duplicate_validator(self):
        with self.assertRaises(ValidationError) as ctx:
            AccuracyConfig(tasks=[{"id": "a", "task": "mmlu"}, {"id": "a"}])
        msg = str(ctx.exception)
        # The missing required 'task' field on element index 1 is what surfaces.
        # Assert the fully-qualified error location "tasks.1.task" rather than a
        # bare "task": the parent field name "tasks" means a plain "task"
        # substring would also match "tasks.1.id" (i.e. the *other* field being
        # the one missing), so it cannot tell which required field failed.
        self.assertIn("tasks.1.task", msg)
        self.assertTrue(
            ("Field required" in msg) or ("missing" in msg),
            f"expected a missing-required-field marker, got: {msg}",
        )
        # ...and the duplicate-id validator must NOT have run.
        self.assertNotIn("duplicate task id(s):", msg)


class TestModelFieldMembership(unittest.TestCase):
    """AC33, AC34 + regression constraints: closed-set field membership."""

    def test_accuracy_task_fields_exact(self):
        # AC33.
        self.assertEqual(
            set(AccuracyTask.model_fields),
            {
                "id",
                "task",
                "num_fewshot",
                "metadata",
                "include_path",
                "num_concurrent",
                "apply_chat_template",
                "gen_kwargs",
            },
        )

    def test_accuracy_config_fields_exact(self):
        # AC34.
        self.assertEqual(set(AccuracyConfig.model_fields), {"tasks"})

    def test_no_threshold_or_gate_fields(self):
        # Regression constraint: no gating/threshold wiring exists on these models.
        forbidden = {"threshold", "gate", "min_score", "accuracy_gate", "accuracy"}
        self.assertEqual(set(AccuracyTask.model_fields) & forbidden, set())
        self.assertEqual(set(AccuracyConfig.model_fields) & forbidden, set())


if __name__ == "__main__":
    unittest.main()
