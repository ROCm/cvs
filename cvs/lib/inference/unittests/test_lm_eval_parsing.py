'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.inference.utils.lm_eval_parsing (_is_real_number, project).

Both public units are PURE functions (no state, output depends only on args, no
I/O, no side effects). Per the authoring discipline this means:
  - range/equivalence testing via a (input -> expected) subTest table, not N
    copy-pasted methods;
  - boundary cases as their own equivalence classes;
  - an invariant test where one plausibly exists (return-type invariant for the
    predicate; value-type / comma-free-key / determinism / non-mutation
    invariants for the flattener).
There is no lifecycle class because neither unit carries mutable state.

Tests are authored black-box from the behavioral spec only; the implementation
was not read. Written greenfield (RED before implementation).
'''

import copy
import inspect
import math
import typing
import unittest
from typing import Any, Dict
from unittest.mock import patch

from cvs.lib.inference.utils.lm_eval_parsing import _is_real_number, project


class TestIsRealNumber(unittest.TestCase):
    """Pure predicate: any object -> bool (never raises). Spec AC 1-6, 22."""

    def test_is_real_number_ranges(self):
        # (value, expected) equivalence-class + boundary table. Spec AC 1-5, 22.
        # Sentinel objects for nan/inf are constructed inline so identity is
        # unambiguous.
        cases = [
            # real numbers -> True
            (1, True),                 # AC1 int
            (1.5, True),               # AC1 float
            (0, True),                 # boundary: zero int is a real number
            (0.0, True),               # boundary: zero float is a real number
            (-3, True),                # AC22 negative int
            (-1.25, True),             # AC22 negative float
            (float("inf"), True),      # AC4 +inf is a real number
            (float("-inf"), True),     # AC4 -inf is a real number
            # not real numbers -> False
            (True, False),             # AC2 bool excluded despite subclass of int
            (False, False),            # AC2 bool excluded
            (float("nan"), False),     # AC3 NaN excluded
            ("0.71", False),           # AC5 numeric-looking string excluded
            (None, False),             # AC5 None excluded
            ({}, False),               # AC5 dict excluded
            ([], False),               # AC5 list excluded
            (complex(1, 2), False),    # complex number is numeric but NOT real
            (1 + 2j, False),           # same boundary, literal form
            (complex(3, 0), False),    # zero-imaginary complex is still not real
        ]
        for value, expected in cases:
            with self.subTest(value=repr(value)):
                self.assertEqual(_is_real_number(value), expected)

    def test_is_real_number_always_returns_plain_bool(self):
        # Invariant (AC6): result is a genuine bool, never a truthy/falsy
        # non-bool. `type(...) is bool` is stricter than isinstance and would
        # reject e.g. returning the int 0/1 or the object itself.
        samples = [
            1, 1.5, 0, -3, -1.25, float("inf"), float("-inf"), float("nan"),
            True, False, "0.71", None, {}, [], object(),
        ]
        for value in samples:
            with self.subTest(value=repr(value)):
                result = _is_real_number(value)
                self.assertIs(type(result), bool)

    def test_is_real_number_bool_is_not_a_real_number(self):
        # Regression constraint: bool is a subclass of int but must be excluded.
        # Pinned separately from the table so a bool-passthrough mutant is killed
        # explicitly.
        self.assertIs(_is_real_number(True), False)
        self.assertIs(_is_real_number(False), False)

    def test_is_real_number_nan_excluded_but_inf_included(self):
        # Boundary between the two float special values (AC3 vs AC4).
        self.assertIs(_is_real_number(float("nan")), False)
        self.assertIs(_is_real_number(float("inf")), True)
        self.assertIs(_is_real_number(float("-inf")), True)

    def test_is_real_number_complex_is_not_a_real_number(self):
        # The function's whole purpose (its name) is real-vs-not-real: complex is
        # the one unambiguously-numeric Python type that is NOT real. Pinned
        # separately so a widened type check (e.g. numbers.Number/Complex instead
        # of (int, float)) that admits complex values is killed explicitly.
        self.assertIs(_is_real_number(complex(1, 2)), False)
        self.assertIs(_is_real_number(1 + 2j), False)
        # even a complex whose imaginary part is exactly zero is still complex,
        # not real, and must be rejected on type, not value.
        self.assertIs(_is_real_number(complex(3, 0)), False)


class TestProject(unittest.TestCase):
    """Pure flattener: payload -> {'task.metric': float}. Spec AC 7-22."""

    def test_project_flatten_cases(self):
        # (payload, expected) table covering the enumerated spec behaviors.
        # Expected dicts are asserted whole (structured-output assertion, not
        # substring-in-blob).
        cases = [
            # AC7: no 'results' key
            ({}, {}),
            # AC8: empty results dict
            ({"results": {}}, {}),
            # AC21: task whose value is an empty dict
            ({"results": {"empty_task": {}}}, {}),
            # AC9: only alias, no numeric metrics
            ({"results": {"mmlu": {"alias": "mmlu"}}}, {}),
            # AC10: single numeric metric alongside alias
            (
                {"results": {"mmlu": {"acc,none": 0.71, "alias": "mmlu"}}},
                {"mmlu.acc__none": 0.71},
            ),
            # Output Contract: only metric_key EXACTLY == 'alias' is excluded.
            # A real-numeric metric whose key merely CONTAINS 'alias' as a
            # substring (prefix/infix/suffix) must survive, while the literal
            # 'alias' key alongside it is dropped. Distinguishes exact-key
            # exclusion from a substring/startswith exclusion.
            (
                {
                    "results": {
                        "t": {
                            "alias_score,none": 0.9,   # prefix substring, survives
                            "task_alias": 0.8,          # suffix substring, survives
                            "has_alias,none": 0.7,      # infix substring, survives
                            "alias": "mmlu",            # exact key, excluded
                        }
                    }
                },
                {
                    "t.alias_score__none": 0.9,
                    "t.task_alias": 0.8,
                    "t.has_alias__none": 0.7,
                },
            ),
            # AC11: bool value excluded
            ({"results": {"mmlu": {"acc,none": True}}}, {}),
            # AC12: non-numeric string value excluded
            ({"results": {"mmlu": {"acc,none": "not a number"}}}, {}),
            # AC13: every comma in the metric key replaced (not just the first)
            (
                {"results": {"t": {"a,b,c": 1.0}}},
                {"t.a__b__c": 1.0},
            ),
            # metric key with zero commas is used as-is after the '.' join
            (
                {"results": {"t": {"nocomma": 0.5}}},
                {"t.nocomma": 0.5},
            ),
            # AC14: RULER-style numeric-string metric-key prefixes, two metrics
            (
                {"results": {"niah_single_1": {"4096,none": 0.5, "32768,none": 0.9}}},
                {"niah_single_1.4096__none": 0.5, "niah_single_1.32768__none": 0.9},
            ),
            # AC15: two tasks, multiple metrics each, no loss/merge across tasks
            (
                {
                    "results": {
                        "mmlu": {"acc,none": 0.71, "alias": "mmlu"},
                        "gsm8k": {
                            "exact_match,strict-match": 0.5,
                            "exact_match,flexible-extract": 0.6,
                            "alias": "gsm8k",
                        },
                    }
                },
                {
                    "mmlu.acc__none": 0.71,
                    "gsm8k.exact_match__strict-match": 0.5,
                    "gsm8k.exact_match__flexible-extract": 0.6,
                },
            ),
            # Boundary: metric value of exactly zero is a real, meaningful
            # score (a fully-failing task) and must survive -- a truthiness-
            # based short-circuit (`not value`) would silently drop it. The
            # int 0 is coerced to the float 0.0.
            ({"results": {"t": {"m,none": 0}}}, {"t.m__none": 0.0}),
            # Boundary: float zero likewise survives and stays 0.0.
            ({"results": {"t": {"m,none": 0.0}}}, {"t.m__none": 0.0}),
            # AC17: int metric value coerced to float
            ({"results": {"t": {"m": 3}}}, {"t.m": 3.0}),
            # AC22: negative value preserved and coerced
            ({"results": {"t": {"m,none": -1.25}}}, {"t.m__none": -1.25}),
            # AC20: comma in the TASK name sanitized the same way
            (
                {"results": {"group,4096": {"acc,none": 0.5}}},
                {"group__4096.acc__none": 0.5},
            ),
            # AC19: sibling top-level keys ignored, incl. 'versions' number
            (
                {
                    "results": {"mmlu": {"acc,none": 0.71, "alias": "mmlu"}},
                    "versions": {"mmlu": 2},
                    "configs": {"mmlu": {"task": "mmlu"}},
                    "n-shot": {"mmlu": 5},
                    "model_name": "meta-llama/Llama-3.1-70B",
                },
                {"mmlu.acc__none": 0.71},
            ),
        ]
        for payload, expected in cases:
            with self.subTest(payload=repr(payload)):
                self.assertEqual(project(payload), expected)

    def test_project_infinity_included_nan_excluded(self):
        # inf/-inf are real numbers and survive coercion; NaN is dropped.
        result = project(
            {"results": {"t": {"good,none": float("inf"),
                               "bad,none": float("nan"),
                               "neg,none": float("-inf")}}}
        )
        self.assertEqual(
            set(result.keys()), {"t.good__none", "t.neg__none"}
        )
        self.assertEqual(result["t.good__none"], float("inf"))
        self.assertEqual(result["t.neg__none"], float("-inf"))
        self.assertNotIn("t.bad__none", result)

    def test_project_alias_never_contributes(self):
        # 'alias' key is excluded regardless of value type (str, number, etc.).
        for alias_val in ("mmlu", 0.5, 7, None, {"x": 1}):
            with self.subTest(alias=repr(alias_val)):
                out = project({"results": {"t": {"alias": alias_val}}})
                self.assertEqual(out, {})

    # --- invariants -------------------------------------------------------

    def test_project_values_are_native_float(self):
        # Invariant (AC17): every output value is a native float, even when the
        # source was an int. isinstance(3, float) is False, so this distinguishes
        # real coercion from a same-type passthrough.
        out = project(
            {"results": {"t": {"i,none": 3, "f,none": 0.71, "neg": -2}}}
        )
        self.assertEqual(set(out.keys()), {"t.i__none", "t.f__none", "t.neg"})
        for key, val in out.items():
            with self.subTest(key=key):
                self.assertIs(type(val), float)
        # explicit int->float coercion pin (AC17)
        self.assertEqual(out["t.i__none"], 3.0)
        self.assertFalse(isinstance(3, float))

    def test_project_keys_never_contain_commas(self):
        # Invariant: no ',' survives in any produced key (from task or metric).
        payload = {
            "results": {
                "group,4096": {"a,b,c": 1.0, "plain": 2.0},
                "gsm8k": {"exact_match,strict-match": 0.5},
            }
        }
        out = project(payload)
        for key in out:
            with self.subTest(key=key):
                self.assertNotIn(",", key)

    def test_project_one_entry_per_real_numeric_metric(self):
        # Invariant (AC15): output size equals the count of (task, real-numeric,
        # non-alias) metric pairs -- nothing lost or merged across tasks.
        payload = {
            "results": {
                "t1": {"a,none": 0.1, "b,none": 0.2, "alias": "t1"},
                "t2": {"c,none": 0.3, "bad": "x", "flag": True, "alias": "t2"},
            }
        }
        out = project(payload)
        self.assertEqual(len(out), 3)
        self.assertEqual(
            set(out.keys()),
            {"t1.a__none", "t1.b__none", "t2.c__none"},
        )

    def test_project_is_deterministic(self):
        # Invariant: pure function -> repeated calls on equal input yield equal
        # output (and distinct dict objects each call, since a NEW dict is
        # returned per contract).
        payload = {"results": {"mmlu": {"acc,none": 0.71, "alias": "mmlu"}}}
        first = project(payload)
        second = project(payload)
        self.assertEqual(first, second)
        self.assertIsNot(first, second)

    def test_project_returns_new_empty_dict_not_none(self):
        # Output contract: never returns None; empty case is a real {} dict.
        out = project({})
        self.assertIsInstance(out, dict)
        self.assertEqual(out, {})

    def test_project_does_not_mutate_payload(self):
        # AC16: in-contract payload (nested dicts, mixed value kinds) is not
        # mutated -- deep-equality against a pre-call snapshot.
        payload = {
            "results": {
                "mmlu": {"acc,none": 0.71, "alias": "mmlu"},
                "gsm8k": {
                    "exact_match,strict-match": 0.5,
                    "flag": True,
                    "note": "text",
                },
                "empty": {},
            },
            "versions": {"mmlu": 2},
            "model_name": "x",
        }
        snapshot = copy.deepcopy(payload)
        project(payload)
        self.assertEqual(payload, snapshot)

    def test_project_performs_no_file_io(self):
        # AC23: with builtins.open patched to raise, project still works ->
        # proves the flattener performs no file I/O.
        with patch("builtins.open", side_effect=AssertionError("no I/O allowed")):
            out = project(
                {"results": {"mmlu": {"acc,none": 0.71, "alias": "mmlu"}}}
            )
            self.assertEqual(out, {"mmlu.acc__none": 0.71})
            self.assertIs(_is_real_number(1), True)


class TestSignaturePreservation(unittest.TestCase):
    """AC24 / Regression Constraints: signatures + annotations are frozen."""

    def test_is_real_number_type_hints(self):
        self.assertEqual(
            typing.get_type_hints(_is_real_number),
            {"value": typing.Any, "return": bool},
        )

    def test_project_type_hints(self):
        self.assertEqual(
            typing.get_type_hints(project),
            {"payload": typing.Dict[str, typing.Any],
             "return": typing.Dict[str, float]},
        )

    def test_is_real_number_parameter_names(self):
        self.assertEqual(
            list(inspect.signature(_is_real_number).parameters), ["value"]
        )

    def test_project_parameter_names(self):
        self.assertEqual(
            list(inspect.signature(project).parameters), ["payload"]
        )


if __name__ == "__main__":
    unittest.main()
