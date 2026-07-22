'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.inference.utils.inference_suite_lifecycle.test_accuracy_eval.

Isolated via unittest.mock.patch on run_accuracy_tasks (imported into
inference_suite_lifecycle at module load time) so these tests never touch a
real orch or the network -- only the stage's own selection/gating/skip logic
is under test.
'''

import unittest
from types import SimpleNamespace
from unittest import mock

import pytest

from cvs.lib.inference.utils import inference_suite_lifecycle as lifecycle_mod
from cvs.lib.utils.verdict import ThresholdViolation


def _variant_config(tasks=(), thresholds=None):
    return SimpleNamespace(
        accuracy=SimpleNamespace(tasks=list(tasks)) if tasks is not None else None,
        params=SimpleNamespace(base_url="http://0.0.0.0", port_no="8000"),
        paths=SimpleNamespace(log_dir="/logs"),
        model=SimpleNamespace(id="meta-llama/Llama-3-8b"),
        thresholds=thresholds or {},
    )


def _task(id_):
    return SimpleNamespace(id=id_)


class _Lifecycle:
    def __init__(self, failed=False):
        self.failed = failed
        self.report = {}

    def record(self, nodeid, label, value, unit="s"):
        self.report.setdefault(nodeid, []).append((label, value, unit))


class _Request:
    class _Node:
        nodeid = "test_accuracy_eval"

    node = _Node()


class TestAccuracyEvalSkip(unittest.TestCase):
    def test_skips_when_prior_stage_failed(self):
        with self.assertRaises(pytest.skip.Exception):
            lifecycle_mod.test_accuracy_eval(
                orch=object(), variant_config=_variant_config(), lifecycle=_Lifecycle(failed=True), request=_Request()
            )

    def test_skips_when_accuracy_block_absent(self):
        vc = _variant_config(tasks=None)
        with self.assertRaises(pytest.skip.Exception):
            lifecycle_mod.test_accuracy_eval(orch=object(), variant_config=vc, lifecycle=_Lifecycle(), request=_Request())

    def test_skips_when_tasks_empty(self):
        vc = _variant_config(tasks=[])
        with self.assertRaises(pytest.skip.Exception):
            lifecycle_mod.test_accuracy_eval(orch=object(), variant_config=vc, lifecycle=_Lifecycle(), request=_Request())


class TestAccuracyEvalRun(unittest.TestCase):
    def test_calls_run_accuracy_tasks_with_expected_args(self):
        vc = _variant_config(tasks=[_task("mmlu")])
        lc = _Lifecycle()
        with mock.patch.object(lifecycle_mod, "run_accuracy_tasks", return_value={"mmlu": {"mmlu.acc__none": 0.7}}) as m:
            lifecycle_mod.test_accuracy_eval(orch="ORCH", variant_config=vc, lifecycle=lc, request=_Request())
        m.assert_called_once()
        kwargs = m.call_args.kwargs
        self.assertEqual(kwargs["orch"], "ORCH")
        self.assertEqual(kwargs["base_url"], "http://0.0.0.0:8000")
        self.assertEqual(kwargs["model_id"], "meta-llama/Llama-3-8b")
        self.assertEqual(kwargs["model_path"], "meta-llama/Llama-3-8b")
        self.assertEqual(kwargs["output_dir"], "/logs/accuracy")
        self.assertEqual([t.id for t in kwargs["tasks"]], ["mmlu"])

    def test_record_only_when_no_threshold_entry(self):
        vc = _variant_config(tasks=[_task("mmlu")], thresholds={})
        lc = _Lifecycle()
        with mock.patch.object(lifecycle_mod, "run_accuracy_tasks", return_value={"mmlu": {"mmlu.acc__none": 0.7}}):
            lifecycle_mod.test_accuracy_eval(orch="ORCH", variant_config=vc, lifecycle=lc, request=_Request())
        self.assertFalse(lc.failed)
        recorded = dict((label, (value, unit)) for label, value, unit in lc.report["test_accuracy_eval"])
        self.assertEqual(recorded["mmlu.mmlu.acc__none"], (0.7, ""))

    def test_threshold_pass_does_not_raise(self):
        vc = _variant_config(
            tasks=[_task("mmlu")],
            thresholds={"accuracy": {"mmlu": {"mmlu.acc__none": {"kind": "min", "value": 0.5}}}},
        )
        lc = _Lifecycle()
        with mock.patch.object(lifecycle_mod, "run_accuracy_tasks", return_value={"mmlu": {"mmlu.acc__none": 0.7}}):
            lifecycle_mod.test_accuracy_eval(orch="ORCH", variant_config=vc, lifecycle=lc, request=_Request())
        self.assertFalse(lc.failed)

    def test_threshold_miss_raises_threshold_violation(self):
        vc = _variant_config(
            tasks=[_task("mmlu")],
            thresholds={"accuracy": {"mmlu": {"mmlu.acc__none": {"kind": "min", "value": 0.9}}}},
        )
        lc = _Lifecycle()
        with mock.patch.object(lifecycle_mod, "run_accuracy_tasks", return_value={"mmlu": {"mmlu.acc__none": 0.7}}):
            with self.assertRaises(ThresholdViolation):
                lifecycle_mod.test_accuracy_eval(orch="ORCH", variant_config=vc, lifecycle=lc, request=_Request())

    def test_removed_task_stale_threshold_entry_ignored(self):
        # config.json only selects "mmlu"; threshold.json still has a stale
        # "gsm8k" entry from a since-removed task -- must not be looked up.
        vc = _variant_config(
            tasks=[_task("mmlu")],
            thresholds={
                "accuracy": {
                    "mmlu": {"mmlu.acc__none": {"kind": "min", "value": 0.5}},
                    "gsm8k": {"exact_match__strict-match": {"kind": "min", "value": 0.99}},
                }
            },
        )
        lc = _Lifecycle()
        with mock.patch.object(lifecycle_mod, "run_accuracy_tasks", return_value={"mmlu": {"mmlu.acc__none": 0.7}}):
            lifecycle_mod.test_accuracy_eval(orch="ORCH", variant_config=vc, lifecycle=lc, request=_Request())
        self.assertFalse(lc.failed)

    def test_run_failure_marks_lifecycle_failed_and_pytest_fails(self):
        vc = _variant_config(tasks=[_task("mmlu")])
        lc = _Lifecycle()
        with mock.patch.object(lifecycle_mod, "run_accuracy_tasks", side_effect=RuntimeError("boom")):
            with self.assertRaises(pytest.fail.Exception):
                lifecycle_mod.test_accuracy_eval(orch="ORCH", variant_config=vc, lifecycle=lc, request=_Request())
        self.assertTrue(lc.failed)

    def test_multiple_tasks_each_gated_independently(self):
        vc = _variant_config(
            tasks=[_task("mmlu"), _task("gsm8k")],
            thresholds={"accuracy": {"gsm8k": {"gsm8k.exact_match__strict-match": {"kind": "min", "value": 0.99}}}},
        )
        lc = _Lifecycle()
        with mock.patch.object(
            lifecycle_mod,
            "run_accuracy_tasks",
            return_value={
                "mmlu": {"mmlu.acc__none": 0.7},
                "gsm8k": {"gsm8k.exact_match__strict-match": 0.6},
            },
        ):
            with self.assertRaises(ThresholdViolation):
                lifecycle_mod.test_accuracy_eval(orch="ORCH", variant_config=vc, lifecycle=lc, request=_Request())


if __name__ == "__main__":
    unittest.main()
