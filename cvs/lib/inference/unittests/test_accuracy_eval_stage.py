'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.inference.utils.inference_suite_lifecycle.test_accuracy_eval.

Isolated via unittest.mock.patch on run_accuracy_tasks (imported into
inference_suite_lifecycle at module load time) so these tests never touch a
real orch or the network -- only the stage's own selection/gating/skip logic
is under test.

test_accuracy_eval is parametrized by `accuracy_task` (one pytest node per
task id, mirroring test_metric/test_gpu_metric), so it is invoked here once
per task under test rather than once per variant_config as before.
'''

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import pytest

from cvs.lib.inference.atom.atom_config_loader import AtomVariantConfig, placeholder_gated_threshold_cell
from cvs.lib.inference.utils import inference_suite_lifecycle as lifecycle_mod
from cvs.lib.inference.utils.inferencing_config_loader import Run, SeqCombo, Sweep
from cvs.lib.utils.verdict import ThresholdViolation
from cvs.parsers.schemas import InferenceVariantConfig


def _variant_config(tasks=(), thresholds=None, enforce_thresholds=True):
    accuracy = (thresholds or {}).get("accuracy", {})
    return SimpleNamespace(
        lm_eval_tasks=lambda: list(tasks),
        lm_eval_thresholds=lambda task_id: accuracy.get(task_id, {}),
        params=SimpleNamespace(base_url="http://0.0.0.0", port_no="8000"),
        paths=SimpleNamespace(log_dir="/logs"),
        model=SimpleNamespace(id="meta-llama/Llama-3-8b"),
        enforce_thresholds=enforce_thresholds,
    )


def _task(id_):
    return SimpleNamespace(id=id_)


class _Lifecycle:
    def __init__(self, failed=False):
        self.failed = failed
        self.report = {}

    def record(self, nodeid, label, value, unit="s"):
        self.report.setdefault(nodeid, []).append((label, value, unit))


def _request(nodeid="test_accuracy_eval"):
    node = SimpleNamespace(nodeid=nodeid)
    return SimpleNamespace(node=node)


class TestAccuracyEvalSkip(unittest.TestCase):
    def test_skips_when_prior_stage_failed(self):
        with self.assertRaises(pytest.skip.Exception):
            lifecycle_mod.test_accuracy_eval(
                orch=object(),
                variant_config=_variant_config(tasks=[_task("mmlu")]),
                accuracy_task="mmlu",
                lifecycle=_Lifecycle(failed=True),
                request=_request(),
            )

    def test_skips_when_tasks_empty(self):
        vc = _variant_config(tasks=[])
        with self.assertRaises(pytest.skip.Exception):
            lifecycle_mod.test_accuracy_eval(
                orch=object(), variant_config=vc, accuracy_task="mmlu", lifecycle=_Lifecycle(), request=_request()
            )

    def test_skips_when_task_id_not_in_configured_tasks(self):
        # config.json's accuracy.tasks no longer includes this id (e.g. removed
        # after collection-time parametrization but before this node ran).
        vc = _variant_config(tasks=[_task("mmlu")])
        with self.assertRaises(pytest.skip.Exception):
            lifecycle_mod.test_accuracy_eval(
                orch=object(), variant_config=vc, accuracy_task="gsm8k", lifecycle=_Lifecycle(), request=_request()
            )


class TestAccuracyEvalRun(unittest.TestCase):
    def test_calls_run_accuracy_tasks_with_only_this_task(self):
        vc = _variant_config(tasks=[_task("mmlu"), _task("gsm8k")])
        lc = _Lifecycle()
        with mock.patch.object(
            lifecycle_mod, "run_accuracy_tasks", return_value={"mmlu": {"mmlu.acc__none": 0.7}}
        ) as m:
            lifecycle_mod.test_accuracy_eval(
                orch="ORCH", variant_config=vc, accuracy_task="mmlu", lifecycle=lc, request=_request()
            )
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
            lifecycle_mod.test_accuracy_eval(
                orch="ORCH", variant_config=vc, accuracy_task="mmlu", lifecycle=lc, request=_request()
            )
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
            lifecycle_mod.test_accuracy_eval(
                orch="ORCH", variant_config=vc, accuracy_task="mmlu", lifecycle=lc, request=_request()
            )
        self.assertFalse(lc.failed)

    def test_threshold_miss_raises_threshold_violation(self):
        vc = _variant_config(
            tasks=[_task("mmlu")],
            thresholds={"accuracy": {"mmlu": {"mmlu.acc__none": {"kind": "min", "value": 0.9}}}},
        )
        lc = _Lifecycle()
        with mock.patch.object(lifecycle_mod, "run_accuracy_tasks", return_value={"mmlu": {"mmlu.acc__none": 0.7}}):
            with self.assertRaises(ThresholdViolation):
                lifecycle_mod.test_accuracy_eval(
                    orch="ORCH", variant_config=vc, accuracy_task="mmlu", lifecycle=lc, request=_request()
                )
        # A threshold violation on this task's own node must NOT flip the
        # shared lifecycle flag -- sibling task nodes must still run.
        self.assertFalse(lc.failed)

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
            lifecycle_mod.test_accuracy_eval(
                orch="ORCH", variant_config=vc, accuracy_task="mmlu", lifecycle=lc, request=_request()
            )
        self.assertFalse(lc.failed)

    def test_run_failure_pytest_fails_without_setting_shared_lifecycle_failed(self):
        vc = _variant_config(tasks=[_task("mmlu")])
        lc = _Lifecycle()
        with mock.patch.object(lifecycle_mod, "run_accuracy_tasks", side_effect=RuntimeError("boom")):
            with self.assertRaises(pytest.fail.Exception):
                lifecycle_mod.test_accuracy_eval(
                    orch="ORCH", variant_config=vc, accuracy_task="mmlu", lifecycle=lc, request=_request()
                )
        # Independent failure isolation: a run failure in one task's node must
        # not set the shared lifecycle.failed flag, or every sibling task node
        # (parametrized on the same fixture) would skip instead of running.
        self.assertFalse(lc.failed)

    def test_threshold_miss_recorded_but_not_raised_when_enforce_thresholds_false(self):
        vc = _variant_config(
            tasks=[_task("mmlu")],
            thresholds={"accuracy": {"mmlu": {"mmlu.acc__none": {"kind": "min", "value": 0.9}}}},
            enforce_thresholds=False,
        )
        lc = _Lifecycle()
        with mock.patch.object(lifecycle_mod, "run_accuracy_tasks", return_value={"mmlu": {"mmlu.acc__none": 0.7}}):
            lifecycle_mod.test_accuracy_eval(
                orch="ORCH", variant_config=vc, accuracy_task="mmlu", lifecycle=lc, request=_request()
            )
        self.assertFalse(lc.failed)
        recorded = dict((label, (value, unit)) for label, value, unit in lc.report["test_accuracy_eval"])
        self.assertEqual(recorded["mmlu.mmlu.acc__none"], (0.7, ""))

    def test_two_tasks_gated_independently_one_fails_other_unaffected(self):
        # Simulates two parametrized nodes sharing one lifecycle object, in
        # collection order: mmlu's node raises ThresholdViolation, but gsm8k's
        # node (called after, as pytest would for the next parametrized item)
        # must still run and pass on its own merits.
        vc = _variant_config(
            tasks=[_task("mmlu"), _task("gsm8k")],
            thresholds={
                "accuracy": {
                    "mmlu": {"mmlu.acc__none": {"kind": "min", "value": 0.9}},
                    "gsm8k": {"gsm8k.exact_match__strict-match": {"kind": "min", "value": 0.5}},
                }
            },
        )
        lc = _Lifecycle()

        with mock.patch.object(lifecycle_mod, "run_accuracy_tasks", return_value={"mmlu": {"mmlu.acc__none": 0.7}}):
            with self.assertRaises(ThresholdViolation):
                lifecycle_mod.test_accuracy_eval(
                    orch="ORCH",
                    variant_config=vc,
                    accuracy_task="mmlu",
                    lifecycle=lc,
                    request=_request("test_accuracy_eval[mmlu]"),
                )
        self.assertFalse(lc.failed)

        with mock.patch.object(
            lifecycle_mod, "run_accuracy_tasks", return_value={"gsm8k": {"gsm8k.exact_match__strict-match": 0.6}}
        ):
            lifecycle_mod.test_accuracy_eval(
                orch="ORCH",
                variant_config=vc,
                accuracy_task="gsm8k",
                lifecycle=lc,
                request=_request("test_accuracy_eval[gsm8k]"),
            )
        self.assertFalse(lc.failed)


def _atom_variant(threshold_spec):
    cell = "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16"
    return AtomVariantConfig(
        schema_version=1,
        framework="atom",
        gpu_arch="mi3xx",
        paths={"shared_fs": "/fs", "models_dir": "/fs/models", "log_dir": "/logs", "hf_token_file": "/fs/.hf"},
        model={"id": "openai/gpt-oss-120b", "remote": 0, "precision": "bf16"},
        container={"name": "c", "image": "img", "runtime": {"name": "docker", "args": {}}},
        params={"tensor_parallelism": "8"},
        sweep=Sweep(
            sequence_combinations=[SeqCombo(name="short", isl="1024", osl="1024")],
            runs=[Run(combo="short", concurrency=16)],
        ),
        accuracy={"tasks": [{"id": "mmlu", "task": "mmlu"}, {"id": "gsm8k", "task": "gsm8k"}]},
        thresholds={cell: placeholder_gated_threshold_cell(), "accuracy": {"mmlu": {"mmlu.acc__none": threshold_spec}}},
    )


def _inference_variant(threshold_spec):
    cell = "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16"
    return InferenceVariantConfig.model_validate(
        {
            "paths": {"shared_fs": "/fs", "models_dir": "/fs/models", "log_dir": "/logs", "hf_token_file": "/fs/.hf"},
            "container": {"name": "c", "image": "img"},
            "server_params": {"model": "meta-llama/Llama-3-8b", "tp": 8},
            "benchmark_params": {"num_prompts": 16},
            "sweeps": {cell: {}},
            "runs": [cell],
            "accuracy": {"lm_eval": {"tasks": [{"id": "mmlu", "task": "mmlu"}, {"id": "gsm8k", "task": "gsm8k"}]}},
            "thresholds": {cell: {}, "accuracy": {"lm_eval": {"mmlu": {"mmlu.acc__none": threshold_spec}}}},
        }
    )


class TestAccuracyEvalModels(unittest.TestCase):
    def test_atom_and_inference_configs_gate_the_same(self):
        spec = {"kind": "min", "value": 0.9}
        for name, build in (("atom", _atom_variant), ("inference", _inference_variant)):
            for task, actual, violates in (("mmlu", 0.7, True), ("mmlu", 0.95, False), ("gsm8k", 0.1, False)):
                with self.subTest(model=name, task=task, actual=actual):
                    results = {task: {f"{task}.acc__none": actual}}
                    expected = self.assertRaises(ThresholdViolation) if violates else contextlib.nullcontext()
                    with mock.patch.object(lifecycle_mod, "run_accuracy_tasks", return_value=results) as run:
                        with expected:
                            lifecycle_mod.test_accuracy_eval(
                                orch="ORCH",
                                variant_config=build(spec),
                                accuracy_task=task,
                                lifecycle=_Lifecycle(),
                                request=_request(),
                            )
                    self.assertEqual([t.id for t in run.call_args.kwargs["tasks"]], [task])


if __name__ == "__main__":
    unittest.main()
