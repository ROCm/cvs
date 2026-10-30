'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.utils.config_blocks.
'''

import math
import unittest

from pydantic import ValidationError

from cvs.lib.utils.config_blocks import ContainerSpec, Paths, RuntimeArgs, ThresholdFile


def _container(**overrides):
    value = {
        "name": "bench",
        "image": "rocm/vllm:latest",
        "runtime": {"name": "docker", "args": {}},
    }
    value.update(overrides)
    return value


class TestCommentKeys(unittest.TestCase):
    def test_comment_keys_are_ignored_at_every_model_level_but_kept_in_env(self):
        spec = ContainerSpec.model_validate(
            _container(
                _comment="container",
                runtime={"_comment": "runtime", "name": "docker", "args": {}},
                env={"_KEEP_ME": "1", "NCCL_DEBUG": "INFO"},
            )
        )
        self.assertEqual(spec.env, {"_KEEP_ME": "1", "NCCL_DEBUG": "INFO"})
        self.assertNotIn("_comment", spec.runtime.model_dump())


class TestPlaceholdersWithoutContext(unittest.TestCase):
    def test_template_values_validate_when_no_placeholder_policy_is_given(self):
        paths = Paths(
            shared_fs="<changeme>",
            models_dir="<changeme-models-mount>",
            log_dir="/logs",
            hf_token_file="/home/x/.hf_token",
        )
        spec = ContainerSpec.model_validate(_container(image="<CHANGEME>", env={"NCCL_SOCKET_IFNAME": "<changeme>"}))
        self.assertEqual(paths.models_dir, "<changeme-models-mount>")
        self.assertEqual(spec.image, "<CHANGEME>")


class TestRuntimeArgs(unittest.TestCase):
    def test_dump_has_only_the_keys_the_config_set(self):
        args = RuntimeArgs.model_validate(
            {"_comment": "mounts", "volumes": ["/data:/data"], "privileged": True, "custom_flag": "x"}
        )
        self.assertEqual(args.model_dump(), {"volumes": ["/data:/data"], "privileged": True, "custom_flag": "x"})

    def test_docker_keys_with_the_wrong_shape_are_rejected(self):
        for key, value in (
            ("volumes", "/data:/data"),
            ("env", ["A=1"]),
            ("privileged", "sometimes"),
            ("ulimit", "memlock=-1"),
        ):
            with self.subTest(key=key):
                with self.assertRaises(ValidationError):
                    RuntimeArgs.model_validate({key: value})


class TestThresholdFile(unittest.TestCase):
    def test_every_kind_loads_with_its_own_fields(self):
        cell = {
            "a": {"kind": "min", "value": 1},
            "b": {"kind": "max", "value": 2.5},
            "c": {"kind": "max_ms", "value": 3},
            "d": {"kind": "min_tok_s", "value": 4},
            "e": {"kind": "info", "value": 0},
            "f": {"kind": "within", "value": 5, "tolerance_pct": 10},
            "g": {"kind": "min_ratio", "value": 0.9, "reference": "a"},
            "h": {"kind": "min", "value": -1, "optional": True},
        }
        loaded = ThresholdFile.model_validate({"CELL": cell})
        self.assertEqual(set(loaded.cells()["CELL"]), set(cell))
        self.assertTrue(loaded.cells()["CELL"]["h"].optional)
        self.assertFalse(loaded.cells()["CELL"]["a"].optional)

    def test_comment_cells_and_metrics_are_dropped_and_accuracy_is_not_checked(self):
        loaded = ThresholdFile.model_validate(
            {
                "_comment": "calibration notes",
                "CELL": {"_example": {"kind": "bogus"}, "a": {"kind": "min", "value": 1}},
                "accuracy": {"gsm8k": {"gsm8k.exact_match": {"kind": "min", "value": 0, "tolerance_pct": 5}}},
            }
        )
        self.assertEqual(list(loaded.cells()), ["CELL"])
        self.assertEqual(list(loaded.cells()["CELL"]), ["a"])

    def test_malformed_specs_are_rejected(self):
        for spec in (
            {"value": 1},
            {"kind": "other", "value": 1},
            {"kind": "min"},
            {"kind": "within", "value": 1},
            {"kind": "min_ratio", "value": 1},
            {"kind": "min", "value": 1, "unit": "tok/s"},
            {"kind": "min", "value": True},
            {"kind": "min", "value": "1"},
            {"kind": "min", "value": None},
            {"kind": "min", "value": math.nan},
            {"kind": "min", "value": math.inf},
        ):
            with self.subTest(spec=spec):
                with self.assertRaises(ValidationError):
                    ThresholdFile.model_validate({"CELL": {"a": spec}})


if __name__ == "__main__":
    unittest.main()
