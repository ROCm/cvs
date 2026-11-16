'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.config_validation.schemas.common_training_inference.
'''

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from cvs.config_validation.schemas.common_training_inference import (
    ContainerSpec,
    Paths,
    RuntimeArgs,
    WorkloadConfig,
)
from cvs.lib.conf_lib import config_blocks

REJECT = {"placeholders": "reject"}


def _paths(**overrides):
    value = {
        "shared_fs": "/data/shared",
        "models_dir": "/data/models",
        "log_dir": "/data/logs",
        "hf_token_file": "/data/.hf_token",  # nosec B105 - a file path, not a secret
    }
    value.update(overrides)
    return value


def _container(**overrides):
    value = {"name": "bench", "image": "rocm/vllm:latest", "runtime": {"name": "docker"}}
    value.update(overrides)
    return value


def _errors(model, data, context=None):
    try:
        model.model_validate(data, context=context)
    except ValidationError as exc:
        return sorted((error["loc"], error["type"]) for error in exc.errors())
    raise AssertionError(f"{model.__name__} accepted {data!r}")


def _fields(model):
    return [
        (name, "required" if field.is_required() else field.get_default(call_default_factory=True))
        for name, field in model.model_fields.items()
    ]


class TestMatchesConfLib(unittest.TestCase):
    def test_field_names_order_and_defaults_match_the_conf_lib_blocks(self):
        for new, old in (
            (Paths, config_blocks.Paths),
            (RuntimeArgs, config_blocks.RuntimeArgs),
            (ContainerSpec, config_blocks.ContainerSpec),
        ):
            with self.subTest(block=new.__name__):
                self.assertEqual(_fields(new), _fields(old))


class TestRuntimeArgs(unittest.TestCase):
    def test_dump_has_only_the_keys_the_config_set(self):
        args = RuntimeArgs.model_validate(
            {"_comment_bnxt_mount": "mounts", "volumes": ["/data:/data"], "network": None, "pids_limit": 100}
        )
        self.assertEqual(args.model_dump(), {"volumes": ["/data:/data"], "pids_limit": 100})
        self.assertEqual(RuntimeArgs().model_dump(), {})


class TestContainerSpec(unittest.TestCase):
    def test_minimal_block_dumps_every_default(self):
        # This dump is what a suite merges onto the cluster file's container block.
        self.assertEqual(
            ContainerSpec.model_validate(_container()).model_dump(),
            {
                "lifetime": "per_run",
                "name": "bench",
                "image": "rocm/vllm:latest",
                "runtime": {"name": "docker", "args": {}},
                "env": {},
            },
        )

    def test_comment_keys_are_dropped_at_each_block_but_kept_in_env(self):
        spec = ContainerSpec.model_validate(
            _container(
                _comment="container",
                runtime={"_comment": "runtime", "name": "docker", "args": {"_comment_ionic_mounts": "x"}},
                env={"_KEEP_ME": "1", "NCCL_DEBUG": "INFO"},
            )
        )
        self.assertEqual(spec.env, {"_KEEP_ME": "1", "NCCL_DEBUG": "INFO"})
        self.assertEqual(spec.runtime.model_dump(), {"name": "docker", "args": {}})

    def test_placeholders_are_rejected_only_under_the_reject_context(self):
        value = _container(
            name="vllm-<changeme>",
            image="<CHANGEME>",
            env={"NCCL_SOCKET_IFNAME": "<changeme>"},
            runtime={"name": "docker", "args": {"volumes": ["<changeme-models-mount>:/models"]}},
        )
        self.assertEqual(
            _errors(ContainerSpec, value, context=REJECT),
            [
                (("env", "NCCL_SOCKET_IFNAME"), "value_error"),
                (("image",), "value_error"),
                (("name",), "value_error"),
                (("runtime", "args", "volumes", 0), "value_error"),
            ],
        )
        self.assertEqual(ContainerSpec.model_validate(value).image, "<CHANGEME>")


class TestWorkloadConfig(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.config_file = Path(self._tmp.name) / "workload.json"

    def test_reject_context_reaches_every_nested_block(self):
        raw = {
            "paths": _paths(models_dir="<changeme>"),
            "container": _container(runtime={"name": "docker", "args": {"env": {"HF_HOME": "<changeme>"}}}),
        }
        self.assertEqual(
            _errors(WorkloadConfig, raw, context=REJECT),
            [
                (("container", "runtime", "args", "env", "HF_HOME"), "value_error"),
                (("paths", "models_dir"), "value_error"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
