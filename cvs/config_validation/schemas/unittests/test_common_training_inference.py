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
    RuntimeSpec,
    WorkloadConfig,
)
from cvs.lib.conf_lib import config_blocks
from cvs.lib.conf_lib.loader import ConfigError
from cvs.unittests.support import FAKE_CRED

REJECT = {"placeholders": "reject"}
PATH_KEYS = ("shared_fs", "models_dir", "log_dir", "hf_token_file")


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


class TestPaths(unittest.TestCase):
    def test_valid_paths_load_with_temp_dir_defaulting_to_empty(self):
        paths = Paths.model_validate(_paths())
        self.assertEqual((paths.log_dir, paths.temp_dir), ("/data/logs", ""))

    def test_each_required_path_is_named_when_missing(self):
        for key in PATH_KEYS:
            value = _paths()
            del value[key]
            with self.subTest(key=key):
                self.assertEqual(_errors(Paths, value), [((key,), "missing")])

    def test_wrong_type_and_extra_key_are_rejected(self):
        self.assertEqual(_errors(Paths, _paths(log_dir=5)), [(("log_dir",), "string_type")])
        self.assertEqual(_errors(Paths, _paths(cache_dir="/c")), [(("cache_dir",), "extra_forbidden")])

    def test_comment_keys_are_dropped(self):
        self.assertEqual(Paths.model_validate(_paths(_comment="mounts")).model_dump(), {**_paths(), "temp_dir": ""})

    def test_every_path_rejects_placeholders_only_under_the_reject_context(self):
        value = _paths(**{key: "<changeme-mount>" for key in PATH_KEYS}, temp_dir="/tmp/<changeme>")
        expected = sorted(((key,), "value_error") for key in (*PATH_KEYS, "temp_dir"))
        self.assertEqual(_errors(Paths, value, context=REJECT), expected)
        self.assertEqual(Paths.model_validate(value).temp_dir, "/tmp/<changeme>")


class TestRuntimeArgs(unittest.TestCase):
    def test_dump_has_only_the_keys_the_config_set(self):
        args = RuntimeArgs.model_validate(
            {"_comment_bnxt_mount": "mounts", "volumes": ["/data:/data"], "network": None, "pids_limit": 100}
        )
        self.assertEqual(args.model_dump(), {"volumes": ["/data:/data"], "pids_limit": 100})
        self.assertEqual(RuntimeArgs().model_dump(), {})

    def test_every_docker_key_accepts_its_shape(self):
        value = {
            "volumes": ["/data:/data"],
            "devices": ["/dev/kfd"],
            "env": {"HF_HOME": "/data/hf"},
            "cap_add": ["SYS_PTRACE"],
            "security_opt": ["seccomp=unconfined"],
            "group_add": ["video", 44],
            "ulimit": ["memlock=-1"],
            "network": "host",
            "ipc": "host",
            "user": 1000,
            "shm_size": "64g",
            "privileged": True,
            "registry": {"username": "jdoe"},
        }
        self.assertEqual(RuntimeArgs.model_validate(value).model_dump(), value)
        for key, other in (("user", "1000:1000"), ("shm_size", 68719476736)):
            with self.subTest(key=key):
                self.assertEqual(RuntimeArgs.model_validate({key: other}).model_dump(), {key: other})

    def test_docker_keys_with_the_wrong_shape_are_rejected(self):
        for key, value in (
            ("volumes", "/data:/data"),
            ("devices", "/dev/kfd"),
            ("env", ["A=1"]),
            ("cap_add", "SYS_PTRACE"),
            ("ulimit", "memlock=-1"),
            ("network", ["host"]),
            ("privileged", "sometimes"),
            ("registry", {"port": 5000}),
        ):
            with self.subTest(key=key):
                with self.assertRaises(ValidationError):
                    RuntimeArgs.model_validate({key: value})

    def test_placeholders_in_volumes_and_env_are_rejected_under_the_reject_context(self):
        value = {"volumes": ["<changeme-models-mount>:/models"], "env": {"HF_HOME": "<changeme>"}}
        self.assertEqual(
            _errors(RuntimeArgs, value, context=REJECT),
            [(("env", "HF_HOME"), "value_error"), (("volumes", 0), "value_error")],
        )
        self.assertEqual(RuntimeArgs.model_validate(value).model_dump(), value)


class TestRuntimeSpec(unittest.TestCase):
    def test_args_default_to_an_empty_object(self):
        spec = RuntimeSpec.model_validate({"name": "docker"})
        self.assertIsInstance(spec.args, RuntimeArgs)
        self.assertEqual(spec.model_dump(), {"name": "docker", "args": {}})

    def test_name_is_required_and_args_are_typed(self):
        self.assertEqual(_errors(RuntimeSpec, {}), [(("name",), "missing")])
        self.assertEqual(
            _errors(RuntimeSpec, {"name": "docker", "args": {"volumes": "/data:/data"}}),
            [(("args", "volumes"), "list_type")],
        )

    def test_extra_keys_are_kept_and_comment_keys_dropped(self):
        spec = RuntimeSpec.model_validate({"name": "docker", "_comment": "x", "socket": "/run/docker.sock"})
        self.assertEqual(spec.model_extra, {"socket": "/run/docker.sock"})


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

    def test_each_lifetime_loads_and_any_other_is_rejected(self):
        for lifetime in ("no_launch", "per_run", "persistent"):
            with self.subTest(lifetime=lifetime):
                self.assertEqual(ContainerSpec.model_validate(_container(lifetime=lifetime)).lifetime, lifetime)
        for lifetime in ("forever", "", None, True):
            with self.subTest(lifetime=lifetime):
                self.assertEqual(
                    _errors(ContainerSpec, _container(lifetime=lifetime)), [(("lifetime",), "literal_error")]
                )

    def test_each_required_field_is_named_when_missing(self):
        for key in ("name", "image", "runtime"):
            value = _container()
            del value[key]
            with self.subTest(key=key):
                self.assertEqual(_errors(ContainerSpec, value), [((key,), "missing")])

    def test_wrong_type_and_extra_key_are_rejected(self):
        self.assertEqual(_errors(ContainerSpec, _container(env={"A": 1})), [(("env", "A"), "string_type")])
        self.assertEqual(_errors(ContainerSpec, _container(enabled=True)), [(("enabled",), "extra_forbidden")])

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

    def test_config_without_a_container_block_has_none(self):
        config = WorkloadConfig.validate_config({"_comment": "notes", "paths": _paths()}, self.config_file)
        self.assertIsNone(config.container)
        self.assertEqual(config.paths.models_dir, "/data/models")

    def test_container_block_is_validated_as_a_container_spec(self):
        config = WorkloadConfig.validate_config({"paths": _paths(), "container": _container()}, self.config_file)
        self.assertIsInstance(config.container, ContainerSpec)
        self.assertEqual(config.container.runtime.name, "docker")

    def test_errors_name_nested_and_unknown_fields_without_their_values(self):
        paths = _paths()
        del paths["log_dir"]
        raw = {"paths": paths, "container": _container(image=[FAKE_CRED]), "ib_netdev": FAKE_CRED}
        with self.assertRaises(ConfigError) as ctx:
            WorkloadConfig.validate_config(raw, self.config_file)
        self.assertEqual(
            sorted(line.split(":", 1)[0] for line in ctx.exception.problems),
            ["container.image", "ib_netdev", "paths.log_dir"],
        )
        self.assertNotIn(FAKE_CRED, str(ctx.exception))

    def test_missing_paths_block_is_named(self):
        with self.assertRaises(ConfigError) as ctx:
            WorkloadConfig.validate_config({"container": _container()}, self.config_file)
        self.assertEqual([line.split(":", 1)[0] for line in ctx.exception.problems], ["paths"])

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
