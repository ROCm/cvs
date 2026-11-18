'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.config_validation.loaders.suite_config.
'''

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Annotated
from unittest import mock

from pydantic import Field, model_validator

import cvs
from cvs.config_validation.loaders.suite_config import (
    MISSING_THRESHOLD_MESSAGE,
    PLACEHOLDER_KEY_MESSAGE,
    UNMAPPED_BLOCK_MESSAGE,
    UNMAPPED_FIELD_MESSAGE,
    LoadedSuiteConfig,
    load_suite_config,
    mode_from_module,
    old_threshold_paths,
    read_cluster,
)
from cvs.config_validation.schemas.base import ConfigStr, OpenBlock, StrictBlock, SuiteConfig
from cvs.config_validation.schemas.thresholds import ThresholdFile, Thresholds
from cvs.lib.conf_lib.config_blocks import PLACEHOLDER_MESSAGE
from cvs.lib.conf_lib.loader import ConfigError
from cvs.lib.utils.config_loader import substitute_placeholders
from cvs.unittests.support import FAKE_CRED

PACKAGED_VLLM_DIR = Path(cvs.__file__).resolve().parent / "input" / "config_file" / "inference" / "vllm"
CELL = "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16"
CLUSTER = {
    "username": "{user-id}",
    "priv_key_file": "/home/{user-id}/.ssh/id_rsa",
    "node_dict": {"node0": {}, "node1": {}},
}
RESOLVED_CLUSTER = {
    "username": "jdoe",
    "priv_key_file": "/home/jdoe/.ssh/id_rsa",
    "node_dict": {"node0": {}, "node1": {}},
}
VALID = {"_comment": "notes", "threshold_json": "", "server_params": {"model": "org/model", "tp": 8}}
OLD_SERVER_KEYS = ("tensor_parallel_size", "pipeline_parallel_size")
OLD_KINDS = {
    "max_ms": {"kind": "max_ms", "value": 250},
    "min_tok_s": {"kind": "min_tok_s", "value": 1200},
    "within": {"kind": "within", "value": 0.8, "tolerance_pct": 5},
    "min_ratio": {"kind": "min_ratio", "value": 0.9, "reference": "output_throughput"},
}
PLACEHOLDER_FORMS = ("<changeme>", "/data/<changeme>/models")


class _Server(StrictBlock):
    model: ConfigStr
    tp: int
    token: str = ""


class _Serving(SuiteConfig, Thresholds):
    suite = "demo"
    thresholds: ThresholdFile = Field(default_factory=ThresholdFile)
    server_params: _Server
    mounts: list = []
    env: dict = {}

    @classmethod
    def old_keys(cls, raw):
        server = raw.get("server_params", {})
        return [f"server_params.{key}" for key in OLD_SERVER_KEYS if key in server]


class _Unnamed(_Serving):
    suite = None


class _Hooked(_Serving):
    @classmethod
    def read_raw(cls, config_file, cluster):
        return substitute_placeholders(config_file, cluster, all_placeholders=True)

    @classmethod
    def orch_inputs(cls, raw, config, cluster):
        return raw, config, list(config.thresholds.model_extra), cluster


class _Coverage(_Serving):
    runs: list = []

    @model_validator(mode="after")
    def _enforced_runs_need_threshold_cells(self):
        if self.enforce_thresholds and any(run not in self.thresholds.model_extra for run in self.runs):
            raise ValueError("with enforce_thresholds, every run needs a threshold cell")
        return self


class _GpuFile(OpenBlock):
    __pydantic_extra__: dict[str, dict[str, float]]


class _GpuSuite(SuiteConfig, Thresholds):
    thresholds: _GpuFile = Field(default_factory=_GpuFile)


class _OneSectionSuite(SuiteConfig, Thresholds):
    thresholds: Annotated[dict, Field(max_length=1)] = {}


class _NoThresholdsField(SuiteConfig, Thresholds):
    pass


class _Plain(SuiteConfig):
    name: str


class _Modes(SuiteConfig):
    modes = {"_distributed": "distributed", "_single": "single", "_disagg_distributed": "disaggregated"}


class _ModesWithDefault(_Modes):
    default_mode = "distributed"


class _NoModes(SuiteConfig):
    pass


class _NoModesWithDefault(SuiteConfig):
    default_mode = "single"


def _spec(kind="min", value=1):
    return {"kind": kind, "value": value}


class _LoaderCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name).resolve()
        user = mock.patch.dict(os.environ, {"USER": "jdoe"})
        user.start()
        self.addCleanup(user.stop)
        self.cluster_file = str(self._write("cluster.json", CLUSTER))
        self.config_file = str(self.tmp / "configs" / "suite.json")
        self.log = mock.Mock()

    def _write(self, name, data):
        path = self.tmp / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return path

    def _load(self, config, schema=_Serving):
        self._write("configs/suite.json", config)
        return load_suite_config(schema, self.config_file, self.cluster_file, self.log)

    def _refused(self, config, schema=_Serving):
        with self.assertRaises(ConfigError) as ctx:
            self._load(config, schema)
        exc = ctx.exception
        # pydantic's ValidationError and a lookup's FileNotFoundError print values, so neither may ride along.
        self.assertIsNone(exc.__cause__)
        self.assertTrue(exc.__context__ is None or exc.__suppress_context__)
        return exc

    def _threshold_config(self, thresholds, name="demo_threshold.json", **config):
        self._write(f"configs/{name}", thresholds)
        return {**VALID, "threshold_json": name, **config}


class TestReadCluster(_LoaderCase):
    def test_user_id_is_resolved_from_the_environment_as_the_orch_fixture_does(self):
        self.assertEqual(read_cluster(self.cluster_file), RESOLVED_CLUSTER)

    def test_a_changeme_in_the_cluster_file_exits_before_the_config_is_read(self):
        cluster_file = self._write("bad_cluster.json", {**CLUSTER, "priv_key_file": "<changeme>"})
        with mock.patch("cvs.lib.utils_lib.log") as utils_log, self.assertRaises(SystemExit) as ctx:
            load_suite_config(_Serving, str(self.tmp / "no_such_config.json"), str(cluster_file), self.log)
        self.assertEqual(ctx.exception.code, 1)
        utils_log.error.assert_called_once()


class TestLoadSuiteConfig(_LoaderCase):
    def test_a_valid_config_returns_schema_config_resolved_cluster_and_the_raw_config_as_orch_inputs(self):
        loaded = self._load(VALID)
        self.assertIsInstance(loaded, LoadedSuiteConfig)
        self.assertIs(loaded.schema, _Serving)
        self.assertIsInstance(loaded.config, _Serving)
        self.assertEqual((loaded.config.server_params.model, loaded.config.server_params.tp), ("org/model", 8))
        self.assertEqual(loaded.cluster, RESOLVED_CLUSTER)
        self.assertEqual(loaded.orch_inputs, VALID)

    def test_hooks_get_the_resolved_cluster_and_orch_inputs_gets_read_raws_config_and_the_validated_one(self):
        config = self._threshold_config({CELL: {"output_throughput": _spec()}})
        config["server_params"] = {"model": "{user-id}/model", "tp": 8}
        loaded = self._load(config, _Hooked)
        raw, validated, cells_when_called, cluster = loaded.orch_inputs
        # No thresholds or enforce_thresholds key: those went only to validate_config.
        self.assertEqual(
            raw, {"threshold_json": "demo_threshold.json", "server_params": {"model": "jdoe/model", "tp": 8}}
        )
        self.assertIs(validated, loaded.config)
        self.assertEqual(cells_when_called, [CELL])
        self.assertIs(cluster, loaded.cluster)


class TestCheckOrder(_LoaderCase):
    def _config(self, old=True, unmapped=True, placeholder=True):
        server = {"model": "org/model", "tp": "eight"}
        config = {**VALID, "server_params": server}
        if old:
            server.update(tensor_parallel_size="4", pipeline_parallel_size="1")
        if unmapped:
            config["_unmapped"] = {"server_params.max_model_len": 4096}
        if placeholder:
            server["model"] = "<changeme>"
        return config

    def test_old_keys_are_reported_alone_with_the_migrate_command(self):
        exc = self._refused(self._config())
        self.assertEqual(
            exc.problems,
            [
                "old config format: server_params.tensor_parallel_size, server_params.pipeline_parallel_size; "
                f"run cvs config migrate demo {self.config_file} <new config>"
            ],
        )
        self.assertTrue(str(exc).startswith(f"{self.config_file} is not a valid config:\n  old config format: "))

    def test_unmapped_is_reported_alone_when_no_old_key_is_left(self):
        exc = self._refused(self._config(old=False))
        self.assertEqual(exc.problems, [f"_unmapped.server_params.max_model_len: {UNMAPPED_FIELD_MESSAGE}"])

    def test_placeholders_are_reported_alone_when_no_unmapped_block_is_left(self):
        exc = self._refused(self._config(old=False, unmapped=False))
        self.assertEqual(exc.problems, [f"server_params.model: {PLACEHOLDER_MESSAGE}"])

    def test_schema_problems_come_last(self):
        exc = self._refused(self._config(old=False, unmapped=False, placeholder=False))
        self.assertEqual([line.split(":", 1)[0] for line in exc.problems], ["server_params.tp"])

    def test_a_schema_without_a_suite_name_still_names_the_command(self):
        exc = self._refused(self._config(), _Unnamed)
        self.assertIn(f"run cvs config migrate <suite> {self.config_file} <new config>", exc.problems[0])

    def test_the_threshold_file_is_checked_after_the_configs_own_checks_and_before_its_schema(self):
        thresholds = {CELL: {"output_throughput": OLD_KINDS["max_ms"]}}
        placeholder = self._threshold_config(thresholds, server_params={"model": "<changeme>", "tp": 8})
        self.assertEqual(self._refused(placeholder).problems, [f"server_params.model: {PLACEHOLDER_MESSAGE}"])
        schema_problem = self._threshold_config(thresholds, server_params={"model": "org/model", "tp": "eight"})
        self.assertTrue(self._refused(schema_problem).problems[0].startswith(f"old config format: {CELL}."))


class TestUnmapped(_LoaderCase):
    def test_each_field_in_a_top_level_block_is_named_beside_comment_keys_and_never_its_value(self):
        config = {
            "_comment": "notes",
            "_comment_unmapped": "move these",
            "_unmapped": {"server_params.foo": 1, "bar": [FAKE_CRED]},
            **VALID,
        }
        exc = self._refused(config)
        self.assertEqual(
            exc.problems,
            [f"_unmapped.server_params.foo: {UNMAPPED_FIELD_MESSAGE}", f"_unmapped.bar: {UNMAPPED_FIELD_MESSAGE}"],
        )
        self.assertNotIn(FAKE_CRED, str(exc))

    def test_blocks_nested_in_an_object_or_a_list_item_are_found(self):
        config = {
            **VALID,
            "server_params": {"model": "org/model", "tp": 8, "_unmapped": {"foo": 1}},
            "mounts": [{"a": 1}, {"_unmapped": {"b": 2}}],
        }
        self.assertEqual(
            self._refused(config).problems,
            [
                f"server_params._unmapped.foo: {UNMAPPED_FIELD_MESSAGE}",
                f"mounts.1._unmapped.b: {UNMAPPED_FIELD_MESSAGE}",
            ],
        )

    def test_a_top_level_block_is_found_even_when_read_raw_drops_comment_keys(self):
        # substitute_placeholders, which _Hooked reads with as vLLM does today, drops every top-level "_" key.
        exc = self._refused({**VALID, "_unmapped": {"server_params.foo": 1}}, _Hooked)
        self.assertEqual(exc.problems, [f"_unmapped.server_params.foo: {UNMAPPED_FIELD_MESSAGE}"])

    def test_an_empty_or_non_object_block_is_named_as_a_whole(self):
        for block in ({}, FAKE_CRED):
            with self.subTest(block=block):
                exc = self._refused({**VALID, "_unmapped": block})
                self.assertEqual(exc.problems, [f"_unmapped: {UNMAPPED_BLOCK_MESSAGE}"])
                self.assertNotIn(FAKE_CRED, str(exc))

    def test_a_comment_key_that_only_starts_with_unmapped_loads(self):
        self.assertEqual(self._load({**VALID, "_unmapped_note": "moved"}).config.server_params.tp, 8)


class TestPlaceholders(_LoaderCase):
    def test_placeholders_are_named_at_the_top_level_nested_and_in_lists(self):
        for form in PLACEHOLDER_FORMS:
            with self.subTest(form=form):
                config = {
                    **VALID,
                    "threshold_json": form,
                    "server_params": {"model": "org/model", "tp": 8, "token": form},
                    "mounts": ["/a:/a", form, {"host": [form]}],
                }
                self.assertEqual(
                    self._refused(config).problems,
                    [
                        f"threshold_json: {PLACEHOLDER_MESSAGE}",
                        f"server_params.token: {PLACEHOLDER_MESSAGE}",
                        f"mounts.1: {PLACEHOLDER_MESSAGE}",
                        f"mounts.2.host.0: {PLACEHOLDER_MESSAGE}",
                    ],
                )

    def test_a_placeholder_key_is_named(self):
        exc = self._refused({**VALID, "env": {"<changeme-var>": "1"}})
        self.assertEqual(exc.problems, [f"env.<changeme-var>: {PLACEHOLDER_KEY_MESSAGE}"])

    def test_a_placeholder_read_raw_brings_in_from_the_cluster_is_named(self):
        # The cluster read exits only on the literal <changeme>, so a suffixed form there reaches the config.
        cluster = {**CLUSTER, "home_mount_dir_name": "<changeme-mount>"}
        self.cluster_file = str(self._write("cluster_mount.json", cluster))
        exc = self._refused({**VALID, "mounts": ["/{home-mount-dir}/data"]}, _Hooked)
        self.assertEqual(exc.problems, [f"mounts.0: {PLACEHOLDER_MESSAGE}"])

    def test_the_value_is_never_printed(self):
        exc = self._refused({**VALID, "env": {"HF_TOKEN": f"{FAKE_CRED}<changeme>"}})
        self.assertEqual(exc.problems, [f"env.HF_TOKEN: {PLACEHOLDER_MESSAGE}"])
        self.assertNotIn(FAKE_CRED, str(exc))

    def test_placeholders_in_comments_load(self):
        # Shipped configs explain placeholders in comments, such as <changeme-libionic-soname> in _comment_ionic_mounts.
        config = {
            **VALID,
            "_comment": "Replace every <changeme> before running.",
            "server_params": {"model": "org/model", "tp": 8, "_comment_model": "<changeme-models-mount>"},
            "mounts": [{"_example": "<changeme>"}],
        }
        self.assertEqual(self._load(config).config.mounts, [{"_example": "<changeme>"}])


class TestThresholdFile(_LoaderCase):
    def test_a_named_file_next_to_the_config_is_attached_and_enforce_stays_as_written(self):
        thresholds = {
            "_comment": "notes",
            CELL: {"output_throughput": _spec("min", 900), "mean_ttft_ms": _spec("max", 80)},
            "accuracy": {"lm_eval": {"gsm8k_flex": {"exact_match": _spec("min", 0.5)}}},
        }
        for enforce in (True, False):
            with self.subTest(enforce=enforce):
                config = self._load(self._threshold_config(thresholds, enforce_thresholds=enforce)).config
                self.assertEqual(config.thresholds, ThresholdFile.model_validate(thresholds))
                self.assertEqual(config.thresholds.model_extra[CELL]["mean_ttft_ms"].value, 80)
                self.assertIs(config.enforce_thresholds, enforce)
        self.log.warning.assert_not_called()

    def test_a_relative_name_resolves_against_the_config_folder_and_an_absolute_one_as_written(self):
        self._write("configs/gpu/demo_threshold.json", {CELL: {"output_throughput": _spec()}})
        # A decoy one folder up catches resolving against the config folder's parent.
        decoy = self._write("gpu/demo_threshold.json", {"decoy": {"output_throughput": _spec()}})
        for name, cells in (("gpu/demo_threshold.json", [CELL]), (str(decoy), ["decoy"])):
            with self.subTest(name=name):
                config = self._load({**VALID, "threshold_json": name}).config
                self.assertEqual(list(config.thresholds.model_extra), cells)

    def test_the_file_replaces_inline_thresholds(self):
        config = self._threshold_config({CELL: {"output_throughput": _spec()}})
        config["thresholds"] = {"inline": {"output_throughput": _spec()}}
        self.assertEqual(list(self._load(config).config.thresholds.model_extra), [CELL])

    def test_an_empty_name_records_only_and_never_reads_a_lone_sibling(self):
        # The older loaders read a lone *threshold.json sibling; this one, refused if read, must stay unread.
        self._write("configs/sibling_threshold.json", {CELL: {"output_throughput": OLD_KINDS["max_ms"]}})
        for name in ("", "  "):
            with self.subTest(name=name):
                self.log.reset_mock()
                config = {**VALID, "threshold_json": name, "enforce_thresholds": True}
                config["thresholds"] = {"inline": {"output_throughput": _spec()}}
                loaded = self._load(config).config
                self.assertEqual(loaded.thresholds, ThresholdFile())
                self.assertIs(loaded.enforce_thresholds, False)
                self.log.warning.assert_called_once_with(
                    "%s names no threshold_json; metrics are recorded without gating", self.config_file
                )

    def test_a_name_missing_beside_the_config_falls_back_to_the_packaged_copy(self):
        packaged = sorted(PACKAGED_VLLM_DIR.glob("*_threshold.json"))[0]
        written = json.loads(packaged.read_text())
        if old_threshold_paths(written):
            # Shipped files still in the old layout are found, then refused as old.
            with self.assertWarns(UserWarning):
                exc = self._refused({**VALID, "threshold_json": packaged.name})
            self.assertTrue(str(exc).startswith(f"{packaged} is not a valid config:"))
            self.assertIn("old config format:", exc.problems[0])
            return
        with self.assertWarns(UserWarning):
            config = self._load({**VALID, "threshold_json": packaged.name}).config
        self.assertEqual(config.thresholds, ThresholdFile.model_validate(written))
        self.assertTrue(config.thresholds.model_extra)

    def test_a_missing_file_with_no_packaged_copy_names_threshold_json_and_never_its_value(self):
        exc = self._refused({**VALID, "threshold_json": f"{FAKE_CRED}_threshold.json"})
        self.assertEqual(exc.problems, [f"threshold_json: {MISSING_THRESHOLD_MESSAGE}"])
        self.assertTrue(str(exc).startswith(f"{self.config_file} is not a valid config:"))
        self.assertNotIn(FAKE_CRED, str(exc))

    def test_old_kinds_are_reported_alone_by_cell_and_metric_with_the_migrate_command(self):
        threshold_path = self.tmp / "configs" / "demo_threshold.json"
        for kind in ("max_ms", "within"):
            spec = OLD_KINDS[kind]
            with self.subTest(kind=kind):
                thresholds = {
                    "_unmapped": {"x": 1},
                    CELL: {"output_throughput": _spec("min", "<changeme>"), "mean_ttft_ms": spec},
                    "accuracy": {"lm_eval": {"gsm8k_flex": {"exact_match": spec, "stderr": _spec("max", "0.1")}}},
                }
                exc = self._refused(self._threshold_config(thresholds))
                self.assertEqual(
                    exc.problems,
                    [
                        f"old config format: {CELL}.mean_ttft_ms, accuracy.lm_eval.gsm8k_flex.exact_match; "
                        f"run cvs config migrate demo {threshold_path} <new config>"
                    ],
                )
                self.assertTrue(str(exc).startswith(f"{threshold_path} is not a valid config:"))

    def test_accuracy_sections_by_task_id_are_reported_as_the_old_layout(self):
        threshold_path = self.tmp / "configs" / "demo_threshold.json"
        thresholds = {
            CELL: {"output_throughput": _spec()},
            "accuracy": {
                "_comment": "notes",
                "gsm8k_flex": {"exact_match": _spec("min", 0.5)},
                "lm_eval": {},
                "gsm8k_strict": {},
            },
        }
        self.assertEqual(
            self._refused(self._threshold_config(thresholds)).problems,
            [
                "old config format: accuracy.gsm8k_flex, accuracy.gsm8k_strict; "
                f"run cvs config migrate demo {threshold_path} <new config>"
            ],
        )

    def test_old_kinds_under_comment_keys_load(self):
        thresholds = {
            "_example": {"output_throughput": OLD_KINDS["min_tok_s"]},
            CELL: {"_old": OLD_KINDS["max_ms"], "output_throughput": _spec()},
        }
        config = self._load(self._threshold_config(thresholds)).config
        self.assertEqual(list(config.thresholds.model_extra), [CELL])
        self.assertEqual(list(config.thresholds.model_extra[CELL]), ["output_throughput"])

    def test_unmapped_is_reported_before_placeholders_and_an_old_kind_inside_it_is_not_old_format(self):
        # The converter moves a within or min_ratio spec, which has no new kind, into _unmapped.
        thresholds = {
            "_comment": "notes",
            "_unmapped": {f"{CELL}.p99_itl_ms": OLD_KINDS["within"]},
            CELL: {"output_throughput": _spec("min", "<changeme>")},
        }
        exc = self._refused(self._threshold_config(thresholds))
        self.assertEqual(exc.problems, [f"_unmapped.{CELL}.p99_itl_ms: {UNMAPPED_FIELD_MESSAGE}"])

    def test_schema_problems_name_each_spec_field_never_its_value_and_an_unknown_kind_is_not_old_format(self):
        threshold_path = self.tmp / "configs" / "demo_threshold.json"
        thresholds = {
            CELL: {
                "output_throughput": _spec("bogus", 1),
                "mean_ttft_ms": _spec("max", FAKE_CRED),
                "p99_ttft_ms": {"kind": "max", "value": 1, "unit": FAKE_CRED},
            }
        }
        exc = self._refused(self._threshold_config(thresholds))
        self.assertEqual(
            [line.split(":", 1)[0] for line in exc.problems],
            [f"{CELL}.output_throughput.kind", f"{CELL}.mean_ttft_ms.value", f"{CELL}.p99_ttft_ms.unit"],
        )
        self.assertTrue(str(exc).startswith(f"{threshold_path} is not a valid config:"))
        self.assertNotIn("old config format", str(exc))
        self.assertNotIn(FAKE_CRED, str(exc))

    def test_a_schema_without_threshold_fields_reads_no_file_and_logs_nothing(self):
        self._write("configs/sibling_threshold.json", {CELL: {"output_throughput": OLD_KINDS["max_ms"]}})
        loaded = self._load({"name": "a"}, _Plain)
        self.assertEqual(loaded.config.model_dump(), {"name": "a"})
        self.log.warning.assert_not_called()

    def test_schema_validators_see_the_threshold_file(self):
        thresholds = {CELL: {"output_throughput": _spec()}}
        covered = self._threshold_config(thresholds, enforce_thresholds=True, runs=[CELL])
        self.assertEqual(self._load(covered, _Coverage).config.runs, [CELL])
        uncovered = self._threshold_config(thresholds, enforce_thresholds=True, runs=[CELL, "uncovered"])
        self.assertEqual(
            self._refused(uncovered, _Coverage).problems,
            ["(config): with enforce_thresholds, every run needs a threshold cell"],
        )

    def test_schema_validators_see_enforce_thresholds_false_when_recording_only(self):
        config = self._load({**VALID, "enforce_thresholds": True, "runs": [CELL]}, _Coverage).config
        self.assertEqual((config.thresholds, config.enforce_thresholds), (ThresholdFile(), False))

    def test_the_file_is_validated_as_the_schemas_own_thresholds_type(self):
        self._write("configs/gpu_threshold.json", {"_comment": "x", "mi355": {"pipe_s": 7.0}, "auto": {"pipe_s": 9}})
        loaded = self._load({"threshold_json": "gpu_threshold.json"}, _GpuSuite).config.thresholds
        self.assertIsInstance(loaded, _GpuFile)
        self.assertEqual(loaded.model_extra, {"mi355": {"pipe_s": 7.0}, "auto": {"pipe_s": 9.0}})
        self.assertEqual(self._load({"threshold_json": ""}, _GpuSuite).config.thresholds, _GpuFile())

    def test_problems_from_the_schemas_thresholds_type_name_the_threshold_file(self):
        for name, thresholds, schema, locations in (
            ("gpu_threshold.json", {"mi355": {"pipe_s": FAKE_CRED}}, _GpuSuite, ["mi355.pipe_s"]),
            ("two_threshold.json", {"mi355": {}, "auto": {}}, _OneSectionSuite, ["(config)"]),
        ):
            with self.subTest(schema=schema.__name__):
                path = self._write(f"configs/{name}", thresholds)
                exc = self._refused({"threshold_json": name}, schema)
                self.assertTrue(str(exc).startswith(f"{path} is not a valid config:"))
                self.assertEqual([line.split(":", 1)[0] for line in exc.problems], locations)
                self.assertNotIn(FAKE_CRED, str(exc))

    def test_a_non_string_name_or_a_non_object_config_is_left_for_the_schema_to_name(self):
        for config, location in (({"threshold_json": 5}, "threshold_json"), ([FAKE_CRED], "(config)")):
            with self.subTest(config=config):
                exc = self._refused(config, _GpuSuite)
                self.assertEqual([line.split(":", 1)[0] for line in exc.problems], [location])
                self.assertNotIn(FAKE_CRED, str(exc))

    def test_a_thresholds_schema_without_a_thresholds_field_is_a_type_error_naming_it(self):
        self._write("configs/demo_threshold.json", {})
        for name in ("", "demo_threshold.json"):
            with self.subTest(name=name):
                with self.assertRaises(TypeError) as ctx:
                    self._load({"threshold_json": name}, _NoThresholdsField)
                self.assertIn("_NoThresholdsField", str(ctx.exception))
        self.log.warning.assert_not_called()


class TestModeFromModule(unittest.TestCase):
    def test_suffixes_map_to_their_mode_and_the_longest_suffix_wins(self):
        for module, mode in (
            ("cvs.tests.inference.demo.demo_distributed", "distributed"),
            ("cvs.tests.inference.demo.demo_disagg_distributed", "disaggregated"),
        ):
            with self.subTest(module=module):
                self.assertEqual(mode_from_module(_Modes, module), mode)

    def test_no_match_falls_back_to_the_default_mode(self):
        self.assertEqual(mode_from_module(_ModesWithDefault, "cvs.tests.inference.demo.demo"), "distributed")

    def test_no_match_without_a_default_names_the_module_and_every_suffix(self):
        module = "cvs.tests.inference.demo.demo"
        with self.assertRaises(ValueError) as ctx:
            mode_from_module(_Modes, module)
        message = str(ctx.exception)
        self.assertIn(module, message)
        for suffix in _Modes.modes:
            self.assertIn(suffix, message)

    def test_a_schema_with_no_modes_has_its_default_mode_or_none(self):
        self.assertIsNone(mode_from_module(_NoModes, "cvs.tests.inference.demo.demo_single"))
        self.assertEqual(mode_from_module(_NoModesWithDefault, "cvs.tests.inference.demo.demo"), "single")


if __name__ == "__main__":
    unittest.main()
