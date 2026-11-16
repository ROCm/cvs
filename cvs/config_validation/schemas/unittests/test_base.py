'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.config_validation.schemas.base.
'''

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError, model_validator

from cvs.config_validation.schemas.base import (
    ConfigStr,
    OpenBlock,
    StrictBlock,
    SuiteConfig,
    drop_comment_keys,
    problem_lines,
)
from cvs.lib.conf_lib.config_blocks import PLACEHOLDER_MESSAGE
from cvs.lib.conf_lib.loader import ConfigError
from cvs.unittests.support import FAKE_CRED

REJECT = {"placeholders": "reject"}

# Forms cvs.lib.conf_lib.config_blocks treats as a placeholder, and near misses it does not.
PLACEHOLDER_FORMS = ("<CHANGEME>", "<changeme-models-mount>", "/data/<changeme>/models")
NOT_PLACEHOLDERS = ("changeme", "<changemenow>")


class _Strict(StrictBlock):
    name: str
    env: dict = {}


class _Open(OpenBlock):
    name: str


class _Text(StrictBlock):
    value: ConfigStr


class _Outer(StrictBlock):
    inner: _Text
    mounts: list[ConfigStr] = []


class _Suite(SuiteConfig):
    name: str
    count: int = 1


class _SuiteText(SuiteConfig):
    value: ConfigStr


class _ContextProbe(SuiteConfig):
    config_dir: str = ""
    context_keys: list = []

    @model_validator(mode="before")
    @classmethod
    def _from_context(cls, data, info):
        return {"config_dir": str(info.context["config_path"].parent), "context_keys": sorted(info.context)}


class _TokenError(Exception):
    pass


class _RaisesTokenError(SuiteConfig):
    @model_validator(mode="before")
    @classmethod
    def _refuse(cls, data):
        raise _TokenError("refused")


class TestDropCommentKeys(unittest.TestCase):
    def test_only_underscore_string_keys_at_the_top_level_are_dropped(self):
        value = {"_comment": "x", "_": 0, "name_": 1, "nested": {"_keep": 2}, 3: "int key"}
        self.assertEqual(drop_comment_keys(value), {"name_": 1, "nested": {"_keep": 2}, 3: "int key"})

    def test_non_dict_input_is_returned_unchanged_for_pydantic_to_reject(self):
        for value in (["_comment"], None):
            with self.subTest(value=value):
                self.assertIs(drop_comment_keys(value), value)


class TestProblemLines(unittest.TestCase):
    def _lines(self, model, data, context=None):
        with self.assertRaises(ValidationError) as ctx:
            model.model_validate(data, context=context)
        return problem_lines(ctx.exception)

    def test_each_line_names_the_dotted_location_and_never_the_value(self):
        secret = f"{FAKE_CRED}<changeme>"
        lines = self._lines(
            _Outer, {"inner": {"value": secret}, "mounts": ["/a:/a", secret], "token": FAKE_CRED}, context=REJECT
        )
        self.assertEqual(lines[:2], [f"inner.value: {PLACEHOLDER_MESSAGE}", f"mounts.1: {PLACEHOLDER_MESSAGE}"])
        self.assertEqual([line.split(":", 1)[0] for line in lines], ["inner.value", "mounts.1", "token"])
        for line in lines:
            self.assertNotIn(FAKE_CRED, line)

    def test_an_error_on_the_whole_input_is_located_at_config_level(self):
        (line,) = self._lines(_Text, [FAKE_CRED])
        self.assertTrue(line.startswith("(config): "))
        self.assertNotIn(FAKE_CRED, line)


class TestStrictBlock(unittest.TestCase):
    def test_comment_keys_are_dropped_at_its_own_level_but_kept_in_a_free_form_map(self):
        loaded = _Strict.model_validate({"_comment": "x", "_note": {"a": 1}, "name": "a", "env": {"_KEEP": "1"}})
        self.assertEqual(loaded.model_dump(), {"name": "a", "env": {"_KEEP": "1"}})


class TestOpenBlock(unittest.TestCase):
    def test_comment_keys_are_dropped_rather_than_kept_as_extras(self):
        loaded = _Open.model_validate({"name": "a", "_comment": "x", "_comment_mounts": ["y"]})
        self.assertEqual(loaded.model_extra, {})


class TestConfigStr(unittest.TestCase):
    def test_placeholder_forms_are_rejected_under_the_reject_context(self):
        for value in PLACEHOLDER_FORMS:
            with self.subTest(value=value):
                with self.assertRaises(ValidationError) as ctx:
                    _Text.model_validate({"value": value}, context=REJECT)
                (error,) = ctx.exception.errors(include_input=False)
                self.assertEqual(error["loc"], ("value",))
                self.assertIn("still a <changeme> placeholder", error["msg"])

    def test_placeholders_load_verbatim_without_the_reject_context(self):
        for context in (None, {"placeholders": "allow"}):
            for value in PLACEHOLDER_FORMS:
                with self.subTest(context=context, value=value):
                    self.assertEqual(_Text.model_validate({"value": value}, context=context).value, value)

    def test_near_misses_are_not_placeholders(self):
        for value in NOT_PLACEHOLDERS:
            with self.subTest(value=value):
                self.assertEqual(_Text.model_validate({"value": value}, context=REJECT).value, value)


class TestSuiteConfigHooks(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.config_file = self.tmp / "suite.json"

    def test_read_raw_returns_the_file_as_written(self):
        # The loader looks for _unmapped and <changeme> in the raw config, so nothing may be dropped or substituted.
        written = {
            "_comment": "notes",
            "_unmapped": {"old_key": 1},
            "name": "/home/{user-id}/<changeme>",
            "nested": {"_keep": True},
        }
        self.config_file.write_text(json.dumps(written))
        self.assertEqual(_Suite.read_raw(self.config_file, {"username": "jdoe"}), written)

    def test_old_keys_reports_none_by_default(self):
        self.assertEqual(_Suite.old_keys({"tensor_parallel_size": "4", "_unmapped": {"a": 1}}), [])

    def test_orch_inputs_is_the_raw_config_not_the_validated_dump(self):
        raw = {"_comment": "notes", "name": "a"}
        config = _Suite.validate_config(raw, self.config_file)
        self.assertEqual(_Suite.orch_inputs(raw, config, {"node_dict": {}}), {"_comment": "notes", "name": "a"})

    def test_validate_config_returns_the_model_with_the_config_path_in_context(self):
        config = _ContextProbe.validate_config({}, str(self.config_file))
        self.assertIsInstance(config, _ContextProbe)
        self.assertEqual(config.config_dir, str(self.tmp))
        self.assertEqual(config.context_keys, ["config_path"])

    def test_validate_config_does_not_reject_placeholders_by_default(self):
        self.assertEqual(_SuiteText.validate_config({"value": "<changeme>"}, self.config_file).value, "<changeme>")

    def test_validate_config_errors_name_each_field_and_never_its_value(self):
        raw = {"name": [FAKE_CRED], "count": FAKE_CRED, "token": FAKE_CRED}
        with self.assertRaises(ConfigError) as ctx:
            _Suite.validate_config(raw, self.config_file)
        exc = ctx.exception
        self.assertCountEqual([line.split(":", 1)[0] for line in exc.problems], ["name", "count", "token"])
        self.assertTrue(str(exc).startswith(f"{self.config_file} is not a valid config:\n  "))
        self.assertNotIn(FAKE_CRED, str(exc))
        # pydantic's ValidationError prints input values, so it must not ride along as the cause.
        self.assertIsNone(exc.__cause__)
        self.assertTrue(exc.__suppress_context__)

    def test_validate_config_lets_non_validation_errors_escape_unwrapped(self):
        with self.assertRaises(_TokenError):
            _RaisesTokenError.validate_config({}, self.config_file)


if __name__ == "__main__":
    unittest.main()
