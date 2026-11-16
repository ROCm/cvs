"""Unit tests for the ANC exe_path validator."""

import contextlib
import importlib.util
import io
import os
import tempfile
import unittest


def _load_validator():
    script = os.path.join(os.path.dirname(__file__), "..", "resources", "validate_exe_paths.py")
    spec = importlib.util.spec_from_file_location("validate_exe_paths", os.path.abspath(script))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()


def _write_yaml(content_dir, tool, filename, exe_path):
    exe_dir = os.path.join(content_dir, tool, "exe")
    os.makedirs(exe_dir, exist_ok=True)
    text = VALIDATOR.yaml.safe_dump({"exe_path": exe_path})
    with open(os.path.join(exe_dir, filename), "w", encoding="utf-8") as handle:
        handle.write(text)


def _run(content_dir):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = VALIDATOR.validate(content_dir)
    return code, buf.getvalue().splitlines()


def _lines(output, prefix):
    return [line for line in output if line.startswith(prefix)]


class TestValidateExePaths(unittest.TestCase):
    def assert_passed(self, code, output):
        self.assertEqual(code, 0)
        self.assertIn("VALIDATION_SUCCESS", output)
        self.assertNotIn("VALIDATION_FAILED", output)

    def assert_failed(self, code, output):
        self.assertEqual(code, 1)
        self.assertIn("VALIDATION_FAILED", output)
        self.assertNotIn("VALIDATION_SUCCESS", output)

    def test_existing_directory_is_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            present = os.path.join(tmp, "sbin")
            os.makedirs(present)
            _write_yaml(tmp, "ainic-nda", "ethtool.yml", present)
            code, output = _run(tmp)
        self.assert_passed(code, output)
        self.assertEqual(_lines(output, "OK:"), [f"OK: {present} (from ethtool.yml)"])

    def test_missing_directory_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "no-such-dir")
            _write_yaml(tmp, "ainic-nda", "ethtool.yml", missing)
            code, output = _run(tmp)
        self.assert_failed(code, output)
        self.assertEqual(_lines(output, "MISSING:"), [f"MISSING: {missing} (from ethtool.yml)"])

    def test_list_is_ok_when_any_directory_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            absent = os.path.join(tmp, "bin")
            present = os.path.join(tmp, "sbin")
            os.makedirs(present)
            _write_yaml(tmp, "ainic-nda", "lspci.yml", [absent, present])
            code, output = _run(tmp)
        self.assert_passed(code, output)
        self.assertEqual(_lines(output, "OK:"), [f"OK: {present} (from lspci.yml)"])

    def test_list_is_missing_when_no_directory_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "bin")
            second = os.path.join(tmp, "sbin")
            _write_yaml(tmp, "ainic-nda", "lspci.yml", [first, second])
            code, output = _run(tmp)
        self.assert_failed(code, output)
        self.assertEqual(_lines(output, "MISSING:"), [f"MISSING: {first}, {second} (from lspci.yml)"])

    def test_null_exe_path_is_invalid_and_scan_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            present = os.path.join(tmp, "sbin")
            os.makedirs(present)
            _write_yaml(tmp, "ainic-nda", "ethtool.yml", present)
            _write_yaml(tmp, "ainic-nda", "lspci.yml", None)
            code, output = _run(tmp)
        self.assert_failed(code, output)
        self.assertEqual(_lines(output, "OK:"), [f"OK: {present} (from ethtool.yml)"])
        self.assertEqual(
            _lines(output, "INVALID:"),
            ["INVALID: exe_path is NoneType, expected a path string or list of paths: None (from lspci.yml)"],
        )

    def test_no_exe_directory_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, output = _run(tmp)
        self.assertEqual(code, 2)
        self.assertEqual(output, [f"ERROR: No exe directories found under {tmp}"])
