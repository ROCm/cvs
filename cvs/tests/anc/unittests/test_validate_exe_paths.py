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


def _write_yaml(content_dir, tool, filename, body):
    exe_dir = os.path.join(content_dir, tool, "exe")
    os.makedirs(exe_dir, exist_ok=True)
    with open(os.path.join(exe_dir, filename), "w", encoding="utf-8") as handle:
        handle.write(body)


def _run(content_dir):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = VALIDATOR.validate(content_dir)
    return code, buf.getvalue()


class TestValidateExePaths(unittest.TestCase):
    # TODO: exe_path: null

    def test_string_path_that_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            present = os.path.join(tmp, "sbin")
            os.makedirs(present)
            _write_yaml(tmp, "ainic-nda", "ethtool.yml", f"exe_path: '{present}'\n")
            code, output = _run(os.path.join(tmp))
        self.assertEqual(code, 0)
        self.assertIn(f"OK: {present} (from ethtool.yml)", output)
        self.assertIn("VALIDATION_SUCCESS", output)

    def test_string_path_that_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "no-such-dir")
            _write_yaml(tmp, "ainic-nda", "ethtool.yml", f"exe_path: '{missing}'\n")
            code, output = _run(tmp)
        self.assertEqual(code, 1)
        self.assertIn(f"MISSING: {missing} (from ethtool.yml)", output)
        self.assertIn("VALIDATION_FAILED", output)

    def test_list_passes_when_any_directory_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            present = os.path.join(tmp, "bin")
            absent = os.path.join(tmp, "sbin")
            os.makedirs(present)
            _write_yaml(
                tmp,
                "ainic-nda",
                "lspci.yml",
                f"exe_path: ['{absent}', '{present}']\n",
            )
            code, output = _run(tmp)
        self.assertEqual(code, 0)
        self.assertIn(f"OK: {present} (from lspci.yml)", output)
        self.assertNotIn("MISSING", output)
        self.assertIn("VALIDATION_SUCCESS", output)

    def test_list_fails_when_no_directory_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "bin")
            second = os.path.join(tmp, "sbin")
            _write_yaml(
                tmp,
                "ainic-nda",
                "lspci.yml",
                f"exe_path: ['{first}', '{second}']\n",
            )
            code, output = _run(tmp)
        self.assertEqual(code, 1)
        self.assertIn(f"MISSING: {first}, {second} (from lspci.yml)", output)
        self.assertIn("VALIDATION_FAILED", output)

    def test_non_string_list_entry_is_reported_and_scan_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            present = os.path.join(tmp, "sbin")
            os.makedirs(present)
            _write_yaml(tmp, "ainic-nda", "ethtool.yml", f"exe_path: '{present}'\n")
            _write_yaml(tmp, "ainic-nda", "lspci.yml", "exe_path: ['/usr/bin', 1]\n")
            code, output = _run(tmp)
        self.assertEqual(code, 1)
        self.assertIn("OK: ", output)
        self.assertIn("INVALID: exe_path[1] is int, expected a path string: 1 (from lspci.yml)", output)
        self.assertIn("VALIDATION_FAILED", output)
        self.assertNotIn("Traceback", output)
