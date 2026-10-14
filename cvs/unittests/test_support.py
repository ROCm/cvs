'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs/unittests/support.py:
  - run_trusted runs an argv list and passes kwargs through to subprocess.run.
  - run_trusted refuses command strings and shell=True.
  - parse_trusted_xml returns the parsed tree of a test-generated file.
'''

import os
import sys
import tempfile
import unittest

from cvs.unittests.support import parse_trusted_xml, run_trusted


class TestRunTrusted(unittest.TestCase):
    def test_runs_argv_and_passes_kwargs_through(self):
        proc = run_trusted(
            [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read()); sys.exit(3)"],
            input="echoed",
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.stdout, "echoed")
        self.assertEqual(proc.returncode, 3)

    def test_rejects_command_string(self):
        with self.assertRaises(TypeError):
            run_trusted("echo hi")

    def test_rejects_shell_true(self):
        with self.assertRaises(ValueError):
            run_trusted(["echo", "hi"], shell=True)  # nosec B604 - asserts the helper rejects shell=True

    def test_shell_false_is_allowed(self):
        proc = run_trusted([sys.executable, "-c", "pass"], shell=False)
        self.assertEqual(proc.returncode, 0)


class TestParseTrustedXml(unittest.TestCase):
    def test_parses_generated_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.xml")
            with open(path, "w") as handle:
                handle.write('<testsuite tests="2"><testcase name="a"/><testcase name="b"/></testsuite>')
            root = parse_trusted_xml(path).getroot()
        self.assertEqual(root.tag, "testsuite")
        self.assertEqual(root.get("tests"), "2")
        self.assertEqual([case.get("name") for case in root], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
