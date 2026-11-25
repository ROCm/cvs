'''Tests for the cvs consolidate-report command.'''

import argparse
import io
import logging
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from cvs.cli_plugins.consolidate_report_plugin import ConsolidateReportPlugin
from cvs.lib.report.consolidate.unittests.deck_fixtures import deck, write_run


class TestConsolidateReportPlugin(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.plugin = ConsolidateReportPlugin()
        basic_config = patch("cvs.cli_plugins.consolidate_report_plugin.logging.basicConfig")
        self.basic_config = basic_config.start()
        self.addCleanup(basic_config.stop)

    def parse(self, *argv):
        parser = argparse.ArgumentParser()
        self.plugin.get_parser(parser.add_subparsers(dest="command"))
        return parser.parse_args(["consolidate-report", *argv])

    def run_command(self, *argv):
        args = self.parse(*argv)
        with self.assertLogs(level="INFO") as logs:
            args._plugin.run(args)
        return "\n".join(logs.output)

    def run_failing_command(self, *argv):
        args = self.parse(*argv)
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            args._plugin.run(args)
        return raised.exception.code, stderr.getvalue()

    def test_name_and_examples(self):
        self.assertEqual(self.plugin.get_name(), "consolidate-report")
        self.assertIn("cvs consolidate-report --log-dir <dir1>,<dir2>", self.plugin.get_epilog())

    def test_log_dir_is_required(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            self.parse()
        self.assertEqual(raised.exception.code, 2)

    def test_one_directory_writes_the_zip_into_it(self):
        write_run(self.tmp / "ws", "rvs_cvs", deck("rvs_cvs"))
        output = self.run_command("--log-dir", str(self.tmp / "ws"))
        zips = list((self.tmp / "ws").glob("consolidated_report_*.zip"))
        self.assertEqual(len(zips), 1)
        self.assertIn("Collection verdict: all_passed", output)
        self.assertIn(str(zips[0]), output)
        self.basic_config.assert_called_once_with(level=logging.INFO, format="%(levelname)s: %(message)s")

    def test_several_directories_with_output(self):
        write_run(self.tmp / "health", "rvs_cvs", deck("rvs_cvs"))
        write_run(self.tmp / "network", "rccl_perf", deck("rccl", "record", "series"))
        log_dirs = "%s,%s" % (self.tmp / "health", self.tmp / "network")
        self.run_command("--log-dir", log_dirs, "--output", str(self.tmp / "out"))
        self.assertEqual(len(list((self.tmp / "out").glob("consolidated_report_*.zip"))), 1)

    def test_several_directories_default_to_the_current_directory(self):
        write_run(self.tmp / "health", "rvs_cvs", deck("rvs_cvs"))
        write_run(self.tmp / "network", "rccl_perf", deck("rccl", "record", "series"))
        (self.tmp / "cwd").mkdir()
        with patch("pathlib.Path.cwd", return_value=self.tmp / "cwd"):
            self.run_command("--log-dir", "%s,%s" % (self.tmp / "health", self.tmp / "network"))
        self.assertEqual(len(list((self.tmp / "cwd").glob("consolidated_report_*.zip"))), 1)

    def test_missing_directory_exits_1(self):
        code, stderr = self.run_failing_command("--log-dir", str(self.tmp / "missing"))
        self.assertEqual(code, 1)
        self.assertIn("Error: Directory not found", stderr)

    def test_empty_directory_exits_1(self):
        (self.tmp / "empty").mkdir()
        code, stderr = self.run_failing_command("--log-dir", str(self.tmp / "empty"))
        self.assertEqual(code, 1)
        self.assertIn("Error: No Run Decks found under", stderr)


if __name__ == "__main__":
    unittest.main()
