import argparse
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import tempfile
import json

# Add the parent directory to sys.path to import cli_plugins
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cvs.cli_plugins.run_plugin import RunPlugin


class TestRunPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = RunPlugin()
        # run_test() now resolves the run layout before handing off to pytest.
        # Stub it so these tests create no directories and stay independent of
        # the ambient scheduler environment.
        patcher = patch("cvs.cli_plugins.run_plugin.RunLayout")
        patcher.start()
        self.addCleanup(patcher.stop)

    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    def test_run_test_single_function(self, mock_exit, mock_pytest_main):
        """Test running a single test function"""
        args = MagicMock()
        args.test = "agfhc_cvs"
        args.function = ["test_func"]
        args.cluster_file = "/path/to/cluster.json"
        args.config_file = "/path/to/config.json"
        args.html = None
        args.self_contained_html = False
        args.log_file = "/tmp/test.log"
        args.log_level = None
        args.capture = "tee-sys"
        args.extra_pytest_args = []

        mock_pytest_main.return_value = 0  # Mock successful pytest run

        with patch.object(self.plugin, "get_test_file", return_value="/mock/path/test.py"):
            with patch.object(self.plugin, "_validate_json_config"):
                self.plugin.run(args)

        # Verify pytest.main was called with correct arguments
        expected_args = [
            "/mock/path/test.py::test_func",
            "--cluster_file=/path/to/cluster.json",
            "--config_file=/path/to/config.json",
            "--log-file=/tmp/test.log",
            "--capture=tee-sys",
        ]
        mock_pytest_main.assert_called_once_with(expected_args)
        mock_exit.assert_called_once_with(0)

    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    def test_run_test_multiple_functions(self, mock_exit, mock_pytest_main):
        """Test running multiple test functions"""
        args = MagicMock()
        args.test = "agfhc_cvs"
        args.function = ["test_func1", "test_func2", "test_func3"]
        args.cluster_file = "/path/to/cluster.json"
        args.config_file = "/path/to/config.json"
        args.html = None
        args.self_contained_html = False
        args.log_file = "/tmp/test.log"
        args.log_level = None
        args.capture = "tee-sys"
        args.extra_pytest_args = []

        mock_pytest_main.return_value = 0

        with patch.object(self.plugin, "get_test_file", return_value="/mock/path/test.py"):
            with patch.object(self.plugin, "_validate_json_config"):
                self.plugin.run(args)

        # Verify pytest.main was called with multiple function targets
        expected_args = [
            "/mock/path/test.py::test_func1",
            "/mock/path/test.py::test_func2",
            "/mock/path/test.py::test_func3",
            "--cluster_file=/path/to/cluster.json",
            "--config_file=/path/to/config.json",
            "--log-file=/tmp/test.log",
            "--capture=tee-sys",
        ]
        mock_pytest_main.assert_called_once_with(expected_args)
        mock_exit.assert_called_once_with(0)

    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    def test_run_test_omits_log_file_when_not_set(self, mock_exit, mock_pytest_main):
        """No --log-file is passed to pytest when the user does not request file logging."""
        args = MagicMock()
        args.test = "agfhc_cvs"
        args.function = []
        args.cluster_file = "/path/to/cluster.json"
        args.config_file = "/path/to/config.json"
        args.html = None
        args.self_contained_html = False
        args.log_file = None
        args.log_level = None
        args.capture = None
        args.extra_pytest_args = []

        mock_pytest_main.return_value = 0

        with patch.object(self.plugin, "get_test_file", return_value="/mock/path/test.py"):
            with patch.object(self.plugin, "_validate_json_config"):
                self.plugin.run(args)

        expected_args = [
            "/mock/path/test.py",
            "--cluster_file=/path/to/cluster.json",
            "--config_file=/path/to/config.json",
        ]
        mock_pytest_main.assert_called_once_with(expected_args)
        mock_exit.assert_called_once_with(0)


class TestRunPluginJsonValidation(unittest.TestCase):
    """Tests for RunPlugin._validate_json_config pre-flight checks."""

    def setUp(self):
        self.plugin = RunPlugin()

    @patch("cvs.cli_plugins.run_plugin.sys.exit", side_effect=SystemExit(1))
    @patch("cvs.cli_plugins.run_plugin.print")
    def test_missing_file(self, mock_print, mock_exit):
        """A missing config file should print a clean error and exit."""
        with self.assertRaises(SystemExit) as ctx:
            self.plugin._validate_json_config("/nonexistent/path.json", "--cluster_file")
        self.assertEqual(ctx.exception.code, 1)
        printed = " ".join(str(c) for c in mock_print.call_args_list[0][0])
        self.assertIn("does not exist", printed)
        self.assertIn("/nonexistent/path.json", printed)

    @patch("cvs.cli_plugins.run_plugin.sys.exit", side_effect=SystemExit(1))
    @patch("cvs.cli_plugins.run_plugin.print")
    def test_malformed_json(self, mock_print, mock_exit):
        """A malformed JSON file should print a clean error and exit."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{not valid json")
            path = f.name
        try:
            with self.assertRaises(SystemExit) as ctx:
                self.plugin._validate_json_config(path, "--config_file")
        finally:
            os.unlink(path)
        self.assertEqual(ctx.exception.code, 1)
        messages = " ".join(str(c[0][0]) for c in mock_print.call_args_list)
        self.assertIn("is not valid JSON", messages)
        self.assertIn(path, messages)

    def test_valid_json(self):
        """A valid JSON file should pass validation without exiting."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"host": {}}, f)
            path = f.name
        try:
            self.plugin._validate_json_config(path, "--config_file")
        finally:
            os.unlink(path)


class TestRunPluginWorkspace(unittest.TestCase):
    """--workspace and the RunLayout handoff.

    Worker ranks in a Slurm/Spur job never enter pytest, so the run layout has
    to be resolved by the CLI before pytest.main() is reached.
    """

    def setUp(self):
        self.plugin = RunPlugin()

    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        self.plugin.get_parser(subparsers)
        return parser.parse_args(["run", "health", "--cluster_file", "c.json", "--config_file", "f.json"] + extra)

    def test_parser_accepts_workspace(self):
        self.assertEqual(self._parse(["--workspace", "/shared/ws"]).workspace, "/shared/ws")

    def test_workspace_defaults_to_none(self):
        self.assertIsNone(self._parse([]).workspace)

    def _run_with_workspace(self, workspace, mock_pytest_main, test_name="agfhc_cvs"):
        args = MagicMock()
        args.test = test_name
        args.function = []
        args.cluster_file = "/path/to/cluster.json"
        args.config_file = "/path/to/config.json"
        args.html = None
        args.self_contained_html = False
        args.log_file = None
        args.log_level = None
        args.capture = None
        args.extra_pytest_args = []
        args.workspace = workspace
        mock_pytest_main.return_value = 0
        with patch.object(self.plugin, "get_test_file", return_value="/mock/path/test.py"):
            with patch.object(self.plugin, "_validate_json_config"):
                self.plugin.run(args)

    @patch("cvs.cli_plugins.run_plugin.RunLayout")
    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    def test_layout_resolved_with_workspace(self, mock_exit, mock_pytest_main, mock_layout):
        self._run_with_workspace("/shared/ws", mock_pytest_main)
        mock_layout.get.assert_called_once_with("/shared/ws")

    @patch("cvs.cli_plugins.run_plugin.RunLayout")
    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    def test_layout_resolved_with_none_when_not_given(self, mock_exit, mock_pytest_main, mock_layout):
        self._run_with_workspace(None, mock_pytest_main)
        mock_layout.get.assert_called_once_with(None)

    @patch("cvs.cli_plugins.run_plugin.RunLayout")
    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    def test_layout_resolved_before_pytest_runs(self, mock_exit, mock_pytest_main, mock_layout):
        # Ordering is the whole point: the layout must be resolved and the agent
        # directory must exist before any fixture or agent looks for them.
        manager = MagicMock()
        manager.attach_mock(mock_layout.get, "get_layout")
        manager.attach_mock(mock_pytest_main, "pytest_main")
        self._run_with_workspace("/shared/ws", mock_pytest_main)
        called = [name for name, _args, _kwargs in manager.mock_calls]
        self.assertLess(called.index("get_layout"), called.index("pytest_main"))

    @patch("cvs.cli_plugins.run_plugin.RunLayout")
    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    def test_workspace_is_not_forwarded_to_pytest(self, mock_exit, mock_pytest_main, mock_layout):
        # The layout reaches suites through RunLayout, not as a pytest option.
        # Matched on substring rather than one exact literal, so forwarding it as
        # a separate ["--workspace", value] pair is caught too.
        self._run_with_workspace("/shared/ws", mock_pytest_main)
        forwarded = mock_pytest_main.call_args[0][0]
        self.assertEqual([arg for arg in forwarded if "workspace" in arg], [])

    @patch("cvs.cli_plugins.run_plugin.RunLayout")
    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    def test_unknown_test_creates_no_run_directory(self, mock_pytest_main, mock_layout):
        # The workspace is shared storage on a real cluster, so a mistyped suite
        # name must not leave an empty run tree behind on it. sys.exit has to
        # raise here as it really does, or the early return does not happen.
        with patch("cvs.cli_plugins.run_plugin.sys.exit", side_effect=SystemExit(1)):
            with self.assertRaises(SystemExit):
                self._run_with_workspace("/shared/ws", mock_pytest_main, test_name="no_such_suite_xyz")
        mock_layout.get.assert_not_called()
        mock_pytest_main.assert_not_called()

    @patch("cvs.cli_plugins.run_plugin.RunLayout")
    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    @patch("cvs.cli_plugins.run_plugin.print")
    def test_unusable_workspace_exits_cleanly(self, mock_print, mock_pytest_main, mock_layout):
        # RunLayout raises RuntimeError for an unwritable or non-venv workspace.
        # The user should get that message, not a traceback, and pytest must not run.
        mock_layout.get.side_effect = RuntimeError("workspace is not writable")
        with patch("cvs.cli_plugins.run_plugin.sys.exit", side_effect=SystemExit(1)) as mock_exit:
            with self.assertRaises(SystemExit):
                self._run_with_workspace("/shared/ws", mock_pytest_main)
        mock_exit.assert_called_once_with(1)
        mock_pytest_main.assert_not_called()
        printed = " ".join(str(c) for call in mock_print.call_args_list for c in call[0])
        self.assertIn("workspace is not writable", printed)


if __name__ == "__main__":
    unittest.main()
