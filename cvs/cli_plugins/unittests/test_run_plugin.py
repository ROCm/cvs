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


if __name__ == "__main__":
    unittest.main()
