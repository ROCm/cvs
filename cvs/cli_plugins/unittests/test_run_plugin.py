import unittest
from unittest.mock import MagicMock, patch
import sys
import os

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


if __name__ == "__main__":
    unittest.main()
