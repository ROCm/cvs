import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from pkgutil import ModuleInfo
from importlib.machinery import FileFinder

# Add the parent directory to sys.path to import main
sys.path.insert(0, os.path.dirname(__file__))

import cvs.main as main


class TestMain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up shared test data"""
        cls.expected_ordered_plugins = ["copy-config", "generate", "list", "run", "monitor", "exec"]

    def test_get_version_success(self):
        """Test successful version retrieval"""
        # Read the expected version from version.txt
        version_file = os.path.join(os.path.dirname(__file__), "..", "..", "version.txt")
        with open(version_file) as f:
            expected_version = f.read().strip()

        version = main.get_version()
        # get_version() now returns 'cvs: <version>' format
        self.assertIn(f"cvs: {expected_version}", version)

    @patch("cvs.main.metadata.version", side_effect=main.metadata.PackageNotFoundError)
    def test_get_version_fallback_package_dir(self, mock_version):
        """Test version fallback when metadata fails - checks package dir first"""
        # Read the expected version from version.txt in package directory
        version_file = os.path.join(os.path.dirname(__file__), "..", "version.txt")
        # If it exists, it should be used
        if os.path.exists(version_file):
            with open(version_file) as f:
                expected_version = f.read().strip()
            version = main.get_version()
            self.assertIn(f"cvs: {expected_version}", version)

    @patch("cvs.main.metadata.version", side_effect=main.metadata.PackageNotFoundError)
    def test_get_version_fallback_parent_dir(self, mock_version):
        """Test version fallback to parent directory when package dir version.txt missing"""
        # This tests the fallback logic: tries package dir first, then parent
        version = main.get_version()
        # Should contain a version (either from parent dir or "unknown")
        self.assertIn("cvs:", version)
        # Version should not be empty
        self.assertTrue(len(version) > 5)

    def test_load_plugins_success(self):
        """Test successful plugin loading"""
        plugins = main.discover_plugins()
        self.assertIsInstance(plugins, list)
        # Ensure all expected plugins are loaded in the correct order
        plugin_names = [plugin.get_name() for plugin in plugins]
        self.assertEqual(plugin_names, self.expected_ordered_plugins)

    @patch("cvs.main.pkgutil.iter_modules")
    def test_partial_loading_of_plugins(self, mock_iter_modules):
        """Test plugin loading with partial import errors"""
        # Mock iter_modules to return only the plugins that should load
        mock_iter_modules.return_value = [
            ModuleInfo(FileFinder("/fake/path"), "generate_plugin", False),
            ModuleInfo(FileFinder("/fake/path"), "list_plugin", False),
            ModuleInfo(FileFinder("/fake/path"), "run_plugin", False),
        ]

        plugins = main.discover_plugins()

        # Should load only the plugins that were "found"
        plugin_names = [plugin.get_name() for plugin in plugins]
        expected_plugins = ["generate", "list", "run"]
        self.assertEqual(plugin_names, expected_plugins)

    def test_main_plugin_execution(self):
        """Test main function with plugin execution using real plugins with mocked run methods"""
        # Discover real plugins
        real_plugins = main.discover_plugins()

        # Check that all expected plugins are loaded
        plugin_names = [plugin.get_name() for plugin in real_plugins]
        self.assertEqual(plugin_names, self.expected_ordered_plugins)

        # Mock the run method for each plugin
        for plugin in real_plugins:
            plugin.run = MagicMock()

        # Test each plugin dispatch
        for plugin in real_plugins:
            with self.subTest(plugin_name=plugin.get_name()):
                # Reset all run mocks
                for p in real_plugins:
                    p.run.reset_mock()

                # Mock args to point to this plugin
                mock_args = MagicMock()
                mock_args._plugin = plugin

                with patch("cvs.main.build_arg_parser") as mock_build_parser:
                    mock_parser = MagicMock()
                    mock_parser.parse_known_args.return_value = (mock_args, [])
                    mock_build_parser.return_value = mock_parser

                    with patch("cvs.main.sys.argv", ["cvs", plugin.get_name()]):
                        main.main(plugins=real_plugins)

                        # Only this plugin's run method should be called
                        plugin.run.assert_called_once_with(mock_args)

                        # Other plugins' run methods should not be called
                        for other_plugin in real_plugins:
                            if other_plugin is not plugin:
                                other_plugin.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
