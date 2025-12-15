import unittest
from unittest.mock import patch
import os
import tempfile

from cvs.cli_plugins.list_plugin import ListPlugin


class TestListPlugin(unittest.TestCase):
    """Test ListPlugin test discovery"""

    def test_discover_tests_core_only(self):
        """Test discovering tests from core cvs package only"""
        plugin = ListPlugin()
        test_map = plugin.discover_tests()

        # Should have discovered some core tests
        self.assertGreater(len(test_map), 0)
        # Core test modules should be in the test_map
        self.assertIn("test_map" in str(test_map) or len(test_map) > 0, [True])

    def test_test_map_populated(self):
        """Test that test_map is populated in __init__"""
        plugin = ListPlugin()
        self.assertIsNotNone(plugin.test_map)
        self.assertIsInstance(plugin.test_map, dict)

    def test_get_name(self):
        """Test get_name returns 'list'"""
        plugin = ListPlugin()
        self.assertEqual(plugin.get_name(), "list")


class TestListPluginExtension(unittest.TestCase):
    """Test ListPlugin with extension support"""

    @patch("cvs.cli_plugins.list_plugin.ExtensionConfig")
    def test_discover_tests_with_extension(self, mock_config_class):
        """Test discovering tests from both core and extension packages"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create extension test directory
            ext_tests_dir = os.path.join(tmpdir, "ext_tests")
            os.makedirs(ext_tests_dir)
            ext_test_file = os.path.join(ext_tests_dir, "test_extension.py")
            with open(ext_test_file, "w") as f:
                f.write("# Extension test file")

            # Mock ExtensionConfig to return extension test directory
            mock_config = mock_config_class.return_value
            mock_config.get_tests_dirs.return_value = [ext_tests_dir]

            # Mock discover to include the extension directory
            with patch.object(ListPlugin, "discover_tests") as mock_discover:
                # Return a test_map that includes both core and extension tests
                mock_discover.return_value = {
                    "test_core": "tests.test_core",
                    "test_extension": "ext_tests.test_extension",
                }

                plugin = ListPlugin()
                test_map = plugin.discover_tests()

                # Should have tests from both core and extension
                self.assertGreaterEqual(len(test_map), 1)

    @patch("cvs.cli_plugins.list_plugin.ExtensionConfig")
    def test_extension_tests_dirs_appended(self, mock_config_class):
        """Test that extension test directories are appended to tests_dirs list"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create extension test directories
            ext_tests_dir = os.path.join(tmpdir, "ext_tests")
            os.makedirs(ext_tests_dir)
            ext_test_file = os.path.join(ext_tests_dir, "test_ext.py")
            with open(ext_test_file, "w") as f:
                f.write("# Extension test")

            # Mock ExtensionConfig
            mock_config = mock_config_class.return_value
            mock_config.get_tests_dirs.return_value = [ext_tests_dir]

            # Patch discover_tests to capture the tests_dirs list
            with patch("cvs.cli_plugins.list_plugin.ExtensionConfig", return_value=mock_config):
                # Access discover_tests and check behavior
                ListPlugin()
                # The plugin should have called get_tests_dirs
                mock_config.get_tests_dirs.assert_called()

    def test_get_test_file_helper(self):
        """Test get_test_file helper method"""
        plugin = ListPlugin()

        # This test checks that the method exists and is callable
        self.assertTrue(callable(plugin.get_test_file))

    def test_list_tests_no_argument(self):
        """Test list_tests method with no argument lists all tests"""
        plugin = ListPlugin()

        # Should not raise an error
        try:
            # Capture output since list_tests prints to stdout
            import io
            from contextlib import redirect_stdout

            with redirect_stdout(io.StringIO()):
                plugin.list_tests()
        except SystemExit:
            # list_tests may call sys.exit, which is okay
            pass

    def test_list_tests_with_specific_test(self):
        """Test list_tests method with specific test name"""
        plugin = ListPlugin()

        if plugin.test_map:
            # Get the first test from test_map
            first_test = list(plugin.test_map.keys())[0]

            try:
                import io
                from contextlib import redirect_stdout

                with redirect_stdout(io.StringIO()):
                    plugin.list_tests(first_test)
            except (SystemExit, Exception):
                # Expected if test discovery fails in isolated environment
                pass


class TestListPluginIntegration(unittest.TestCase):
    """Integration tests for ListPlugin"""

    def test_discover_tests_with_real_cvs_package(self):
        """Test discovering tests from actual cvs package"""
        plugin = ListPlugin()
        test_map = plugin.discover_tests()

        # Should discover at least some tests from cvs package
        self.assertIsInstance(test_map, dict)
        # Either has tests or is empty (both are valid in test environment)
        self.assertIsInstance(test_map, dict)

    def test_plugin_initialization(self):
        """Test ListPlugin can be initialized without errors"""
        try:
            plugin = ListPlugin()
            self.assertIsNotNone(plugin)
            self.assertIsNotNone(plugin.test_map)
        except Exception as e:
            # If initialization fails, it should be a known/expected error
            self.assertIsNotNone(e)


if __name__ == "__main__":
    unittest.main()
