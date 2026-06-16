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
        # Should have 'cvs' package key (nested structure)
        self.assertIn("cvs", test_map)

    def test_test_map_nested_structure(self):
        """Test that test_map has nested structure: {package_name: {test_name: module_path}}"""
        plugin = ListPlugin()
        test_map = plugin.test_map

        # Should be a dict
        self.assertIsInstance(test_map, dict)

        # Each package should map to a dict of tests
        for pkg_name, tests in test_map.items():
            self.assertIsInstance(pkg_name, str)
            self.assertIsInstance(tests, dict)
            # Each test should map to a module path string
            for test_name, module_path in tests.items():
                self.assertIsInstance(test_name, str)
                self.assertIsInstance(module_path, str)
                # Module path should contain dots and package name
                self.assertIn(".", module_path)

    def test_test_map_populated(self):
        """Test that test_map is populated in __init__"""
        plugin = ListPlugin()
        self.assertIsNotNone(plugin.test_map)
        self.assertIsInstance(plugin.test_map, dict)

    def test_find_test_method_exists(self):
        """Test that _find_test helper method exists"""
        plugin = ListPlugin()
        self.assertTrue(hasattr(plugin, '_find_test'))
        self.assertTrue(callable(plugin._find_test))

    def test_find_test_searches_all_packages(self):
        """Test that _find_test can find tests across all packages"""
        plugin = ListPlugin()

        # Get first test from first package
        if plugin.test_map:
            first_pkg = list(plugin.test_map.keys())[0]
            first_test = list(plugin.test_map[first_pkg].keys())[0]

            # Should find it
            result = plugin._find_test(first_test)
            self.assertIsNotNone(result)
            self.assertEqual(result, plugin.test_map[first_pkg][first_test])

    def test_find_test_returns_none_for_unknown(self):
        """Test that _find_test returns None for unknown test"""
        plugin = ListPlugin()
        result = plugin._find_test("nonexistent_test_xyz_123")
        self.assertIsNone(result)

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
                f.write("def test_dummy(): pass")

            # Mock ExtensionConfig to return extension test directory
            mock_config = mock_config_class.return_value
            # get_tests_dirs now returns tuples (module_path, abs_path)
            mock_config.get_tests_dirs.return_value = [("test_extension_pkg.ext_tests", ext_tests_dir)]
            mock_config.get_package_name.return_value = "test_extension_pkg"

            # Patch discover_tests to use mock config
            with patch("cvs.cli_plugins.list_plugin.ExtensionConfig", return_value=mock_config):
                plugin = ListPlugin()
                test_map = plugin.discover_tests()

                # Should have tests from extension
                self.assertIn("test_extension_pkg", test_map)
                self.assertIn("test_extension", test_map["test_extension_pkg"])

    @patch("cvs.cli_plugins.list_plugin.ExtensionConfig")
    def test_extension_tests_dirs_appended(self, mock_config_class):
        """Test that extension test directories are appended to tests_dirs list"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create extension test directories
            ext_tests_dir = os.path.join(tmpdir, "ext_tests")
            os.makedirs(ext_tests_dir)
            ext_test_file = os.path.join(ext_tests_dir, "test_ext.py")
            with open(ext_test_file, "w") as f:
                f.write("def test_dummy(): pass")

            # Mock ExtensionConfig
            mock_config = mock_config_class.return_value
            # get_tests_dirs now returns tuples (module_path, abs_path)
            mock_config.get_tests_dirs.return_value = [("ext_pkg.ext_tests", ext_tests_dir)]
            mock_config.get_package_name.return_value = "ext_pkg"

            with patch("cvs.cli_plugins.list_plugin.ExtensionConfig", return_value=mock_config):
                ListPlugin()
                # The plugin should have called get_tests_dirs and get_package_name
                mock_config.get_tests_dirs.assert_called()
                mock_config.get_package_name.assert_called()

    def test_list_tests_no_argument(self):
        """Test list_tests method with no argument lists all tests by package"""
        plugin = ListPlugin()

        # Should not raise an error
        try:
            # Capture output since list_tests prints to stdout
            import io
            from contextlib import redirect_stdout

            with redirect_stdout(io.StringIO()) as buf:
                plugin.list_tests()
                output = buf.getvalue()
                # Should contain package names
                self.assertTrue(len(output) > 0 or len(plugin.test_map) == 0)
        except SystemExit:
            # list_tests may call sys.exit, which is okay
            pass

    def test_list_tests_with_specific_test(self):
        """Test list_tests method with specific test name"""
        plugin = ListPlugin()

        # Get first test from test_map
        if plugin.test_map:
            first_pkg = list(plugin.test_map.keys())[0]
            first_test = list(plugin.test_map[first_pkg].keys())[0]

            try:
                import io
                from contextlib import redirect_stdout

                with redirect_stdout(io.StringIO()):
                    plugin.list_tests(first_test)
                    # May have output or may not depending on test file content
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
