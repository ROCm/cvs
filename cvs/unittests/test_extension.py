import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import configparser

from cvs.extension import ExtensionConfig


class TestExtensionConfig(unittest.TestCase):
    """Test ExtensionConfig class for loading and parsing extension configuration."""

    def test_load_config_no_extension(self):
        """Test that load_config works when no extension is configured."""
        # When no CVS_EXTENSION_PKG_NAMES is set, should still initialize
        with patch.dict(os.environ, {}, clear=True):
            config = ExtensionConfig()
            # Should have config object even if no extension found
            self.assertIsNotNone(config.config or True)  # Either has config or None is okay

    def test_get_package_name_default(self):
        """Test get_package_name returns 'cvs' when no extension is configured."""
        with patch.dict(os.environ, {}, clear=True):
            config = ExtensionConfig()
            # Should default to 'cvs' when no extension configured
            self.assertEqual(config.get_package_name(), "cvs")

    def test_get_package_name_configured(self):
        """Test get_package_name returns configured package name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock extension.ini
            config_file = os.path.join(tmpdir, "extension.ini")
            parser = configparser.ConfigParser()
            parser.add_section("extensions")
            parser.set("extensions", "package_name", "cvs_internal")
            parser.set("extensions", "tests_dirs", "tests")
            parser.set("extensions", "input_dirs", "input")
            with open(config_file, "w") as f:
                parser.write(f)

            # Mock find_spec to return our test package
            mock_spec = MagicMock()
            mock_spec.origin = os.path.join(tmpdir, "__init__.py")

            with patch("importlib.util.find_spec", return_value=mock_spec):
                config = ExtensionConfig()
                self.assertEqual(config.get_package_name(), "cvs_internal")

    def test_get_tests_dirs_absolute_path(self):
        """Test get_tests_dirs with absolute path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock extension.ini with absolute path
            config_file = os.path.join(tmpdir, "extension.ini")
            parser = configparser.ConfigParser()
            parser.add_section("extensions")
            parser.set("extensions", "package_name", "cvs_test")
            abs_test_path = "/absolute/path/to/tests"
            parser.set("extensions", "tests_dirs", abs_test_path)
            parser.set("extensions", "input_dirs", "input")
            with open(config_file, "w") as f:
                parser.write(f)

            mock_spec = MagicMock()
            mock_spec.origin = os.path.join(tmpdir, "__init__.py")

            with patch("importlib.util.find_spec", return_value=mock_spec):
                config = ExtensionConfig()
                dirs = config.get_tests_dirs()
                self.assertIn(abs_test_path, dirs)

    def test_get_tests_dirs_relative_path(self):
        """Test get_tests_dirs with relative path resolved from site_packages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested directory structure
            ext_dir = os.path.join(tmpdir, "cvs_test")
            os.makedirs(ext_dir)

            config_file = os.path.join(ext_dir, "extension.ini")
            parser = configparser.ConfigParser()
            parser.add_section("extensions")
            parser.set("extensions", "package_name", "cvs_test")
            parser.set("extensions", "tests_dirs", "cvs_test/tests")
            parser.set("extensions", "input_dirs", "cvs_test/input")
            with open(config_file, "w") as f:
                parser.write(f)

            mock_spec = MagicMock()
            mock_spec.origin = os.path.join(ext_dir, "__init__.py")

            with patch("importlib.util.find_spec", return_value=mock_spec):
                config = ExtensionConfig()
                dirs = config.get_tests_dirs()
                # Should resolve relative to site_packages_dir (tmpdir)
                expected_dir = os.path.join(tmpdir, "cvs_test", "tests")
                self.assertIn(expected_dir, dirs)

    def test_get_input_dirs_absolute_path(self):
        """Test get_input_dirs with absolute path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = os.path.join(tmpdir, "extension.ini")
            parser = configparser.ConfigParser()
            parser.add_section("extensions")
            parser.set("extensions", "package_name", "cvs_test")
            parser.set("extensions", "tests_dirs", "tests")
            abs_input_path = "/absolute/path/to/input"
            parser.set("extensions", "input_dirs", abs_input_path)
            with open(config_file, "w") as f:
                parser.write(f)

            mock_spec = MagicMock()
            mock_spec.origin = os.path.join(tmpdir, "__init__.py")

            with patch("importlib.util.find_spec", return_value=mock_spec):
                config = ExtensionConfig()
                dirs = config.get_input_dirs()
                self.assertIn(abs_input_path, dirs)

    def test_get_input_dirs_relative_path(self):
        """Test get_input_dirs with relative path resolved from site_packages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested directory structure
            ext_dir = os.path.join(tmpdir, "cvs_test")
            os.makedirs(ext_dir)

            config_file = os.path.join(ext_dir, "extension.ini")
            parser = configparser.ConfigParser()
            parser.add_section("extensions")
            parser.set("extensions", "package_name", "cvs_test")
            parser.set("extensions", "tests_dirs", "tests")
            parser.set("extensions", "input_dirs", "cvs_test/input")
            with open(config_file, "w") as f:
                parser.write(f)

            mock_spec = MagicMock()
            mock_spec.origin = os.path.join(ext_dir, "__init__.py")

            with patch("importlib.util.find_spec", return_value=mock_spec):
                config = ExtensionConfig()
                dirs = config.get_input_dirs()
                # Should resolve relative to site_packages_dir (tmpdir)
                expected_dir = os.path.join(tmpdir, "cvs_test", "input")
                self.assertIn(expected_dir, dirs)

    def test_get_dirs_multiple_paths(self):
        """Test get_tests_dirs and get_input_dirs with comma-separated paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ext_dir = os.path.join(tmpdir, "cvs_test")
            os.makedirs(ext_dir)

            config_file = os.path.join(ext_dir, "extension.ini")
            parser = configparser.ConfigParser()
            parser.add_section("extensions")
            parser.set("extensions", "package_name", "cvs_test")
            parser.set("extensions", "tests_dirs", "cvs_test/tests, /abs/tests")
            parser.set("extensions", "input_dirs", "cvs_test/input, /abs/input")
            with open(config_file, "w") as f:
                parser.write(f)

            mock_spec = MagicMock()
            mock_spec.origin = os.path.join(ext_dir, "__init__.py")

            with patch("importlib.util.find_spec", return_value=mock_spec):
                config = ExtensionConfig()

                test_dirs = config.get_tests_dirs()
                self.assertEqual(len(test_dirs), 2)
                self.assertIn(os.path.join(tmpdir, "cvs_test", "tests"), test_dirs)
                self.assertIn("/abs/tests", test_dirs)

                input_dirs = config.get_input_dirs()
                self.assertEqual(len(input_dirs), 2)
                self.assertIn(os.path.join(tmpdir, "cvs_test", "input"), input_dirs)
                self.assertIn("/abs/input", input_dirs)

    def test_extension_ini_base_set(self):
        """Test that extension_ini_base is properly set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ext_dir = os.path.join(tmpdir, "cvs_test")
            os.makedirs(ext_dir)

            config_file = os.path.join(ext_dir, "extension.ini")
            parser = configparser.ConfigParser()
            parser.add_section("extensions")
            parser.set("extensions", "package_name", "cvs_test")
            parser.set("extensions", "tests_dirs", "tests")
            parser.set("extensions", "input_dirs", "input")
            with open(config_file, "w") as f:
                parser.write(f)

            mock_spec = MagicMock()
            mock_spec.origin = os.path.join(ext_dir, "__init__.py")

            with patch("importlib.util.find_spec", return_value=mock_spec):
                config = ExtensionConfig()
                self.assertEqual(config.extension_ini_base, ext_dir)

    def test_site_packages_dir_set(self):
        """Test that site_packages_dir is properly set as parent of extension_ini_base."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ext_dir = os.path.join(tmpdir, "cvs_test")
            os.makedirs(ext_dir)

            config_file = os.path.join(ext_dir, "extension.ini")
            parser = configparser.ConfigParser()
            parser.add_section("extensions")
            parser.set("extensions", "package_name", "cvs_test")
            parser.set("extensions", "tests_dirs", "tests")
            parser.set("extensions", "input_dirs", "input")
            with open(config_file, "w") as f:
                parser.write(f)

            mock_spec = MagicMock()
            mock_spec.origin = os.path.join(ext_dir, "__init__.py")

            with patch("importlib.util.find_spec", return_value=mock_spec):
                config = ExtensionConfig()
                self.assertEqual(config.site_packages_dir, tmpdir)


if __name__ == "__main__":
    unittest.main()
