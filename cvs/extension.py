"""
Extension configuration loader for cvs.

This module provides functionality to load and parse extension configuration
from extension.ini files in installed extension packages (e.g., cvs-extension).
It allows extensions to override default behavior such as package name,
and test discovery directories.
"""

import os
import configparser
import sys
import importlib.util


CORE_PKG_NAME = "cvs"
CORE_TESTS_DIR = "tests"


class ExtensionConfig:
    """Load and parse extension configuration from extension.ini files."""

    def __init__(self):
        """Initialize the extension config loader."""
        self.config = None
        self.extension_ini_base = None
        self.site_packages_dir = None
        self.load_config()

    def load_config(self):
        """
        Load extension.ini from the cvs package or from packages specified via environment variable.

        Search order:
        1. If CVS_EXTENSION_PKG_NAMES environment variable is set (comma-separated list):
           Look for extension.ini in each site-packages/<extension_pkg>/extension.ini
        2. Otherwise, look for extension.ini in the cvs package directory
           (placed there by extension package's setup.py)

        Also discovers the site-packages directory to resolve relative paths to sibling
        extension packages.
        """
        # Check for extension packages specified via environment variable (comma-separated)
        config_file = None
        extension_pkg_names = os.environ.get("CVS_EXTENSION_PKG_NAMES")
        if extension_pkg_names:
            # Try each package in order
            for pkg_name in extension_pkg_names.split(","):
                pkg_name = pkg_name.strip()
                if pkg_name:
                    config_file = self._find_config_in_package(pkg_name)
                    if config_file:
                        break

        # Fall back to looking in cvs package
        if not config_file:
            config_file = self._find_config_in_package(CORE_PKG_NAME)

        if config_file and os.path.exists(config_file):
            try:
                self.config = configparser.ConfigParser()
                self.config.read(config_file)
                self.extension_ini_base = os.path.dirname(config_file)
                # Determine site-packages directory (parent of extension package)
                self.site_packages_dir = os.path.dirname(self.extension_ini_base)
            except Exception as e:
                print(f"Warning: Could not parse extension config {config_file}: {e}", file=sys.stderr)

    def _find_config_in_package(self, package_name):
        """
        Find extension.ini in an installed package.

        Args:
            package_name (str): The name of the package to search (e.g., 'cvs', 'cvs_extension')

        Returns:
            str or None: Path to extension.ini if found, None otherwise
        """
        try:
            spec = importlib.util.find_spec(package_name)
            if spec and spec.origin:
                # Get the package directory
                pkg_dir = os.path.dirname(spec.origin)
                # Look for extension.ini in the package directory itself
                config_path = os.path.join(pkg_dir, "extension.ini")
                if os.path.exists(config_path):
                    return os.path.abspath(config_path)
        except Exception:
            pass
        return None

    def get_package_name(self):
        """
        Get the package name from config or default to 'cvs'.

        Returns:
            str: Package name to use (e.g., 'cvs' or 'cvs_extension')
        """
        if self.config and self.config.has_option("extensions", "package_name"):
            return self.config.get("extensions", "package_name")
        return CORE_PKG_NAME

    def get_tests_dirs(self):
        """
        Get list of test directories from config.

        Resolves relative paths from the site-packages directory, allowing
        references to sibling extension packages like cvs_extension/tests.

        Returns:
            list: List of tuples (module_path, absolute_path) where:
                  - module_path: Python import path (e.g., 'cvs_internal.tests')
                  - absolute_path: Filesystem path for walking directories
        """
        dirs = []
        if self.config and self.config.has_option("extensions", "tests_dirs"):
            tests_dirs_str = self.config.get("extensions", "tests_dirs")
            test_paths = [d.strip() for d in tests_dirs_str.split(",")]

            for test_path in test_paths:
                # Always treat as relative path: convert to module path and absolute path
                module_path = test_path.replace(os.sep, ".")
                abs_path = os.path.join(self.site_packages_dir, test_path)
                dirs.append((module_path, abs_path))
        return dirs

    def get_input_dirs(self):
        """
        Get list of input directories from config.

        Resolves relative paths from the site-packages directory, allowing
        references to sibling extension packages like cvs_extension/input.

        Returns:
            list: List of input directories (relative or absolute paths)
        """
        dirs = []
        if self.config and self.config.has_option("extensions", "input_dirs"):
            input_dirs_str = self.config.get("extensions", "input_dirs")
            input_paths = [d.strip() for d in input_dirs_str.split(",")]

            for input_path in input_paths:
                if os.path.isabs(input_path):
                    dirs.append(input_path)
                else:
                    # Resolve relative to site-packages directory
                    dirs.append(os.path.join(self.site_packages_dir, input_path))
        return dirs
