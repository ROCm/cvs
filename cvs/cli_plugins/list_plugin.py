import os
import sys
import importlib.resources as resources
import re
import pytest
from io import StringIO
import contextlib

from .base import SubcommandPlugin
from cvs.extension import ExtensionConfig, CORE_PKG_NAME, CORE_TESTS_DIR


class ListPlugin(SubcommandPlugin):
    @staticmethod
    def discover_tests():
        """
        Dynamically discover all test files organized by package.
        Returns a nested dict: {package_name: {test_name: module_path}}

        Supports extension packages (e.g., cvs-extension) that provide additional test directories
        via extension.ini configuration.
        """
        test_map = {}
        config = ExtensionConfig()

        # Collect all test directories: core + extensions
        all_tests_dirs = []

        # Core CVS tests directory
        base_dir = os.path.dirname(os.path.dirname(__file__))
        cvs_tests_dir = os.path.join(base_dir, CORE_TESTS_DIR)
        if os.path.exists(cvs_tests_dir):
            # For core CVS, convert path to module path
            cvs_tests_path = f"{CORE_PKG_NAME}.{CORE_TESTS_DIR}"
            all_tests_dirs.append((CORE_PKG_NAME, cvs_tests_path, cvs_tests_dir))

        # Extension tests directories - get_tests_dirs now returns tuples (module_path, abs_path)
        for module_path, abs_path in config.get_tests_dirs():
            if os.path.exists(abs_path):
                all_tests_dirs.append((config.get_package_name(), module_path, abs_path))

        # Discover tests from all directories
        for pkg_name, tests_path, tests_dir in all_tests_dirs:
            test_map[pkg_name] = {}
            for root, dirs, files in os.walk(tests_dir):
                for file in files:
                    if file.endswith(".py") and file != "__init__.py":
                        rel_path = os.path.relpath(os.path.join(root, file), tests_dir)
                        module_parts = os.path.splitext(rel_path)[0].split(os.sep)
                        # Module path: <tests_path>.<test_name>
                        module_path = f"{tests_path}." + ".".join(module_parts)
                        test_name = os.path.splitext(file)[0]
                        test_map[pkg_name][test_name] = module_path

        return test_map

    @staticmethod
    def get_test_file(module_path):
        """Helper to get the test file path from module path."""
        try:
            # Module path is already correct (e.g., cvs.tests.<category>.<test> or extension_package.tests.<test>)
            # Just split and use it directly
            module_parts = module_path.split(".")
            package = ".".join(module_parts[:-1])

            # Try to locate the test file
            test_file = None
            try:
                # For Python 3.9+
                files = resources.files(package)
                test_file = str(files / f"{module_parts[-1]}.py")
            except AttributeError:
                # Fallback for older Python versions
                with resources.path(package, f"{module_parts[-1]}.py") as p:
                    test_file = str(p)
            return test_file
        except Exception as e:
            print(f"Error locating test file: {e}")
            sys.exit(1)

    def __init__(self):
        self.test_map = self.discover_tests()  # Nested: {pkg_name: {test_name: module_path}}

    def _find_test(self, test_name):
        """Find test module path by name across all packages."""
        for pkg_name, tests in self.test_map.items():
            if test_name in tests:
                return tests[test_name]
        return None

    def list_tests(self, test_name=None):
        if test_name:
            # List specific tests within a test file
            module_path = self._find_test(test_name)
            if not module_path:
                print(f"Error: Unknown test '{test_name}'")
                print("Use 'cvs list' to see available tests.")
                sys.exit(1)

            test_file = self.get_test_file(module_path)

            # Use pytest to collect tests, but add dummy arguments for required options
            pytest_args = [
                test_file,
                "--collect-only",
                "-q",
                "--cluster_file=dummy",  # Dummy value to satisfy argparse
                "--config_file=dummy",  # Dummy value to satisfy argparse
            ]
            # Capture pytest output
            buf = StringIO()
            with contextlib.redirect_stdout(buf):
                pytest.main(pytest_args)
            output = buf.getvalue()
            # Parse output for test functions
            test_rows = []
            for line in output.splitlines():
                m = re.match(r"(.+\.py)::(test_\w+)", line.strip())
                if m:
                    test_file_path, test_func = m.groups()
                    test_rows.append(test_func)
            print(f"\nAvailable tests in {test_name}:")
            for func in test_rows:
                print(f"  - {func}")
            if not test_rows:
                print(output)
        else:
            # List all test files, categorized by package
            print("Available tests:")
            for pkg_name in sorted(self.test_map.keys()):
                print(f"{pkg_name}:")
                for test_name in sorted(self.test_map[pkg_name].keys()):
                    print(f"  - {test_name}")
                print()  # Blank line between packages

    def get_name(self):
        return "list"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser("list", help="List available tests")
        parser.add_argument("test", nargs="?", help="Optional: specific test file to list tests from")
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
List Commands:
  cvs list                           List all available test files
  cvs list agfhc_cvs                 List all tests in agfhc_cvs"""

    def run(self, args):
        self.list_tests(args.test)
