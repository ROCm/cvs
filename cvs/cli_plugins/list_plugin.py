import os
import sys
import importlib.resources as resources

from .base import SubcommandPlugin

class ListPlugin(SubcommandPlugin):
    @staticmethod
    def discover_tests():
        """
        Dynamically discover all test files in the tests/ directory.
        Returns a dict mapping test names to their module paths.
        """
        test_map = {}
        # Get the directory where this script is located
        base_dir = os.path.dirname(os.path.dirname(__file__))
        tests_dir = os.path.join(base_dir, 'tests')

        if not os.path.exists(tests_dir):
            return test_map

        # Walk through tests directory
        for root, dirs, files in os.walk(tests_dir):
            for file in files:
                if file.endswith('.py') and file != '__init__.py':
                    # Get relative path from tests directory
                    rel_path = os.path.relpath(os.path.join(root, file), tests_dir)
                    # Convert to module path
                    module_parts = os.path.splitext(rel_path)[0].split(os.sep)
                    module_path = 'tests.' + '.'.join(module_parts)
                    # Use filename without .py as test name
                    test_name = os.path.splitext(file)[0]
                    test_map[test_name] = module_path

        return test_map

    @staticmethod
    def get_test_file(module_path):
        """Helper to get the test file path from module path."""
        try:
            # Get the package path for the test module
            module_parts = module_path.split('.')
            package = '.'.join(['cvs'] + module_parts[:-1])

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
        self.test_map = self.discover_tests()

    def list_tests(self, test_name=None):
        import re
        import pytest
        from io import StringIO
        import contextlib
        if test_name:
            # List specific tests within a test file
            if test_name not in self.test_map:
                print(f"Error: Unknown test '{test_name}'")
                print("Use 'cvs list' to see available tests.")
                sys.exit(1)

            module_path = self.test_map[test_name]
            test_file = self.get_test_file(module_path)

            # Use pytest to collect tests, but add dummy arguments for required options
            pytest_args = [
                test_file,
                "--collect-only",
                "-q",
                "--cluster_file=dummy",  # Dummy value to satisfy argparse
                "--config_file=dummy"    # Dummy value to satisfy argparse
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
            # List all test files
            print("Available tests:")
            for test_name in sorted(self.test_map.keys()):
                print(f"  - {test_name}")

    def get_name(self):
        return "list"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser('list', help='List available tests')
        parser.add_argument('test', nargs='?', help='Optional: specific test file to list tests from')
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
List Commands:
  cvs list                           List all available test files
  cvs list agfhc_cvs                 List all tests in agfhc_cvs"""

    def run(self, args):
        self.list_tests(args.test)
