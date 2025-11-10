#!/usr/bin/env python3
import argparse
import sys
import os
import pytest
import importlib.resources as resources

from cvs.input.generate.base import GeneratorPlugin, _discover_generators, _run_generator


class CVSExecutor(object):
    def __init__(self):
        self.test_map = self._discover_tests()

    def _discover_tests(self):
        """
        Dynamically discover all test files in the tests/ directory.
        Returns a dict mapping test names to their module paths.
        """
        test_map = {}
        # Get the directory where this script is located
        base_dir = os.path.dirname(__file__)
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

    def list_tests(self, test_name=None):
        if test_name:
            # List specific tests within a test file
            if test_name not in self.test_map:
                print(f"Error: Unknown test '{test_name}'")
                print("Use 'cvs list' to see available tests.")
                sys.exit(1)

            module_path = self.test_map[test_name]
            test_file = self._get_test_file(module_path)

            print(f"Available tests in {test_name}:")
            # Use pytest to collect tests, but add dummy arguments for required options
            pytest_args = [
                test_file,
                "--collect-only",
                "-q",
                "--cluster_file=dummy",  # Dummy value to satisfy argparse
                "--config_file=dummy"    # Dummy value to satisfy argparse
            ]
            pytest.main(pytest_args)
        else:
            # List all test files
            print("Available tests:")
            for test_name in sorted(self.test_map.keys()):
                print(f"  - {test_name}")

    def _get_test_file(self, module_path):
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

    def run_test(self, test_name, test_function, cluster_file, config_file, html, self_contained_html,
                 log_file, log_level, capture, extra_pytest_args):
        if test_name not in self.test_map:
            print(f"Error: Unknown test '{test_name}'")
            print("Use 'cvs list' to see available tests.")
            sys.exit(1)

        module_path = self.test_map[test_name]
        test_file = self._get_test_file(module_path)

        # Build pytest arguments
        if test_function:
            # Run specific test function
            test_target = f"{test_file}::{test_function}"
        else:
            # Run all tests in the file
            test_target = test_file

        pytest_args = [test_target]

        # Add CVS-specific arguments
        if cluster_file:
            pytest_args.append(f"--cluster_file={cluster_file}")
        elif "--collect-only" in extra_pytest_args:
            pytest_args.append("--cluster_file=dummy")

        if config_file:
            pytest_args.append(f"--config_file={config_file}")
        elif "--collect-only" in extra_pytest_args:
            pytest_args.append("--config_file=dummy")

        # Add pytest arguments
        if html:
            pytest_args.append(f"--html={html}")
            if self_contained_html:
                pytest_args.append("--self-contained-html")

        if log_file:
            pytest_args.append(f"--log-file={log_file}")

        if log_level:
            pytest_args.append(f"--log-level={log_level}")

        if capture:
            pytest_args.append(f"--capture={capture}")

        # Add any extra pytest args
        pytest_args.extend(extra_pytest_args)

        # Run pytest
        exit_code = pytest.main(pytest_args)
        sys.exit(exit_code)

def parse_arguments():
    # Special handling for generate command - parse manually if needed
    if len(sys.argv) >= 2 and sys.argv[1] == 'generate':
        # Manual parsing for generate command
        if len(sys.argv) == 2:
            # Just 'cvs generate'
            args = argparse.Namespace()
            args.command = 'generate'
            args.generator = None
            args.generator_args = []
            args.extra_pytest_args = []
            return args
        elif len(sys.argv) >= 3:
            # 'cvs generate <generator> [args...]'
            args = argparse.Namespace()
            args.command = 'generate'
            args.generator = sys.argv[2]
            args.generator_args = sys.argv[3:]
            args.extra_pytest_args = []
            return args
    
    # Normal parsing for other commands
    parser = argparse.ArgumentParser(
        description="Cluster Validation Suite (CVS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cvs list                           List all available test files
  cvs list agfhc_cvs                 List all tests in agfhc_cvs
  cvs run agfhc                      Run all tests in agfhc
  cvs run agfhc test_function        Run specific test function
  cvs run agfhc --html report.html   Run test and generate HTML report
  cvs run                            List all available test files
  cvs generate                       List available generators
  cvs generate cluster_json -h       Show help for cluster_json generator
  cvs generate cluster_json --input_hosts_file hosts.txt --username user --key_file key --output_json_file cluster.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available tests')
    list_parser.add_argument('test', nargs='?', help='Optional: specific test file to list tests from')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a specific test (wrapper over pytest)')
    run_parser.add_argument('test', nargs='?', help='Name of the test file to run (omit to list available tests)')
    run_parser.add_argument('function', nargs='?', help='Optional: specific test function to run')
    run_parser.add_argument('--cluster_file', help='Path to cluster configuration JSON file')
    run_parser.add_argument('--config_file', help='Path to test configuration JSON file')
    run_parser.add_argument('--html', help='Pytest: Create HTML report file at given path')
    run_parser.add_argument('--self-contained-html', action='store_true',
                          help='Pytest: Create a self-contained HTML file containing all the HTML report')
    run_parser.add_argument('--log-file', default='/tmp/test.log',
                          help='Pytest: Path to file for logging output (default: /tmp/test.log)')
    run_parser.add_argument('--log-level',
                          choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                          help='Pytest: Level of messages to catch/display')
    run_parser.add_argument('--capture',
                          choices=['no', 'tee-sys', 'tee-merged', 'fd', 'sys'],
                          help='Per-test capturing method for stdout/stderr')
    run_parser.epilog = """All pytest supported arguments will be accepted by 'cvs run'.
                           For the complete list of pytest options, run: pytest --help"""

    # Use parse_known_args to capture extra pytest args
    args, extra_pytest_args = parser.parse_known_args()
    args.extra_pytest_args = extra_pytest_args
    return args

def main():
    args = parse_arguments()

    executor = CVSExecutor()

    if args.command == 'list':
        executor.list_tests(args.test)
    elif args.command == 'run':
        if args.test is None:
            executor.list_tests()
        else:
            executor.run_test(
                args.test,
                args.function,
                args.cluster_file,
                args.config_file,
                args.html,
                args.self_contained_html,
                args.log_file,
                args.log_level,
                args.capture,
                args.extra_pytest_args
            )
    elif args.command == 'generate':
        generators = _discover_generators()
        if args.generator is None:
            # List available generators
            if generators:
                print("Available generators:")
                for name, plugin in sorted(generators.items()):
                    print(f"  {name} - {plugin.get_description()}")
            else:
                print("No generators found in cvs/generate/ directory.")
        else:
            # For generate command, move extra_pytest_args to generator_args
            if args.extra_pytest_args:
                args.generator_args.extend(args.extra_pytest_args)
                args.extra_pytest_args = []
            
            # Handle help requests for generators
            if args.generator_args and args.generator_args[0] in ['-h', '--help']:
                # Show help for the specific generator
                _run_generator(args.generator, ['-h'])
            else:
                # Run the specified generator
                _run_generator(args.generator, args.generator_args)
    else:
        # No command specified, show help
        parser = argparse.ArgumentParser()
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
