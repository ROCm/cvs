import pytest
import sys

from .list_plugin import ListPlugin

class RunPlugin(ListPlugin):
    def get_name(self):
        return "run"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser('run', help='Run a specific test (wrapper over pytest)')
        parser.add_argument('test', nargs='?', help='Name of the test file to run (omit to list available tests)')
        parser.add_argument('function', nargs='?', help='Optional: specific test function to run')
        parser.add_argument('--cluster_file', help='Path to cluster configuration JSON file')
        parser.add_argument('--config_file', help='Path to test configuration JSON file')
        parser.add_argument('--html', help='Pytest: Create HTML report file at given path')
        parser.add_argument('--self-contained-html', action='store_true',
                          help='Pytest: Create a self-contained HTML file containing all the HTML report')
        parser.add_argument('--log-file', default='/tmp/test.log',
                          help='Pytest: Path to file for logging output (default: /tmp/test.log)')
        parser.add_argument('--log-level',
                          choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                          help='Pytest: Level of messages to catch/display')
        parser.add_argument('--capture',
                          choices=['no', 'tee-sys', 'tee-merged', 'fd', 'sys'],
                          help='Per-test capturing method for stdout/stderr')
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
Run Commands:
  cvs run agfhc                      Run all tests in agfhc
  cvs run agfhc test_function        Run specific test function
  cvs run agfhc --html report.html   Run test and generate HTML report
  cvs run                            List all available test files"""

    def run(self, args):
        if args.test is None:
            self.list_tests()
        else:
            self.run_test(
                args.test,
                args.function,
                args.cluster_file,
                args.config_file,
                args.html,
                args.self_contained_html,
                args.log_file,
                args.log_level,
                args.capture,
                getattr(args, 'extra_pytest_args', [])
            )

    def run_test(self, test_name, test_function, cluster_file, config_file, html, self_contained_html,
                 log_file, log_level, capture, extra_pytest_args):
        if test_name not in self.test_map:
            print(f"Error: Unknown test '{test_name}'")
            print("Use 'cvs list' to see available tests.")
            sys.exit(1)

        module_path = self.test_map[test_name]
        test_file = self.get_test_file(module_path)

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

        # If --collect-only is in args, delegate to list_tests for consistent output
        if '--collect-only' in pytest_args:
            self.list_tests(test_name)
            sys.exit(0)
        else:
            # Run pytest normally
            exit_code = pytest.main(pytest_args)
            sys.exit(exit_code)
