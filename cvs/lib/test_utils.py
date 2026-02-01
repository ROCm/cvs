'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Test Utilities for CVS Framework
=================================
This module provides utilities for running multiple test suites sequentially
with per-suite customization and comprehensive result tracking.
'''

import os
import glob
import sys
import inspect
import pytest


class CVSTestSuiteRunner:
    """
    Orchestrates sequential execution of multiple test suites with per-suite customization.

    This class provides a framework for discovering test modules, filtering/customizing
    CLI arguments, running individual test suites, and generating summary reports.

    Typical usage (simple - auto-configured):
        runner = CVSTestSuiteRunner(log, pytestconfig, test_pattern='test_*.py')
        runner.run()
        runner.generate_summary_report()

    Advanced usage (custom configuration):
        runner = CVSTestSuiteRunner(log, pytestconfig, test_pattern='test_*.py')
        # Access discovered modules
        for suite_name, test_module in runner.test_modules.items():
            # Custom per-suite logic
            pytest_args = runner.prepare_suite_arguments(
                suite_name,
                customize_paths={'--html': custom_html_path}
            )
            result = runner.run_single_test_suite(test_module, pytest_args)
            # Custom result handling...

        # Generate custom report
        runner.generate_summary_report(html_path=custom_path)
    """

    def __init__(self, logger, pytestconfig, test_pattern='*.py'):
        """
        Initialize the test suite runner.

        Args:
            logger: Logger instance for output (e.g., globals.log)
            pytestconfig: Built-in pytest fixture with config/options
            test_pattern (str): Glob pattern to match test files (default: '*.py')
        """
        self.log = logger
        self.pytestconfig = pytestconfig

        # Extract HTML and log file paths from pytest config
        self.pytest_html_path = pytestconfig.getoption("htmlpath", None)
        self.pytest_log_file = pytestconfig.getoption("log_file", None)

        # Capture original CLI arguments at initialization
        self.original_args = sys.argv[1:]  # Skip the script name

        # Automatically detect the calling test file to exclude it from arguments
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back  # Get the immediate caller (the test file)
            if caller_frame:
                caller_file = caller_frame.f_code.co_filename
                caller_basename = os.path.basename(caller_file).replace('.py', '')
                # Exclude the calling test file and 'run' command (from 'cvs run')
                self.exclude_patterns = [caller_basename, 'run']
                # Also capture the directory where the calling test file is located
                self.test_dir = os.path.dirname(caller_file)
            else:
                self.exclude_patterns = []
                self.test_dir = os.getcwd()
        finally:
            del frame

        # Default flags that should be excluded (and their following arguments skipped)
        # Note: --collect-only is NOT excluded so 'cvs list' works correctly
        self.exclude_flags = ['-k', '--setup-only', '--setup-show', '--setup-plan']

        # Discover test modules matching the pattern
        self.test_modules = self.discover_test_modules(pattern=test_pattern)

        # Track test results
        self.passed_suites = []
        self.failed_suites = []

    def discover_test_modules(self, pattern='*.py'):
        """
        Discover test modules matching the specified pattern in the calling test's directory.

        Args:
            pattern (str): Glob pattern to match test files (default: '*.py')

        Returns:
            dict: Dictionary mapping suite_name to full module path
        """
        pattern_path = os.path.join(self.test_dir, pattern)
        module_paths = sorted(glob.glob(pattern_path))
        # Create dict mapping suite_name -> module_path, excluding files that match exclude_patterns
        test_modules = {}
        for path in module_paths:
            suite_name = os.path.basename(path).replace('.py', '')
            # Skip if this suite name matches any exclude pattern
            if not any(pattern in suite_name for pattern in self.exclude_patterns):
                test_modules[suite_name] = path
        return test_modules

    def prepare_suite_arguments(self, suite_name, exclude_patterns=None, exclude_flags=None, customize_paths=None):
        """
        Filter and customize CLI arguments for a specific test suite.

        This function performs two operations:
        1. Filters out unwanted arguments (test file names, internal pytest flags)
        2. Customizes file path arguments by inserting suite name before extension

        Args:
            suite_name (str): Suite name to insert into customized paths
            exclude_patterns (list): List of string patterns to exclude from arguments.
                                    If None, uses the exclude patterns detected at initialization.
            exclude_flags (list): List of flags that require skipping the next argument too
            customize_paths (dict): Dict mapping arg names to their original paths for customization
                                   e.g., {'--html': '/path/report.html', '--log-file': '/tmp/log.txt'}

        Returns:
            list: Filtered and customized arguments ready for suite execution
        """
        if exclude_patterns is None:
            exclude_patterns = self.exclude_patterns

        if exclude_flags is None:
            exclude_flags = self.exclude_flags

        if customize_paths is None:
            customize_paths = {}

        # Phase 1: Filter unwanted arguments
        filtered_args = []
        skip_next = False

        for i, arg in enumerate(self.original_args):
            if skip_next:
                skip_next = False
                continue

            # Skip arguments matching exclude patterns
            if any(pattern in arg for pattern in exclude_patterns):
                continue

            # Skip flags and their following arguments
            if arg in exclude_flags:
                skip_next = True
                continue

            filtered_args.append(arg)

        # Phase 2: Customize path arguments
        for arg_name, original_path in customize_paths.items():
            if not original_path:
                continue

            # Create suite-specific path
            path_base, path_ext = os.path.splitext(original_path)
            suite_path = f"{path_base}_{suite_name}{path_ext}"

            # Replace --arg=value format
            arg_prefix = f"{arg_name}="
            filtered_args = [
                arg if not arg.startswith(arg_prefix) else f"{arg_name}={suite_path}" for arg in filtered_args
            ]

            # Replace --arg value format (two separate arguments)
            for i, arg in enumerate(filtered_args):
                if arg == arg_name and i + 1 < len(filtered_args):
                    filtered_args[i + 1] = suite_path

        return filtered_args

    def run(self, html_path=None, log_file=None):
        """
        Run all discovered test suites sequentially.

        Args:
            html_path (str): Optional HTML report path for per-suite customization.
                            If None, uses self.pytest_html_path from pytestconfig.
            log_file (str): Optional log file path for per-suite customization.
                           If None, uses self.pytest_log_file from pytestconfig.

        Returns:
            None
        """
        # Use instance attributes if not provided
        if html_path is None:
            html_path = self.pytest_html_path
        if log_file is None:
            log_file = self.pytest_log_file

        # Reset results
        self.passed_suites = []
        self.failed_suites = []

        for suite_name, test_module in self.test_modules.items():
            # Prepare arguments: filter and customize paths for this suite
            customize_paths = {
                '--html': html_path,
                '--log-file': log_file,
            }

            pytest_args = self.prepare_suite_arguments(suite_name, customize_paths=customize_paths)

            # Add the test module as the first argument
            pytest_args = [test_module] + pytest_args

            # Run the test suite
            result = self.run_single_test_suite(test_module, pytest_args)

            # Track results
            if result == 0:
                self.passed_suites.append(suite_name)
            else:
                self.failed_suites.append(suite_name)

    def run_single_test_suite(self, test_module, pytest_args):
        """
        Run a single test module with the provided pytest arguments.

        Args:
            test_module (str): Path to the test module to run
            pytest_args (list): List of arguments to pass to pytest.main()

        Returns:
            int: Exit code from pytest.main() (0 = success, non-zero = failure)
        """
        suite_name = os.path.basename(test_module).replace('.py', '')
        self.log.info(f"\n{'=' * 80}")
        self.log.info(f"Running test suite: {suite_name}")
        self.log.info(f"{'=' * 80}\n")

        # Run the test module
        result = pytest.main(pytest_args)

        if result == 0:
            self.log.info(f"✓ {suite_name} PASSED")
        else:
            self.log.error(f"✗ {suite_name} FAILED (exit code: {result})")

        return result

    def generate_summary_report(self, html_path=None):
        """
        Generate and log a summary report of all test suite results.

        Args:
            html_path (str): Optional HTML report path for listing generated reports.
                            If None, uses self.pytest_html_path from pytestconfig.

        Returns:
            None

        Raises:
            pytest.fail: If any test suites failed
        """
        # Use instance attribute if not provided
        if html_path is None:
            html_path = self.pytest_html_path

        self.log.info(f"\n{'=' * 80}")
        self.log.info("TEST SUITE SUMMARY")
        self.log.info(f"{'=' * 80}")
        self.log.info(f"Total suites: {len(self.test_modules)}")
        self.log.info(f"Passed: {len(self.passed_suites)}")
        self.log.info(f"Failed: {len(self.failed_suites)}")

        if self.passed_suites:
            self.log.info("\nPassed suites:")
            for suite in self.passed_suites:
                self.log.info(f"  ✓ {suite}")

        if self.failed_suites:
            self.log.error("\nFailed suites:")
            for suite in self.failed_suites:
                self.log.error(f"  ✗ {suite}")

            # Log HTML report locations for debugging
            if html_path:
                self.log.info("\nHTML reports generated:")
                for suite in self.passed_suites + self.failed_suites:
                    html_base, html_ext = os.path.splitext(html_path)
                    suite_html = f"{html_base}_{suite}{html_ext}"
                    self.log.info(f"  - {suite_html}")

            pytest.fail(f"{len(self.failed_suites)} test suite(s) failed")

        self.log.info("\n✓ All test suites passed!")

        # Log HTML report locations
        if html_path:
            self.log.info("\nHTML reports generated:")
            for suite in self.passed_suites:
                html_base, html_ext = os.path.splitext(html_path)
                suite_html = f"{html_base}_{suite}{html_ext}"
                self.log.info(f"  - {suite_html}")
