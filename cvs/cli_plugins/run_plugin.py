import pytest
import sys
import os
import json
from pathlib import Path

from cvs.core.agent.lifecycle import AgentRunner
from cvs.core.run_layout import RunLayout
from cvs.core.scheduler import is_managed_compute

from .list_plugin import ListPlugin

# Legacy preflight pytest entry points — resolved at CLI time only so full-module
# collection does not register duplicate tests (same function object, two names).
LEGACY_PREFLIGHT_TEST_ALIASES = {
    "test_node_smoke": "test_node_smoke_tier1",
    "test_tier3_info": "test_node_smoke_tier3",
}


def resolve_test_function_name(name: str) -> str:
    """Map deprecated preflight test function names to their canonical pytest targets."""
    return LEGACY_PREFLIGHT_TEST_ALIASES.get(name, name)


def resolve_test_function_names(names):
    """Resolve legacy aliases and drop duplicates while preserving order."""
    resolved = []
    seen = set()
    for name in names:
        canonical = resolve_test_function_name(name)
        if canonical not in seen:
            seen.add(canonical)
            resolved.append(canonical)
    return resolved


class RunPlugin(ListPlugin):
    def get_name(self):
        return "run"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser("run", help="Run a specific test (wrapper over pytest)")
        parser.add_argument("test", help="Name of the test file to run")
        parser.add_argument("function", nargs="*", help="Optional: specific test functions to run")
        parser.add_argument(
            "--cluster_file",
            help="Path to cluster configuration JSON; optional on a scheduler-managed run",
        )
        parser.add_argument("--config_file", required=True, help="Path to test configuration JSON file")
        parser.add_argument(
            "--workspace",
            default=None,
            metavar="PATH",
            help=(
                "Shared-filesystem root for this run's artifacts; the run directory "
                "becomes <workspace>/cvs_runs/<run_id> and is exposed to configs as "
                "{run_dir}. Falls back to $CVS_WORKSPACE, then to the venv's parent "
                "directory. Use this to override the default workspace location. "
                "HTML and log reports always land in <run_dir> regardless of how the "
                "workspace is resolved; use --no-html / --no-log-file to suppress them."
            ),
        )
        parser.add_argument(
            "--html",
            help=(
                "Pytest: HTML report path. Defaults to <run_dir>/<test-file-stem>.html "
                "(self-contained). Parent directories are created automatically. "
                "Disabled with --no-html."
            ),
        )
        parser.add_argument(
            "--no-html",
            action="store_true",
            help="Do not pass --html to pytest (overrides auto-derive)",
        )
        parser.add_argument(
            "--self-contained-html",
            action="store_true",
            help=("Pytest: embed CSS/JS in the HTML report. Implied when the HTML path is auto-derived."),
        )
        parser.add_argument(
            "--log-file",
            default=None,
            metavar="PATH",
            help=(
                "Pytest: write logging output to this file. Defaults to "
                "<run_dir>/<test-file-stem>.log. Parent directories are created "
                "automatically. Disabled with --no-log-file. Console logging is unaffected."
            ),
        )
        parser.add_argument(
            "--no-log-file",
            action="store_true",
            help="Do not pass --log-file to pytest (overrides auto-derive)",
        )
        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Pytest: Level of messages to catch/display",
        )
        parser.add_argument(
            "--capture",
            choices=["no", "tee-sys", "tee-merged", "fd", "sys"],
            help="Per-test capturing method for stdout/stderr",
        )
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
Run Commands:
  cvs run agfhc                      Run all tests in agfhc (HTML + log auto-generated)
  cvs run agfhc test1                Run specific test function
  cvs run agfhc test1 test2 test3    Run multiple specific test functions
  cvs run agfhc --no-html            Run without generating an HTML report"""

    def run(self, args):
        managed = is_managed_compute()
        if not managed and not args.cluster_file:
            print("Error: --cluster_file is required outside a scheduler-managed run")
            return sys.exit(1)
        if args.cluster_file:
            self._validate_json_config(args.cluster_file, "--cluster_file")
        self._validate_json_config(args.config_file, "--config_file")
        test_file = self._resolve_test_file(args.test)

        try:
            layout = RunLayout.get(args.workspace)
        except RuntimeError as e:
            print(f"Error: {e}")
            sys.exit(1)

        if not managed:
            return sys.exit(self._run_pytest(layout, test_file, args, args.cluster_file))

        try:
            runner = AgentRunner(layout, cluster_file=args.cluster_file)
        except RuntimeError as e:
            print(f"Error: {e}")
            return sys.exit(1)

        if not runner.is_rank0:
            # Workers host the HTTP agent; only rank 0 coordinates and runs pytest.
            return sys.exit(runner.start())

        runner.start()
        try:
            try:
                cluster_file = runner.wait()
            except (ValueError, OSError) as e:
                print(f"Error: {e}")
                return sys.exit(1)
            exit_code = self._run_pytest(layout, test_file, args, cluster_file)
        finally:
            runner.stop()
        return sys.exit(exit_code)

    def _resolve_test_file(self, test_name):
        """Map a suite name to the file pytest should collect."""
        module_path = self._find_test(test_name)
        if not module_path:
            print(f"Error: Unknown test '{test_name}'")
            print("Use 'cvs list' to see available tests.")
            sys.exit(1)
        return self.get_test_file(module_path)

    def _run_pytest(self, layout, test_file, args, cluster_file):
        """Resolve report paths and hand off to pytest. Rank-0 / unmanaged only."""
        try:
            html, log_file, self_contained_html = self._resolve_report_paths(layout.run_dir, test_file, args)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        return self.run_test(
            test_file,
            args.function,
            cluster_file,
            args.config_file,
            html,
            self_contained_html,
            log_file,
            args.log_level,
            args.capture,
            getattr(args, "extra_pytest_args", []),
        )

    @staticmethod
    def _nonempty_str(value):
        """Blank CLI/env strings are unset so auto-derive still applies."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    def _resolve_report_paths(self, run_dir, test_file, args):
        """Decide pytest --html / --log-file for this run.

        Always auto-derives under run_dir unless the user passes an explicit path
        or --no-html / --no-log-file.
        Raises ValueError when an explicit path is combined with its --no-* flag.
        """
        html = self._nonempty_str(getattr(args, "html", None))
        log_file = self._nonempty_str(getattr(args, "log_file", None))
        no_html = bool(getattr(args, "no_html", False))
        no_log_file = bool(getattr(args, "no_log_file", False))

        if html and no_html:
            raise ValueError("cannot combine --html and --no-html")
        if log_file and no_log_file:
            raise ValueError("cannot combine --log-file and --no-log-file")

        contained = bool(getattr(args, "self_contained_html", False))
        file_stem = Path(test_file).stem

        if no_html:
            html = None
            contained = False
        elif not html:
            html = str(Path(run_dir) / f"{file_stem}.html")
            contained = True

        if no_log_file:
            log_file = None
        elif not log_file:
            log_file = str(Path(run_dir) / f"{file_stem}.log")

        return html, log_file, contained

    def _validate_json_config(self, path, label):
        """Validate that a config file exists and is valid JSON."""
        if not os.path.exists(path):
            print(f"Error: {label} does not exist: {path}")
            sys.exit(1)
        if not os.path.isfile(path):
            print(f"Error: {label} is not a file: {path}")
            sys.exit(1)
        try:
            with open(path, "r", encoding="utf-8") as f:
                json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: {label} is not valid JSON: {path}")
            print(f"  {e}")
            sys.exit(1)
        except OSError as e:
            print(f"Error: unable to read {label}: {path}")
            print(f"  {e}")
            sys.exit(1)

    def run_test(
        self,
        test_file,
        test_functions,
        cluster_file,
        config_file,
        html,
        self_contained_html,
        log_file,
        log_level,
        capture,
        extra_pytest_args,
    ):
        # Build pytest arguments
        pytest_args = []
        if test_functions:
            for func in resolve_test_function_names(test_functions):
                pytest_args.append(f"{test_file}::{func}")
        else:
            # Run all tests in the file
            pytest_args.append(test_file)

        # Add CVS-specific arguments
        pytest_args.append(f"--cluster_file={cluster_file}")
        pytest_args.append(f"--config_file={config_file}")

        # Ensure log directory exists
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
        if html:
            html_dir = os.path.dirname(html)
            if html_dir:
                os.makedirs(html_dir, exist_ok=True)

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

        # Run pytest normally
        return pytest.main(pytest_args)
