import asyncio
import pytest
import sys
import os
import json

from cvs.core.agent.lifecycle import REGISTRATION_TIMEOUT_SECONDS, managed_rank, run_worker, start_rank0
from cvs.core.agent.mesh import AgentMesh
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
        parser.add_argument("--cluster_file", required=True, help="Path to cluster configuration JSON file")
        parser.add_argument("--config_file", required=True, help="Path to test configuration JSON file")
        parser.add_argument(
            "--workspace",
            default=None,
            metavar="PATH",
            help=(
                "Shared-filesystem root for this run's artifacts; the run directory "
                "becomes <workspace>/cvs_runs/<run_id> and is exposed to configs as "
                "{run_dir}. Falls back to $CVS_WORKSPACE, then to the venv's parent "
                "directory. Scheduler-managed runs in a container must set this "
                "explicitly, since the venv's parent is not on shared storage there."
            ),
        )
        parser.add_argument("--html", help="Pytest: Create HTML report file at given path")
        parser.add_argument(
            "--self-contained-html",
            action="store_true",
            help="Pytest: Create a self-contained HTML file containing all the HTML report",
        )
        parser.add_argument(
            "--log-file",
            default=None,
            metavar="PATH",
            help=(
                "Pytest: write logging output to this file (optional). "
                "Parent directories are created automatically when set."
            ),
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
  cvs run agfhc                      Run all tests in agfhc
  cvs run agfhc test1                Run specific test function
  cvs run agfhc test1 test2 test3    Run multiple specific test functions
  cvs run agfhc --html report.html   Run test and generate HTML report"""

    def run(self, args):
        # Pre-flight, in this order, before pytest is reached at all. Worker ranks
        # in a Slurm/Spur job never enter pytest, so the run layout -- the
        # rendezvous those ranks read -- has to be resolved here rather than in the
        # handoff that launches it. Inputs are checked first because creating
        # directories for a mistyped suite name would litter shared storage.
        self._validate_json_config(args.cluster_file, "--cluster_file")
        self._validate_json_config(args.config_file, "--config_file")
        test_file = self._resolve_test_file(args.test)

        try:
            layout = RunLayout.get(args.workspace)
        except RuntimeError as e:
            print(f"Error: {e}")
            sys.exit(1)
        # --workspace is a CLI argument, so any process we later spawn (pytest shard
        # workers) would re-derive the layout from its own defaults and land on a
        # different run_dir. Publishing the resolved answer keeps every descendant on
        # the one directory this run already committed to.
        os.environ["CVS_WORKSPACE"] = str(layout.workspace)

        test_args = (
            test_file,
            args.function,
            args.cluster_file,
            args.config_file,
            args.html,
            args.self_contained_html,
            args.log_file,
            args.log_level,
            args.capture,
            getattr(args, "extra_pytest_args", []),
        )
        if not is_managed_compute():
            return sys.exit(self.run_test(*test_args))

        try:
            rank, world_size = managed_rank()
        except RuntimeError as e:
            print(f"Error: {e}")
            return sys.exit(1)
        if rank != 0:
            return sys.exit(run_worker(layout.agent_dir, rank, world_size))

        coordinator = start_rank0(layout.agent_dir, world_size)
        try:
            try:
                snapshot = coordinator.wait_for_registrations(REGISTRATION_TIMEOUT_SECONDS)
            except (TimeoutError, asyncio.TimeoutError):
                print("Warning: registration timeout expired; continuing with available ranks.")
                snapshot = coordinator.registered_agents()
            try:
                AgentMesh.install_from_agent_dir(snapshot, layout.agent_dir)
            except (ValueError, OSError) as e:
                print(f"Error: {e}")
                return sys.exit(1)
            exit_code = self.run_test(*test_args)
        finally:
            AgentMesh.reset()
            coordinator.close()
        return sys.exit(exit_code)

    def _resolve_test_file(self, test_name):
        """Map a suite name to the file pytest should collect."""
        module_path = self._find_test(test_name)
        if not module_path:
            print(f"Error: Unknown test '{test_name}'")
            print("Use 'cvs list' to see available tests.")
            sys.exit(1)
        return self.get_test_file(module_path)

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
            os.makedirs(log_dir, exist_ok=True)

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
