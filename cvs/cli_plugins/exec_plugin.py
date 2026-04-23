import logging
import os
import sys

from .base import SubcommandPlugin
from cvs.core import (
    ExecScope,
    ExecTarget,
    OrchestratorConfigError,
    create_orchestrator,
    load_config,
)


class ExecPlugin(SubcommandPlugin):
    def get_name(self):
        return "exec"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser("exec", help="Execute a command on all nodes in the cluster")
        parser.add_argument("--cmd", required=True, help="Command to execute on all nodes")
        parser.add_argument(
            "--cluster_file", help="Path to cluster configuration JSON file (overrides CLUSTER_FILE env var)"
        )
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
Exec Commands:
  cvs exec --cmd "hostname" --cluster_file cluster.json    Execute hostname on all nodes
  CLUSTER_FILE=cluster.json cvs exec --cmd "hostname"      Use env var for cluster file"""

    def run(self, args):
        # Determine cluster file: env var takes precedence, then --cluster_file
        cluster_file = os.environ.get('CLUSTER_FILE') or args.cluster_file
        if not cluster_file:
            print("Error: No cluster file specified. Set CLUSTER_FILE environment variable or use --cluster_file.")
            sys.exit(1)

        # Single canonical loader. Reads JSON, applies the additive overlay
        # rules, dispatches per-axis validation, aggregates errors. The
        # bespoke key-by-key sys.exit(1) chain that this method used to do
        # is now one OrchestratorConfigError catch.
        try:
            cfg = load_config(cluster_file)
        except OrchestratorConfigError as e:
            print(f"Error: {e}")
            sys.exit(1)

        log = logging.getLogger(__name__)
        try:
            orch = create_orchestrator(cfg, log=log)
        except OrchestratorConfigError as e:
            print(f"Error initializing orchestrator: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error initializing orchestrator: {e}")
            sys.exit(1)

        # `cvs exec --cmd` runs on the bare host shell. We do not bring the
        # runtime up (no orch.setup) because this is meant to be a quick
        # ad hoc command across nodes; if you want commands inside the
        # runtime, use a pytest fixture that calls orch.setup().
        try:
            results = orch.exec(args.cmd, scope=ExecScope.ALL, target=ExecTarget.HOST)
        except Exception as e:
            print(f"Error executing command: {e}")
            sys.exit(1)

        for host, result in results.items():
            print(f"Host: {host}")
            print(result.output)
            print("---")
