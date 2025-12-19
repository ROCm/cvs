import json
import sys
import os

from .base import SubcommandPlugin
from cvs.lib.parallel_ssh_lib import Pssh


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

        # Load cluster file
        try:
            with open(cluster_file, 'r') as f:
                cluster = json.load(f)
        except FileNotFoundError:
            print(f"Error: Cluster file '{cluster_file}' not found.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in cluster file: {e}")
            sys.exit(1)

        username = cluster.get('username')
        if not username:
            print("Error: 'username' not found in cluster file.")
            sys.exit(1)

        pkey = cluster.get('priv_key_file')
        if not pkey:
            print("Error: 'priv_key_file' not found in cluster file.")
            sys.exit(1)

        node_dict = cluster.get('node_dict', {})
        hosts = list(node_dict.keys())
        if not hosts:
            print("Error: No hosts found in cluster file.")
            sys.exit(1)

        # Create Pssh instance
        try:
            pssh = Pssh(log=None, host_list=hosts, user=username, pkey=pkey, stop_on_errors=False)
        except Exception as e:
            print(f"Error initializing Pssh: {e}")
            sys.exit(1)

        # Execute command
        try:
            output = pssh.exec(args.cmd)
        except Exception as e:
            print(f"Error executing command: {e}")
            sys.exit(1)

        # Print output
        for host, out in output.items():
            print(f"Host: {host}")
            print(out)
            print("---")
