import json
import logging
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import SubcommandPlugin
from cvs.lib.parallel_ssh_lib import Pssh


class ScpPlugin(SubcommandPlugin):
    def get_name(self):
        return "scp"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser("scp", help="Copy files to all nodes in the cluster")
        parser.add_argument("--file", required=True, help="Local file to copy to all nodes")
        parser.add_argument("--dest", help="Remote destination path (defaults to same as source)")
        parser.add_argument("--recurse", action="store_true", help="Copy directories recursively")
        parser.add_argument(
            "--cluster_file", help="Path to cluster configuration JSON file (overrides CLUSTER_FILE env var)"
        )
        parser.add_argument("--parallel", type=int, default=20, help="Number of parallel SCP operations (default: 20)")
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
SCP Commands:
  cvs scp --file /path/to/file.txt --cluster_file cluster.json           Copy file to same path on all nodes
  cvs scp --file /local/file.txt --dest /remote/file.txt                Copy file to specific remote path
  cvs scp --file /local/dir --dest /remote/dir --recurse                Copy directory recursively
  cvs scp --file /path/to/file.txt --parallel 10                        Use 10 parallel SCP operations
  CLUSTER_FILE=cluster.json cvs scp --file /path/to/file.txt            Use env var for cluster file"""

    def copy_file_to_host(self, log, host, username, pkey, local_file, remote_path, recurse=False):
        """Copy file to a single host via SCP."""
        try:
            pssh = Pssh(log=log, host_list=[host], user=username, pkey=pkey, stop_on_errors=False)
            pssh.scp_file(local_file, remote_path, recurse=recurse)
            return f"{host}: SUCCESS - File copied to {remote_path}"
        except Exception as e:
            return f"{host}: FAILED - {str(e)}"

    def run(self, args):
        # Determine cluster file: env var takes precedence, then --cluster_file
        cluster_file = os.environ.get('CLUSTER_FILE') or args.cluster_file
        if not cluster_file:
            print("Error: No cluster file specified. Set CLUSTER_FILE environment variable or use --cluster_file.")
            sys.exit(1)

        # Validate local file exists
        if not os.path.exists(args.file):
            print(f"Error: Local file '{args.file}' not found.")
            sys.exit(1)

        # Set destination path (default to same as source)
        dest_path = args.dest or args.file

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

        # Create logger
        log = logging.getLogger(__name__)

        print("SCP Configuration:")
        print(f"  Cluster file: {cluster_file}")
        print(f"  Local file: {args.file}")
        print(f"  Remote destination: {dest_path}")
        print(f"  Recurse: {args.recurse}")
        print(f"  Number of hosts: {len(hosts)}")
        print(f"  Parallel operations: {args.parallel}")
        print()

        # Copy file to all hosts in parallel
        print(f"Copying '{args.file}' to '{dest_path}' on {len(hosts)} hosts...")
        results = []
        successful = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            future_to_host = {
                executor.submit(
                    self.copy_file_to_host, log, host, username, pkey, args.file, dest_path, args.recurse
                ): host
                for host in hosts
            }

            # Collect results as they complete
            for future in as_completed(future_to_host):
                host = future_to_host[future]
                try:
                    result = future.result()
                    results.append(result)
                    if "SUCCESS" in result:
                        successful += 1
                    else:
                        failed += 1
                    print(result)
                except Exception as e:
                    error_msg = f"{host}: FAILED - Exception: {str(e)}"
                    results.append(error_msg)
                    failed += 1
                    print(error_msg)

        print()
        print("SCP Summary:")
        print(f"  Total hosts: {len(hosts)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")

        if successful > 0:
            print(f"\nSuccessfully copied file to {successful} host(s).")

        # Exit with non-zero code if there were failures
        if failed > 0:
            sys.exit(1)
