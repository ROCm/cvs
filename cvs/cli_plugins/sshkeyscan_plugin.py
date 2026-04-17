import json
import sys
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import SubcommandPlugin


class SSHKeyScanPlugin(SubcommandPlugin):
    def get_name(self):
        return "sshkeyscan"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser("sshkeyscan", help="Scan and update SSH host keys for all cluster nodes")
        parser.add_argument(
            "--cluster_file", help="Path to cluster configuration JSON file (overrides CLUSTER_FILE env var)"
        )
        parser.add_argument(
            "--known_hosts", default="~/.ssh/known_hosts", help="Path to known_hosts file (default: ~/.ssh/known_hosts)"
        )
        parser.add_argument("--remove-existing", action="store_true", help="Remove existing host keys before scanning")
        parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
        parser.add_argument(
            "--parallel", type=int, default=20, help="Number of parallel ssh-keyscan processes (default: 20)"
        )
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
SSH Key Scan Commands:
  cvs sshkeyscan --cluster_file cluster.json                    Scan all cluster nodes
  cvs sshkeyscan --remove-existing --cluster_file cluster.json  Remove old keys and rescan
  cvs sshkeyscan --dry-run --cluster_file cluster.json          Show what would be done
  CLUSTER_FILE=cluster.json cvs sshkeyscan                      Use env var for cluster file"""

    def scan_host_key(self, host, known_hosts_file, remove_existing=False, dry_run=False):
        """Scan SSH host key for a single host."""
        try:
            # Remove existing key if requested
            if remove_existing and not dry_run:
                remove_cmd = ['ssh-keygen', '-f', known_hosts_file, '-R', host]
                subprocess.run(remove_cmd, capture_output=True, text=True)

            if dry_run:
                return f"{host}: Would scan and add SSH host key"

            # Scan for SSH host key
            scan_cmd = ['ssh-keyscan', '-H', host]
            result = subprocess.run(scan_cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                return f"{host}: FAILED - {result.stderr.strip()}"

            if not result.stdout.strip():
                return f"{host}: FAILED - No host key received"

            # Append to known_hosts file
            with open(os.path.expanduser(known_hosts_file), 'a') as f:
                f.write(result.stdout)

            # Count the number of keys added
            key_count = len([line for line in result.stdout.strip().split('\n') if line.strip()])
            return f"{host}: SUCCESS - Added {key_count} host key(s)"

        except subprocess.TimeoutExpired:
            return f"{host}: FAILED - Connection timeout"
        except FileNotFoundError:
            return f"{host}: FAILED - ssh-keyscan command not found"
        except Exception as e:
            return f"{host}: FAILED - {str(e)}"

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

        node_dict = cluster.get('node_dict', {})
        hosts = list(node_dict.keys())
        if not hosts:
            print("Error: No hosts found in cluster file.")
            sys.exit(1)

        known_hosts_file = os.path.expanduser(args.known_hosts)

        print("SSH Key Scan Configuration:")
        print(f"  Cluster file: {cluster_file}")
        print(f"  Known hosts file: {known_hosts_file}")
        print(f"  Number of hosts: {len(hosts)}")
        print(f"  Parallel processes: {args.parallel}")
        print(f"  Remove existing keys: {args.remove_existing}")
        print(f"  Dry run: {args.dry_run}")
        print()

        if args.dry_run:
            print("DRY RUN - No changes will be made:")
        else:
            # Create known_hosts directory if it doesn't exist
            known_hosts_dir = os.path.dirname(known_hosts_file)
            os.makedirs(known_hosts_dir, exist_ok=True)

            # Create known_hosts file if it doesn't exist
            if not os.path.exists(known_hosts_file):
                open(known_hosts_file, 'a').close()

        # Scan hosts in parallel
        print("Scanning SSH host keys...")
        results = []
        successful = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all tasks
            future_to_host = {
                executor.submit(self.scan_host_key, host, known_hosts_file, args.remove_existing, args.dry_run): host
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
        print("SSH Key Scan Summary:")
        print(f"  Total hosts: {len(hosts)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")

        if not args.dry_run and successful > 0:
            print(f"  SSH host keys updated in: {known_hosts_file}")
            print()
            print("You can now run SSH commands without host key verification warnings.")
            print("Consider running 'cvs sshkeyscan' periodically if host keys change.")

        # Exit with non-zero code if there were failures
        if failed > 0:
            sys.exit(1)
