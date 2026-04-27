import json
import sys
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import SubcommandPlugin
from cvs.lib.parallel_ssh_lib import Pssh


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
        parser.add_argument(
            "--at",
            choices=['local', 'head'],
            default='local',
            help="Where to run the scan: 'local' (current node) or 'head' (head node from cluster)",
        )
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
SSH Key Scan Commands:
  cvs sshkeyscan --cluster_file cluster.json                    Scan locally (default)
  cvs sshkeyscan --cluster_file cluster.json --at head         Scan from head node (MPI)
  cvs sshkeyscan --at head --remove-existing                   Remove old keys and rescan from head
  cvs sshkeyscan --at head --dry-run                           Show what would be done on head node
  CLUSTER_FILE=cluster.json cvs sshkeyscan --at head           Use env var for cluster file"""

    def scan_host_key(self, host, known_hosts_file, remove_existing=False, dry_run=False):
        """Scan SSH host key for a single host (local execution)."""
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

    def scan_host_key_remote(self, phdl, host, known_hosts_file, remove_existing=False, dry_run=False):
        """Scan SSH host key for a single host (remote execution via pssh)."""
        try:
            # Remove existing key if requested
            if remove_existing and not dry_run:
                remove_cmd = f"ssh-keygen -f {known_hosts_file} -R {host}"
                phdl.exec(remove_cmd, print_console=False)

            if dry_run:
                return f"{host}: Would scan and add SSH host key (on remote)"

            # Scan for SSH host key
            scan_cmd = f"timeout 30 ssh-keyscan -H {host}"
            result_dict = phdl.exec(scan_cmd, timeout=35, print_console=False)

            # Get result for the head node (should be only one host in phdl)
            head_node = list(result_dict.keys())[0]
            output = result_dict[head_node]

            # Check if command succeeded - ssh-keyscan failure usually shows in output
            if 'Connection refused' in output or 'No route to host' in output or 'Connection timed out' in output:
                return f"{host}: FAILED - {output.strip()}"

            if not output.strip():
                return f"{host}: FAILED - No host key received"

            # Check if output looks like SSH host keys (should contain ssh key types or be hashed format)
            lines = [line.strip() for line in output.strip().split('\n') if line.strip() and not line.startswith('#')]
            if not lines or not any(
                line
                for line in lines
                if ('ssh-' in line or line.startswith('|1|') or 'ecdsa-sha2' in line or 'ed25519' in line)
            ):
                return f"{host}: FAILED - Invalid output: {output.strip()}"

            # Append to known_hosts file - escape single quotes in the output
            escaped_output = output.strip().replace("'", "'\"'\"'")
            append_cmd = f"echo '{escaped_output}' >> {known_hosts_file}"
            append_result = phdl.exec(append_cmd, print_console=False)

            # Check if append succeeded by checking if there were any obvious errors
            append_output = list(append_result.values())[0]
            if 'Permission denied' in append_output or 'No such file' in append_output:
                return f"{host}: FAILED - Could not write to known_hosts file: {append_output.strip()}"

            # Count the number of keys added
            key_count = len(lines)
            return f"{host}: SUCCESS - Added {key_count} host key(s) (on remote)"

        except Exception as e:
            return f"{host}: FAILED - {str(e)}"

    def scan_host_key_remote_threadsafe(
        self, cluster_config, host, known_hosts_file, remove_existing=False, dry_run=False
    ):
        """Scan SSH host key for a single host (remote execution via thread-local pssh)."""
        try:
            # Create thread-local pssh handle to avoid race conditions
            head_node = cluster_config['head_node']
            username = cluster_config['username']
            priv_key_file = cluster_config['priv_key_file']
            env_vars = cluster_config.get('env_vars')

            thread_pssh = Pssh(None, [head_node], user=username, pkey=priv_key_file, env_vars=env_vars)

            # Use the same logic as scan_host_key_remote but with thread_pssh
            return self.scan_host_key_remote(thread_pssh, host, known_hosts_file, remove_existing, dry_run)

        except Exception as e:
            return f"{host}: FAILED - Thread creation error: {str(e)}"

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

        # Handle --at head option
        phdl = None
        if args.at == 'head':
            head_node_dict = cluster.get('head_node_dict', {})
            head_node = head_node_dict.get('mgmt_ip')
            if not head_node:
                print("Error: No head node found in cluster file (head_node_dict.mgmt_ip).")
                sys.exit(1)

            username = cluster.get('username')
            priv_key_file = cluster.get('priv_key_file')
            env_vars = cluster.get('env_vars')

            if not username or not priv_key_file:
                print("Error: username and priv_key_file required in cluster file for remote execution.")
                sys.exit(1)

            # Create pssh handle to head node
            phdl = Pssh(None, [head_node], user=username, pkey=priv_key_file, env_vars=env_vars)

        known_hosts_file = args.known_hosts  # Use as-is for remote execution
        if args.at == 'local':
            known_hosts_file = os.path.expanduser(args.known_hosts)

        print("SSH Key Scan Configuration:")
        print(f"  Cluster file: {cluster_file}")
        print(f"  Execution location: {args.at}")
        if args.at == 'head':
            print(f"  Head node: {head_node}")
        print(f"  Known hosts file: {known_hosts_file}")
        print(f"  Number of hosts: {len(hosts)}")
        print(f"  Parallel processes: {args.parallel}")
        print(f"  Remove existing keys: {args.remove_existing}")
        print(f"  Dry run: {args.dry_run}")
        print()

        if args.dry_run:
            print("DRY RUN - No changes will be made:")
        else:
            if args.at == 'local':
                # Create known_hosts directory if it doesn't exist (local only)
                known_hosts_dir = os.path.dirname(known_hosts_file)
                os.makedirs(known_hosts_dir, exist_ok=True)

                # Create known_hosts file if it doesn't exist (local only)
                if not os.path.exists(known_hosts_file):
                    open(known_hosts_file, 'a').close()
            else:
                # For remote execution, ensure the .ssh directory and known_hosts file exist
                setup_cmd = f"mkdir -p $(dirname {known_hosts_file}) && touch {known_hosts_file}"
                phdl.exec(setup_cmd, print_console=False)

        # Scan hosts in parallel
        print("Scanning SSH host keys...")
        results = []
        successful = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all tasks
            if args.at == 'local':
                future_to_host = {
                    executor.submit(
                        self.scan_host_key, host, known_hosts_file, args.remove_existing, args.dry_run
                    ): host
                    for host in hosts
                }
            else:
                # Create cluster config for thread-safe pssh creation
                cluster_config = {
                    'head_node': head_node,
                    'username': cluster.get('username'),
                    'priv_key_file': cluster.get('priv_key_file'),
                    'env_vars': cluster.get('env_vars'),
                }
                future_to_host = {
                    executor.submit(
                        self.scan_host_key_remote_threadsafe,
                        cluster_config,
                        host,
                        known_hosts_file,
                        args.remove_existing,
                        args.dry_run,
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
        print("SSH Key Scan Summary:")
        print(f"  Total hosts: {len(hosts)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")

        if not args.dry_run and successful > 0:
            location = "head node" if args.at == 'head' else "local machine"
            print(f"  SSH host keys updated on: {location}")
            print(f"  Known hosts file: {known_hosts_file}")
            print()
            if args.at == 'head':
                print("Head node can now run SSH/MPI commands without host key verification warnings.")
            else:
                print("You can now run SSH commands without host key verification warnings.")
            print("Consider running 'cvs sshkeyscan' periodically if host keys change.")

        # Exit with non-zero code if there were failures
        if failed > 0:
            sys.exit(1)
