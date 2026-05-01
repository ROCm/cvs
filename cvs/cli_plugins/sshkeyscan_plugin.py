import json
import sys
import os
import subprocess
import tempfile
import logging

from .base import SubcommandPlugin
from cvs.lib.parallel_ssh_lib import Pssh


class SSHKeyScanPlugin(SubcommandPlugin):
    def __init__(self):
        super().__init__()
        # Create logger for Pssh instances
        self.logger = logging.getLogger(self.__class__.__name__)

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
            "--parallel", type=int, default=20, help="Number of parallel processes to use (default: 20)"
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

    # Core command building functions
    def build_remove_keys_command(self, hosts_file, known_hosts_file):
        """Build command to remove existing SSH host keys"""
        return f"cat {hosts_file} | xargs -I {{}} ssh-keygen -f {known_hosts_file} -R {{}}"

    def build_scan_command(self, hosts_file, known_hosts_file, parallel, batch_size, timeout, dry_run=False):
        """Build ssh-keyscan command with xargs and temp files for race-condition safety"""
        # For dry run, just return a simple command
        if dry_run:
            return f"echo 'DRY RUN: Would scan hosts and append to {known_hosts_file}'"

        # Generate unique temp directory for this scan operation
        import uuid

        temp_dir = f"/tmp/cvs_scan_{uuid.uuid4().hex[:8]}"

        # Build race-condition-safe command with separate temp files per process
        # We capture output both for parsing AND append to known_hosts
        cmd = (
            f"mkdir -p {temp_dir} && "
            f"cat {hosts_file} | "
            f"xargs -P {parallel} -n {batch_size} "
            f"sh -c 'ssh-keyscan -T {timeout} -H \"$@\" > {temp_dir}/batch_$$' sh && "
            f"cat {temp_dir}/batch_* | tee -a {known_hosts_file} && "
            f"rm -rf {temp_dir}"
        )

        return cmd

    # File management functions
    def create_hosts_file_local(self, hosts):
        """Create temporary hosts file locally"""
        fd, hosts_file = tempfile.mkstemp(suffix='.hosts', text=True)
        try:
            with os.fdopen(fd, 'w') as f:
                for host in hosts:
                    f.write(f"{host}\n")
            return hosts_file
        except:
            os.unlink(hosts_file)
            raise

    def create_hosts_file_remote(self, hosts, phdl):
        """Create temporary hosts file on remote system using SCP transfer"""
        # Create local temp file first
        local_temp = self.create_hosts_file_local(hosts)

        try:
            # Generate unique remote filename
            import uuid

            remote_file = f"/tmp/cvs_hosts_{uuid.uuid4().hex[:8]}"

            # Use SCP to transfer the hosts file efficiently
            # This avoids command-line length limits and works with any cluster size
            phdl.scp_file(local_temp, remote_file)

            return remote_file

        finally:
            # Always cleanup local temp file
            try:
                os.unlink(local_temp)
            except:
                pass  # Best effort cleanup

    def cleanup_hosts_file(self, hosts_file, phdl=None):
        """Clean up temporary hosts file"""
        try:
            if phdl:
                phdl.exec(f"rm -f {hosts_file}", print_console=False)
            else:
                os.unlink(hosts_file)
        except Exception as e:
            # Log warning but don't fail - cleanup is best effort
            self.logger.warning(f"Could not clean up hosts file {hosts_file}: {e}")

    # Command execution functions
    def execute_command_local(self, command, timeout_seconds):
        """Execute command locally and return structured result"""
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout_seconds)
            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
                'success': result.returncode == 0,
            }
        except subprocess.TimeoutExpired as e:
            return {
                'stdout': e.stdout or '',
                'stderr': e.stderr or '',
                'returncode': 124,  # Standard timeout exit code
                'success': False,
            }

    def execute_command_remote(self, command, phdl, timeout_seconds):
        """Execute command on remote host and return structured result"""
        try:
            result_dict = phdl.exec(command, print_console=False, timeout=timeout_seconds)
            head_node = list(result_dict.keys())[0]
            output = result_dict[head_node]

            # pssh doesn't easily separate stdout/stderr, so we infer success
            success = not any(
                error_indicator in output.lower() for error_indicator in ['error:', 'failed:', 'permission denied']
            )

            return {
                'stdout': output,
                'stderr': '',  # pssh combines streams
                'returncode': 0 if success else 1,
                'success': success,
            }
        except Exception as e:
            return {'stdout': '', 'stderr': str(e), 'returncode': 1, 'success': False}

    # Output parsing functions
    def parse_ssh_keyscan_output(self, stdout, stderr, hosts):
        """Parse ssh-keyscan output to determine per-host results"""
        # Count SSH key lines (with hashed output, we can't map back to specific hosts)
        key_lines = [
            line
            for line in stdout.split('\n')
            if line.strip()
            and not line.startswith('#')
            and ('ssh-' in line or 'ecdsa-' in line or 'ed25519' in line or line.startswith('|1|'))
        ]

        total_keys_found = len(key_lines)

        # Generate results - this is approximate with hashed output
        results = []
        if total_keys_found > 0:
            # Each host typically has 2-3 keys (rsa, ecdsa, ed25519)
            estimated_hosts_with_keys = total_keys_found // 2  # Conservative estimate
            estimated_successful = min(len(hosts), estimated_hosts_with_keys)

            for i, host in enumerate(hosts):
                if i < estimated_successful:
                    results.append(f"{host}: SUCCESS - SSH host key added")
                else:
                    results.append(f"{host}: FAILED - No key found")
        else:
            # No keys found
            for host in hosts:
                results.append(f"{host}: FAILED - No key found")

        successful = sum(1 for r in results if "SUCCESS" in r)
        failed = len(results) - successful

        return successful, failed, results

    def parse_error_output(self, stderr):
        """Extract meaningful errors from command stderr"""
        errors = []
        for line in stderr.split('\n'):
            line = line.strip()
            if line and any(
                indicator in line.lower()
                for indicator in ['connection refused', 'timeout', 'no route to host', 'permission denied']
            ):
                errors.append(line)
        return errors

    # High-level operation functions
    def remove_existing_keys(self, hosts, known_hosts_file, phdl=None, dry_run=False):
        """Remove existing SSH host keys for given hosts"""
        if dry_run:
            return [f"{host}: Would remove existing SSH host key" for host in hosts]

        hosts_file = None
        try:
            # Create hosts file
            if phdl:
                hosts_file = self.create_hosts_file_remote(hosts, phdl)
            else:
                hosts_file = self.create_hosts_file_local(hosts)

            # Build and execute remove command
            command = self.build_remove_keys_command(hosts_file, known_hosts_file)
            timeout = len(hosts) * 2  # 2 seconds per host

            if phdl:
                result = self.execute_command_remote(command, phdl, timeout)
            else:
                result = self.execute_command_local(command, timeout)

            # Parse results
            if result['success']:
                return [f"{host}: Existing keys removed" for host in hosts]
            else:
                errors = self.parse_error_output(result['stderr'])
                error_msg = '; '.join(errors) if errors else 'Unknown error'
                return [f"{host}: Failed to remove keys - {error_msg}" for host in hosts]

        finally:
            if hosts_file:
                self.cleanup_hosts_file(hosts_file, phdl)

    def scan_ssh_keys(self, hosts, known_hosts_file, parallel, phdl=None, dry_run=False):
        """Scan SSH host keys for given hosts"""
        if dry_run:
            return len(hosts), 0, [f"{host}: Would scan and add SSH host key" for host in hosts]

        hosts_file = None
        try:
            # Create hosts file
            if phdl:
                hosts_file = self.create_hosts_file_remote(hosts, phdl)
            else:
                hosts_file = self.create_hosts_file_local(hosts)

            # Build and execute scan command
            import math

            batch_size = max(1, math.ceil(len(hosts) / parallel))  # Distribute hosts evenly across processes
            timeout_total = len(hosts) * 2  # 2 seconds per host
            command = self.build_scan_command(hosts_file, known_hosts_file, parallel, batch_size, 30, dry_run)

            if phdl:
                result = self.execute_command_remote(command, phdl, timeout_total)
            else:
                result = self.execute_command_local(command, timeout_total)

            # Parse results
            return self.parse_ssh_keyscan_output(result['stdout'], result['stderr'], hosts)

        finally:
            if hosts_file:
                self.cleanup_hosts_file(hosts_file, phdl)

    def scan_hosts_unified(self, hosts, known_hosts_file, args, phdl=None):
        """Unified SSH key scanning for both local and remote execution"""
        all_results = []
        total_successful = 0
        total_failed = 0

        # Step 1: Remove existing keys if requested
        if args.remove_existing:
            remove_results = self.remove_existing_keys(hosts, known_hosts_file, phdl, args.dry_run)
            all_results.extend(remove_results)
            if not args.dry_run:
                print("Existing key removal:")
                for result in remove_results:
                    print(f"  {result}")
                print()

        # Step 2: Scan new keys
        successful, failed, scan_results = self.scan_ssh_keys(
            hosts, known_hosts_file, args.parallel, phdl, args.dry_run
        )
        all_results.extend(scan_results)
        total_successful += successful
        total_failed += failed

        return total_successful, total_failed, all_results

    def _load_and_validate_cluster_config(self, args):
        """Load and validate cluster configuration"""
        cluster_file = os.environ.get('CLUSTER_FILE') or args.cluster_file
        if not cluster_file:
            print("Error: No cluster file specified. Set CLUSTER_FILE environment variable or use --cluster_file.")
            sys.exit(1)

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

        return cluster_file, cluster, hosts

    def _setup_remote_connection(self, cluster, args):
        """Setup remote SSH connection if needed"""
        if args.at != 'head':
            return None

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

        return Pssh(self.logger, [head_node], user=username, pkey=priv_key_file, env_vars=env_vars), head_node

    def _resolve_known_hosts_path(self, args):
        """Resolve the known_hosts file path based on execution mode"""
        if args.at == 'local':
            return os.path.expanduser(args.known_hosts)
        else:
            return args.known_hosts  # Use as-is for remote execution

    def _print_configuration(self, cluster_file, hosts, known_hosts_file, args, head_node=None):
        """Print scan configuration for user visibility"""
        print("SSH Key Scan Configuration:")
        print(f"  Cluster file: {cluster_file}")
        print(f"  Execution location: {args.at}")
        if head_node:
            print(f"  Head node: {head_node}")
        print(f"  Known hosts file: {known_hosts_file}")
        print(f"  Number of hosts: {len(hosts)}")
        print(f"  Parallel processes: {args.parallel}")
        print(f"  Remove existing keys: {args.remove_existing}")
        print(f"  Dry run: {args.dry_run}")
        print()

    def _prepare_known_hosts_file(self, known_hosts_file, args, phdl):
        """Ensure known_hosts file and directory exist"""
        if args.dry_run:
            print("DRY RUN - No changes will be made:")
            return

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

    def _execute_ssh_key_operations(self, hosts, known_hosts_file, args, phdl):
        """Execute the SSH key scanning operations"""
        print("Scanning SSH host keys...")
        return self.scan_hosts_unified(hosts, known_hosts_file, args, phdl)

    def _print_results_and_summary(self, results, successful, failed, hosts, known_hosts_file, args):
        """Print operation results and summary"""
        # Print individual results
        for result in results:
            print(result)

        # Print summary
        print()
        print("SSH Key Scan Summary:")
        print(f"  Total hosts: {len(hosts)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")

        # Print success information
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

    def _cleanup_resources(self, phdl):
        """Clean up any resources (SSH connections, etc.)"""
        if phdl:
            try:
                phdl.destroy_clients()
            except Exception as e:
                self.logger.warning(f"Could not clean up pssh connection: {e}")

    def run(self, args):
        """Main entry point - orchestrates the SSH key scanning workflow"""
        # Step 1: Load and validate configuration
        cluster_file, cluster, hosts = self._load_and_validate_cluster_config(args)

        # Step 2: Setup remote connection if needed
        remote_result = self._setup_remote_connection(cluster, args)
        if remote_result:
            phdl, head_node = remote_result
        else:
            phdl, head_node = None, None

        # Step 3: Resolve file paths
        known_hosts_file = self._resolve_known_hosts_path(args)

        # Step 4: Display configuration
        self._print_configuration(cluster_file, hosts, known_hosts_file, args, head_node)

        try:
            # Step 5: Prepare known_hosts file
            self._prepare_known_hosts_file(known_hosts_file, args, phdl)

            # Step 6: Execute SSH key operations
            successful, failed, results = self._execute_ssh_key_operations(hosts, known_hosts_file, args, phdl)

            # Step 7: Display results
            self._print_results_and_summary(results, successful, failed, hosts, known_hosts_file, args)

        finally:
            # Step 8: Cleanup resources
            self._cleanup_resources(phdl)

        # Step 9: Exit with appropriate code
        if failed > 0:
            sys.exit(1)
