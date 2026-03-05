"""
Jump Host Parallel SSH using paramiko + pssh.
Based on working test_auth_script.py approach.
"""

import paramiko
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class JumpHostPssh:
    """
    SSH to nodes via jump host using ParallelSSHClient.

    Approach:
    1. Connect to jump host with paramiko
    2. Create proxy socket function using jump_transport.open_channel
    3. Inject proxy socket into ParallelSSHClient
    4. Use node key file that's in the container
    """

    def __init__(
        self,
        jump_host: str,
        jump_user: str,
        jump_password: Optional[str] = None,
        jump_pkey: Optional[str] = None,
        target_hosts: List[str] = None,
        target_user: str = None,
        target_pkey: str = None,
        max_parallel: int = 32,
        timeout: int = 30,
    ):
        self.jump_host = jump_host
        self.jump_user = jump_user
        self.jump_password = jump_password
        self.jump_pkey = jump_pkey
        self.target_hosts = target_hosts or []
        self.target_user = target_user
        self.target_pkey = target_pkey
        self.max_parallel = max_parallel
        self.timeout = timeout

        self.jump_client = None
        self.jump_transport = None
        self.client = None

        # Properties for compatibility
        self.host_list = self.target_hosts
        self.reachable_hosts = self.target_hosts.copy()
        self.unreachable_hosts = []

        # Initialize
        self._connect_to_jump_host()
        self._create_parallel_client()

    def _connect_to_jump_host(self):
        """Connect to jump host using paramiko."""
        logger.info(f"Connecting to jump host: {self.jump_host}")
        logger.info(f"  Jump user: {self.jump_user}")
        logger.info(
            f"  Jump password: {'***SET*** (length={len(self.jump_password)})' if self.jump_password else 'NOT SET'}"
        )
        logger.info(f"  Jump pkey: {self.jump_pkey if self.jump_pkey else 'NOT SET'}")

        self.jump_client = paramiko.SSHClient()
        self.jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            if self.jump_password:
                logger.info(f"Attempting password authentication to {self.jump_host}...")
                logger.info(
                    f"  Password value check: {self.jump_password[:3]}*** (showing first 3 chars for verification)"
                )
                logger.info("Using password authentication for jump host")
                self.jump_client.connect(
                    hostname=self.jump_host,
                    username=self.jump_user,
                    password=self.jump_password,
                    timeout=self.timeout,
                    banner_timeout=60,
                )
            else:
                logger.info(f"Using key authentication for jump host: {self.jump_pkey}")
                self.jump_client.connect(
                    hostname=self.jump_host,
                    username=self.jump_user,
                    key_filename=self.jump_pkey,
                    timeout=self.timeout,
                    banner_timeout=60,
                )

            self.jump_transport = self.jump_client.get_transport()
            logger.info(f"✅ Connected to jump host: {self.jump_host}")

        except Exception as e:
            logger.error(f"❌ Failed to connect to jump host: {e}")
            raise

    def _make_proxy(self, host, port):
        """Create proxy socket through jump host."""
        logger.debug(f"Creating proxy socket for {host}:{port}")
        return self.jump_transport.open_channel(
            "direct-tcpip",
            (host, port),
            ("", 0),
        )

    def _create_parallel_client(self):
        """Setup for parallel execution - key file is ON the jump host."""
        logger.info(f"Ready for parallel SSH execution to {len(self.target_hosts)} nodes")
        logger.info(f"  Target user: {self.target_user}")
        logger.info(f"  Target pkey (on jump host): {self.target_pkey}")
        logger.info(f"  Max parallel: {self.max_parallel}")
        logger.info("  Method: Execute SSH commands on jump host using key file there")
        logger.info("✅ Ready to execute commands via jump host")

    def _execute_on_node(self, node: str, cmd: str, timeout: Optional[int] = None) -> str:
        """Execute command on a single node via jump host."""
        try:
            # Build SSH command to execute on jump host with timeout
            # Format: ssh -i /path/on/jumphost user@node "command"
            # Add ConnectTimeout to prevent hanging
            ssh_cmd = f"timeout {timeout or 60} ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -i {self.target_pkey} {self.target_user}@{node} '{cmd}'"

            logger.debug(f"[{node}] Executing via jump host: {ssh_cmd[:150]}...")

            # Execute on jump host with timeout
            stdin, stdout, stderr = self.jump_client.exec_command(ssh_cmd, timeout=(timeout or 60) + 10)

            # Collect output
            output = stdout.read().decode('utf-8', errors='ignore')
            error = stderr.read().decode('utf-8', errors='ignore')

            if error and not output:
                logger.warning(f"[{node}] stderr: {error[:200]}")
                return f"ERROR: {error}"

            return output

        except Exception as e:
            logger.error(f"[{node}] Exception: {e}")
            return f"ERROR: {str(e)}"

    def exec(self, cmd: str, timeout: Optional[int] = None, print_console: bool = True) -> Dict[str, str]:
        """
        Execute command on all nodes in parallel via jump host.
        Uses ThreadPoolExecutor for parallel execution.
        """
        logger.info(f"Executing command: {cmd[:100]}...")
        logger.info(f"Running on {len(self.target_hosts)} nodes (max {self.max_parallel} parallel)...")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}
        success_count = 0
        fail_count = 0

        try:
            # Execute in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
                # Submit all tasks
                future_to_node = {
                    executor.submit(self._execute_on_node, node, cmd, timeout): node for node in self.target_hosts
                }

                # Collect results as they complete
                for future in as_completed(future_to_node):
                    node = future_to_node[future]
                    try:
                        output = future.result()
                        results[node] = output

                        if output.startswith("ERROR"):
                            logger.error(f"❌ [{node}] FAILED: {output[:200]}")
                            fail_count += 1
                        else:
                            # Log first 3 lines
                            lines = output.split('\n')[:3]
                            logger.info(f"✅ [{node}] SUCCESS (first 3 lines):")
                            for line in lines:
                                if line.strip():
                                    logger.info(f"    {line[:150]}")
                            success_count += 1

                    except Exception as e:
                        results[node] = f"ERROR: {str(e)}"
                        logger.error(f"❌ [{node}] Exception: {e}")
                        fail_count += 1

            logger.info(f"Results: {success_count} successful, {fail_count} failed")
            return results

        except Exception as e:
            logger.error(f"❌ Parallel execution failed: {e}", exc_info=True)
            raise

    async def exec_async(self, cmd, timeout=None, print_console=True):
        """Async wrapper - just calls exec() directly."""
        return self.exec(cmd, timeout, print_console)

    def get_reachable_hosts(self):
        """Return list of reachable hosts."""
        return self.reachable_hosts.copy()

    def get_unreachable_hosts(self):
        """Return list of unreachable hosts."""
        return self.unreachable_hosts.copy()

    def destroy_clients(self):
        """Clean up connections."""
        logger.info("Closing connections...")
        if self.client:
            try:
                self.client.disconnect()
            except:
                pass
        if self.jump_client:
            try:
                self.jump_client.close()
                logger.info("✅ Jump host connection closed")
            except:
                pass
