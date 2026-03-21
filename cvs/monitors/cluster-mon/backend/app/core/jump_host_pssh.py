"""
Jump Host Parallel SSH using paramiko + pssh.
Based on working test_auth_script.py approach.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, AsyncIterator
import logging
import socket
import threading
import time

import paramiko

from app.core.ssh_port_forward import _run_bridge

# TCP probe for fast reachability detection
from app.core.host_probe import probe_from_bastion

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

        # Initialize - connect to jump host first
        self._connect_to_jump_host()

        # Probe cluster nodes via jump host for reachability
        logger.info(f"Probing {len(self.target_hosts)} cluster nodes via jump host...")
        probe_start = time.time()
        try:
            self.reachable_hosts, self.unreachable_hosts = probe_from_bastion(
                self.jump_client, self.target_hosts, port=22, timeout=5
            )
        except Exception as e:
            logger.error(f"Failed to probe nodes via jump host: {e}")
            logger.warning("Assuming all nodes are reachable (probe failed)")
            self.reachable_hosts = self.target_hosts.copy()
            self.unreachable_hosts = []

        probe_duration = time.time() - probe_start
        logger.info(
            f"Probe via jump host completed in {probe_duration:.2f}s: "
            f"{len(self.reachable_hosts)} reachable, {len(self.unreachable_hosts)} unreachable"
        )

        self._create_parallel_client()

        self._exec_lock = threading.Lock()

    def _is_jump_host_alive(self):
        """Check if jump host connection is still active."""
        if not self.jump_client:
            return False
        try:
            transport = self.jump_client.get_transport()
            return transport is not None and transport.is_active()
        except:
            return False

    def _ensure_jump_host_connection(self):
        """Ensure jump host connection is active, reconnect if needed."""
        if self._is_jump_host_alive():
            return True

        logger.warning("Jump host connection is not active - reconnecting...")
        try:
            # Close old connection if exists
            if self.jump_client:
                try:
                    self.jump_client.close()
                except:
                    pass

            # Reconnect
            self._connect_to_jump_host()
            return self._is_jump_host_alive()
        except Exception as e:
            logger.error(f"Failed to reconnect to jump host: {e}")
            return False

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
        # Skip if node is in unreachable list
        if node in self.unreachable_hosts:
            logger.debug(f"[{node}] Skipping - marked as unreachable")
            return "ABORT: Host Unreachable Error"

        try:
            # Build SSH command to execute on jump host with timeout
            # Format: ssh -i /path/on/jumphost user@node "command"
            # Add ConnectTimeout=30 and ConnectionAttempts=2 for faster failure
            ssh_cmd = f"timeout {timeout or 60} ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 -o ConnectionAttempts=2 -i {self.target_pkey} {self.target_user}@{node} '{cmd}'"

            logger.debug(f"[{node}] Executing via jump host: {ssh_cmd[:150]}...")

            # Execute on jump host with timeout
            stdin, stdout, stderr = self.jump_client.exec_command(ssh_cmd, timeout=(timeout or 60) + 10)

            # Collect output
            output = stdout.read().decode('utf-8', errors='ignore')
            error = stderr.read().decode('utf-8', errors='ignore')

            # Check for connection failures and mark node as unreachable
            if error:
                # Connection timeout or refused = unreachable node
                if any(
                    x in error.lower()
                    for x in ['connection timed out', 'connection refused', 'no route to host', 'host is down']
                ):
                    if node not in self.unreachable_hosts:
                        logger.warning(f"[{node}] Marking as unreachable: {error[:200]}")
                        self.unreachable_hosts.append(node)
                        if node in self.reachable_hosts:
                            self.reachable_hosts.remove(node)
                    return f"ABORT: Host Unreachable Error - {error[:100]}"
                elif not output:
                    logger.warning(f"[{node}] stderr: {error[:200]}")
                    return f"ERROR: {error}"

            return output

        except Exception as e:
            # Check if it's a timeout exception
            error_str = str(e).lower()
            if 'timeout' in error_str or 'timed out' in error_str:
                if node not in self.unreachable_hosts:
                    logger.warning(f"[{node}] Marking as unreachable due to timeout: {e}")
                    self.unreachable_hosts.append(node)
                    if node in self.reachable_hosts:
                        self.reachable_hosts.remove(node)
                return "ABORT: Host Unreachable Error - Timeout"

            logger.error(f"[{node}] Exception: {e}")
            return f"ERROR: {str(e)}"

    def exec(self, cmd: str, timeout: Optional[int] = None, print_console: bool = True) -> Dict[str, str]:
        """
        Execute command on all nodes in parallel via jump host.
        Uses ThreadPoolExecutor for parallel execution.
        Skips unreachable nodes and reports them separately.
        """
        with self._exec_lock:
            # Ensure jump host connection is active before executing
            if not self._ensure_jump_host_connection():
                logger.error("Cannot execute command - jump host connection failed")
                return {node: "ERROR: Jump host connection failed" for node in self.target_hosts}

            logger.info(f"Executing command: {cmd[:100]}...")
            logger.info(
                f"Total nodes: {len(self.target_hosts)}, Reachable: {len(self.reachable_hosts)}, Unreachable: {len(self.unreachable_hosts)}"
            )

            from concurrent.futures import ThreadPoolExecutor, as_completed

            results = {}
            success_count = 0
            fail_count = 0

            # First, add unreachable hosts to results
            for node in self.unreachable_hosts:
                results[node] = "ABORT: Host Unreachable Error"
                fail_count += 1

            try:
                # Execute in parallel using ThreadPoolExecutor on reachable hosts only
                with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
                    # Submit tasks only for reachable nodes
                    future_to_node = {
                        executor.submit(self._execute_on_node, node, cmd, timeout): node for node in self.reachable_hosts
                    }

                    # Collect results as they complete
                    for future in as_completed(future_to_node):
                        node = future_to_node[future]
                        try:
                            output = future.result()
                            results[node] = output

                            if output.startswith("ERROR") or output.startswith("ABORT"):
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

                # If too many failures, trigger re-probe (connection issue detection)
                failure_rate = fail_count / len(self.target_hosts) if self.target_hosts else 0
                if failure_rate > 0.5 and fail_count > 5:  # More than 50% failed and at least 5 failures
                    logger.warning(f"High failure rate ({failure_rate:.1%}) - triggering re-probe")
                    self._handle_connection_failure()

                return results

            except Exception as e:
                logger.error(f"❌ Parallel execution failed: {e}", exc_info=True)
                # Check if it's a connection error to jump host
                if "connection" in str(e).lower() or "transport" in str(e).lower():
                    logger.warning("Jump host connection issue detected - triggering re-probe")
                    self._handle_connection_failure()
                raise

    async def exec_async(self, cmd: str, timeout: int = 30, print_console: bool = False) -> dict:
        """Non-blocking wrapper around exec() using asyncio.to_thread."""
        return await asyncio.to_thread(self.exec, cmd, timeout, print_console)

    def get_reachable_hosts(self):
        """Return list of reachable hosts."""
        return self.reachable_hosts.copy()

    def get_unreachable_hosts(self):
        """Return list of unreachable hosts."""
        return self.unreachable_hosts.copy()

    def refresh_host_reachability(self):
        """
        Re-probe all cluster nodes via jump host and update reachable/unreachable lists.
        Returns True if the reachable host list changed.

        This is called periodically (every 5 minutes) and on mid-execution failures
        to detect nodes that have come online or gone offline.
        """
        logger.info("Refreshing host reachability via jump host...")

        # Ensure jump host connection is active before probing
        if not self._ensure_jump_host_connection():
            logger.error("Cannot refresh reachability - jump host connection failed")
            return False

        old_reachable = set(self.reachable_hosts)

        try:
            # Re-probe all target hosts via jump host
            new_reachable, new_unreachable = probe_from_bastion(self.jump_client, self.target_hosts, port=22, timeout=5)

            # Check for changes
            new_reachable_set = set(new_reachable)
            newly_reachable = new_reachable_set - old_reachable
            newly_unreachable = old_reachable - new_reachable_set

            if newly_reachable or newly_unreachable:
                logger.info("Host reachability changed:")
                if newly_reachable:
                    logger.info(f"  Newly reachable ({len(newly_reachable)}): {list(newly_reachable)[:10]}")
                if newly_unreachable:
                    logger.info(f"  Newly unreachable ({len(newly_unreachable)}): {list(newly_unreachable)[:10]}")

            # Update lists
            self.reachable_hosts = new_reachable
            self.unreachable_hosts = new_unreachable

            return len(old_reachable) != len(new_reachable_set) or old_reachable != new_reachable_set

        except Exception as e:
            logger.error(f"Failed to refresh reachability: {e}")
            # Keep existing lists on error
            return False

    def recreate_client(self):
        """
        Recreate client connection (no-op for JumpHostPssh).

        For JumpHostPssh, we don't need to recreate anything because:
        1. We maintain a single persistent jump host connection
        2. Commands are executed via SSH from jump host (no direct connections to nodes)
        3. Reachable/unreachable filtering happens in exec() by skipping unreachable nodes

        This method exists for API compatibility with Pssh class.
        """
        logger.info("recreate_client called (no-op for JumpHostPssh)")
        # No action needed - we use the same jump_client regardless of node reachability

    def _handle_connection_failure(self):
        """
        Handle connection failures during command execution.
        Re-probes all hosts via jump host.
        """
        logger.info("Handling connection failure - re-probing hosts via jump host...")
        changed = self.refresh_host_reachability()

        if changed:
            logger.info("Host reachability changed")
            # Note: No need to recreate client for JumpHostPssh
        else:
            logger.info("No reachability changes detected")

    @asynccontextmanager
    async def open_port_forward(
        self, node: str, remote_port: int
    ) -> AsyncIterator[tuple]:
        """
        Open a two-hop SSH tunnel: monitoring_host -> jump_host -> node:remote_port.

        Yields (asyncio.StreamReader, asyncio.StreamWriter) ready for asyncio use.
        Uses a Unix socketpair() -- no ephemeral TCP port allocation, no TOCTOU race.

        Security note: Uses AutoAddPolicy() for host key verification (TOFU).
        See plan for hardening options (pre-distributed known_hosts, SSH certificates).

        Args:
            node: Target node hostname/IP
            remote_port: Port on the target node to forward to

        Yields:
            (reader, writer) connected to node:remote_port via the jump host
        """
        asyncio_end, thread_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            channel = await asyncio.to_thread(
                self.jump_transport.open_channel,
                "direct-tcpip",
                (node, remote_port),
                ("127.0.0.1", 0),
            )
        except Exception:
            asyncio_end.close()
            thread_end.close()
            raise

        _run_bridge(channel, thread_end)

        try:
            reader, writer = await asyncio.open_connection(sock=asyncio_end)
        except Exception:
            asyncio_end.close()
            channel.close()
            raise

        try:
            yield reader, writer
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            channel.close()
            thread_end.close()
            # asyncio_end is owned by the asyncio transport after open_connection();
            # closing writer causes it to be closed automatically.

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
