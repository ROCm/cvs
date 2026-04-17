'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from __future__ import print_function
from pssh.clients import ParallelSSHClient
from pssh.exceptions import Timeout, ConnectionError

import asyncio
import socket
from contextlib import asynccontextmanager
from typing import AsyncIterator
import time
import logging
import threading

# Following used only for scp of file
import paramiko
from paramiko import SSHClient
from scp import SCPClient

# TCP probe for fast reachability detection
from app.core.host_probe import discover_reachable_hosts
from app.core.ssh_port_forward import _run_bridge

# Module-level logger
logger = logging.getLogger(__name__)

# Global lock to prevent concurrent SSH operations (parallel-ssh is not thread-safe)
_ssh_lock = threading.Lock()


class Pssh:
    """
    ParallelSessions - Uses the pssh library that is based of Paramiko, that lets you take
    multiple parallel ssh sessions to hosts and execute commands.

    Input host_config should be in this format ..
    mandatory args =  user, password (or) 'private_key': load_private_key('my_key.pem')
    """

    def __init__(
        self,
        log,
        host_list,
        user=None,
        password=None,
        pkey='id_rsa',
        host_key_check=False,
        stop_on_errors=True,
        timeout=30,
        proxy_host=None,
        proxy_user=None,
        proxy_password=None,
        proxy_pkey=None,
    ):
        self.log = log
        self.host_list = host_list
        self.reachable_hosts = host_list
        self.user = user
        self.pkey = pkey
        self.password = password
        self.host_key_check = host_key_check
        self.stop_on_errors = stop_on_errors
        self.unreachable_hosts = []
        self.proxy_host = proxy_host
        self.timeout = timeout

        # Build client parameters
        # Set num_retries=1 (one retry) for faster failure on unreachable nodes
        # NOTE: Do NOT set 'timeout' here - it acts as default read timeout for ALL commands
        #       Connection timeout is handled by num_retries and SSH protocol defaults
        # pool_size: Balance between parallelism and resource usage (50 for large clusters)
        client_params = {
            'user': self.user,
            'num_retries': 1,  # Only retry once (total 2 attempts) for fast failure on unreachable nodes
            'pool_size': 50,  # Reduced from 100 for stability with 617 hosts
            # keepalive_seconds: Omitted - use library default to avoid interference with long commands
        }

        # Add authentication
        if self.password is None:
            logger.debug(f"Reachable hosts: {self.reachable_hosts}")
            logger.debug(f"SSH user: {self.user}")
            logger.debug(f"SSH key: {self.pkey}")
            client_params['pkey'] = self.pkey
        else:
            client_params['password'] = self.password

        # Add jump host/proxy if configured
        if proxy_host:
            logger.info("Configuring jump host proxy:")
            logger.info(f"  Proxy host: {proxy_host}")
            logger.info(f"  Proxy user: {proxy_user}")
            logger.info(f"  Proxy password: {'***SET***' if proxy_password else 'NOT SET'}")
            logger.info(f"  Proxy pkey: {proxy_pkey if proxy_pkey else 'NOT SET'}")

            client_params['proxy_host'] = proxy_host
            if proxy_user:
                client_params['proxy_user'] = proxy_user
            if proxy_password:
                client_params['proxy_password'] = proxy_password
            elif proxy_pkey:
                client_params['proxy_pkey'] = proxy_pkey

        # Probe hosts for reachability before SSH connection
        logger.info(f"Probing {len(host_list)} hosts for reachability...")
        probe_start = time.time()
        self.reachable_hosts, self.unreachable_hosts = discover_reachable_hosts(
            host_list, port=22, timeout=5, max_workers=100
        )
        probe_duration = time.time() - probe_start
        logger.info(
            f"Probe completed in {probe_duration:.2f}s: "
            f"{len(self.reachable_hosts)} reachable, {len(self.unreachable_hosts)} unreachable"
        )

        self._pf_clients: dict[str, paramiko.SSHClient] = {}
        self._pf_lock = threading.Lock()  # protects _pf_clients dict

        # Only create ParallelSSHClient with reachable hosts
        if not self.reachable_hosts:
            logger.warning("No reachable hosts found! SSH manager will be inactive")
            self.client = None
            return

        logger.info("Creating ParallelSSHClient with params:")
        logger.info(f"  Hosts: {self.reachable_hosts}")
        logger.info(f"  User: {client_params.get('user')}")
        logger.info(f"  Password: {'***SET***' if client_params.get('password') else 'NOT SET'}")
        logger.info(f"  Pkey: {client_params.get('pkey', 'NOT SET')}")
        logger.info(f"  Proxy host: {client_params.get('proxy_host', 'NOT SET')}")
        logger.info(f"  Proxy password: {'***SET***' if client_params.get('proxy_password') else 'NOT SET'}")

        self.client = ParallelSSHClient(self.reachable_hosts, **client_params)
        logger.info("✅ ParallelSSHClient created successfully")
        logger.info(f"Ready to execute commands on {len(self.reachable_hosts)} reachable hosts")

    def check_connectivity(self, hosts):
        """
        Check connectivity for a list of hosts using one ParallelSSHClient.
        Returns a list of TRULY unreachable hosts (connection failures only, not slow hosts).
        Uses generous timeout to avoid false positives.

        NOTE: This method is now primarily used by prune_unreachable_hosts().
        Initial reachability is determined by TCP probes in __init__().
        """
        if not hosts:
            return []
        temp_client = ParallelSSHClient(
            hosts,
            user=self.user,
            pkey=self.pkey if self.password is None else None,
            password=self.password,
            num_retries=0,  # No retries for connectivity check
            pool_size=50,  # Reduced from 100 for stability
        )
        # Use 15 second timeout - enough for slow hosts but fast enough for unreachable detection
        output = temp_client.run_command('echo 1', stop_on_errors=False, read_timeout=15)

        # Only mark hosts with ConnectionError as unreachable (not Timeout - could just be slow)
        unreachable = [item.host for item in output if item.exception and isinstance(item.exception, ConnectionError)]
        return unreachable

    def refresh_host_reachability(self):
        """
        Re-probe all hosts and update reachable/unreachable lists.
        Returns True if the reachable host list changed.

        This is called periodically (every 5 minutes) and on mid-execution failures
        to detect nodes that have come online or gone offline.
        """
        logger.info("Refreshing host reachability...")
        old_reachable = set(self.reachable_hosts)

        # Re-probe all original hosts
        new_reachable, new_unreachable = discover_reachable_hosts(self.host_list, port=22, timeout=5, max_workers=100)

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

    def recreate_client(self):
        """
        Recreate ParallelSSHClient with current reachable_hosts.
        Called after host reachability changes are detected.
        """
        if not self.reachable_hosts:
            logger.warning("No reachable hosts! Clearing client.")
            if self.client:
                try:
                    self.client.disconnect()
                except:
                    pass
                self.client = None
            return

        logger.info(f"Recreating ParallelSSHClient with {len(self.reachable_hosts)} reachable hosts...")

        # Disconnect old client
        if self.client:
            try:
                self.client.disconnect()
            except:
                pass

        # Build client parameters (same as __init__)
        client_params = {
            'user': self.user,
            'num_retries': 1,
            'pool_size': 50,  # Reduced from 100 for stability with 617 hosts
        }

        if self.password is None:
            client_params['pkey'] = self.pkey
        else:
            client_params['password'] = self.password

        # Recreate client
        self.client = ParallelSSHClient(self.reachable_hosts, **client_params)
        logger.info("✅ ParallelSSHClient recreated successfully")

    def _handle_connection_failure(self):
        """
        Handle connection failures during command execution.
        Re-probes all hosts and recreates client if reachability changed.
        """
        logger.info("Handling connection failure - re-probing hosts...")
        changed = self.refresh_host_reachability()

        if changed:
            logger.info("Host reachability changed - recreating client")
            self.recreate_client()
        else:
            logger.info("No reachability changes detected")

    def prune_unreachable_hosts(self, output):
        """
        Prune unreachable hosts from self.reachable_hosts if they have ConnectionError or Timeout exceptions and also fail connectivity check.

        Targeted pruning: Only ConnectionError and Timeout exceptions trigger pruning to avoid removing hosts for transient failures
        like authentication errors or SSH protocol issues, which may succeed on next try. ConnectionErrors and Timeouts are indicative
        of potential unreachability, so we perform an additional connectivity check before pruning. This ensures
        that hosts are not permanently removed from the list for recoverable errors.
        """
        initial_unreachable_len = len(self.unreachable_hosts)
        failed_hosts = [
            item.host for item in output if item.exception and isinstance(item.exception, (ConnectionError, Timeout))
        ]
        unreachable = self.check_connectivity(failed_hosts)
        for host in unreachable:
            logger.info(f"Host {host} is unreachable, pruning from reachable hosts list.")
            self.unreachable_hosts.append(host)
            self.reachable_hosts.remove(host)
        if len(self.unreachable_hosts) > initial_unreachable_len:
            # Recreate client with filtered reachable_hosts, only if hosts were actually pruned
            if self.password is None:
                self.client = ParallelSSHClient(
                    self.reachable_hosts,
                    user=self.user,
                    pkey=self.pkey,
                    num_retries=1,
                    pool_size=50,
                )
            else:
                self.client = ParallelSSHClient(
                    self.reachable_hosts,
                    user=self.user,
                    password=self.password,
                    num_retries=1,
                    pool_size=50,
                )

    def inform_unreachability(self, cmd_output):
        """
        Update cmd_output with "Host Unreachable" for all hosts in self.unreachable_hosts.
        This ensures that the output dictionary reflects the status of pruned hosts.
        """
        for host in self.unreachable_hosts:
            cmd_output[host] = cmd_output.get(host, "") + "\nABORT: Host Unreachable Error"

    def _process_output(self, output, cmd=None, cmd_list=None, print_console=True):
        """
        Helper method to process output from run_command, collect results, and handle pruning.
        Returns cmd_output dictionary.
        """
        cmd_output = {}
        i = 0
        for item in output:
            logger.debug('#----------------------------------------------------------#')
            logger.debug(f'Host == {item.host} ==')
            logger.debug('#----------------------------------------------------------#')
            cmd_out_str = ''
            if cmd_list:
                logger.debug(cmd_list[i])
            else:
                logger.debug(cmd)
            try:
                for line in item.stdout or []:
                    if print_console:
                        logger.debug(line)
                    cmd_out_str += line.replace('\t', '   ') + '\n'
                for line in item.stderr or []:
                    if print_console:
                        logger.debug(line)
                    cmd_out_str += line.replace('\t', '   ') + '\n'
            except Timeout as e:
                if not self.stop_on_errors:
                    self._handle_timeout_exception(output, e)
                else:
                    raise
            if item.exception:
                exc_str = str(item.exception) if str(item.exception) else repr(item.exception)
                exc_str = exc_str.replace('\t', '   ')
                if isinstance(item.exception, Timeout):
                    exc_str += "\nABORT: Timeout Error in Host: " + item.host
                logger.debug(exc_str)
                cmd_out_str += exc_str + '\n'
            if cmd_list:
                i += 1
            cmd_output[item.host] = cmd_out_str

        if not self.stop_on_errors:
            self.prune_unreachable_hosts(output)
            self.inform_unreachability(cmd_output)

        # Log summary
        success = sum(1 for v in cmd_output.values() if not ("ERROR" in v or "ABORT" in v))
        failed = len(cmd_output) - success
        logger.info(f"✅ CVS Pssh completed: {success} successful, {failed} failed")

        # Log individual results
        for host, output_str in cmd_output.items():
            if "ERROR" in output_str or "ABORT" in output_str:
                logger.error(f"❌ [{host}] FAILED: {output_str[:150]}")
            else:
                lines = output_str.split('\n')[:3]
                logger.info(f"✅ [{host}] SUCCESS (first 3 lines):")
                for line in lines:
                    if line.strip():
                        logger.info(f"    {line[:150]}")

        return cmd_output

    def _handle_timeout_exception(self, output, e):
        """
        Helper method to handle Timeout exceptions by setting exceptions for all hosts in output.
        Since Timeout is raised once for the operation, assume all hosts are affected.
        """
        if output is not None and isinstance(e, Timeout):
            for item in output:
                if item.exception is None:
                    item.exception = e

    def exec(self, cmd, timeout=None, print_console=True):
        """
        Returns a dictionary of host as key and command output as values.
        Thread-safe: Uses lock to prevent concurrent SSH operations.
        """
        # Check if client is available
        if not self.client:
            logger.warning("No SSH client available (no reachable hosts)")
            # Return error for all original hosts
            return {host: "ABORT: Host Unreachable Error" for host in self.host_list}

        # CRITICAL: Acquire lock to prevent concurrent SSH operations
        # parallel-ssh/paramiko/libssh2 are NOT thread-safe
        with _ssh_lock:
            # Re-check after acquiring the lock: destroy_clients() may have run
            # while we were waiting, setting self.client = None.
            if not self.client:
                logger.info("SSH client destroyed before command could run (shutdown race) — skipping")
                return {host: "ABORT: Host Unreachable Error" for host in self.host_list}

            logger.info(f"CVS Pssh executing: {cmd[:100]}...")
            logger.info(f"Calling ParallelSSHClient.run_command() on {len(self.reachable_hosts)} reachable nodes...")
            logger.info(f"  Timeout: {timeout if timeout else 'default'}")
            logger.info(f"  Stop on errors: {self.stop_on_errors}")

            logger.debug(f'cmd = {cmd}')

            try:
                if timeout is None:
                    logger.info("Starting run_command (no timeout)...")
                    output = self.client.run_command(cmd, stop_on_errors=self.stop_on_errors)
                else:
                    logger.info(f"Starting run_command (read_timeout={timeout})...")
                    output = self.client.run_command(cmd, read_timeout=timeout, stop_on_errors=self.stop_on_errors)

                logger.info(f"✅ run_command returned {len(list(output))} results")
                cmd_output = self._process_output(output, cmd=cmd, print_console=print_console)
            except ConnectionError as e:
                # Connection error during execution - trigger re-probe
                logger.warning(f"ConnectionError during execution: {e}")
                logger.info("Triggering host re-probe...")
                self._handle_connection_failure()
                raise
            except Exception as e:
                logger.error(f"❌ run_command raised exception: {e}", exc_info=True)
                raise
            return cmd_output

    def exec_cmd_list(self, cmd_list, timeout=None, print_console=True):
        """
        Run different commands on different hosts compared to to exec
        which runs the same command on all hosts.
        Returns a dictionary of host as key and command output as values
        """
        logger.debug(cmd_list)
        if timeout is None:
            output = self.client.run_command('%s', host_args=cmd_list, stop_on_errors=self.stop_on_errors)
        else:
            output = self.client.run_command(
                '%s', host_args=cmd_list, read_timeout=timeout, stop_on_errors=self.stop_on_errors
            )
        cmd_output = self._process_output(output, cmd_list=cmd_list, print_console=print_console)
        return cmd_output

    def scp_file(self, local_file, remote_file, recurse=False):
        logger.info('About to copy local file {} to remote {} on all Hosts'.format(local_file, remote_file))
        cmds = self.client.copy_file(local_file, remote_file, recurse=recurse)
        self.client.pool.join()
        for cmd in cmds:
            try:
                cmd.get()
            except IOError:
                raise Exception("Expected IOError exception, got none")
        return

    def get_reachable_hosts(self):
        """Return list of reachable hosts."""
        return self.reachable_hosts.copy()

    def get_unreachable_hosts(self):
        """Return list of unreachable hosts."""
        return self.unreachable_hosts.copy()

    def reboot_connections(self):
        logger.info('Rebooting Connections')
        self.client.run_command('reboot -f', stop_on_errors=self.stop_on_errors)

    def _get_pf_transport(self, node: str) -> paramiko.Transport:
        """
        Get or create a dedicated paramiko SSH client for port-forwarding to node.
        Thread-safe. Separate from the pssh connection pool (which does not expose transports).

        Security note: Uses AutoAddPolicy() (TOFU). See plan for hardening options.
        """
        with self._pf_lock:
            client = self._pf_clients.get(node)
            transport = client.get_transport() if client else None
            if transport is None or not transport.is_active():
                new_client = paramiko.SSHClient()
                new_client.set_missing_host_key_policy(
                    paramiko.AutoAddPolicy()
                    # Security note: AutoAddPolicy() accepts any host key without
                    # verification (TOFU). Production hardening: pre-distribute known
                    # host keys via Ansible/Puppet and use RejectPolicy() + a
                    # pre-populated known_hosts file, or use OpenSSH certificate auth.
                )
                new_client.connect(
                    node,
                    username=self.user,
                    key_filename=self.pkey,
                    password=self.password,
                    timeout=self.timeout,
                )
                if client:
                    try:
                        client.close()  # close the stale connection before replacing
                    except Exception:
                        pass
                self._pf_clients[node] = new_client
            return self._pf_clients[node].get_transport()

    @asynccontextmanager
    async def open_port_forward(
        self, node: str, remote_port: int
    ) -> AsyncIterator[tuple]:
        """
        Open a single-hop SSH tunnel: monitoring_host -> node:remote_port.

        Yields (asyncio.StreamReader, asyncio.StreamWriter) ready for asyncio use.
        Uses a Unix socketpair() -- no ephemeral TCP port allocation, no TOCTOU race.
        """
        asyncio_end, thread_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            transport = await asyncio.to_thread(self._get_pf_transport, node)
            channel = await asyncio.to_thread(
                transport.open_channel,
                "direct-tcpip",
                ("::1", remote_port),
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

    def destroy_clients(self):
        logger.info('Destroying Current phdl connections ..')
        with _ssh_lock:
            self.client = None  # set to None (not del) so exec() guard stays valid
        with self._pf_lock:
            for c in self._pf_clients.values():
                try:
                    c.close()
                except Exception:
                    pass
            self._pf_clients.clear()

    async def exec_async(self, cmd, timeout=None, print_console=True):
        """
        Async wrapper for exec() that runs in a thread pool to avoid blocking the event loop.

        This allows async API endpoints to call SSH commands without blocking other requests.
        """
        import asyncio

        return await asyncio.to_thread(self.exec, cmd, timeout, print_console)


def scp(src, dst, srcusername, srcpassword, dstusername=None, dstpassword=None):
    """
    This method gets/puts files from one server to another
    :param arg: These are sub arguments for scp command
    :return: None
    :examples:
        To get remote file '/tmp/x' from 1.1.1.1 to local server '/home/user/x'
        scp('1.1.1.1:/tmp/x', '/home/user/x', 'root', 'docker')
        To put local file  '/home/user/x to remote server-B's /tmp/x'
        scp('/home/user/x', '1.1.1.1:/tmp/x', 'root', 'docker')
        To copy remote file '/tmp/x' from 1.1.1.1 to remote server 1.1.1.2 '/home/user/x'
        scp('1.1.1.1:/tmp/x','1.1.1.2:/home/user/x','root','docker','root','docker')
    """

    ssh = SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.load_system_host_keys()
    srclist = src.split(":")
    dstlist = dst.split(":")
    # 0 means get, 1 means put, 2 means server A to server B
    get_put = 1
    srcip = None
    dstip = None

    if len(srclist) == 2:
        srcip = srclist[0]
        srcfile = srclist[1]
        ssh.connect(srcip, username=srcusername, password=srcpassword)
        get_put = 0
    else:
        srcfile = srclist[0]

    if len(dstlist) == 2:
        dstip = dstlist[0]
        dstfile = dstlist[1]
        if get_put == 0:
            get_put = 2
        else:
            get_put = 1
            ssh.connect(dstip, username=srcusername, password=srcpassword)
    else:
        dstfile = dstlist[0]
    if get_put < 2:
        scp = SCPClient(ssh.get_transport())
        if not get_put:
            scp.get(srcfile, dstfile)
        else:
            scp.put(srcfile, dstfile)
        scp.close()
    else:
        if dstusername is None:
            dstusername = srcusername
        if dstpassword is None:
            dstpassword = srcpassword
        # This is to handle if ssh keys in the known_hosts is empty or incorrect
        # Need better way to handle in the future
        ssh.exec_command('ssh-keygen -R %s' % (dstip))
        time.sleep(1)
        ssh.exec_command('ssh-keyscan %s >> ~/.ssh/known_hosts' % (dstip))
        time.sleep(1)
        ssh.exec_command('sshpass -p %s scp %s %s@%s:%s' % (dstpassword, srcfile, dstusername, dstip, dstfile))
