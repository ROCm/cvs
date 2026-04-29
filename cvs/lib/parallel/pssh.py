'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from __future__ import print_function

import warnings
from pssh.clients import ParallelSSHClient
from pssh.exceptions import Timeout, ConnectionError, SessionError

from cvs.lib.env_lib import build_env_prefix
from cvs.lib import globals

global_log = globals.log


class SimpleHostOutput:
    """Simple HostOutput-compatible object for consistent processing across process boundaries."""

    def __init__(self, host, stdout_lines, stderr_lines, exception, exit_code=0):
        self.host = host
        self.stdout = iter(stdout_lines)  # Convert list to iterator for _process_output compatibility
        self.stderr = iter(stderr_lines)  # Convert list to iterator for _process_output compatibility
        self.exception = exception
        self.exit_code = exit_code


class Pssh:
    """
    Single-process parallel SSH: one ParallelSSHClient (one gevent hub) over a host list.

    For large host counts, use PsshSharded (see cvs.lib.parallel.pssh_sharded), which shards hosts across processes.
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
        env_vars=None,
        process_output=True,
    ):
        # Backward compatibility warning for log parameter
        if log is not None:
            warnings.warn(
                "Passing 'log' parameter is deprecated. "
                "Configure logging via cvs.lib.globals instead. "
                "This parameter will be removed in future versions.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Always use global logger but maintain self.log for backward compatibility
        self.log = global_log
        self.host_list = host_list
        self.reachable_hosts = host_list
        self.user = user
        self.pkey = pkey
        self.password = password
        self.host_key_check = host_key_check
        self.stop_on_errors = stop_on_errors
        self.process_output = process_output
        self.unreachable_hosts = []
        self.env_prefix = build_env_prefix(env_vars)
        self.log.debug(f"Environ vars: {self.env_prefix}")

        if self.password is None:
            self.log.info("%s", self.reachable_hosts)
            self.log.info("%s", self.user)
            self.log.info("%s", self.pkey)
            self.client = ParallelSSHClient(self.reachable_hosts, user=self.user, pkey=self.pkey, keepalive_seconds=30)
        else:
            self.client = ParallelSSHClient(
                self.reachable_hosts, user=self.user, password=self.password, keepalive_seconds=30
            )

    def check_connectivity(self, hosts):
        """
        Check connectivity for a list of hosts using one ParallelSSHClient.
        Returns a list of unreachable hosts.
        """
        if not hosts:
            return []
        temp_client = ParallelSSHClient(
            hosts,
            user=self.user,
            pkey=self.pkey if self.password is None else None,
            password=self.password,
            num_retries=0,
            timeout=2,
        )
        output = temp_client.run_command('echo 1', stop_on_errors=False, read_timeout=2)
        unreachable = [item.host for item in output if item.exception]
        return unreachable

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
            item.host
            for item in output
            if item.exception and isinstance(item.exception, (ConnectionError, Timeout, SessionError))
        ]
        unreachable = self.check_connectivity(failed_hosts)
        for host in unreachable:
            self.log.warning(f"Host {host} is unreachable, pruning from reachable hosts list.")
            self.unreachable_hosts.append(host)
            self.reachable_hosts.remove(host)
        if len(self.unreachable_hosts) > initial_unreachable_len:
            # Recreate client with filtered reachable_hosts, only if hosts were actually pruned
            if self.password is None:
                self.client = ParallelSSHClient(
                    self.reachable_hosts, user=self.user, pkey=self.pkey, keepalive_seconds=30
                )
            else:
                self.client = ParallelSSHClient(
                    self.reachable_hosts, user=self.user, password=self.password, keepalive_seconds=30
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
        Returns cmd_output dictionary OR SimpleHostOutput objects based on process_output flag.
        """
        # Check if we should return raw data for sharding
        if not self.process_output:
            return self._extract_simple_data(output)
        cmd_output = {}
        i = 0
        for item in output:
            self.log.info('#----------------------------------------------------------#')
            self.log.info(f'Host == {item.host} ==')
            self.log.info('#----------------------------------------------------------#')
            cmd_out_str = ''
            if cmd_list:
                self.log.debug("%s", cmd_list[i])
            else:
                self.log.debug("%s", cmd)
            try:
                for line in item.stdout or []:
                    if print_console:
                        self.log.info("%s", line)
                    cmd_out_str += line.replace('\t', '   ') + '\n'
                for line in item.stderr or []:
                    if print_console:
                        self.log.info("%s", line)
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
                self.log.warning("%s", exc_str)
                cmd_out_str += exc_str + '\n'
            if cmd_list:
                i += 1
            cmd_output[item.host] = cmd_out_str

        if not self.stop_on_errors:
            self.prune_unreachable_hosts(output)
            self.inform_unreachability(cmd_output)

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
        Returns a dictionary of host as key and command output as values
        """
        if self.env_prefix:
            full_cmd = f"{self.env_prefix} ; {cmd}"
        else:
            full_cmd = cmd

        self.log.info(f'cmd = {full_cmd}')

        # Log command execution
        if self.log:
            if timeout is not None:
                self.log.debug(
                    f"Executing command on {len(self.reachable_hosts)} host(s) [timeout={timeout}s]: {full_cmd}"
                )
            else:
                self.log.debug(f"Executing command on {len(self.reachable_hosts)} host(s): {full_cmd}")

        if timeout is None:
            output = self.client.run_command(full_cmd, stop_on_errors=self.stop_on_errors)
        else:
            output = self.client.run_command(full_cmd, read_timeout=timeout, stop_on_errors=self.stop_on_errors)

        cmd_output = self._process_output(output, cmd=full_cmd, print_console=print_console)

        # Log per-host execution completion (only for processed output)
        if self.process_output and self.log and isinstance(cmd_output, dict):
            for host in cmd_output.keys():
                self.log.debug(f"Command completed on {host}: {cmd}")

        return cmd_output

    def exec_cmd_list(self, cmd_list, timeout=None, print_console=True):
        """
        Run different commands on different hosts compared to to exec
        which runs the same command on all hosts.
        Returns a dictionary of host as key and command output as values
        """
        if self.env_prefix:
            cmd_list = [f"{self.env_prefix} ; {cmd}" for cmd in cmd_list]
        else:
            cmd_list = cmd_list

        self.log.info("%s", cmd_list)

        # Log command list execution
        if self.log:
            if timeout is not None:
                self.log.debug(f"Executing command list on {len(self.reachable_hosts)} host(s) [timeout={timeout}s]")
            else:
                self.log.debug(f"Executing command list on {len(self.reachable_hosts)} host(s)")

        if timeout is None:
            output = self.client.run_command('%s', host_args=cmd_list, stop_on_errors=self.stop_on_errors)
        else:
            output = self.client.run_command(
                '%s', host_args=cmd_list, read_timeout=timeout, stop_on_errors=self.stop_on_errors
            )

        cmd_output = self._process_output(output, cmd_list=cmd_list, print_console=print_console)

        # Log per-host command execution (only for processed output)
        if self.process_output and self.log and isinstance(cmd_output, dict):
            for host, cmd in zip(self.reachable_hosts, cmd_list):
                self.log.debug(f"Command on {host}: {cmd}")

        return cmd_output

    def scp_file(self, local_file, remote_file, recurse=False):
        self.log.info('About to copy local file {} to remote {} on all Hosts'.format(local_file, remote_file))
        cmds = self.client.copy_file(local_file, remote_file, recurse=recurse)
        self.client.pool.join()
        for cmd in cmds:
            try:
                cmd.get()
            except IOError:
                raise Exception("Expected IOError exception, got none")
        return

    def reboot_connections(self):
        self.log.info('Rebooting Connections')
        self.client.run_command('reboot -f', stop_on_errors=self.stop_on_errors)

    def _extract_simple_data(self, output):
        """
        Extract essential data from pssh.output.HostOutput objects for safe IPC transfer.

        Converts non-picklable HostOutput objects (containing active SSH channels,
        generators, and C extension objects) into picklable SimpleHostOutput objects
        by consuming stdout/stderr generators into lists and extracting core attributes.
        """

        simple_outputs = []
        for item in output:
            # Consume generators to lists for pickling
            stdout_lines = list(item.stdout or [])
            stderr_lines = list(item.stderr or [])

            simple_output = SimpleHostOutput(
                host=item.host,
                stdout_lines=stdout_lines,
                stderr_lines=stderr_lines,
                exception=item.exception,
                exit_code=getattr(item, 'exit_code', 0),
            )
            simple_outputs.append(simple_output)

        return simple_outputs

    def destroy_clients(self):
        self.log.info('Destroying Current phdl connections ..')
        del self.client
