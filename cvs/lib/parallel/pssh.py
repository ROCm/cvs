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
        # Jump host parameters
        jump_host=None,
        jump_user=None,
        jump_password=None,
        jump_pkey=None,
        jump_port=22,
        **ssh_client_kwargs,
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
        self.host_list = list(host_list)
        self.reachable_hosts = list(host_list)
        self.user = user
        self.pkey = pkey
        self.password = password
        self.host_key_check = host_key_check
        self.stop_on_errors = stop_on_errors
        self.process_output = process_output
        self.unreachable_hosts = []
        self.ssh_client_kwargs = ssh_client_kwargs
        self.env_prefix = build_env_prefix(env_vars)
        self.log.debug(f"Environ vars: {self.env_prefix}")

        # Jump host (bastion) support: the native ParallelSSHClient tunnels
        # target connections through its built-in proxy_* parameters, so no
        # extra SSH library (paramiko) is needed.
        if jump_host:
            self.ssh_client_kwargs['proxy_host'] = jump_host
            if jump_user:
                self.ssh_client_kwargs['proxy_user'] = jump_user
            if jump_password:
                self.ssh_client_kwargs['proxy_password'] = jump_password
            if jump_pkey:
                self.ssh_client_kwargs['proxy_pkey'] = jump_pkey
            if jump_port != 22:
                self.ssh_client_kwargs['proxy_port'] = jump_port

        if self.password is None:
            self.log.info("%s", self.reachable_hosts)
            self.log.info("%s", self.user)
            self.log.info("%s", self.pkey)
            self.client = ParallelSSHClient(
                self.reachable_hosts, user=self.user, pkey=self.pkey, keepalive_seconds=30, **self.ssh_client_kwargs
            )
        else:
            self.client = ParallelSSHClient(
                self.reachable_hosts,
                user=self.user,
                password=self.password,
                keepalive_seconds=30,
                **self.ssh_client_kwargs,
            )

    def check_connectivity(self, hosts):
        """
        Check connectivity for a list of hosts using one ParallelSSHClient.
        Returns a list of unreachable hosts.
        """
        if not hosts:
            return []
        temp_ssh_client_kwargs = self.ssh_client_kwargs.copy()
        temp_ssh_client_kwargs['timeout'] = 2
        temp_ssh_client_kwargs['num_retries'] = 0
        temp_client = ParallelSSHClient(
            hosts,
            user=self.user,
            pkey=self.pkey if self.password is None else None,
            password=self.password,
            **temp_ssh_client_kwargs,
        )
        output = temp_client.run_command('echo 1', stop_on_errors=False, read_timeout=2)
        unreachable = [item.host for item in output if item.exception]
        return unreachable

    def prune_nodes(self, nodes_to_remove):
        """
        Explicitly prune hosts from this Pssh instance and rebuild client.

        Args:
            nodes_to_remove: Iterable of hostnames/IPs to remove.

        Returns:
            list: Hosts actually removed from reachable_hosts.
        """
        if not nodes_to_remove:
            return []

        remove_set = {h for h in nodes_to_remove if h}
        removed = [h for h in self.reachable_hosts if h in remove_set]
        if not removed:
            return []

        for host in removed:
            self.log.warning(f"Host {host} is unreachable, pruning from reachable hosts list.")

        self.reachable_hosts = [h for h in self.reachable_hosts if h not in remove_set]
        for host in removed:
            if host not in self.unreachable_hosts:
                self.unreachable_hosts.append(host)

        if self.password is None:
            self.client = ParallelSSHClient(
                self.reachable_hosts, user=self.user, pkey=self.pkey, keepalive_seconds=30, **self.ssh_client_kwargs
            )
        else:
            self.client = ParallelSSHClient(
                self.reachable_hosts,
                user=self.user,
                password=self.password,
                keepalive_seconds=30,
                **self.ssh_client_kwargs,
            )
        return removed

    def prune_unreachable_hosts(self, output):
        """
        Prune unreachable hosts from self.reachable_hosts if they have ConnectionError or Timeout exceptions and also fail connectivity check.

        Targeted pruning: Only ConnectionError and Timeout exceptions trigger pruning to avoid removing hosts for transient failures
        like authentication errors or SSH protocol issues, which may succeed on next try. ConnectionErrors and Timeouts are indicative
        of potential unreachability, so we perform an additional connectivity check before pruning. This ensures
        that hosts are not permanently removed from the list for recoverable errors.
        """
        failed_hosts = [
            item.host
            for item in output
            if item.exception and isinstance(item.exception, (ConnectionError, Timeout, SessionError))
        ]
        unreachable = self.check_connectivity(failed_hosts)
        self.prune_nodes(unreachable)

    def inform_unreachability(self, cmd_output, include_exit_codes=False):
        """
        Update cmd_output with "Host Unreachable" for all hosts in self.unreachable_hosts.
        Handles both string and structured formats based on include_exit_codes.
        """
        for host in self.unreachable_hosts:
            if include_exit_codes:
                existing = cmd_output.get(host, {'output': '', 'exit_code': -1})
                existing['output'] += "\nABORT: Host Unreachable Error"
                existing['exit_code'] = -1  # Ensure unreachable hosts have error exit code
                cmd_output[host] = existing
            else:
                cmd_output[host] = cmd_output.get(host, "") + "\nABORT: Host Unreachable Error"

    def _process_output(self, output, cmd=None, cmd_list=None, print_console=True, include_exit_codes=False):
        """
        Helper method to process output from run_command, collect results, and handle pruning.
        Returns cmd_output dictionary. If include_exit_codes=True, returns structured format.
        """
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

            if include_exit_codes:
                # -1 means "unknown / aborted": either the command raised before
                # an exit code was available, or the SSH channel hasn't reached
                # EOF yet (parallel-ssh's HostOutput.exit_code property returns
                # None in that case, which would silently break consumers that
                # compare against 0).
                if item.exception is not None:
                    exit_code = -1
                else:
                    exit_code = item.exit_code
                    if exit_code is None:
                        exit_code = -1
                cmd_output[item.host] = {'output': cmd_out_str, 'exit_code': exit_code}
            else:
                cmd_output[item.host] = cmd_out_str

        if not self.stop_on_errors:
            self.prune_unreachable_hosts(output)
            self.inform_unreachability(cmd_output, include_exit_codes)

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

    def exec(self, cmd, timeout=None, print_console=True, detailed=False):
        """
        Returns a dictionary of host as key and command output as values.
        If detailed=True, returns structured dict with 'output' and 'exit_code' keys.
        If detailed=False (default), returns output strings.
        """
        # Jump host connection handled by ParallelSSHClient proxy parameters

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
        cmd_output = self._process_output(
            output, cmd=full_cmd, print_console=print_console, include_exit_codes=detailed
        )

        # Log per-host execution completion
        if self.log:
            for host in cmd_output.keys():
                self.log.debug(f"Command completed on {host}: {cmd}")

        return cmd_output

    def exec_cmd_list(self, cmd_list, timeout=None, print_console=True):
        """
        Run different commands on different hosts compared to to exec
        which runs the same command on all hosts.
        Returns a dictionary of host as key and command output as values
        """
        # Jump host connection handled by ParallelSSHClient proxy parameters

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
        """
        Backward-compatible alias for upload_file.

        Kept so existing callers (and log-grep tooling looking for the legacy
        "About to copy local file..." line) keep working. New code should call
        upload_file directly.
        """
        self.log.info('About to copy local file {} to remote {} on all Hosts'.format(local_file, remote_file))
        self.upload_file(local_file, remote_file, recurse=recurse)

    def upload_file(self, local_file, remote_file, recurse=False):
        """
        SFTP-upload a local file from the runner node to `remote_file` on every host
        in this Pssh's reachable_hosts. Wraps ParallelSSHClient.copy_file.

        Use this instead of embedding file contents in `exec()` command strings
        (e.g. heredoc cat>EOF), which is bounded by libssh2's ~30 KB exec_request
        cap. SFTP transfers go over a separate channel with no such limit and
        auto-create missing parent directories when recurse=True.

        For a 1:1 runner->head_node push, construct the Pssh with [head_node].

        Parameters:
          local_file: Path on the runner node to read from.
          remote_file: Absolute destination path on each remote host.
          recurse: If True, copy a directory tree (parent dirs auto-created).

        Raises:
          IOError: If transfer fails on any host. Message lists offending hosts.
        """
        self.log.info('SFTP upload %s -> %s on %s', local_file, remote_file, self.reachable_hosts)
        cmds = self.client.copy_file(local_file, remote_file, recurse=recurse)
        self.client.pool.join()
        errors = []
        for cmd, host in zip(cmds, self.reachable_hosts):
            try:
                cmd.get()
            except Exception as e:
                errors.append((host, e))
        if errors:
            raise IOError(
                f"upload_file '{local_file}' -> '{remote_file}' failed on "
                f"{len(errors)}/{len(self.reachable_hosts)} hosts: {errors}"
            ) from errors[0][1]

    def download_file(self, remote_file, local_file, recurse=False, suffix_separator='_'):
        """
        SFTP-download `remote_file` from every host in this Pssh's reachable_hosts
        to the runner node. Wraps ParallelSSHClient.copy_remote_file.

        Use this instead of `exec('cat <file>')` + parsing stdout when the file
        may exceed a few KB. `cat`-over-exec reassembles bytes through the
        line-oriented stdout pipeline of `_process_output`, which is fragile for
        binary content and unnecessary work for any text payload of nontrivial
        size.

        Note: parallel-ssh suffixes the local filename with `<suffix_separator><host>`
        to avoid collisions when downloading the same path from multiple hosts.
        Use the returned dict to look up the actual on-disk path per host.

        Parameters:
          remote_file: Absolute path on each remote host.
          local_file: Local path prefix on the runner node (host name will be appended).
          recurse: If True, recursively download a directory tree.
          suffix_separator: Separator placed between local_file and host. Default '_'.

        Returns:
          dict: {host: actual_local_path} for each host in reachable_hosts.

        Raises:
          IOError: If transfer fails on any host. Message lists offending hosts.
        """
        self.log.info('SFTP download %s -> %s from %s', remote_file, local_file, self.reachable_hosts)
        cmds = self.client.copy_remote_file(remote_file, local_file, recurse=recurse, suffix_separator=suffix_separator)
        self.client.pool.join()
        errors = []
        result = {}
        for cmd, host in zip(cmds, self.reachable_hosts):
            try:
                cmd.get()
                result[host] = f'{local_file}{suffix_separator}{host}'
            except Exception as e:
                errors.append((host, e))
        if errors:
            raise IOError(
                f"download_file '{remote_file}' -> '{local_file}' failed on "
                f"{len(errors)}/{len(self.reachable_hosts)} hosts: {errors}"
            ) from errors[0][1]
        return result

    def upload_file_list(self, node_path_map):
        """
        Upload different files to different hosts using SFTP.

        Args:
            node_path_map: dict mapping {host: (local_path, remote_path)}

        Returns:
            dict: {host: "host: SUCCESS" | "host: FAILED - ..."}
        """
        if not node_path_map:
            return {}

        # Build copy_args for each host
        copy_args = []
        valid_hosts = []

        for host in self.reachable_hosts:
            if host in node_path_map:
                local_path, remote_path = node_path_map[host]
                copy_args.append({'local_file': local_path, 'remote_file': remote_path})
                valid_hosts.append(host)

        if not copy_args:
            return {}

        self.log.info(f"Uploading {len(copy_args)} different files to {len(valid_hosts)} hosts")

        # Use copy_file with copy_args - returns greenlets
        cmds = self.client.copy_file("%(local_file)s", "%(remote_file)s", copy_args=copy_args)

        # Wait for greenlets to complete (like upload_file does)
        self.client.pool.join()

        # Process greenlet results
        results = {}
        for cmd, host in zip(cmds, valid_hosts):
            try:
                cmd.get()  # This will raise if the greenlet failed
                results[host] = f"{host}: SUCCESS"
            except Exception as e:
                results[host] = f"{host}: FAILED - {e}"

        return results

    def reboot_connections(self):
        self.log.info('Rebooting Connections')
        self.client.run_command('reboot -f', stop_on_errors=self.stop_on_errors)

    def destroy_clients(self):
        self.log.info('Destroying Current phdl connections ..')
        if hasattr(self, 'client'):
            del self.client
