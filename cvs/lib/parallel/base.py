'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from __future__ import print_function

from pssh.clients import ParallelSSHClient
from pssh.exceptions import Timeout, ConnectionError, SessionError

from cvs.lib.env_lib import build_env_prefix


def _chunk_hosts(host_list, chunk_size):
    """Yield successive slices of host_list of length up to chunk_size."""
    for i in range(0, len(host_list), chunk_size):
        yield host_list[i : i + chunk_size]


class PsshShard:
    """
    Single-process parallel SSH: one ParallelSSHClient (one gevent hub) over a host list.

    For large host counts, use Pssh (see cvs.lib.parallel.ssh), which shards hosts across processes.
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
        self.env_vars = env_vars
        self.env_prefix = build_env_prefix(env_vars)
        if self.log:
            self.log.debug(f"Environ vars: {self.env_prefix}")

        if self.password is None:
            print(self.reachable_hosts)
            print(self.user)
            print(self.pkey)
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
            print(f"Host {host} is unreachable, pruning from reachable hosts list.")
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
        Returns cmd_output dictionary.
        """
        cmd_output = {}
        i = 0
        for item in output:
            print('#----------------------------------------------------------#')
            print(f'Host == {item.host} ==')
            print('#----------------------------------------------------------#')
            cmd_out_str = ''
            if cmd_list:
                print(cmd_list[i])
            else:
                print(cmd)
            try:
                for line in item.stdout or []:
                    if print_console:
                        print(line)
                    cmd_out_str += line.replace('\t', '   ') + '\n'
                for line in item.stderr or []:
                    if print_console:
                        print(line)
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
                print(exc_str)
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

        print(f'cmd = {full_cmd}')

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
        if self.env_prefix:
            cmd_list = [f"{self.env_prefix} ; {cmd}" for cmd in cmd_list]
        else:
            cmd_list = cmd_list

        print(cmd_list)

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

        # Log per-host command execution
        if self.log:
            for host, cmd in zip(self.reachable_hosts, cmd_list):
                self.log.debug(f"Command on {host}: {cmd}")

        return cmd_output

    def scp_file(self, local_file, remote_file, recurse=False):
        print('About to copy local file {} to remote {} on all Hosts'.format(local_file, remote_file))
        cmds = self.client.copy_file(local_file, remote_file, recurse=recurse)
        self.client.pool.join()
        for cmd in cmds:
            try:
                cmd.get()
            except IOError:
                raise Exception("Expected IOError exception, got none")
        return

    def reboot_connections(self):
        print('Rebooting Connections')
        self.client.run_command('reboot -f', stop_on_errors=self.stop_on_errors)

    def destroy_clients(self):
        print('Destroying Current phdl connections ..')
        del self.client


def _shard_dispatch_exec(shard, payload):
    return shard.exec(payload['cmd'], timeout=payload.get('timeout'), print_console=False)


def _shard_dispatch_cmd_list(shard, payload):
    return shard.exec_cmd_list(payload['cmd_list'], timeout=payload.get('timeout'), print_console=False)


def _shard_dispatch_scp(shard, payload):
    shard.scp_file(payload['local_file'], payload['remote_file'], recurse=payload.get('recurse', False))
    return None


def _shard_dispatch_reboot(shard, payload):
    shard.reboot_connections()
    return None


_SHARD_MODE_HANDLERS = {
    'exec': _shard_dispatch_exec,
    'cmd_list': _shard_dispatch_cmd_list,
    'scp': _shard_dispatch_scp,
    'reboot': _shard_dispatch_reboot,
}


def _pssh_shard_worker(payload):
    """
    Top-level worker for ProcessPoolExecutor (must be picklable).
    payload['mode']: ``exec`` | ``cmd_list`` | ``scp`` | ``reboot``
    """
    mode = payload['mode']
    if not isinstance(mode, str):
        raise TypeError('payload["mode"] must be str, got %r' % (type(mode),))
    init = payload['init']
    handler = _SHARD_MODE_HANDLERS.get(mode)
    if handler is None:
        raise ValueError('unknown mode: %r' % (mode,))
    shard = PsshShard(**init)
    try:
        result = handler(shard, payload)
        return {
            'result': result,
            'reachable_hosts': list(shard.reachable_hosts),
            'unreachable_hosts': list(shard.unreachable_hosts),
        }
    finally:
        try:
            shard.destroy_clients()
        except Exception:
            pass
