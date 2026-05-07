'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from __future__ import print_function
from pssh.clients import ParallelSSHClient
from pssh.exceptions import Timeout, ConnectionError, SessionError

import time

# Following used only for scp of file
import paramiko
from paramiko import SSHClient
from scp import SCPClient
from cvs.lib.env_lib import build_env_prefix


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
        Returns cmd_output dictionary.
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

        # Log per-host command execution
        if self.log:
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
        self.log.info(
            'SFTP upload %s -> %s on %s', local_file, remote_file, self.reachable_hosts
        )
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
                f'upload_file failed on {len(errors)}/{len(self.reachable_hosts)} hosts: {errors}'
            )

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
        self.log.info(
            'SFTP download %s -> %s from %s', remote_file, local_file, self.reachable_hosts
        )
        cmds = self.client.copy_remote_file(
            remote_file, local_file, recurse=recurse, suffix_separator=suffix_separator
        )
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
                f'download_file failed on {len(errors)}/{len(self.reachable_hosts)} hosts: {errors}'
            )
        return result

    def reboot_connections(self):
        self.log.info('Rebooting Connections')
        self.client.run_command('reboot -f', stop_on_errors=self.stop_on_errors)

    def destroy_clients(self):
        self.log.info('Destroying Current phdl connections ..')
        del self.client


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
