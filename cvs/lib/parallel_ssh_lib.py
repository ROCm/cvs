'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from __future__ import print_function
from pssh.clients import ParallelSSHClient
from pssh.exceptions import Timeout, ConnectionError

import time

# Following used only for scp of file
import paramiko
from paramiko import SSHClient
from scp import SCPClient


class Pssh:
    """
    ParallelSessions - Uses the pssh library that is based of Paramiko, that lets you take
    multiple parallel ssh sessions to hosts and execute commands.

    Input host_config should be in this format ..
    mandatory args =  user, password (or) 'private_key': load_private_key('my_key.pem')
    """

    def __init__(
        self, log, host_list, user=None, password=None, pkey='id_rsa', host_key_check=False, stop_on_errors=True
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
            item.host for item in output if item.exception and isinstance(item.exception, (ConnectionError, Timeout))
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
        print(f'cmd = {cmd}')
        if timeout is None:
            output = self.client.run_command(cmd, stop_on_errors=self.stop_on_errors)
        else:
            output = self.client.run_command(cmd, read_timeout=timeout, stop_on_errors=self.stop_on_errors)
        cmd_output = self._process_output(output, cmd=cmd, print_console=print_console)
        return cmd_output

    def exec_cmd_list(self, cmd_list, timeout=None, print_console=True):
        """
        Run different commands on different hosts compared to to exec
        which runs the same command on all hosts.
        Returns a dictionary of host as key and command output as values
        """
        print(cmd_list)
        if timeout is None:
            output = self.client.run_command('%s', host_args=cmd_list, stop_on_errors=self.stop_on_errors)
        else:
            output = self.client.run_command(
                '%s', host_args=cmd_list, read_timeout=timeout, stop_on_errors=self.stop_on_errors
            )
        cmd_output = self._process_output(output, cmd_list=cmd_list, print_console=print_console)
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
