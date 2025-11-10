'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from __future__ import print_function
from pssh.clients import ParallelSSHClient

import sys
import os
import re
import ast
import json

# Following used only for scp of file
import paramiko
from paramiko import SSHClient
from scp import SCPClient

class Pssh():
    """
    ParallelSessions - Uses the pssh library that is based of Paramiko, that lets you take
    multiple parallel ssh sessions to hosts and execute commands.

    Input host_config should be in this format ..
    mandatory args =  user, password (or) 'private_key': load_private_key('my_key.pem')
    """

    def __init__(self, log, host_list, user=None, password=None, pkey='id_rsa', host_key_check=False ):

        self.log            = log
        self.host_list      = host_list
        self.user           = user
        self.pkey           = pkey
        self.password       = password
        self.host_key_check = host_key_check

        if self.password is None:
            print(self.host_list)
            print(self.user)
            print(self.pkey)
            self.client         = ParallelSSHClient( self.host_list, user=self.user, pkey=self.pkey, keepalive_seconds=30 )
        else:
            self.client         = ParallelSSHClient( self.host_list, user=self.user, password=self.password, keepalive_seconds=30 )


    def exec(self, cmd, timeout=None ):
        """
        Returns a dictionary of host as key and command output as values
        """
        cmd_output = {}
        print(f'cmd = {cmd}')
        if timeout is None:
            output = self.client.run_command(cmd )
        else:
            output = self.client.run_command(cmd, read_timeout=timeout )
        for item in output:
            print('#----------------------------------------------------------#')
            print(f'Host == {item.host} ==')
            print('#----------------------------------------------------------#')
            cmd_out_str = ''
            print(cmd)
            for line in item.stdout:
                print(line)
                cmd_out_str = cmd_out_str + line.replace( '\t', '   ')
                cmd_out_str = cmd_out_str + '\n'
            if item.stderr:
                for line in item.stderr:
                    print(line)
                    cmd_out_str = cmd_out_str + line.replace( '\t', '   ')
                    cmd_out_str = cmd_out_str + '\n'
            cmd_output[item.host] = cmd_out_str

        return cmd_output


    def exec_cmd_list(self, cmd_list, timeout=None ):
        """
        Run different commands on different hosts compared to to exec 
        which runs the same command on all hosts.
        Returns a dictionary of host as key and command output as values
        """
        cmd_output = {}
        print(cmd_list)
        if timeout is None:
            output = self.client.run_command( '%s', host_args=cmd_list )
        else:
            output = self.client.run_command( '%s', host_args=cmd_list, read_timeout=timeout )
        i = 0
        for item in output:
            print('#----------------------------------------------------------#')
            print(f'Host == {item.host} ==')
            print('#----------------------------------------------------------#')
            cmd_out_str = ''
            cmd_out_err = ''
            print(cmd_list[i])
            for line in item.stdout:
                print(line)
                cmd_out_str = cmd_out_str + line.replace( '\t', '   ')
                cmd_out_str = cmd_out_str + '\n'
            if item.stderr:
                for line in item.stderr:
                    print(line)
                    cmd_out_str = cmd_out_str + line.replace( '\t', '   ')
                    cmd_out_str = cmd_out_str + '\n'
            i=i+1
            cmd_output[item.host] = cmd_out_str

        return cmd_output



    def scp_file(self, local_file, remote_file, recurse=False ):
        print('About to copy local file {} to remote {} on all Hosts'.format(local_file, remote_file))
        cmds = self.client.copy_file( local_file, remote_file, recurse=recurse )
        self.client.pool.join()
        for cmd in cmds:
            try:
               cmd.get()
            except IOError:
               raise Exception("Expected IOError exception, got none")
        return


    def reboot_connections(self ):
        print('Rebooting Connections')
        self.client.run_command( 'reboot -f' ) 

    def destroy_clients(self ):
        print('Destroying Current phdl connections ..')
        del self.client



def scp( src, dst, srcusername, srcpassword, dstusername = None, dstpassword = None):
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
    srcip   = None
    dstip   = None

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
           scp.get(srcfile,dstfile)
       else:
           scp.put(srcfile,dstfile)
       scp.close()
    else:
        if dstusername is None:
            dstusername = srcusername
        if dstpassword is None:
            dstpassword = srcpassword
        # This is to handle if ssh keys in the known_hosts is empty or incorrect
        # Need better way to handle in the future
        output = ssh.exec_command('ssh-keygen -R %s'%(dstip))
        # print('ssh-keygen output is {0}'.format(output))
        time.sleep(1)
        output = ssh.exec_command('ssh-keyscan %s >> ~/.ssh/known_hosts'%(dstip))
        # print('ssh-keyscan output is {0}'.format(output))
        time.sleep(1)
        output = ssh.exec_command('sshpass -p %s scp %s %s@%s:%s'%(dstpassword, srcfile, dstusername, dstip, dstfile))
