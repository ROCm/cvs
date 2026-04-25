'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from __future__ import print_function

import time

import paramiko
from paramiko import SSHClient
from scp import SCPClient


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
        scp_client = SCPClient(ssh.get_transport())
        if not get_put:
            scp_client.get(srcfile, dstfile)
        else:
            scp_client.put(srcfile, dstfile)
        scp_client.close()
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
