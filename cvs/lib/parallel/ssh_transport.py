'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from gevent import killall

from cvs.lib import globals
from cvs.lib.parallel.transport import BaseTransport

log = globals.log


def _get_parallel_ssh_client():
    """Resolve ParallelSSHClient from phandle so unit-test patches apply."""
    from cvs.lib.parallel import phandle as phandle_mod

    return phandle_mod.ParallelSSHClient


class SshTransport(BaseTransport):
    """SSH wire protocol via parallel-ssh ParallelSSHClient."""

    def __init__(self, hosts, user=None, password=None, pkey='id_rsa', **ssh_client_kwargs):
        self.hosts = list(hosts)
        self.user = user
        self.password = password
        self.pkey = pkey
        self.ssh_client_kwargs = ssh_client_kwargs
        self.client = self._make_client(self.hosts)

    def _make_client(self, hosts):
        ParallelSSHClient = _get_parallel_ssh_client()
        if self.password is None:
            return ParallelSSHClient(
                hosts, user=self.user, pkey=self.pkey, keepalive_seconds=30, **self.ssh_client_kwargs
            )
        return ParallelSSHClient(
            hosts,
            user=self.user,
            password=self.password,
            keepalive_seconds=30,
            **self.ssh_client_kwargs,
        )

    def rebuild(self, hosts):
        self.hosts = list(hosts)
        self.client = self._make_client(self.hosts)

    def check_connectivity(self, hosts):
        if not hosts:
            return []
        temp_ssh_client_kwargs = self.ssh_client_kwargs.copy()
        temp_ssh_client_kwargs['timeout'] = 2
        temp_ssh_client_kwargs['num_retries'] = 0
        temp_client = self._make_client_with_kwargs(hosts, temp_ssh_client_kwargs)
        output = temp_client.run_command('echo 1', stop_on_errors=False, read_timeout=2)
        return [item.host for item in output if item.exception]

    def _make_client_with_kwargs(self, hosts, client_kwargs):
        ParallelSSHClient = _get_parallel_ssh_client()
        if self.password is None:
            return ParallelSSHClient(
                hosts,
                user=self.user,
                pkey=self.pkey,
                keepalive_seconds=30,
                **client_kwargs,
            )
        return ParallelSSHClient(
            hosts,
            user=self.user,
            password=self.password,
            keepalive_seconds=30,
            **client_kwargs,
        )

    def client_for_hosts(self, hosts):
        return SshTransport(
            hosts,
            user=self.user,
            password=self.password,
            pkey=self.pkey,
            **self.ssh_client_kwargs,
        )

    def destroy(self):
        client = getattr(self, 'client', None)
        if client is None:
            return

        pending = getattr(client, 'cmds', None)
        if pending:
            try:
                killall(pending, block=True, timeout=5)
            except Exception as exc:
                log.debug("Error killing pending SSH greenlets: %s", exc)
            client.cmds = None

        host_clients = getattr(client, '_host_clients', None) or {}
        for key, host_client in list(host_clients.items()):
            try:
                host_client._disconnect()
            except Exception as exc:
                log.debug("Error disconnecting SSH client %s: %s", key, exc)
        host_clients.clear()

        del self.client
