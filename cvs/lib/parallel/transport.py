'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from abc import ABC, abstractmethod


class BaseTransport(ABC):
    """Owns client construction, rebuild, connectivity probe, and teardown.

    Fan-out operations (run_command, copy_file, …) live on self.client.
    ParallelHTTPClient must match ParallelSSHClient's API or ParallelHandle breaks.
    """

    @abstractmethod
    def rebuild(self, hosts):
        """Recreate self.client for a new host list."""

    @abstractmethod
    def check_connectivity(self, hosts):
        """Return hosts that fail a lightweight reachability probe."""

    @abstractmethod
    def client_for_hosts(self, hosts):
        """Return a transport scoped to an exact host subset."""

    @abstractmethod
    def destroy(self):
        """Tear down connections and release resources."""


def create_transport(
    hosts,
    *,
    transport='ssh',
    user=None,
    password=None,
    pkey='id_rsa',
    **transport_kwargs,
):
    """Instantiate a transport for the given host list.

    Args:
        hosts: Hostnames/IPs to connect to.
        transport: Wire protocol selector. ``'ssh'`` today; ``'http'`` later.
        user, password, pkey: SSH credentials (ignored by non-SSH transports).
        transport_kwargs: Protocol-specific options forwarded to the transport.
    """
    if transport == 'ssh':
        from cvs.lib.parallel.ssh_transport import SshTransport

        return SshTransport(
            hosts,
            user=user,
            password=password,
            pkey=pkey,
            **transport_kwargs,
        )
    if transport == 'http':
        raise NotImplementedError('HTTP transport is not implemented yet')
    raise ValueError(f'Unknown transport: {transport!r}')
