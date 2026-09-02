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

    The two class attributes below let ParallelHandle stay protocol-agnostic:
    it asks the transport what it can do instead of branching on a name.
    """

    # Exceptions on a per-host result that mean "this host may be down", making it a
    # candidate for the connectivity re-check in ParallelHandle.prune_unreachable_hosts.
    # Anything outside this tuple (auth, protocol, a command that merely failed) is a
    # recoverable error and must never prune.
    prune_exception_types = ()

    # True when the far side enforces the inactivity timeout itself and returns finished
    # output. False when the timeout has to be applied locally while reading a live stream.
    remote_inactivity_timeout = False

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
        transport: Wire protocol selector (``'ssh'`` or ``'http'``).
        user, password, pkey: SSH credentials (ignored by non-SSH transports).
        transport_kwargs: Protocol-specific options forwarded to the transport.
            HTTP requires ``agent_urls`` (host → base URL) and ``token``.
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
        from cvs.lib.parallel.http_transport import HttpTransport

        return HttpTransport(hosts, **transport_kwargs)
    raise ValueError(f'Unknown transport: {transport!r}')
