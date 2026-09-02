'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import ipaddress
import socket
from pathlib import Path

from cvs.core.agent import messages


def _short(name):
    return name.split('.')[0].lower()


def _addresses(name):
    """Non-loopback addresses for *name* (IPs pass through as themselves).

    Loopback is dropped so a rank-0 ``/etc/hosts`` line like ``127.0.0.1 node01``
    cannot claim a cluster management IP. Every A/AAAA record is kept so a
    hostname that publishes both management and compute addresses still matches
    a cluster file that names either one.
    """
    try:
        infos = socket.getaddrinfo(name, None)
    except OSError:
        return []
    seen = []
    for family, _, _, _, sockaddr in infos:
        if family not in (socket.AF_INET, socket.AF_INET6):
            continue
        ip = sockaddr[0]
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
            ip = str(addr)
        if addr.is_loopback:
            continue
        if ip not in seen:
            seen.append(ip)
    return seen


# Marks an index key that more than one agent claims. Kept in the index rather than
# dropped so a lookup landing on it can say "ambiguous" instead of "not found".
_AMBIGUOUS = object()


def _claim(index, key, url):
    """Map *key* to *url*, marking the key ambiguous if a different agent already claims it."""
    existing = index.get(key)
    if existing is None:
        index[key] = url
    elif existing != url:
        index[key] = _AMBIGUOUS


def _ptr_names(ip):
    try:
        primary, aliases, _ = socket.gethostbyaddr(ip)
    except OSError:
        return []
    names = []
    for name in (primary, *aliases):
        if name and name not in names:
            names.append(name)
    return names


class AgentMesh:
    """Rank-0 view of registered HTTP agents: hostname → base URL plus the job token.

    Installed once after registration (or registration timeout) so pytest-side
    orchestrators can construct ParallelHandle(transport='http', ...) without SSH.
    One agent per hostname is required; two ranks on the same node collide in
    host-keyed fan-out.

    Agents register under the name the scheduler knows them by (SLURM_NODENAME, else
    the node's own hostname), while the cluster file names the same machines however
    the operator wrote it -- commonly management IPs. resolve() bridges the two.
    """

    _instance = None

    def __init__(self, urls_by_host, token):
        self.urls_by_host = dict(urls_by_host)
        self.token = token
        self._indexes = None

    @classmethod
    def install(cls, snapshot, token):
        if not token:
            raise ValueError("HTTP agent mesh requires a non-empty auth token")
        urls_by_host = {}
        for info in snapshot.values():
            host = info.hostname
            if host in urls_by_host:
                raise ValueError(
                    f"duplicate agent hostname {host!r}; HTTP fan-out requires one agent per node "
                    f"(ntasks must equal nnodes)"
                )
            urls_by_host[host] = f"http://{info.hostname}:{info.port}"
        cls._instance = cls(urls_by_host, token)
        return cls._instance

    @classmethod
    def install_from_agent_dir(cls, snapshot, agent_dir):
        token_path = Path(agent_dir) / messages.AUTH_TOKEN_FILENAME
        token = token_path.read_text(encoding="utf-8").strip()
        return cls.install(snapshot, token)

    def _lookup_indexes(self):
        """Short-name and address indexes over the registered agents, built once.

        The registered set is fixed at install time, so this is computed on first use and
        reused. resolve() runs once per orchestrator and orchestrators are module-scoped,
        so rebuilding would repeat every agent's forward lookup for each test module.
        """
        if self._indexes is None:
            by_short = {}
            by_ip = {}
            for host, url in self.urls_by_host.items():
                _claim(by_short, _short(host), url)
                for ip in _addresses(host):
                    _claim(by_ip, ip, url)
            self._indexes = (by_short, by_ip)
        return self._indexes

    def resolve(self, hosts):
        """Map cluster-file host names onto registered agent URLs.

        Tried in order, most to least certain: the name as registered, then the short
        name case-folded (a cluster file saying node01.cluster.local against an agent
        registered as node01), then any non-loopback address shared with a registered
        name (a cluster file of management IPs against agents registered by name, including
        extra A records), then a PTR of the cluster name matching a registered name
        (forward DNS of the compute hostname often returns the IB address while the
        cluster file has the management IP).

        A name that more than one agent claims is refused rather than resolved to whichever
        agent was registered first, and anything unmatched is fatal and names both sides.
        Both are hard failures for the same reason: the alternative is an HTTP fan-out that
        silently skips -- or silently mis-addresses -- nodes the caller believes it is testing.
        """
        by_short, by_ip = self._lookup_indexes()

        resolved = {}
        unresolved = []
        ambiguous = []
        for host in hosts:
            url = self.urls_by_host.get(host)
            if url is None:
                url = by_short.get(_short(host))
            # Resolved once and reused by both address-based rungs below: on the failure
            # path every lookup here can block until DNS times out, and paying that twice
            # per host turns a clear error into a slow one.
            addresses = []
            if url is None:
                addresses = _addresses(host)
                for ip in addresses:
                    url = by_ip.get(ip)
                    if url is not None:
                        break
            if url is None:
                for ip in addresses:
                    for name in _ptr_names(ip):
                        url = self.urls_by_host.get(name) or by_short.get(_short(name))
                        if url is not None:
                            break
                    if url is not None:
                        break
            if url is _AMBIGUOUS:
                ambiguous.append(host)
            elif url is None:
                unresolved.append(host)
            else:
                resolved[host] = url
        if ambiguous:
            raise ValueError(
                f"Cluster host(s) {ambiguous} match more than one registered agent. "
                f"Registered agents: {sorted(self.urls_by_host)}. Name these hosts in the cluster "
                f"file exactly as the agents registered so the mapping is unambiguous."
            )
        if unresolved:
            raise ValueError(
                f"No HTTP agent registered for cluster host(s): {unresolved}. "
                f"Registered agents: {sorted(self.urls_by_host)}. In a scheduler-managed run every "
                f"host in the cluster file (including the head node) must be a node of the job step, "
                f"and must be reachable under the name the agent registered with. Matching by "
                f"address or PTR needs working forward/reverse DNS for these names; if the cluster "
                f"has neither, name the hosts in the cluster file as the agents registered."
            )
        return resolved

    @classmethod
    def get(cls):
        if cls._instance is None:
            raise RuntimeError("AgentMesh is not installed; rank 0 must record HTTP agent registrations before pytest")
        return cls._instance

    @classmethod
    def reset(cls):
        """Drop the installed mesh. Called at the end of a managed run, and by unit tests."""
        cls._instance = None
