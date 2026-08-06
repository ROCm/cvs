"""
/etc/hosts Consistency Checking Module

Ports the ``ansible/checketchosts/check_hosts.yml`` playbook. The original
playbook compared ``/etc/hosts`` against a hardcoded, deployment-specific
128-line block. Since preflight must not bake deployment-specific hostnames
into the codebase, this module instead validates that every cluster node's
address (from ``cluster_dict['node_dict']``) has *some* entry in
``/etc/hosts`` -- i.e. that DNS-independent, static hostname resolution has
been provisioned for the whole cluster -- plus any operator-supplied
``extra_entries`` that must match an exact ``ip -> hostname`` pair (e.g. for
out-of-cluster infrastructure hosts).
"""

from cvs.lib.preflight.base import PreflightCheck


class EtcHostsConsistencyCheck(PreflightCheck):
    """Validate /etc/hosts covers every cluster node address."""

    def __init__(self, phdl, expected_ips, extra_entries=None, config_dict=None):
        """
        Initialize the /etc/hosts consistency check.

        Args:
            phdl: Parallel SSH handle for cluster nodes.
            expected_ips: List of IP/hostname addresses that must have a
                static entry in ``/etc/hosts`` on every node (typically the
                cluster's node addresses and ``vpc_ip`` values).
            extra_entries: Optional list of ``{"hostname": ..., "ip": ...}``
                dicts that must appear verbatim (exact ip+hostname pairing).
            config_dict: Optional configuration dictionary.
        """
        super().__init__(phdl, config_dict)
        self.expected_ips = sorted(set(expected_ips or []))
        self.extra_entries = extra_entries or []

    @staticmethod
    def _parse_hosts_file(output):
        """Parse ``/etc/hosts`` content into ``{ip: {hostnames...}}``."""
        ip_to_hostnames = {}
        for raw_line in (output or '').splitlines():
            line = raw_line.split('#', 1)[0].strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            ip, hostnames = fields[0], fields[1:]
            ip_to_hostnames.setdefault(ip, set()).update(hostnames)
        return ip_to_hostnames

    def run(self):
        """
        Execute the /etc/hosts consistency check on every reachable node.

        Returns:
            dict: Per-node results
                ``{node: {status, missing_ips, missing_extra_entries, errors}}``.
        """
        self.results = {}
        out_dict = self.phdl.exec("cat /etc/hosts 2>/dev/null || echo ''")

        for node, output in out_dict.items():
            ip_to_hostnames = self._parse_hosts_file(output)

            missing_ips = [ip for ip in self.expected_ips if ip not in ip_to_hostnames]
            missing_extra = []
            for entry in self.extra_entries:
                ip = entry.get('ip')
                hostname = entry.get('hostname')
                if hostname not in ip_to_hostnames.get(ip, set()):
                    missing_extra.append(f"{hostname}={ip}")

            errors = []
            if missing_ips:
                errors.append(f"/etc/hosts missing entries for: {', '.join(missing_ips)}")
            if missing_extra:
                errors.append(f"/etc/hosts missing required extra entries: {', '.join(missing_extra)}")

            self.results[node] = {
                'status': 'PASS' if not errors else 'WARNING',
                'missing_ips': missing_ips,
                'missing_extra_entries': missing_extra,
                'errors': errors,
            }
        return self.results
