"""
Node Reachability Checking Module

Ports the ``clschkping`` / ``clschkuptime`` pdsh scripts. Ping is a
driver-host -> node operation (not node -> node), so it is executed as a
local subprocess against each node's address rather than via ``phdl.exec``.
Uptime collection remains a simple ``phdl.exec`` broadcast and is purely
informational (no PASS/FAIL judgement).
"""

import subprocess

from cvs.lib.preflight.base import PreflightCheck


class PingReachabilityCheck(PreflightCheck):
    """ICMP ping reachability check executed from the CVS driver host."""

    def __init__(self, phdl, node_ip_map, count=4, timeout_sec=1, config_dict=None):
        """
        Initialize the ping reachability check.

        Args:
            phdl: Parallel SSH handle for cluster nodes (used only for ``reachable_hosts``
                bookkeeping; ping itself runs locally on the driver host).
            node_ip_map: Mapping of node identifier -> IP/hostname to ping.
            count: Number of ICMP echo requests per node (``ping -c``).
            timeout_sec: Per-ping timeout in seconds (``ping -W``).
            config_dict: Optional configuration dictionary.
        """
        super().__init__(phdl, config_dict)
        self.node_ip_map = node_ip_map or {}
        self.count = count
        self.timeout_sec = timeout_sec

    def _ping_one(self, target):
        """Run a single local ``ping`` subprocess against ``target``.

        Returns a ``(reachable, detail)`` tuple. Any subprocess error
        (missing binary, timeout, etc.) is treated as unreachable rather
        than raised, so one bad node cannot abort the whole check.
        """
        cmd = ["ping", "-c", str(self.count), "-W", str(self.timeout_sec), target]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.count * self.timeout_sec + 10,
            )
            output = proc.stdout.decode(errors="replace") if proc.stdout else ""
            return proc.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, "ping subprocess timed out"
        except OSError as exc:
            return False, f"ping subprocess failed to start: {exc}"

    def run(self):
        """
        Execute ICMP ping reachability checks for every configured node.

        Returns:
            dict: Per-node results ``{node: {status, target, errors}}``.
        """
        self.results = {}
        for node, target in self.node_ip_map.items():
            reachable, output = self._ping_one(target)
            self.results[node] = {
                'status': 'PASS' if reachable else 'FAIL',
                'target': target,
                'errors': [] if reachable else [f"ping to {target} failed: {output.strip()[-500:]}"],
            }
        return self.results


class UptimeCheck(PreflightCheck):
    """Informational ``uptime`` collection across cluster nodes (no PASS/FAIL judgement)."""

    def run(self):
        """
        Collect ``uptime`` output from every reachable node.

        Returns:
            dict: Per-node results ``{node: {status, uptime, errors}}``. Status is
                always 'PASS' unless the remote command produced no output at all.
        """
        self.results = {}
        out_dict = self.phdl.exec("uptime")
        for node, output in out_dict.items():
            text = (output or '').strip()
            self.results[node] = {
                'status': 'PASS' if text else 'FAIL',
                'uptime': text,
                'errors': [] if text else ["uptime returned no output"],
            }
        return self.results
