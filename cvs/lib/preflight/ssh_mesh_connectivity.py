"""
SSH Mesh Connectivity Checking Module

Ports the ``ansible/mutualssh/check_ssh.yml`` playbook. Every reachable node
attempts a passwordless, non-interactive SSH to every other node in the
cluster's VPC address space. This is a diagnostic (WARNING-only) check: a
node that cannot reach some peers is not pruned from further preflight
testing, since it can still be otherwise perfectly healthy (e.g. asymmetric
security-group rules that don't affect RDMA/IFoE at all).
"""

from cvs.lib.preflight.base import PreflightCheck


class SshMeshConnectivityCheck(PreflightCheck):
    """Full node x node passwordless SSH mesh diagnostic."""

    def __init__(self, phdl, peer_map, ssh_timeout_sec=10, config_dict=None):
        """
        Initialize the SSH mesh connectivity check.

        Args:
            phdl: Parallel SSH handle for cluster nodes.
            peer_map: Mapping of node identifier -> SSH target address (typically
                the cluster's ``vpc_ip`` for that node) used to build each
                node's peer list. Every node is tested against every *other*
                node's address; self-pings are skipped.
            ssh_timeout_sec: ``ssh -o ConnectTimeout=<n>`` value.
            config_dict: Optional configuration dictionary.
        """
        super().__init__(phdl, config_dict)
        self.peer_map = peer_map or {}
        self.ssh_timeout_sec = ssh_timeout_sec

    def _build_command_for_node(self, node):
        """Build the mesh-SSH shell snippet a single node runs against all its peers."""
        peers = [addr for peer_node, addr in self.peer_map.items() if peer_node != node]
        peer_list = " ".join(peers)
        return f"""
        PEERS="{peer_list}"
        PASS=0
        FAIL=0
        FAILED_PEERS=""
        for peer in $PEERS; do
            if ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout={self.ssh_timeout_sec} "$peer" true 2>/dev/null; then
                PASS=$((PASS+1))
            else
                FAIL=$((FAIL+1))
                FAILED_PEERS="$FAILED_PEERS $peer"
            fi
        done
        echo "SSH_MESH_TOTAL:$((PASS+FAIL))"
        echo "SSH_MESH_PASS:$PASS"
        echo "SSH_MESH_FAIL:$FAIL"
        echo "SSH_MESH_FAILED_PEERS:${{FAILED_PEERS# }}"
        """

    @staticmethod
    def _parse_output(output):
        total = passed = failed = 0
        failed_peers = []
        for line in (output or '').strip().split('\n'):
            if line.startswith('SSH_MESH_TOTAL:'):
                total = int(line.split(':', 1)[1] or 0)
            elif line.startswith('SSH_MESH_PASS:'):
                passed = int(line.split(':', 1)[1] or 0)
            elif line.startswith('SSH_MESH_FAIL:'):
                failed = int(line.split(':', 1)[1] or 0)
            elif line.startswith('SSH_MESH_FAILED_PEERS:'):
                remainder = line.split(':', 1)[1].strip()
                failed_peers = remainder.split() if remainder else []
        return total, passed, failed, failed_peers

    def run(self):
        """
        Execute the SSH mesh diagnostic on every reachable node.

        Builds one distinct command per reachable host (each targeting every
        *other* node's address) and dispatches them in a single
        ``phdl.exec_cmd_list`` call, since that API requires the command list
        to be positionally aligned with ``phdl.reachable_hosts``.

        Returns:
            dict: Per-node results
                ``{node: {status, total_peers, passed_peers, failed_peers, errors}}``.
        """
        self.results = {}
        nodes = list(self.phdl.reachable_hosts)
        if not nodes:
            return self.results

        cmd_list = [self._build_command_for_node(node) for node in nodes]
        out_dict = self.phdl.exec_cmd_list(cmd_list)

        for node in nodes:
            peers = [peer_node for peer_node in self.peer_map if peer_node != node]
            if not peers:
                self.results[node] = {
                    'status': 'PASS',
                    'total_peers': 0,
                    'passed_peers': 0,
                    'failed_peers': [],
                    'errors': [],
                }
                continue

            output = out_dict.get(node, '')
            total, passed, failed, failed_peers = self._parse_output(output)

            self.results[node] = {
                'status': 'PASS' if failed == 0 and total > 0 else 'WARNING',
                'total_peers': total,
                'passed_peers': passed,
                'failed_peers': failed_peers,
                'errors': (
                    [f"SSH mesh failed to {len(failed_peers)} peer(s): {', '.join(failed_peers)}"]
                    if failed_peers
                    else []
                ),
            }
        return self.results
