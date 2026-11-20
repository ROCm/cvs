"""
Interface Consistency Checking Module

This module provides functionality for checking RDMA interface consistency across cluster nodes.
"""

from cvs.lib.preflight.base import PreflightCheck
from cvs.lib.preflight.node_smoke import resolve_expected_rdma_nic_count
from cvs.lib import linux_utils


class InterfaceConsistencyCheck(PreflightCheck):
    """Check RDMA interface consistency across cluster nodes."""

    def __init__(self, phdl, expected_interfaces=None, config_dict=None):
        """
        Initialize interface consistency check.

        Args:
            phdl: Parallel SSH handle for cluster nodes
            expected_interfaces: List of expected interface names
            config_dict: Optional configuration dictionary
        """
        super().__init__(phdl, config_dict)
        self.expected_interfaces = expected_interfaces or ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"]
        self.count_error = None
        try:
            explicit = resolve_expected_rdma_nic_count(config_dict or {})
        except ValueError as exc:
            self.count_error = str(exc)
            explicit = None
        self.expected_nic_count = explicit or len(self.expected_interfaces)
        if explicit is not None and explicit != len(self.expected_interfaces):
            self.count_error = (
                f"connectivity_check.rdma.expected_nic_count is {explicit} but "
                f"connectivity_check.rdma.interfaces lists {len(self.expected_interfaces)} device(s); "
                "list every backend RDMA device"
            )

    def _evaluate_node_interfaces(self, node, node_rdma_info):
        """Build result dict for one node from RDMA NIC info."""
        found_interfaces = list(node_rdma_info.keys())

        result = {
            'interfaces': [],
            'status': 'PASS',
            'errors': [],
            'expected_interfaces': self.expected_interfaces,
            'found_interfaces': found_interfaces,
            'missing_interfaces': [],
            'unexpected_interfaces': [],
            'inactive_interfaces': [],
            'down_interfaces': [],
            'expected_nic_count': self.expected_nic_count,
            'active_expected_count': 0,
        }

        if not found_interfaces:
            result['status'] = 'FAIL'
            result['errors'].append("No RDMA interfaces found")
            result['missing_interfaces'] = self.expected_interfaces.copy()
            if self.count_error:
                result['errors'].append(self.count_error)
            if self.expected_nic_count:
                result['errors'].append(f"0 of {self.expected_nic_count} backend RDMA NICs present and active")
            return result

        missing = []
        inactive_expected = []
        down_expected = []

        for expected_iface in self.expected_interfaces:
            if expected_iface not in found_interfaces:
                missing.append(expected_iface)
            else:
                iface_info = node_rdma_info[expected_iface]
                device_status = iface_info.get('device_status', 'UNKNOWN')
                link_status = iface_info.get('link_status', 'UNKNOWN')

                if device_status != 'ACTIVE':
                    inactive_expected.append(f"{expected_iface} (state: {device_status})")
                if link_status not in ['LINK_UP', 'LinkUp']:
                    down_expected.append(f"{expected_iface} (physical_state: {link_status})")
                if device_status == 'ACTIVE' and link_status in ('LINK_UP', 'LinkUp'):
                    result['active_expected_count'] += 1

        result['missing_interfaces'] = missing
        result['inactive_interfaces'] = inactive_expected
        result['down_interfaces'] = down_expected

        # Unlisted RDMA devices can be management ports and do not affect the backend count.
        unexpected = []
        for found_iface in found_interfaces:
            if found_iface not in self.expected_interfaces:
                unexpected.append(found_iface)

        result['unexpected_interfaces'] = unexpected

        for interface in found_interfaces:
            iface_info = node_rdma_info.get(interface, {})
            device_status = iface_info.get('device_status', 'UNKNOWN')
            link_status = iface_info.get('link_status', 'UNKNOWN')

            is_expected = interface in self.expected_interfaces
            is_functional = device_status == 'ACTIVE' and link_status in ['LINK_UP', 'LinkUp']

            result['interfaces'].append(
                {
                    'name': interface,
                    'expected': is_expected,
                    'device_status': device_status,
                    'link_status': link_status,
                    'functional': is_functional,
                }
            )

        if missing:
            result['status'] = 'FAIL'
            result['errors'].append(f"Missing expected interfaces: {', '.join(missing)}")

        if inactive_expected:
            result['status'] = 'FAIL'
            result['errors'].append(f"Expected interfaces not active: {', '.join(inactive_expected)}")

        if down_expected:
            result['status'] = 'FAIL'
            result['errors'].append(f"Expected interfaces link down: {', '.join(down_expected)}")

        if self.count_error:
            result['status'] = 'FAIL'
            result['errors'].append(self.count_error)
        if result['active_expected_count'] < self.expected_nic_count:
            result['status'] = 'FAIL'
            result['errors'].append(
                f"{result['active_expected_count']} of {self.expected_nic_count} backend RDMA NICs present and active"
            )

        return result

    def run(self):
        """
        Execute interface consistency check across all cluster nodes.

        Returns:
            dict: Results with per-node interface status
        """
        self.log_info(
            f"Checking RDMA interface presence (expected: {self.expected_interfaces}, count: {self.expected_nic_count})"
        )

        rdma_dict = linux_utils.get_rdma_nic_dict(self.orch.all)

        self.results = {}
        for node in self.orch.all.reachable_hosts:
            node_rdma_info = rdma_dict.get(node, {})
            self.results[node] = self._evaluate_node_interfaces(node, node_rdma_info)

        return self.results
