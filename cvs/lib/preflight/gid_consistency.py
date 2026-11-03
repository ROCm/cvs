"""Check RDMA GID consistency across cluster nodes."""

from cvs.lib.preflight.base import PreflightCheck
from cvs.lib.rdma_gid_lib import (
    DEFAULT_GID_TYPE,
    GID_MISSING,
    GID_OK,
    build_gid_probe_cmd,
    check_gid_entry,
    gid_scope,
    is_infiniband,
    parse_gid_probe_output,
)


class GidConsistencyCheck(PreflightCheck):
    """Check GID values and types across cluster RDMA interfaces."""

    def __init__(
        self, phdl, gid_index="3", expected_interfaces=None, config_dict=None, expected_gid_type=DEFAULT_GID_TYPE
    ):
        """Configure the GID index, interfaces, and expected sysfs type."""
        super().__init__(phdl, config_dict)
        self.gid_index = gid_index
        self.expected_interfaces = expected_interfaces
        self.expected_gid_type = expected_gid_type

    def _build_gid_check_command(self):
        """Build the remote sysfs probe command."""
        if self.expected_interfaces:
            self.log_info(
                f'Checking GID consistency for index {self.gid_index} on specific interfaces: '
                f'{self.expected_interfaces} (expected type {self.expected_gid_type})'
            )
        else:
            self.log_info(
                f'Checking GID consistency for index {self.gid_index} on all interfaces '
                f'(expected type {self.expected_gid_type})'
            )
        return build_gid_probe_cmd(self.gid_index, self.expected_interfaces or None)

    def _parse_gid_output_for_node(self, node, output):
        """Parse and validate each configured interface on a node."""
        result = {'status': 'PASS', 'interfaces': {}, 'errors': [], 'warnings': []}
        entries = parse_gid_probe_output(output)
        devices = self.expected_interfaces or list(entries)
        for device in devices:
            entry = entries.get(device)
            status, message = check_gid_entry(device, entry, self.gid_index, self.expected_gid_type)
            values = entry or {}
            interface = {
                'status': status,
                'gid_index': str(self.gid_index),
                'gid_value': values.get('gid') or '',
                'gid_type': values.get('gid_type') or '',
                'link_layer': values.get('link_layer') or '',
            }
            result['interfaces'][device] = interface
            if status != GID_OK:
                interface['error'] = message
                result['status'] = 'FAIL'
                result['errors'].append(message)
            elif gid_scope(interface['gid_value']) == 'link-local' and not is_infiniband(interface['link_layer']):
                warning = (
                    f'GID index {self.gid_index} on {device} is link-local ({interface["gid_value"]}); '
                    'it is not routable across L3 RoCE fabrics'
                )
                result['warnings'].append(warning)
                self.log_warning(f'Node {node}: {warning}')
        return result

    def _invalid_index_results(self, reason):
        """Fail every node without probing when gid_index is not a sysfs index."""
        message = (
            f'Invalid connectivity_check.rdma.gid_index {self.gid_index!r}: {reason}. '
            'Preflight needs an explicit GID index; "auto" is supported by ibperf only'
        )
        self.log_error(message)
        devices = self.expected_interfaces or []
        return {
            node: {
                'status': 'FAIL',
                'interfaces': {
                    device: {'status': GID_MISSING, 'gid_index': str(self.gid_index), 'error': message}
                    for device in devices
                },
                'errors': [message],
                'warnings': [],
            }
            for node in self.orch.all.reachable_hosts
        }

    def run(self):
        """Execute GID validation across all cluster nodes."""
        try:
            cmd = self._build_gid_check_command()
        except ValueError as exc:
            self.results = self._invalid_index_results(exc)
            return self.results
        self.results = {}
        out_dict = self.orch.all.exec(cmd)
        for node, output in out_dict.items():
            self.results[node] = self._parse_gid_output_for_node(node, output)
        return self.results
