"""
GID Consistency Checking Module

This module provides functionality for checking RDMA GID consistency across cluster nodes.
"""

from cvs.lib.preflight.base import PreflightCheck


class GidConsistencyCheck(PreflightCheck):
    """Check GID consistency across cluster RDMA interfaces."""

    def __init__(self, phdl, gid_index="3", expected_interfaces=None, config_dict=None):
        """
        Initialize GID consistency check.

        Args:
            phdl: Parallel SSH handle for cluster nodes
            gid_index: GID index to check (default: "3")
            expected_interfaces: List of specific interfaces to check (if None, checks all)
            config_dict: Optional configuration dictionary
        """
        super().__init__(phdl, config_dict)
        self.gid_index = gid_index
        self.expected_interfaces = expected_interfaces

    def _build_gid_check_command(self):
        """Build the remote shell snippet that enumerates GIDs per device."""
        if self.expected_interfaces:
            interface_filter = " ".join([f"/sys/class/infiniband/{iface}" for iface in self.expected_interfaces])
            self.log_info(
                f"Checking GID consistency for index {self.gid_index} on specific interfaces: {self.expected_interfaces}"
            )
            return f"""
            for dev in {interface_filter}; do 
                if [ -d "$dev" ]; then
                    dev_name=$(basename "$dev")
                    echo "DEVICE:$dev_name"
                    if [ -f "$dev/ports/1/gids/{self.gid_index}" ]; then
                        gid_value=$(cat "$dev/ports/1/gids/{self.gid_index}" 2>/dev/null)
                        if [ -n "$gid_value" ] && [ "$gid_value" != "0000:0000:0000:0000:0000:0000:0000:0000" ]; then
                            echo "GID_OK:$gid_value"
                        else
                            echo "GID_EMPTY:$gid_value"
                        fi
                    else
                        echo "GID_MISSING:No GID file"
                    fi
                else
                    dev_name=$(basename "$dev")
                    echo "DEVICE:$dev_name"
                    echo "DEVICE_MISSING:Interface not found"
                fi
            done
            """
        self.log_info(f"Checking GID consistency for index {self.gid_index} on all interfaces")
        return f"""
            for dev in /sys/class/infiniband/*/; do 
                if [ -d "$dev" ]; then
                    dev_name=$(basename "$dev")
                    echo "DEVICE:$dev_name"
                    if [ -f "$dev/ports/1/gids/{self.gid_index}" ]; then
                        gid_value=$(cat "$dev/ports/1/gids/{self.gid_index}" 2>/dev/null)
                        if [ -n "$gid_value" ] && [ "$gid_value" != "0000:0000:0000:0000:0000:0000:0000:0000" ]; then
                            echo "GID_OK:$gid_value"
                        else
                            echo "GID_EMPTY:$gid_value"
                        fi
                    else
                        echo "GID_MISSING:No GID file"
                    fi
                fi
            done
            """

    def _parse_gid_output_for_node(self, node, output):
        """Parse remote command output into a per-node result dict."""
        node_result = {'status': 'PASS', 'interfaces': {}, 'errors': []}
        current_device = None
        for line in output.strip().split('\n'):
            if line.startswith('DEVICE:'):
                current_device = line.split(':', 1)[1]
                node_result['interfaces'][current_device] = {}
            elif line.startswith('GID_OK:'):
                gid_value = line.split(':', 1)[1]
                node_result['interfaces'][current_device] = {'status': 'OK', 'gid_value': gid_value}
            elif line.startswith('GID_EMPTY:'):
                gid_value = line.split(':', 1)[1]
                node_result['interfaces'][current_device] = {'status': 'EMPTY', 'gid_value': gid_value}
                node_result['status'] = 'FAIL'
                node_result['errors'].append(f"GID index {self.gid_index} is empty on {current_device}")
            elif line.startswith('GID_MISSING:'):
                error_msg = line.split(':', 1)[1]
                node_result['interfaces'][current_device] = {'status': 'MISSING', 'error': error_msg}
                node_result['status'] = 'FAIL'
                node_result['errors'].append(f"GID index {self.gid_index} missing on {current_device}: {error_msg}")
            elif line.startswith('DEVICE_MISSING:'):
                error_msg = line.split(':', 1)[1]
                node_result['interfaces'][current_device] = {'status': 'DEVICE_MISSING', 'error': error_msg}
                node_result['status'] = 'FAIL'
                node_result['errors'].append(f"Interface {current_device} not found: {error_msg}")
        return node_result

    def run(self):
        """
        Execute GID consistency check across all cluster nodes.

        Returns:
            dict: Results with per-node GID status
        """
        cmd = self._build_gid_check_command()
        self.results = {}
        out_dict = self.phdl.exec(cmd)
        for node, output in out_dict.items():
            self.results[node] = self._parse_gid_output_for_node(node, output)
        return self.results
