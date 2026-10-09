"""
RDMA MTU Checking Module

This module checks RDMA port active MTUs and, on RoCE ports, the Ethernet netdev MTU.
"""

import shlex

from cvs.lib.preflight.base import PreflightCheck


DEFAULT_MIN_NETDEV_MTU = 4200
DEFAULT_MIN_ACTIVE_MTU = 4096
VALID_ACTIVE_MTUS = (0, 256, 512, 1024, 2048, 4096)


INFINIBAND = 'InfiniBand'
ETHERNET = 'Ethernet'


def transport_label(link_layer):
    """Name the RDMA transport for a port's link layer: InfiniBand, RoCE, or '' when unknown."""
    if link_layer == INFINIBAND:
        return INFINIBAND
    if link_layer == ETHERNET:
        return 'RoCE'
    return ''


class RdmaMtuCheck(PreflightCheck):
    """Check RDMA active MTU on configured interfaces, plus the netdev MTU on RoCE ports.

    An InfiniBand port has no Ethernet netdev to size. Its IPoIB netdev, when present,
    has its own MTU (often 2044) that does not limit the verbs MTU, so only active_mtu applies.
    """

    def __init__(
        self,
        orch,
        expected_interfaces,
        min_netdev_mtu=DEFAULT_MIN_NETDEV_MTU,
        min_active_mtu=DEFAULT_MIN_ACTIVE_MTU,
        gid_index="3",
        config_dict=None,
    ):
        super().__init__(orch, config_dict)
        self.expected_interfaces = list(expected_interfaces or [])
        self.min_netdev_mtu = int(min_netdev_mtu)
        self.min_active_mtu = int(min_active_mtu)
        self.gid_index = str(gid_index)

    def _build_mtu_probe_command(self):
        """Build one remote probe for every configured interface."""
        interfaces = " ".join(shlex.quote(dev) for dev in self.expected_interfaces)
        gid_index = shlex.quote(self.gid_index)
        gid_path = f'"/sys/class/infiniband/$dev/ports/1/gid_attrs/ndevs/{gid_index}"'
        if gid_index != self.gid_index:
            gid_path = f'"/sys/class/infiniband/$dev/ports/1/gid_attrs/ndevs/"{gid_index}'
        self.log_info(f"Checking RDMA MTU on interfaces: {self.expected_interfaces}")
        return f"""
for dev in {interfaces}; do
    echo "DEVICE:$dev"
    if [ ! -d "/sys/class/infiniband/$dev" ]; then echo "DEVICE_MISSING:Interface not found"; continue; fi
    ndev=$(rdma link show "$dev/1" 2>/dev/null | sed -n 's/.* netdev \\([^ ]*\\).*/\\1/p' | head -n1)
    [ -z "$ndev" ] && ndev=$(cat {gid_path} 2>/dev/null)
    echo "NETDEV:$ndev"
    if [ -n "$ndev" ] && [ -r "/sys/class/net/$ndev/mtu" ]; then
        echo "NETDEV_MTU:$(cat "/sys/class/net/$ndev/mtu")"
    else
        echo "NETDEV_MTU:"
    fi
    info=$(ibv_devinfo -d "$dev" -i 1 2>/dev/null)
    ll=$(echo "$info" | awk '/link_layer:/ {{print $2; exit}}')
    [ -z "$ll" ] && ll=$(cat "/sys/class/infiniband/$dev/ports/1/link_layer" 2>/dev/null)
    echo "LINK_LAYER:$ll"
    echo "ACTIVE_MTU:$(echo "$info" | awk '/active_mtu:/ {{print $2; exit}}')"
    echo "MAX_MTU:$(echo "$info" | awk '/max_mtu:/ {{print $2; exit}}')"
done
"""

    def _parse_probe_output(self, output):
        """Parse marker lines into interface measurements."""
        interfaces = {}
        current = None
        for raw_line in str(output).splitlines():
            line = raw_line.strip()
            if ':' not in line:
                continue
            marker, value = line.split(':', 1)
            if marker == 'DEVICE':
                current = value
                interfaces[current] = {
                    'link_layer': '',
                    'netdev': '',
                    'netdev_mtu': None,
                    'active_mtu': None,
                    'max_mtu': None,
                    'missing': False,
                }
                continue
            if current is None:
                continue
            entry = interfaces[current]
            if marker == 'DEVICE_MISSING':
                entry['missing'] = True
            elif marker == 'NETDEV':
                entry['netdev'] = value
            elif marker == 'LINK_LAYER':
                entry['link_layer'] = value
            elif marker in ('NETDEV_MTU', 'ACTIVE_MTU', 'MAX_MTU'):
                field = marker.lower()
                try:
                    entry[field] = int(value)
                except ValueError:
                    entry[field] = None
        return interfaces

    def _evaluate_node(self, node, output):
        """Compare measurements on one node with configured minimums."""
        parsed = self._parse_probe_output(output)
        result = {
            'status': 'PASS',
            'interfaces': {},
            'errors': [],
            'min_netdev_mtu': self.min_netdev_mtu,
            'min_active_mtu': self.min_active_mtu,
        }
        if not parsed:
            first_line = str(output).splitlines()
            if first_line:
                result['errors'].append(f"MTU probe returned no devices on {node}: {first_line[0]}")
        for dev in self.expected_interfaces:
            entry = dict(
                parsed.get(dev)
                or {
                    'link_layer': '',
                    'netdev': '',
                    'netdev_mtu': None,
                    'active_mtu': None,
                    'max_mtu': None,
                    'missing': False,
                }
            )
            errors = []
            netdev = entry['netdev']
            netdev_mtu = entry['netdev_mtu']
            active_mtu = entry['active_mtu']
            transport = transport_label(entry['link_layer'])
            if dev not in parsed:
                errors.append(f"No MTU data returned for {dev}")
            elif entry['missing']:
                errors.append(f"Interface {dev} not found")
            else:
                if self.min_netdev_mtu > 0 and transport != INFINIBAND:
                    if netdev_mtu is None:
                        errors.append(f"Could not determine netdev MTU for {dev} (netdev '{netdev}')")
                    elif netdev_mtu < self.min_netdev_mtu:
                        errors.append(
                            f"{dev} netdev {netdev} MTU {netdev_mtu} < {self.min_netdev_mtu} (jumbo frames not enabled)"
                        )
                if self.min_active_mtu > 0:
                    if active_mtu is None:
                        errors.append(f"Could not read active_mtu for {dev} (is ibv_devinfo installed?)")
                    elif active_mtu < self.min_active_mtu:
                        detail = f"(max_mtu {entry['max_mtu']})"
                        if transport != INFINIBAND:
                            detail += f"; netdev {netdev} MTU {netdev_mtu}"
                        errors.append(
                            f"{dev} {transport or 'RDMA'} active_mtu {active_mtu} < {self.min_active_mtu} {detail}"
                        )
            entry.update({'status': 'FAIL' if errors else 'OK', 'errors': errors})
            result['interfaces'][dev] = entry
            result['errors'].extend(errors)
        if result['errors']:
            result['status'] = 'FAIL'
        return result

    def run(self):
        """Probe all nodes in one parallel SSH round."""
        out_dict = self.orch.all.exec(self._build_mtu_probe_command(), timeout=60)
        self.results = {node: self._evaluate_node(node, output) for node, output in out_dict.items()}
        return self.results
