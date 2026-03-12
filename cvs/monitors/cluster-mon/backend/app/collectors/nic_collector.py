"""
NIC metrics collector for RDMA and Ethernet interfaces.
Adapted from CVS linux_utils.py
"""

import re
import json
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class NICMetricsCollector:
    """Collects NIC metrics via rdma, ethtool, and ip commands."""

    async def collect_rdma_links(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect RDMA link information.

        Returns:
            {
                "node1": {
                    "mlx5_0/1": {
                        "state": "ACTIVE",
                        "physical_state": "LINK_UP",
                        "netdev": "ens1"
                    },
                    ...
                },
                ...
            }
        """
        logger.info("Collecting RDMA link info")
        output = ssh_manager.exec("rdma link", timeout=60)

        rdma_dict = {}
        for node, out_str in output.items():
            if out_str.startswith("ERROR") or out_str.startswith("ABORT"):
                rdma_dict[node] = {"error": out_str}
                continue

            rdma_dict[node] = {}

            # Parse rdma link output
            # Format: link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev ens1
            pattern = r"link\s+([\w_]+/\d+)\s+state\s+(\w+)\s+physical_state\s+(\w+)\s+netdev\s+([\w\-\.]+)"

            for line in out_str.split("\n"):
                match = re.search(pattern, line)
                if match:
                    rdma_dev = match.group(1)
                    rdma_dict[node][rdma_dev] = {
                        "state": match.group(2),
                        "physical_state": match.group(3),
                        "netdev": match.group(4),
                    }

        return rdma_dict

    async def collect_rdma_stats(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect RDMA statistics including congestion control metrics.

        Uses 'rdma statistic show --json' for reliable parsing.

        Returns:
            {
                "node1": {
                    "mlx5_0/1": {
                        "port_rcv_data": 123456789,
                        "port_xmit_data": 987654321,
                        "rx_rdma_ecn_pkts": 100,
                        "tx_rdma_ecn_pkts": 50,
                        ...
                    },
                    ...
                },
                ...
            }
        """
        logger.info("Collecting RDMA statistics (includes congestion control metrics)")
        # Use bash -c to properly handle shell redirection and || operator
        output = ssh_manager.exec("bash -c 'rdma statistic show --json 2>/dev/null || echo \"{}\"'", timeout=60)

        logger.info(f"RDMA stats output received from {len(output)} nodes")

        rdma_stats = {}
        for node, out_str in output.items():
            if out_str.startswith("ERROR") or out_str.startswith("ABORT"):
                rdma_stats[node] = {"error": out_str}
                logger.warning(f"Node {node}: RDMA stats error - {out_str[:100]}")
                continue

            try:
                if not out_str.strip() or out_str.strip() == '{}':
                    rdma_stats[node] = {}
                    logger.info(f"Node {node}: Empty RDMA stats output")
                    continue

                data = json.loads(out_str)
                rdma_stats[node] = {}

                # Parse JSON output
                # Actual format: [{ "ifname": "rdma0", "port": 1, "rx_pkts": 123, ... }]
                # Stats are direct key-value pairs, not in a "counters" array
                if isinstance(data, list):
                    for entry in data:
                        if isinstance(entry, dict):
                            ifname = entry.get("ifname", "")
                            port = entry.get("port", "")

                            if ifname:
                                # Create device key with port (e.g., "rdma0/1")
                                dev_key = f"{ifname}/{port}" if port else ifname

                                # Extract all stats (skip metadata fields)
                                rdma_stats[node][dev_key] = {}
                                metadata_keys = {"ifname", "port", "ifindex"}

                                for key, value in entry.items():
                                    if key not in metadata_keys and isinstance(value, (int, float)):
                                        rdma_stats[node][dev_key][key] = value

                                logger.info(
                                    f"Node {node}: Device {dev_key} - parsed {len(rdma_stats[node][dev_key])} stats"
                                )

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse RDMA stats JSON for {node}: {e}")
                logger.warning(f"  Output was: {out_str[:200]}")
                rdma_stats[node] = {}

        logger.info(f"RDMA stats collection complete: {len(rdma_stats)} nodes with data")
        return rdma_stats

    async def collect_ethtool_stats(self, ssh_manager, interfaces: Dict[str, list] = None) -> Dict[str, Any]:
        """
        Collect network interface statistics using 'ip -s link' (optimized).

        OPTIMIZATION: Instead of running ethtool per interface (10+ commands per node),
        we run 'ip -s link' once per node to get all interface stats in one shot.

        Returns:
            {
                "node1": {
                    "ens1": {
                        "rx_bytes": 123456789,
                        "tx_bytes": 987654321,
                        "rx_packets": 456789,
                        "tx_packets": 654321,
                        "rx_errors": 0,
                        "tx_errors": 0,
                        "rx_dropped": 0,
                        "tx_dropped": 0,
                        ...
                    },
                    ...
                },
                ...
            }
        """
        logger.info("Collecting network statistics via 'ip -s link' (optimized)")

        # Run 'ip -s link' once per node to get all interface stats
        cmd = "ip -s link show"
        output = ssh_manager.exec(cmd, timeout=60)

        eth_stats = {}

        for node, out_str in output.items():
            if out_str.startswith("ERROR") or out_str.startswith("ABORT"):
                continue

            eth_stats[node] = {}

            # Parse ip -s link output
            # Format:
            # 2: ens1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 ...
            #     RX: bytes  packets  errors  dropped missed  mcast
            #     12345678   123456   0       0       0       0
            #     TX: bytes  packets  errors  dropped carrier collsns
            #     87654321   654321   0       0       0       0

            lines = out_str.split('\n')
            current_iface = None
            rx_line = False
            tx_line = False

            for i, line in enumerate(lines):
                # Match interface line: "2: ens1: <...>"
                iface_match = re.match(r'^\d+:\s+(\S+):', line)
                if iface_match:
                    current_iface = iface_match.group(1).split('@')[0]  # Remove @... suffix
                    # Skip loopback and virtual interfaces
                    if current_iface == 'lo' or current_iface.startswith('veth'):
                        current_iface = None
                        continue
                    eth_stats[node][current_iface] = {}
                    rx_line = False
                    tx_line = False
                    continue

                if not current_iface:
                    continue

                # Check for RX line
                if 'RX:' in line and 'bytes' in line:
                    rx_line = True
                    continue

                # Check for TX line
                if 'TX:' in line and 'bytes' in line:
                    tx_line = True
                    continue

                # Parse RX stats line
                if rx_line and current_iface:
                    parts = line.strip().split()
                    if len(parts) >= 6 and parts[0].isdigit():
                        eth_stats[node][current_iface]['rx_bytes'] = int(parts[0])
                        eth_stats[node][current_iface]['rx_packets'] = int(parts[1])
                        eth_stats[node][current_iface]['rx_errors'] = int(parts[2])
                        eth_stats[node][current_iface]['rx_dropped'] = int(parts[3])
                        eth_stats[node][current_iface]['rx_missed'] = int(parts[4])
                        eth_stats[node][current_iface]['rx_mcast'] = int(parts[5])
                        rx_line = False
                        continue

                # Parse TX stats line
                if tx_line and current_iface:
                    parts = line.strip().split()
                    if len(parts) >= 6 and parts[0].isdigit():
                        eth_stats[node][current_iface]['tx_bytes'] = int(parts[0])
                        eth_stats[node][current_iface]['tx_packets'] = int(parts[1])
                        eth_stats[node][current_iface]['tx_errors'] = int(parts[2])
                        eth_stats[node][current_iface]['tx_dropped'] = int(parts[3])
                        eth_stats[node][current_iface]['tx_carrier'] = int(parts[4])
                        eth_stats[node][current_iface]['tx_collsns'] = int(parts[5])
                        tx_line = False
                        continue

        return eth_stats

    async def collect_ip_addr(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect IP address information.

        Returns:
            {
                "node1": {
                    "ens1": {
                        "mtu": "1500",
                        "state": "UP",
                        "mac_addr": "aa:bb:cc:dd:ee:ff",
                        "ipv4_addr_list": ["192.168.1.10/24"],
                        "ipv6_addr_list": ["fe80::1/64"]
                    },
                    ...
                },
                ...
            }
        """
        logger.info("Collecting IP address info")
        output = ssh_manager.exec("bash -c 'ip addr show | grep -A 5 mtu --color=never'", timeout=60)

        ip_dict = {}

        for node, out_str in output.items():
            if out_str.startswith("ERROR") or out_str.startswith("ABORT"):
                ip_dict[node] = {"error": out_str}
                continue

            ip_dict[node] = {}
            int_nam = None

            for line in out_str.split("\n"):
                # Match interface header: "2: ens1: <BROADCAST,MULTICAST,UP>"
                pattern = r"[0-9]+:\s+([\w\.\-_/]+):\s+([\<\>\,A-Z0-9_]+)"
                if re.search(pattern, line):
                    match = re.search(pattern, line)
                    int_nam = match.group(1)
                    ip_dict[node][int_nam] = {
                        "ipv4_addr_list": [],
                        "ipv6_addr_list": [],
                        "flags": match.group(2),
                    }

                if not int_nam:
                    continue

                # MTU
                if re.search(r"mtu ([0-9]+)", line):
                    match = re.search(r"mtu ([0-9]+)", line)
                    ip_dict[node][int_nam]["mtu"] = match.group(1)

                # State
                if re.search(r"state ([A-Z]+)", line):
                    match = re.search(r"state ([A-Z]+)", line)
                    ip_dict[node][int_nam]["state"] = match.group(1)

                # MAC address
                pattern = r"link/ether\s+([a-f0-9:]+)"
                if re.search(pattern, line):
                    match = re.search(pattern, line)
                    ip_dict[node][int_nam]["mac_addr"] = match.group(1)

                # IPv4
                pattern = r"inet\s+([0-9\./]+)"
                if re.search(pattern, line):
                    match = re.search(pattern, line)
                    ip_dict[node][int_nam]["ipv4_addr_list"].append(match.group(1))

                # IPv6
                pattern = r"inet6\s+([a-f0-9:/]+)"
                if re.search(pattern, line):
                    match = re.search(pattern, line)
                    ip_dict[node][int_nam]["ipv6_addr_list"].append(match.group(1))

        return ip_dict

    async def collect_lldp(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect LLDP neighbor information (if lldpctl is available).

        Returns:
            {
                "node1": {
                    "ens1": {
                        "neighbor_name": "switch1",
                        "neighbor_port": "Ethernet1/1",
                        ...
                    },
                    ...
                },
                ...
            }
        """
        logger.info("Collecting LLDP info")
        # Use bash -c to properly handle shell redirection and || operator
        output = ssh_manager.exec("bash -c 'sudo lldpctl -f json 2>/dev/null || echo \"{}\"'", timeout=60)

        lldp_dict = {}
        for node, out_str in output.items():
            try:
                if out_str.strip() and not out_str.startswith("ERROR"):
                    lldp_dict[node] = json.loads(out_str)
                else:
                    lldp_dict[node] = {}
            except json.JSONDecodeError:
                lldp_dict[node] = {}

        return lldp_dict

    def _filter_lldp_by_rdma(self, lldp_data: Dict[str, Any], rdma_links_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter LLDP data to only include interfaces that appear in rdma link output.
        This filters out management interfaces and only shows backend network topology.

        Args:
            lldp_data: Raw LLDP data from lldpctl
            rdma_links_data: RDMA link data containing netdev mappings

        Returns:
            Filtered LLDP data with only RDMA interfaces
        """
        filtered_lldp = {}

        for node, lldp_node_data in lldp_data.items():
            if not lldp_node_data or node not in rdma_links_data:
                filtered_lldp[node] = {}
                continue

            # Get list of RDMA network interfaces for this node
            rdma_interfaces = set()
            rdma_node_data = rdma_links_data.get(node, {})

            if isinstance(rdma_node_data, dict) and "error" not in rdma_node_data:
                for rdma_dev, rdma_info in rdma_node_data.items():
                    if isinstance(rdma_info, dict) and "netdev" in rdma_info:
                        rdma_interfaces.add(rdma_info["netdev"])

            logger.info(f"Node {node}: Filtering LLDP to only include RDMA interfaces: {rdma_interfaces}")

            # Filter LLDP data to only include these interfaces
            filtered_node_lldp = {}

            # Handle LLDP data structure: node.lldp.interface
            lldp_info = lldp_node_data.get("lldp", lldp_node_data)
            if "interface" in lldp_info:
                interfaces = lldp_info["interface"]

                # Handle both array and object formats
                if isinstance(interfaces, list):
                    # Array format: [{ "enp195s0": {...} }, { "enp195s0": {...} }]
                    filtered_interfaces = []
                    for interface_entry in interfaces:
                        if isinstance(interface_entry, dict):
                            filtered_entry = {}
                            for ifname, ifdata in interface_entry.items():
                                if ifname in rdma_interfaces:
                                    filtered_entry[ifname] = ifdata
                            if filtered_entry:
                                filtered_interfaces.append(filtered_entry)

                    if filtered_interfaces:
                        filtered_node_lldp = {"lldp": {"interface": filtered_interfaces}}
                else:
                    # Object format: { "enp195s0": {...}, "eth0": {...} }
                    filtered_interface_dict = {}
                    for ifname, ifdata in interfaces.items():
                        if ifname in rdma_interfaces:
                            filtered_interface_dict[ifname] = ifdata

                    if filtered_interface_dict:
                        filtered_node_lldp = {"lldp": {"interface": filtered_interface_dict}}

            filtered_lldp[node] = filtered_node_lldp

        return filtered_lldp

    async def collect_all_metrics(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect all NIC metrics in parallel.

        Returns:
            {
                "timestamp": "2025-02-11T12:00:00Z",
                "rdma_links": {...},
                "rdma_stats": {...},
                "ethtool_stats": {...},
                "ip_addr": {...},
                "lldp": {...}
            }
        """

        logger.info("Collecting all NIC metrics")

        # Collect SEQUENTIALLY (one command completes before next starts)
        rdma_links = await self.collect_rdma_links(ssh_manager)
        rdma_stats = await self.collect_rdma_stats(ssh_manager)
        ip_addr = await self.collect_ip_addr(ssh_manager)
        lldp = await self.collect_lldp(ssh_manager)
        results = [rdma_links, rdma_stats, ip_addr, lldp]

        # Get interface info first for ethtool
        ip_data = results[2] if not isinstance(results[2], Exception) else {}

        # Build interface list for ethtool
        interfaces = {}
        for node, iface_data in ip_data.items():
            if isinstance(iface_data, dict) and "error" not in iface_data:
                interfaces[node] = list(iface_data.keys())

        # Collect ethtool stats with interface list
        ethtool_stats = await self.collect_ethtool_stats(ssh_manager, interfaces)

        # Collect RDMA resources
        rdma_res = await self.collect_rdma_resources(ssh_manager)

        # Filter LLDP data to only include RDMA interfaces
        rdma_links_data = results[0] if not isinstance(results[0], Exception) else {}
        lldp_data = results[3] if not isinstance(results[3], Exception) else {}
        filtered_lldp = self._filter_lldp_by_rdma(lldp_data, rdma_links_data)

        metrics = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "rdma_links": rdma_links_data,
            "rdma_stats": results[1] if not isinstance(results[1], Exception) else {},
            "rdma_resources": rdma_res,
            "ip_addr": ip_data,
            "lldp": filtered_lldp,
            "ethtool_stats": ethtool_stats,
        }

        return metrics

    async def collect_rdma_resources(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect RDMA resources (pd, cq, qp, mr, etc.).

        Returns:
            {
                "node1": {
                    "bnxt_re0": {"pd": 1, "cq": 1, "qp": 1, ...},
                    ...
                },
                ...
            }
        """
        logger.info("Collecting RDMA resources")
        output = ssh_manager.exec("rdma res", timeout=60)

        rdma_res = {}
        for node, out_str in output.items():
            if out_str.startswith("ERROR") or out_str.startswith("ABORT"):
                rdma_res[node] = {"error": out_str}
                continue

            rdma_res[node] = {}

            # Parse rdma res output
            # Format: 0: bnxt_re0: pd 1 cq 1 qp 1 cm_id 0 mr 0 ctx 0 srq 0
            pattern = r"(\d+):\s+([\w_]+):\s+(.+)"

            for line in out_str.split("\n"):
                match = re.search(pattern, line)
                if match:
                    device = match.group(2)
                    resources_str = match.group(3)

                    # Parse resource key-value pairs
                    resources = {}
                    for res_match in re.finditer(r"(\w+)\s+(\d+)", resources_str):
                        key = res_match.group(1)
                        value = int(res_match.group(2))
                        resources[key] = value

                    rdma_res[node][device] = resources

        return rdma_res
