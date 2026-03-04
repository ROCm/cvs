"""
NIC software and statistics collector.
Collects NIC firmware, driver versions, RDMA statistics, and ethtool statistics.
Supports AMD AINIC, NVIDIA CX7, and Broadcom Thor2.
"""

import re
import json
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class NICSoftwareCollector:
    """Collects NIC software, firmware, and detailed statistics."""

    async def collect_nic_firmware_version(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect NIC firmware versions.

        Command: ethtool -i <interface> for firmware version
        """
        logger.info("Collecting NIC firmware versions")

        # First get list of interfaces
        ip_output = await ssh_manager.exec_async("ip -o link show | awk -F': ' '{print $2}' | grep -v lo")

        firmware_info = {}

        for host, ifaces_str in ip_output.items():
            if ifaces_str.startswith("ERROR") or ifaces_str.startswith("ABORT"):
                firmware_info[host] = {"error": ifaces_str}
                continue

            firmware_info[host] = {}
            interfaces = [i.strip() for i in ifaces_str.split("\n") if i.strip() and "@" not in i]

            for iface in interfaces[:10]:  # Limit to first 10 interfaces
                cmd = f"sudo ethtool -i {iface} 2>/dev/null"
                output = await ssh_manager.exec_async(cmd)

                if host in output and output[host]:
                    info = {}
                    for line in output[host].split("\n"):
                        if ":" in line:
                            key, value = line.split(":", 1)
                            key = key.strip().lower().replace(" ", "_")
                            info[key] = value.strip()

                    if info:
                        firmware_info[host][iface] = info

        return firmware_info

    async def collect_nic_driver_version(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect NIC driver versions for different vendors.

        Checks for:
        - AMD AINIC: amd-ainic driver
        - NVIDIA CX7: mlx5_core driver
        - Broadcom Thor2: bnxt_en driver
        """
        logger.info("Collecting NIC driver versions")

        commands = [
            "modinfo mlx5_core 2>/dev/null | grep -E '^version|^firmware' | head -3",
            "modinfo bnxt_en 2>/dev/null | grep -E '^version|^firmware' | head -3",
            "modinfo amd-ainic 2>/dev/null | grep -E '^version|^firmware' | head -3 || echo 'Not loaded'",
        ]

        driver_info = {}

        for host in ssh_manager.reachable_hosts:
            driver_info[host] = {}

            # Check Mellanox (NVIDIA CX7)
            output = await ssh_manager.exec_async(commands[0])
            if host in output and output[host] and "modinfo" not in output[host]:
                mlx_info = {}
                for line in output[host].split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        mlx_info[key.strip()] = value.strip()
                if mlx_info:
                    driver_info[host]["mlx5_core"] = mlx_info

            # Check Broadcom (Thor2)
            output = await ssh_manager.exec_async(commands[1])
            if host in output and output[host] and "modinfo" not in output[host]:
                bnxt_info = {}
                for line in output[host].split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        bnxt_info[key.strip()] = value.strip()
                if bnxt_info:
                    driver_info[host]["bnxt_en"] = bnxt_info

            # Check AMD AINIC
            output = await ssh_manager.exec_async(commands[2])
            if host in output and output[host] and "Not loaded" not in output[host]:
                amd_info = {}
                for line in output[host].split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        amd_info[key.strip()] = value.strip()
                if amd_info:
                    driver_info[host]["amd-ainic"] = amd_info

        return driver_info

    async def collect_rdma_statistics_detailed(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect detailed RDMA statistics from 'rdma statistic show --json'.

        Returns comprehensive RDMA counter statistics.
        """
        logger.info("Collecting detailed RDMA statistics")
        output = await ssh_manager.exec_async("rdma statistic show --json 2>/dev/null || echo '[]'")

        rdma_stats = {}
        for host, out_str in output.items():
            if out_str.startswith("ERROR") or out_str.startswith("ABORT"):
                rdma_stats[host] = {"error": out_str}
                continue

            try:
                if not out_str.strip() or out_str.strip() == '[]':
                    rdma_stats[host] = {}
                    continue

                data = json.loads(out_str)
                rdma_stats[host] = {}

                # Parse JSON output
                # Actual format: [{ "ifname": "rdma0", "port": 1, "rx_pkts": 123, ... }]
                # Stats are direct key-value pairs, not in a "counters" array
                if isinstance(data, list):
                    for entry in data:
                        if isinstance(entry, dict):
                            ifname = entry.get("ifname", "")
                            port = entry.get("port", "")

                            if ifname:
                                # Create device key with port
                                dev_key = f"{ifname}/{port}" if port else ifname
                                rdma_stats[host][dev_key] = {}

                                # Extract all stats (skip metadata)
                                metadata_keys = {"ifname", "port", "ifindex"}

                                for key, value in entry.items():
                                    if key not in metadata_keys and isinstance(value, (int, float)):
                                        rdma_stats[host][dev_key][key] = value

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse RDMA stats JSON for {host}: {e}")
                rdma_stats[host] = {}

        return rdma_stats

    async def collect_ethtool_statistics_detailed(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect detailed ethtool statistics for all interfaces.

        Command: ethtool -S <interface>
        Returns all NIC counters.
        """
        logger.info("Collecting detailed ethtool statistics")

        # Get list of interfaces first
        ip_output = await ssh_manager.exec_async("ip -o link show | awk -F': ' '{print $2}' | grep -v lo")

        eth_stats = {}

        for host, ifaces_str in ip_output.items():
            if ifaces_str.startswith("ERROR") or ifaces_str.startswith("ABORT"):
                eth_stats[host] = {"error": ifaces_str}
                continue

            eth_stats[host] = {}
            interfaces = [i.strip() for i in ifaces_str.split("\n") if i.strip() and "@" not in i]

            for iface in interfaces[:10]:  # Limit to first 10
                cmd = f"sudo ethtool -S {iface} 2>/dev/null"
                output = await ssh_manager.exec_async(cmd)

                if host in output and output[host] and "NOT_AVAILABLE" not in output[host]:
                    stats = {}
                    for line in output[host].split("\n"):
                        # Parse "     stat_name: value"
                        match = re.search(r"^\s+([\w_]+):\s+(\d+)", line)
                        if match:
                            stats[match.group(1)] = int(match.group(2))

                    if stats:
                        eth_stats[host][iface] = stats

        return eth_stats

    async def collect_pci_device_info(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect PCI device information for NICs.

        Command: lspci -nn | grep -i network
        """
        logger.info("Collecting PCI device info for NICs")
        output = await ssh_manager.exec_async("lspci -nn | grep -i 'network\\|ethernet'")

        pci_info = {}
        for host, out_str in output.items():
            if out_str.startswith("ERROR") or out_str.startswith("ABORT"):
                pci_info[host] = {"error": out_str}
                continue

            devices = []
            for line in out_str.split("\n"):
                if line.strip():
                    # Parse PCI address and device info
                    match = re.match(r"([\da-f:\.]+)\s+(.+)", line, re.I)
                    if match:
                        devices.append({"pci_address": match.group(1), "description": match.group(2).strip()})

            pci_info[host] = {"devices": devices}

        return pci_info

    async def collect_all_software_info(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect all NIC software and statistics information.

        Returns consolidated NIC software info.
        """
        import asyncio

        logger.info("Collecting all NIC software information")

        results = await asyncio.gather(
            self.collect_nic_firmware_version(ssh_manager),
            self.collect_nic_driver_version(ssh_manager),
            self.collect_rdma_statistics_detailed(ssh_manager),
            self.collect_ethtool_statistics_detailed(ssh_manager),
            self.collect_pci_device_info(ssh_manager),
            return_exceptions=True,
        )

        software_info = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "nic_firmware": results[0] if not isinstance(results[0], Exception) else {},
            "nic_drivers": results[1] if not isinstance(results[1], Exception) else {},
            "rdma_statistics": results[2] if not isinstance(results[2], Exception) else {},
            "ethtool_statistics": results[3] if not isinstance(results[3], Exception) else {},
            "pci_devices": results[4] if not isinstance(results[4], Exception) else {},
        }

        return software_info
