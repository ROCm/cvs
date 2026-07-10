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

        ip_output = await ssh_manager.exec_async(
            "bash -c \"ip -o link show | awk -F': ' '{print \\$2}' | grep -v lo\"", timeout=60
        )

        # Build per-host interface list and a deduplicated set of all interface
        # names seen across the fleet so we can run ethtool once per unique name
        # rather than once per (host, interface) pair.
        host_ifaces: Dict[str, list] = {}
        all_ifaces: set = set()
        firmware_info: Dict[str, Any] = {}

        for host, ifaces_str in ip_output.items():
            if ifaces_str.startswith("ERROR") or ifaces_str.startswith("ABORT"):
                firmware_info[host] = {"error": ifaces_str}
                continue
            ifaces = [i.strip() for i in ifaces_str.split("\n") if i.strip() and "@" not in i][:10]
            host_ifaces[host] = ifaces
            all_ifaces.update(ifaces)
            firmware_info[host] = {}

        # One fleet-wide exec per unique interface name instead of per (host, iface).
        iface_outputs: Dict[str, Dict[str, str]] = {}
        for iface in sorted(all_ifaces):
            iface_outputs[iface] = await ssh_manager.exec_async(f"sudo ethtool -i {iface} 2>/dev/null", timeout=60)

        for host, ifaces in host_ifaces.items():
            for iface in ifaces:
                raw = iface_outputs.get(iface, {}).get(host, "")
                if raw:
                    info = {}
                    for line in raw.split("\n"):
                        if ":" in line:
                            key, value = line.split(":", 1)
                            info[key.strip().lower().replace(" ", "_")] = value.strip()
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

        # Run each command once fleet-wide — do NOT call exec_async inside the
        # per-host loop; that would repeat the fleet-wide sweep len(hosts) times.
        mlx_output = await ssh_manager.exec_async(commands[0], timeout=60)
        bnxt_output = await ssh_manager.exec_async(commands[1], timeout=60)
        amd_output = await ssh_manager.exec_async(commands[2], timeout=60)

        driver_info = {}
        for host in ssh_manager.reachable_hosts:
            driver_info[host] = {}

            raw = mlx_output.get(host, "")
            if raw and "modinfo" not in raw:
                info = {
                    k.strip(): v.strip() for line in raw.split("\n") if ":" in line for k, v in [line.split(":", 1)]
                }
                if info:
                    driver_info[host]["mlx5_core"] = info

            raw = bnxt_output.get(host, "")
            if raw and "modinfo" not in raw:
                info = {
                    k.strip(): v.strip() for line in raw.split("\n") if ":" in line for k, v in [line.split(":", 1)]
                }
                if info:
                    driver_info[host]["bnxt_en"] = info

            raw = amd_output.get(host, "")
            if raw and "Not loaded" not in raw:
                info = {
                    k.strip(): v.strip() for line in raw.split("\n") if ":" in line for k, v in [line.split(":", 1)]
                }
                if info:
                    driver_info[host]["amd-ainic"] = info

        return driver_info

    async def collect_rdma_statistics_detailed(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect detailed RDMA statistics from 'rdma statistic show --json'.

        Returns comprehensive RDMA counter statistics.
        """
        logger.info("Collecting detailed RDMA statistics")
        output = await ssh_manager.exec_async(
            "bash -c 'rdma statistic show --json 2>/dev/null || echo \"[]\"'", timeout=60
        )

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

        ip_output = await ssh_manager.exec_async(
            "bash -c \"ip -o link show | awk -F': ' '{print \\$2}' | grep -v lo\"", timeout=60
        )

        # Deduplicate interface names across the fleet — run ethtool -S once per
        # unique name rather than once per (host, interface) pair.
        host_ifaces: Dict[str, list] = {}
        all_ifaces: set = set()
        eth_stats: Dict[str, Any] = {}

        for host, ifaces_str in ip_output.items():
            if ifaces_str.startswith("ERROR") or ifaces_str.startswith("ABORT"):
                eth_stats[host] = {"error": ifaces_str}
                continue
            ifaces = [i.strip() for i in ifaces_str.split("\n") if i.strip() and "@" not in i][:10]
            host_ifaces[host] = ifaces
            all_ifaces.update(ifaces)
            eth_stats[host] = {}

        iface_outputs: Dict[str, Dict[str, str]] = {}
        for iface in sorted(all_ifaces):
            iface_outputs[iface] = await ssh_manager.exec_async(f"sudo ethtool -S {iface} 2>/dev/null", timeout=60)

        for host, ifaces in host_ifaces.items():
            for iface in ifaces:
                raw = iface_outputs.get(iface, {}).get(host, "")
                if raw and "NOT_AVAILABLE" not in raw:
                    stats = {}
                    for line in raw.split("\n"):
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
        output = await ssh_manager.exec_async("bash -c \"lspci -nn | grep -i 'network\\|ethernet'\"", timeout=60)

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

        logger.info("Collecting all NIC software information")

        nic_firmware = await self.collect_nic_firmware_version(ssh_manager)
        nic_drivers = await self.collect_nic_driver_version(ssh_manager)
        rdma_statistics = await self.collect_rdma_statistics_detailed(ssh_manager)
        ethtool_statistics = await self.collect_ethtool_statistics_detailed(ssh_manager)
        pci_devices = await self.collect_pci_device_info(ssh_manager)

        software_info = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "nic_firmware": nic_firmware if not isinstance(nic_firmware, Exception) else {},
            "nic_drivers": nic_drivers if not isinstance(nic_drivers, Exception) else {},
            "rdma_statistics": rdma_statistics if not isinstance(rdma_statistics, Exception) else {},
            "ethtool_statistics": ethtool_statistics if not isinstance(ethtool_statistics, Exception) else {},
            "pci_devices": pci_devices if not isinstance(pci_devices, Exception) else {},
        }

        return software_info
