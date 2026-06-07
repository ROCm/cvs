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

    # Combined single-command versions of each collection (used by Go binary fast path)
    _CMD_FIRMWARE = (
        r"bash -c '"
        r"for iface in $(ip -o link show | awk -F\": \" \"{print \$2}\" | grep -v lo | grep -v @ | head -10); do "
        r"printf \"===IFACE:%s===\n\" \"$iface\"; "
        r"sudo ethtool -i \"$iface\" 2>/dev/null; "
        r"done'"
    )
    _CMD_DRIVERS = (
        r"bash -c '"
        r"printf \"===DRIVER:mlx5_core===\n\"; modinfo mlx5_core 2>/dev/null | grep -E \"^version|^firmware\" | head -3; "
        r"printf \"===DRIVER:bnxt_en===\n\"; modinfo bnxt_en 2>/dev/null | grep -E \"^version|^firmware\" | head -3; "
        r"printf \"===DRIVER:amd-ainic===\n\"; modinfo amd-ainic 2>/dev/null | grep -E \"^version|^firmware\" | head -3 || printf \"not loaded\n\"'"
    )
    _CMD_RDMA = "bash -c 'rdma statistic show --json 2>/dev/null || echo \"[]\"'"
    _CMD_ETHTOOL = (
        r"bash -c '"
        r"for iface in $(ip -o link show | awk -F\": \" \"{print \$2}\" | grep -v lo | grep -v @ | head -10); do "
        r"printf \"===IFACE:%s===\n\" \"$iface\"; "
        r"sudo ethtool -S \"$iface\" 2>/dev/null; "
        r"done'"
    )
    _CMD_PCI = "bash -c \"lspci -nn | grep -i 'network\\|ethernet'\""

    @staticmethod
    def _parse_iface_sections(output_str: str) -> Dict[str, str]:
        """Split combined per-interface output into {iface: raw_block} dict."""
        sections: Dict[str, str] = {}
        current: str | None = None
        lines: list = []
        for line in output_str.split('\n'):
            if line.startswith('===IFACE:') and line.endswith('==='):
                if current is not None:
                    sections[current] = '\n'.join(lines)
                current = line[9:-3]
                lines = []
            elif current is not None:
                lines.append(line)
        if current is not None:
            sections[current] = '\n'.join(lines)
        return sections

    async def collect_nic_firmware_version(self, ssh_manager=None, preloaded_output=None) -> Dict[str, Any]:
        """
        Collect NIC firmware versions.

        Command: ethtool -i <interface> for firmware version
        """
        logger.info("Collecting NIC firmware versions")

        if preloaded_output is not None:
            # Combined output: one string per host containing all interfaces
            firmware_info = {}
            for host, combined_str in preloaded_output.items():
                if not combined_str or combined_str.startswith("ERROR") or combined_str.startswith("ABORT"):
                    firmware_info[host] = {"error": combined_str}
                    continue
                firmware_info[host] = {}
                for iface, block in self._parse_iface_sections(combined_str).items():
                    info = {}
                    for line in block.split("\n"):
                        if ":" in line:
                            key, value = line.split(":", 1)
                            key = key.strip().lower().replace(" ", "_")
                            info[key] = value.strip()
                    if info:
                        firmware_info[host][iface] = info
            return firmware_info

        # Fallback: original sequential parallel-ssh approach
        ip_output = await ssh_manager.exec_async(
            "bash -c \"ip -o link show | awk -F': ' '{print \\$2}' | grep -v lo\"", timeout=60
        )
        firmware_info = {}
        for host, ifaces_str in ip_output.items():
            if ifaces_str.startswith("ERROR") or ifaces_str.startswith("ABORT"):
                firmware_info[host] = {"error": ifaces_str}
                continue
            firmware_info[host] = {}
            interfaces = [i.strip() for i in ifaces_str.split("\n") if i.strip() and "@" not in i]
            for iface in interfaces[:10]:
                cmd = f"sudo ethtool -i {iface} 2>/dev/null"
                output = await ssh_manager.exec_async(cmd, timeout=60)
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

    async def collect_nic_driver_version(self, ssh_manager=None, preloaded_output=None) -> Dict[str, Any]:
        """
        Collect NIC driver versions for different vendors.

        Checks for:
        - AMD AINIC: amd-ainic driver
        - NVIDIA CX7: mlx5_core driver
        - Broadcom Thor2: bnxt_en driver
        """
        logger.info("Collecting NIC driver versions")

        def _parse_driver_block(block: str, driver_name: str) -> dict:
            info = {}
            for line in block.split("\n"):
                if ":" in line and "modinfo" not in line and "not loaded" not in line.lower():
                    key, value = line.split(":", 1)
                    info[key.strip()] = value.strip()
            return info

        if preloaded_output is not None:
            driver_info = {}
            for host, combined_str in preloaded_output.items():
                if not combined_str or combined_str.startswith("ERROR") or combined_str.startswith("ABORT"):
                    driver_info[host] = {"error": combined_str}
                    continue
                driver_info[host] = {}
                # Split on ===DRIVER:name=== markers
                current_driver = None
                lines = []
                for line in combined_str.split("\n"):
                    if line.startswith("===DRIVER:") and line.endswith("==="):
                        if current_driver and lines:
                            info = _parse_driver_block("\n".join(lines), current_driver)
                            if info:
                                driver_info[host][current_driver] = info
                        current_driver = line[10:-3]
                        lines = []
                    elif current_driver is not None:
                        lines.append(line)
                if current_driver and lines:
                    info = _parse_driver_block("\n".join(lines), current_driver)
                    if info:
                        driver_info[host][current_driver] = info
            return driver_info

        # Fallback: original sequential parallel-ssh approach
        commands = [
            "modinfo mlx5_core 2>/dev/null | grep -E '^version|^firmware' | head -3",
            "modinfo bnxt_en 2>/dev/null | grep -E '^version|^firmware' | head -3",
            "modinfo amd-ainic 2>/dev/null | grep -E '^version|^firmware' | head -3 || echo 'Not loaded'",
        ]
        driver_info = {}
        for host in ssh_manager.reachable_hosts:
            driver_info[host] = {}
            for driver_name, cmd in zip(["mlx5_core", "bnxt_en", "amd-ainic"], commands):
                output = await ssh_manager.exec_async(cmd, timeout=60)
                if (
                    host in output
                    and output[host]
                    and "Not loaded" not in output[host]
                    and "modinfo" not in output[host]
                ):
                    info = {}
                    for line in output[host].split("\n"):
                        if ":" in line:
                            key, value = line.split(":", 1)
                            info[key.strip()] = value.strip()
                    if info:
                        driver_info[host][driver_name] = info
        return driver_info

    async def collect_rdma_statistics_detailed(self, ssh_manager=None, preloaded_output=None) -> Dict[str, Any]:
        """
        Collect detailed RDMA statistics from 'rdma statistic show --json'.

        Returns comprehensive RDMA counter statistics.
        """
        logger.info("Collecting detailed RDMA statistics")
        output = (
            preloaded_output
            if preloaded_output is not None
            else await ssh_manager.exec_async(
                "bash -c 'rdma statistic show --json 2>/dev/null || echo \"[]\"'", timeout=60
            )
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

    async def collect_ethtool_statistics_detailed(self, ssh_manager=None, preloaded_output=None) -> Dict[str, Any]:
        """
        Collect detailed ethtool statistics for all interfaces.

        Command: ethtool -S <interface>
        Returns all NIC counters.
        """
        logger.info("Collecting detailed ethtool statistics")

        if preloaded_output is not None:
            eth_stats = {}
            for host, combined_str in preloaded_output.items():
                if not combined_str or combined_str.startswith("ERROR") or combined_str.startswith("ABORT"):
                    eth_stats[host] = {"error": combined_str}
                    continue
                eth_stats[host] = {}
                for iface, block in self._parse_iface_sections(combined_str).items():
                    stats = {}
                    for line in block.split("\n"):
                        match = re.search(r"^\s+([\w_]+):\s+(\d+)", line)
                        if match:
                            stats[match.group(1)] = int(match.group(2))
                    if stats:
                        eth_stats[host][iface] = stats
            return eth_stats

        # Fallback: original sequential parallel-ssh approach
        ip_output = await ssh_manager.exec_async(
            "bash -c \"ip -o link show | awk -F': ' '{print \\$2}' | grep -v lo\"", timeout=60
        )
        eth_stats = {}
        for host, ifaces_str in ip_output.items():
            if ifaces_str.startswith("ERROR") or ifaces_str.startswith("ABORT"):
                eth_stats[host] = {"error": ifaces_str}
                continue
            eth_stats[host] = {}
            interfaces = [i.strip() for i in ifaces_str.split("\n") if i.strip() and "@" not in i]
            for iface in interfaces[:10]:
                cmd = f"sudo ethtool -S {iface} 2>/dev/null"
                output = await ssh_manager.exec_async(cmd, timeout=60)
                if host in output and output[host] and "NOT_AVAILABLE" not in output[host]:
                    stats = {}
                    for line in output[host].split("\n"):
                        match = re.search(r"^\s+([\w_]+):\s+(\d+)", line)
                        if match:
                            stats[match.group(1)] = int(match.group(2))
                    if stats:
                        eth_stats[host][iface] = stats
        return eth_stats

    async def collect_pci_device_info(self, ssh_manager=None, preloaded_output=None) -> Dict[str, Any]:
        """
        Collect PCI device information for NICs.

        Command: lspci -nn | grep -i network
        """
        logger.info("Collecting PCI device info for NICs")
        output = (
            preloaded_output
            if preloaded_output is not None
            else await ssh_manager.exec_async("bash -c \"lspci -nn | grep -i 'network\\|ethernet'\"", timeout=60)
        )

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
        from app.core.go_collector import collect_parallel

        logger.info("Collecting all NIC software information")

        commands = {
            "firmware": self._CMD_FIRMWARE,
            "drivers": self._CMD_DRIVERS,
            "rdma": self._CMD_RDMA,
            "ethtool": self._CMD_ETHTOOL,
            "pci": self._CMD_PCI,
        }

        go_results = await asyncio.to_thread(collect_parallel, ssh_manager, commands, 60)

        if go_results is not None:
            logger.info("NIC software collected via Go binary")
            nic_firmware = await self.collect_nic_firmware_version(preloaded_output=go_results.get("firmware", {}))
            nic_drivers = await self.collect_nic_driver_version(preloaded_output=go_results.get("drivers", {}))
            rdma_statistics = await self.collect_rdma_statistics_detailed(preloaded_output=go_results.get("rdma", {}))
            ethtool_statistics = await self.collect_ethtool_statistics_detailed(
                preloaded_output=go_results.get("ethtool", {})
            )
            pci_devices = await self.collect_pci_device_info(preloaded_output=go_results.get("pci", {}))
        else:
            logger.info("Falling back to sequential parallel-ssh for NIC software")
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
