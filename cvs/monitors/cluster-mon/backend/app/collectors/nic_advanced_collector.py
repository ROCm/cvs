"""
Advanced NIC information collector for vendor-specific data.
Supports AMD AINIC, NVIDIA/Mellanox CX7, Broadcom Thor2.
"""

import re
import json
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class NICAdvancedCollector:
    """Collects vendor-specific NIC information and congestion metrics."""

    async def collect_nic_pcie_info(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect PCIe information for all NICs using lspci.
        Optimized: single lspci call per node instead of per NIC.
        """
        logger.info("Collecting NIC PCIe information (optimized)")

        # Get ALL NIC PCIe info in one command per node
        # cmd = "sudo lspci -vvv 2>/dev/null | grep -A 30 -i 'ethernet\\|network' | grep -E '^[0-9a-f]{2}:|Ethernet|Network|LnkCap:|LnkSta:'"
        cmd = "sudo lspci -vvv 2>/dev/null | egrep -A 30 -i 'ethernet\\|network' | egrep '^[0-9a-f]{2}:|Ethernet|Network|LnkCap:|LnkSta:'"
        result = await ssh_manager.exec_async(cmd, timeout=120)

        logger.info(f"NIC PCIe lspci returned results from {len(result)} nodes")

        pcie_info = {}

        for node, output in result.items():
            if output.startswith("ERROR"):
                logger.warning(f"NIC PCIe error from {node}: {output[:100]}")
                continue
            if not output.strip():
                logger.warning(f"NIC PCIe empty output from {node}")
                continue

            logger.debug(f"NIC PCIe processing {node}: {len(output)} bytes")

            pcie_info[node] = {}

            current_bdf = None
            current_device = ''
            link_cap_speed = ''
            link_cap_width = ''
            link_sta_speed = ''
            link_sta_width = ''

            for line in output.split('\n'):
                # New device line (BDF address)
                if re.match(r'^[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]', line):
                    # Save previous device if exists
                    if current_bdf:
                        pcie_gen = 'N/A'
                        if link_cap_speed:
                            speed_val = float(link_cap_speed.replace('GT/s', ''))
                            if speed_val >= 32:
                                pcie_gen = 'Gen5'
                            elif speed_val >= 16:
                                pcie_gen = 'Gen4'
                            elif speed_val >= 8:
                                pcie_gen = 'Gen3'
                            elif speed_val >= 5:
                                pcie_gen = 'Gen2'
                            else:
                                pcie_gen = 'Gen1'

                        pcie_info[node][current_bdf] = {
                            'device': current_device,
                            'pcie_gen': pcie_gen,
                            'link_speed_cap': link_cap_speed,
                            'link_width_cap': link_cap_width,
                            'link_speed_current': link_sta_speed,
                            'link_width_current': link_sta_width,
                        }

                    # Start new device
                    current_bdf = line.split()[0]
                    current_device = line.split(':', 2)[-1].strip() if ':' in line else ''
                    link_cap_speed = ''
                    link_cap_width = ''
                    link_sta_speed = ''
                    link_sta_width = ''

                # Parse LnkCap
                elif 'LnkCap:' in line:
                    speed_match = re.search(r'Speed\s+([0-9.]+GT/s)', line)
                    width_match = re.search(r'Width\s+(x\d+)', line)
                    if speed_match:
                        link_cap_speed = speed_match.group(1)
                    if width_match:
                        link_cap_width = width_match.group(1)

                # Parse LnkSta
                elif 'LnkSta:' in line:
                    speed_match = re.search(r'Speed\s+([0-9.]+GT/s)', line)
                    width_match = re.search(r'Width\s+(x\d+)', line)
                    if speed_match:
                        link_sta_speed = speed_match.group(1)
                    if width_match:
                        link_sta_width = width_match.group(1)

            # Save last device
            if current_bdf:
                pcie_gen = 'N/A'
                if link_cap_speed:
                    speed_val = float(link_cap_speed.replace('GT/s', ''))
                    if speed_val >= 32:
                        pcie_gen = 'Gen5'
                    elif speed_val >= 16:
                        pcie_gen = 'Gen4'
                    elif speed_val >= 8:
                        pcie_gen = 'Gen3'
                    elif speed_val >= 5:
                        pcie_gen = 'Gen2'
                    else:
                        pcie_gen = 'Gen1'

                pcie_info[node][current_bdf] = {
                    'device': current_device,
                    'pcie_gen': pcie_gen,
                    'link_speed_cap': link_cap_speed,
                    'link_width_cap': link_cap_width,
                    'link_speed_current': link_sta_speed,
                    'link_width_current': link_sta_width,
                }

        logger.info(
            f"NIC PCIe collection complete: {len(pcie_info)} nodes, {sum(len(v) for v in pcie_info.values())} devices total"
        )
        return pcie_info

    async def collect_congestion_info(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect congestion control information (PFC, ECN, CNP).

        OPTIMIZATION: Use 'rdma statistic show' instead of per-interface ethtool commands.
        This single command per node provides all statistics including:
        - PFC (Priority Flow Control) stats
        - ECN (Explicit Congestion Notification) packets: rx_rdma_ecn_pkts
        - CNP (Congestion Notification Packets): tx_rdma_cnp_pkts, rx_rdma_cnp_pkts
        - Timeout counters: tx_rdma_ack_timeout, tx_rdma_ccl_cts_ack_timeout
        - Error counters: req_rx_pkt_seq_err, req_rx_rnr_retry_err, etc.
        - Drop counters: rx_rdma_mtu_discard_pkts

        Output format example:
        link rdma0/1 tx_rdma_cnp_pkts 0 rx_rdma_cnp_pkts 0 rx_rdma_ecn_pkts 0 req_rx_pkt_seq_err 14370842 tx_rdma_ack_timeout 2371603 ...

        Looks for fields matching patterns: pfc, pause, ecn, cnp, drop, err, timeout
        """
        logger.info("Collecting congestion control information from rdma statistic (optimized)")

        # Use rdma statistic show with JSON for reliable parsing
        output = await ssh_manager.exec_async(
            "bash -c 'rdma statistic show --json 2>/dev/null || echo \"[]\"'", timeout=60
        )

        congestion_info = {}

        for node, out_str in output.items():
            if out_str.startswith("ERROR") or out_str.startswith("ABORT"):
                continue

            try:
                if not out_str.strip() or out_str.strip() == '[]':
                    congestion_info[node] = {}
                    continue

                data = json.loads(out_str)
                congestion_info[node] = {}

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
                                stats = {}

                                # Extract congestion-related stats (skip metadata)
                                metadata_keys = {"ifname", "port", "ifindex"}

                                for key, value in entry.items():
                                    if key not in metadata_keys and isinstance(value, (int, float)):
                                        # Only keep congestion-related stats
                                        if re.search(r'(pfc|pause|ecn|cnp|drop|err|timeout)', key, re.IGNORECASE):
                                            stats[key] = value

                                if stats:
                                    congestion_info[node][dev_key] = stats

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse RDMA congestion stats JSON for {node}: {e}")
                congestion_info[node] = {}

        return congestion_info

    async def collect_mellanox_info(self, ssh_manager) -> Dict[str, Any]:
        """
        OPTIMIZATION: Disabled per-device devlink collection.

        We rely on 'rdma statistic show' for all NIC statistics (PFC, ECN, CNP, errors).
        Devlink per-device queries are unnecessary overhead.

        Returns minimal info without running per-device commands.
        """
        logger.info("Mellanox NIC info collection skipped (using rdma statistic instead)")
        return {}

    async def collect_broadcom_info(self, ssh_manager) -> Dict[str, Any]:
        """
        OPTIMIZATION: Disabled per-device niccli collection.

        We rely on 'rdma statistic show' for all NIC statistics (PFC, ECN, CNP, errors).
        Per-device niccli queries are unnecessary overhead.

        Returns minimal info without running per-device commands.
        """
        logger.info("Broadcom NIC info collection skipped (using rdma statistic instead)")
        return {}

    async def collect_all_nic_advanced_info(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect all advanced NIC information.
        """

        logger.info("Collecting all advanced NIC information")

        # IMPORTANT: Run commands SEQUENTIALLY to avoid parallel-ssh thread safety issues
        # asyncio.gather() was causing "munmap_chunk(): invalid pointer" crashes
        nic_pcie = await self.collect_nic_pcie_info(ssh_manager)
        congestion = await self.collect_congestion_info(ssh_manager)
        mellanox = await self.collect_mellanox_info(ssh_manager)
        broadcom = await self.collect_broadcom_info(ssh_manager)

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "nic_pcie": nic_pcie if not isinstance(nic_pcie, Exception) else {},
            "congestion": congestion if not isinstance(congestion, Exception) else {},
            "mellanox": mellanox if not isinstance(mellanox, Exception) else {},
            "broadcom": broadcom if not isinstance(broadcom, Exception) else {},
        }
