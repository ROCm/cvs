"""
NIC devlink information collector.
Collects NIC firmware and driver information using devlink.
Supports AMD AINIC, NVIDIA CX7, and Broadcom Thor2.
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class NICDevlinkCollector:
    """Collects NIC information via devlink."""

    async def collect_devlink_info(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect NIC device information using 'devlink dev info --json'.

        Returns normalized data across vendors:
        - Broadcom Thor2: bnxt_en driver
        - NVIDIA CX7: mlx5_core driver
        - AMD AINIC: pds_core and ionic drivers

        Returns:
            {
                "node1": {
                    "pci/0000:76:00.0": {
                        "pci_address": "0000:76:00.0",
                        "driver": "bnxt_en",
                        "vendor": "Broadcom Thor2",
                        "serial_number": "...",
                        "board_serial": "...",
                        "board_id": "...",
                        "asic_id": "...",
                        "asic_rev": "...",
                        "fw_version": "...",
                        "fw_psid": "...",
                        "fw_mgmt": "...",
                    },
                    ...
                },
                ...
            }
        """
        logger.info("Collecting NIC devlink information")
        output = await ssh_manager.exec_async(
            "bash -c 'devlink dev info --json 2>/dev/null || echo \"{}\"'", timeout=60
        )

        devlink_info = {}

        for node, out_str in output.items():
            if out_str.startswith("ERROR") or out_str.startswith("ABORT"):
                devlink_info[node] = {"error": out_str}
                continue

            try:
                if not out_str.strip() or out_str.strip() == '{}':
                    devlink_info[node] = {}
                    continue

                data = json.loads(out_str)
                devlink_info[node] = {}

                # Parse JSON output: {"info": {"pci/0000:76:00.0": {...}, ...}}
                info_dict = data.get("info", {})

                for pci_dev, dev_data in info_dict.items():
                    if not isinstance(dev_data, dict):
                        continue

                    # Extract PCI address from key (e.g., "pci/0000:76:00.0" -> "0000:76:00.0")
                    pci_address = pci_dev.replace("pci/", "")

                    driver = dev_data.get("driver", "-")
                    serial_number = dev_data.get("serial_number", "-")
                    board_serial = dev_data.get("board.serial_number", "-")

                    versions = dev_data.get("versions", {})
                    fixed = versions.get("fixed", {})
                    running = versions.get("running", {})
                    versions.get("stored", {})

                    # Determine vendor based on driver
                    if driver == "bnxt_en":
                        vendor = "Broadcom Thor2"
                    elif driver == "mlx5_core":
                        vendor = "NVIDIA CX7"
                    elif driver in ["pds_core", "ionic"]:
                        vendor = "AMD AINIC"
                    elif driver == "i40e":
                        vendor = "Intel"
                    else:
                        vendor = "Unknown"

                    # Normalize firmware versions across vendors
                    # Broadcom Thor2 and NVIDIA CX7 use "fw" field
                    # AMD AINIC uses specific fields like "fw.a35_fip_a"
                    fw_version = running.get("fw") or running.get("fw.version") or running.get("fw.a35_fip_a") or "-"

                    fw_psid = fixed.get("fw.psid") or "-"
                    fw_mgmt = running.get("fw.mgmt") or "-"
                    fw_mgmt_api = running.get("fw.mgmt.api") or "-"

                    board_id = fixed.get("board.id") or "-"
                    asic_id = fixed.get("asic.id") or "-"
                    asic_rev = fixed.get("asic.rev") or "-"

                    # AMD AINIC specific fields
                    fw_cpld = running.get("fw.cpld") or "-"
                    fw_heartbeat = running.get("fw.heartbeat") or "-"

                    devlink_info[node][pci_dev] = {
                        "pci_address": pci_address,
                        "driver": driver,
                        "vendor": vendor,
                        "serial_number": serial_number,
                        "board_serial": board_serial,
                        "board_id": board_id,
                        "asic_id": asic_id,
                        "asic_rev": asic_rev,
                        "fw_version": fw_version,
                        "fw_psid": fw_psid,
                        "fw_mgmt": fw_mgmt,
                        "fw_mgmt_api": fw_mgmt_api,
                        "fw_cpld": fw_cpld,
                        "fw_heartbeat": fw_heartbeat,
                    }

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse devlink JSON for {node}: {e}")
                devlink_info[node] = {}

        logger.info(f"Devlink info collection complete: {len(devlink_info)} nodes")
        return devlink_info

    async def collect_all_devlink_info(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect all NIC devlink information.
        """
        logger.info("Collecting all NIC devlink information")

        devlink_info = await self.collect_devlink_info(ssh_manager)

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "devlink": devlink_info,
        }
