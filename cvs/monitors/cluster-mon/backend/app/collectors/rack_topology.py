"""
MAC-address-based topology builder.

Correlates compute tray IFoE port MACs (from afmctl show port) against
switch tray MAC tables (from SONiC show mac) to produce a per-port
compute→switch mapping.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _normalise_mac(mac: str) -> str:
    """Lowercase, strip non-hex chars, reformat as xx:xx:xx:xx:xx:xx."""
    raw = re.sub(r"[^0-9a-fA-F]", "", mac).lower()
    if len(raw) == 12:
        return ":".join(raw[i : i + 2] for i in range(0, 12, 2))
    return mac.lower()  # return as-is if not a standard 48-bit MAC


def _build_switch_mac_index(switch_macs: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, Tuple[str, str, str]]:
    """
    Build a flat index: normalised_mac → (switch_tray_ip, asic_name, switch_port).

    switch_macs structure:
      {switch_tray_ip: {"asic0": [mac_rows], "asic1": [mac_rows]}}

    Each mac_row is a dict from parse_show_mac(), expected to have keys like
    "mac_address" (or "macaddress") and "port".
    """
    index: Dict[str, Tuple[str, str, str]] = {}

    for switch_ip, asics in switch_macs.items():
        for asic_name, mac_rows in asics.items():
            for row in mac_rows:
                # Find MAC field (key contains "mac")
                mac_val: Optional[str] = None
                for k, v in row.items():
                    if "mac" in k and v:
                        mac_val = v
                        break
                if not mac_val:
                    continue

                # Find port field
                port_val = row.get("port") or row.get("interface") or ""

                norm_mac = _normalise_mac(mac_val)
                if norm_mac and norm_mac != "00:00:00:00:00:00":
                    index[norm_mac] = (switch_ip, asic_name, str(port_val))

    logger.debug(f"Switch MAC index built: {len(index)} entries")
    return index


def build_topology(
    compute_ports: Dict[str, List[Dict[str, Any]]],
    switch_macs: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    """
    Correlate compute IFoE ports against switch MAC tables.

    Parameters
    ----------
    compute_ports:
        {compute_tray_ip: [port_rows_from_afmctl_show_port]}
    switch_macs:
        {switch_tray_ip: {"asic0": [mac_rows], "asic1": [mac_rows]}}

    Returns
    -------
    List of topology dicts, one per IFoE port across all compute trays:

    {
        "compute_tray":   str,
        "gpu_index":      str,
        "station_index":  str,
        "port_index":     str,
        "ifoe_interface": str,
        "compute_mac":    str (normalised),
        "link_status":    str,
        "speed":          str,
        "switch_tray":    str or "",
        "asic":           str or "",
        "switch_port":    str or "",
        "mapped":         bool,
    }
    """
    mac_index = _build_switch_mac_index(switch_macs)
    topology: List[Dict[str, Any]] = []

    for compute_ip, port_rows in compute_ports.items():
        for row in port_rows:
            # Extract fields from afmctl port row (key names vary by firmware)
            gpu_idx = row.get("gpu_index") or row.get("gpu") or row.get("gpu_id") or ""
            station_idx = row.get("station_index") or row.get("station") or row.get("station_id") or ""
            port_idx = row.get("port_index") or row.get("port") or row.get("port_id") or ""
            iface = row.get("ifoe_interface") or row.get("interface") or row.get("if_name") or row.get("name") or ""
            raw_mac = row.get("mac_address") or row.get("mac") or ""
            link_status = row.get("link_status") or row.get("status") or row.get("state") or ""
            speed = row.get("speed") or row.get("link_speed") or ""

            norm_mac = _normalise_mac(raw_mac) if raw_mac else ""
            match = mac_index.get(norm_mac) if norm_mac else None

            topo_row: Dict[str, Any] = {
                "compute_tray": compute_ip,
                "gpu_index": str(gpu_idx),
                "station_index": str(station_idx),
                "port_index": str(port_idx),
                "ifoe_interface": str(iface),
                "compute_mac": norm_mac,
                "link_status": str(link_status),
                "speed": str(speed),
                "switch_tray": match[0] if match else "",
                "asic": match[1] if match else "",
                "switch_port": match[2] if match else "",
                "mapped": match is not None,
            }
            topology.append(topo_row)

    logger.info(f"build_topology: {len(topology)} rows, {sum(1 for r in topology if r['mapped'])} mapped")
    return topology
