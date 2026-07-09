"""
Parser for `afmctl` command output from compute trays.

Both `afmctl show device` and `afmctl show port` use the same structured
key-value block format — NOT a fixed-width table.

afmctl show device — one block per BDF:
    BDF                              : 0001:01:00.1
    Spec:
      Accelerator id                 : 7
      Config phase                   : ACTIVE
      Capability:
        No. of stations              : 18
    Status:
      Firmware version               : 0.14.15

afmctl show port — one BDF header, then port blocks separated by `---`:
    BDF                           : 0001:01:00.1
      Port id                     : 0
      Spec:
        Station id                : 0
        NW MAC address            : 02:00:00:0e:03:00
      Status:
        Name                      : netport0
        Link status               : LINK_UP
        Speed                     : 400G
        Fault                     : no
      ---
      Port id                     : 1
      ...
"""

from __future__ import annotations

import json as _json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared key-value helpers
# ---------------------------------------------------------------------------


def _key_to_snake(raw: str) -> str:
    """'NW MAC address' → 'nw_mac_address', 'No. of stations' → 'no_of_stations'."""
    s = raw.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


# Matches "  Some key text   : value text" (key may not contain a bare colon)
_KV_RE = re.compile(r"^(\s*)([\w][\w\s./()-]*?)\s*:\s*(.*)$")
# Matches section headers: "  Spec:" / "  Status:" / "  Capability:" (nothing after colon)
_SECTION_RE = re.compile(r"^\s*([\w][\w\s./()-]*?):\s*$")
# Port separator
_SEP_RE = re.compile(r"^\s*---+\s*$")


def _is_section_header(line: str) -> bool:
    return bool(_SECTION_RE.match(line))


def _parse_kv(line: str) -> Optional[tuple[str, str]]:
    """Return (snake_key, value) or None if the line is not a KV pair."""
    m = _KV_RE.match(line)
    if not m:
        return None
    key = _key_to_snake(m.group(2))
    val = m.group(3).strip()
    return key, val


# ---------------------------------------------------------------------------
# afmctl show device
# ---------------------------------------------------------------------------


def parse_afmctl_show_device(output: str) -> List[Dict[str, Any]]:
    """
    Parse `afmctl show device` key-value block output.

    Each BDF block → one flat dict.  Section headers (Spec:, Status:,
    Capability:) are discarded; their children are merged directly into
    the device dict.

    Keys produced (representative):
      bdf, accelerator_id, config_phase, virtualization_mode,
      encap_type, failover_mode, loopback_mode, crypto_mode,
      local_accelerators, vpod_accelerators, whitelisted_accels,
      no_of_stations, no_of_accelerators, no_of_network_ports,
      no_of_n_w_ports_per_station, no_of_paths_per_station,
      library_version, firmware_version, telemetry_version

    Returns [] on error output or empty input.
    """
    if not output or output.strip().startswith("ERROR:"):
        return []

    devices: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # New BDF block
        bdf_m = re.match(r"^BDF\s*:\s*(\S+)", line, re.IGNORECASE)
        if bdf_m:
            if current is not None:
                devices.append(current)
            current = {"bdf": bdf_m.group(1)}
            continue

        if current is None:
            continue

        if _is_section_header(line):
            continue

        kv = _parse_kv(line)
        if kv:
            key, val = kv
            if key and key != "bdf":
                current[key] = val

    if current is not None:
        devices.append(current)

    logger.debug(f"parse_afmctl_show_device: parsed {len(devices)} device blocks")
    return devices


# ---------------------------------------------------------------------------
# afmctl show port
# ---------------------------------------------------------------------------


def parse_afmctl_show_port(output: str) -> List[Dict[str, Any]]:
    """
    Parse `afmctl show port` key-value block output.

    Structure: one BDF header, then N port sub-blocks separated by `---`.
    Each port becomes one flat dict.  Section headers (Spec:, Status:) are
    discarded; their children are merged into the port dict.

    Important fields per port:
      bdf             — from the BDF line (same for all ports of that device)
      port_id         — from "Port id : N"
      station_id      — from Spec / "Station id"
      mac_address     — normalised from Spec / "NW MAC address"  ← topology key
      name            — from Status / "Name"  (e.g. netport0)
      link_status     — from Status / "Link status"  (e.g. LINK_UP or
                         NO_PHY_LINK, PCS_NO_BLOCK_LOCK)
      speed           — from Status / "Speed"
      fault           — from Status / "Fault"
      admin_state     — from Spec / "Admin state"

    Returns [] on error output or empty input.
    """
    if not output or output.strip().startswith("ERROR:"):
        return []

    ports: List[Dict[str, Any]] = []
    current_bdf: str = ""
    current_port: Optional[Dict[str, Any]] = None

    def _flush():
        if current_port:
            _normalise_mac(current_port)
            ports.append(current_port)

    for line in output.splitlines():
        stripped = line.strip()

        # Port separator — flush current port
        if _SEP_RE.match(stripped):
            _flush()
            current_port = None
            continue

        if not stripped:
            continue

        # BDF line (top-level, no indentation)
        bdf_m = re.match(r"^BDF\s*:\s*(\S+)", line, re.IGNORECASE)
        if bdf_m:
            _flush()
            current_port = None
            current_bdf = bdf_m.group(1)
            continue

        # "Port id : N" — starts a new port block
        port_id_m = re.match(r"^\s+Port id\s*:\s*(\d+)", line, re.IGNORECASE)
        if port_id_m:
            _flush()
            current_port = {"bdf": current_bdf, "port_id": port_id_m.group(1)}
            continue

        if current_port is None:
            continue

        if _is_section_header(line):
            continue

        kv = _parse_kv(line)
        if kv:
            key, val = kv
            if key and key not in ("bdf", "port_id"):
                current_port[key] = val

    _flush()

    logger.debug(f"parse_afmctl_show_port: parsed {len(ports)} port blocks")
    return ports


def _normalise_mac(port: Dict[str, Any]) -> None:
    """
    Canonicalise the MAC address in a port dict.

    Renames any key containing "mac" that is NOT "oob_mac_address" to
    "mac_address" (NW MAC address is the topology-relevant one).
    Normalises the value to lowercase colon-delimited hex.
    """
    # Rename: prefer "nw_mac_address" → "mac_address"; ignore OOB
    for key in list(port.keys()):
        if "mac" in key and key != "mac_address" and "oob" not in key:
            port["mac_address"] = port.pop(key)
            break

    raw = port.get("mac_address", "")
    if raw:
        hex_only = re.sub(r"[^0-9a-fA-F]", "", raw).lower()
        if len(hex_only) == 12:
            port["mac_address"] = ":".join(hex_only[i : i + 2] for i in range(0, 12, 2))


# ---------------------------------------------------------------------------
# afmctl show port statistics <type> --json
# ---------------------------------------------------------------------------


def _flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Recursively flatten a nested dict/list into a flat dict."""
    result: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}_{k}" if prefix else k
            if isinstance(v, (dict, list)):
                result.update(_flatten(v, key))
            else:
                result[key] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}_{i}" if prefix else str(i)
            if isinstance(v, (dict, list)):
                result.update(_flatten(v, key))
            else:
                result[key] = v
    else:
        result[prefix] = obj
    return result


def parse_afmctl_stats_gzip_b64(output: str, category: str) -> List[Dict[str, Any]]:
    """
    Decode and parse gzip+base64 encoded afmctl statistics output.

    Command used on remote:
      sudo afmctl show port statistics <type> --json 2>/dev/null | gzip -c | base64 -w0

    The raw JSON is 668 KB which overflows SSH session buffers. Compressed
    via gzip it becomes ~27 KB, well within SSH transfer limits.

    The JSON has a nested structure:
      {"devices": [{"bdf": "0001:01:00.1", "ports": [{"station_id":"0","port_id":"0",...}]}]}

    Returns a flat list of dicts, one per port, with bdf/station_id/port_id
    plus all counter fields.
    """
    if not output or output.strip().startswith(("ERROR", "ABORT")):
        return []
    try:
        import base64 as _b64
        import gzip as _gz

        decoded = _b64.b64decode(output.strip())
        json_str = _gz.decompress(decoded).decode("utf-8")
    except Exception as e:
        logger.warning(f"Stats gzip/base64 decode error ({category}): {e} — raw[:60]: {repr(output[:60])}")
        return []

    try:
        parsed = _json.loads(json_str)
    except _json.JSONDecodeError as e:
        logger.warning(f"Stats JSON parse error ({category}): {e}")
        return []

    rows: List[Dict[str, Any]] = []

    if isinstance(parsed, dict) and "devices" in parsed:
        # Nested structure: {"devices": [{"bdf": "...", "ports": [{counters}]}]}
        for device in parsed["devices"]:
            bdf = device.get("bdf", "")
            for port in device.get("ports", []):
                if not isinstance(port, dict):
                    continue
                if port.get("error"):
                    continue  # "Function not implemented" etc.
                row = _flatten(port)
                row["bdf"] = bdf
                row["_stat_type"] = category
                rows.append(row)
    elif isinstance(parsed, list):
        # Flat array fallback
        for item in parsed:
            if isinstance(item, dict) and not item.get("error"):
                row = _flatten(item)
                row["_stat_type"] = category
                rows.append(row)

    logger.info(
        f"parse_afmctl_stats_gzip_b64 ({category}): {len(rows)} port rows from {len(parsed.get('devices', [])) if isinstance(parsed, dict) else '?'} devices"
    )
    return rows


def parse_afmctl_stats_json(output: str, category: str) -> List[Dict[str, Any]]:
    """
    Parse `afmctl show port statistics <category> --json` output.

    Actual format (verified against live hardware):
      A flat JSON array of port objects:
      [
        {"bdf": "0001:01:00.1", "station": 0, "port": 0, "rx_total_bytes": 1234, ...},
        ...
      ]

    IFCP returns: [{"bdf":..., "station":..., "port":..., "error": "Function not implemented"}]
    These are filtered out (the command is not supported on current firmware).

    Returns a list of flat dicts, one per port. The identifier fields bdf/station/port
    are always present. All counter fields follow.
    """
    if not output or output.startswith(("ERROR", "ABORT")):
        return []

    # Strip any leading WARNING/INFO lines before the JSON array
    clean = output.strip()
    for start_ch in ("[", "{"):
        idx = clean.find(start_ch)
        if idx != -1:
            clean = clean[idx:]
            break
    else:
        return []

    try:
        parsed = _json.loads(clean)
    except _json.JSONDecodeError as e:
        logger.warning(f"afmctl stats JSON parse error ({category}): {e} — first 200 chars: {repr(clean[:200])}")
        return []

    rows: List[Dict[str, Any]] = []

    items = parsed if isinstance(parsed, list) else [parsed]
    for item in items:
        if not isinstance(item, dict):
            continue
        # Skip "Function not implemented" error rows (e.g. IFCP)
        if item.get("error"):
            continue
        # The format is already flat — no nesting needed in practice,
        # but _flatten handles any nested sub-objects defensively.
        row = _flatten(item)
        row["_stat_type"] = category
        rows.append(row)

    return rows
