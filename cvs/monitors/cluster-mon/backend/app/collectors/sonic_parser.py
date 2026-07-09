"""
Parser for SONiC switch command output from switch trays.

Actual output formats observed:

show vlan brief --verbose
    +-----------+--------------+-------------+-----------------+-------------+-----------------------+
    |   VLAN ID | IP Address   | Ports       | Port Tagging    | Proxy ARP   | DHCP Helper Address   |
    +===========+==============+=============+=================+=============+=======================+
    |       100 |              |             |                 | disabled    |                       |
    +-----------+--------------+-------------+-----------------+-------------+-----------------------+
    |       101 |              | Ethernet275 | priority_tagged | disabled    |                       |
    |           |              | Ethernet294 | priority_tagged |             |                       |
    ...

    Multi-row VLANs: VLAN ID only in first row; continuation rows have
    blank VLAN ID cell.  Ports are aggregated into a list per VLAN.

show mac
      No.    Vlan  MacAddress         Port         Type
    -----  ------  -----------------  -----------  ------
        1     101  02:00:00:10:01:01  Ethernet294  Static
    ...
    Total number of entries 48
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

import json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic JSON helper — most new SONiC commands use --json output
# ---------------------------------------------------------------------------


def _parse_json_safe(output: str) -> Any:
    """Parse JSON output; return None on failure."""
    if not output or output.strip().startswith(("ERROR", "ABORT", "command not found")):
        return None
    try:
        # Some commands prepend WARNING lines before JSON
        for start in (output.find("["), output.find("{")):
            if start != -1:
                candidate = output[start:]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
        return json.loads(output.strip())
    except Exception:
        return None


def parse_json_command(output: str) -> Any:
    """Return parsed JSON object/list from SONiC --json output, or None."""
    return _parse_json_safe(output)


# ---------------------------------------------------------------------------
# show platform summary --json
# ---------------------------------------------------------------------------


def parse_platform_summary(output: str) -> Dict[str, Any]:
    data = _parse_json_safe(output)
    if isinstance(data, dict):
        return data
    return {}


# ---------------------------------------------------------------------------
# show platform psustatus --json
# ---------------------------------------------------------------------------


def parse_platform_psu(output: str) -> List[Dict[str, Any]]:
    data = _parse_json_safe(output)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


# ---------------------------------------------------------------------------
# show platform fan --json
# ---------------------------------------------------------------------------


def parse_platform_fan(output: str) -> List[Dict[str, Any]]:
    # Try JSON first, fall back to text table
    data = _parse_json_safe(output)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return parse_text_table(output)


# ---------------------------------------------------------------------------
# show platform temperature  (no --json on this SONiC version)
# ---------------------------------------------------------------------------


def parse_platform_temperature(output: str) -> List[Dict[str, Any]]:
    data = _parse_json_safe(output)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return parse_text_table(output)


# ---------------------------------------------------------------------------
# show system status  (may not support --json on all versions; parse text)
# ---------------------------------------------------------------------------


def parse_show_version(output: str) -> Dict[str, str]:
    """
    Parse `show version` output into a flat dict of key: value pairs.

    Sample output:
      SONiC Software Version: SONiC.202505_1.0.0-104
      SONiC OS Version: 12
      Distribution: Debian 12.14
      Kernel: 6.1.0-29-2-amd64
      Build commit: b856e49d2
      Build date: Thu Jun 11 13:04:47 UTC 2026
      Built by: vm@user-Purico
      Platform: x86_64-amd_anacapa-r0
    """
    if not output or output.strip().startswith(("ERROR", "ABORT")):
        return {}
    result: Dict[str, str] = {}
    for line in output.splitlines():
        if ":" in line and not line.strip().startswith("#"):
            k, _, v = line.partition(":")
            key = k.strip()
            val = v.strip()
            if key and val:
                result[key] = val
    return result


def parse_system_health_summary(output: str) -> Dict[str, str]:
    """
    Parse `sudo show system-health summary` output.

    Sample output:
      System status summary

        System status LED  blue
        Services:
          Status: OK
        Hardware:
          Status: OK
    """
    if not output or output.strip().startswith(("ERROR", "ABORT")):
        return {}
    result: Dict[str, str] = {}
    current_section = ""
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Section headers (end with ":")
        if stripped.endswith(":") and not stripped.startswith("Status"):
            current_section = stripped.rstrip(":").strip()
            continue
        # "System status LED  blue"
        if "System status LED" in stripped:
            result["System LED"] = stripped.replace("System status LED", "").strip()
            continue
        # "Status: OK" inside a section
        if stripped.startswith("Status:"):
            val = stripped.split(":", 1)[1].strip()
            if current_section:
                result[f"{current_section} Status"] = val
            continue
        # Generic "key: value" lines
        if ":" in stripped:
            k, _, v = stripped.partition(":")
            key = k.strip()
            val = v.strip()
            if key and val:
                result[key] = val
    return result


def parse_system_status(output: str) -> Dict[str, Any]:
    """Legacy: parse 'show system status' — delegates to version + health parsers."""
    return parse_system_health_summary(output)


# ---------------------------------------------------------------------------
# sudo docker ps --format '{{json .}}'  (one JSON object per line)
# ---------------------------------------------------------------------------


def parse_docker_ps(output: str) -> List[Dict[str, Any]]:
    """Parse docker ps --format json output (one JSON object per line)."""
    if not output or output.strip().startswith(("ERROR", "ABORT")):
        return []
    rows = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except json.JSONDecodeError:
            continue
    return rows


# ---------------------------------------------------------------------------
# sudo docker stats --no-stream --format '{{json .}}'
# ---------------------------------------------------------------------------


def parse_docker_stats(output: str) -> List[Dict[str, Any]]:
    """Parse docker stats --no-stream --format json output."""
    return parse_docker_ps(output)  # same format


# ---------------------------------------------------------------------------
# free -m  (text parsing)
# ---------------------------------------------------------------------------


def parse_free(output: str) -> Dict[str, Any]:
    """Parse 'free -m' output into a structured dict."""
    if not output or output.strip().startswith(("ERROR", "ABORT")):
        return {}
    result: Dict[str, Any] = {}
    for line in output.splitlines():
        parts = line.split()
        if not parts:
            continue
        label = parts[0].lower().rstrip(":")
        if label in ("mem", "swap") and len(parts) >= 3:
            try:
                result[label] = {
                    "total_mb": int(parts[1]),
                    "used_mb": int(parts[2]),
                    "free_mb": int(parts[3]) if len(parts) > 3 else 0,
                    "available_mb": int(parts[6]) if len(parts) > 6 else 0,
                }
            except (ValueError, IndexError):
                pass
    return result


# ---------------------------------------------------------------------------
# show interfaces status --json
# ---------------------------------------------------------------------------


def parse_interfaces_status(output: str) -> List[Dict[str, Any]]:
    data = _parse_json_safe(output)
    if isinstance(data, dict):
        return [{"interface": k, **v} for k, v in data.items() if isinstance(v, dict)]
    if isinstance(data, list):
        return data
    # Text fallback
    return parse_text_table(output)


# ---------------------------------------------------------------------------
# show pfc counters / show priority-group counters / show queue counters
# / show queue watermark  --json
# All return structured counter dicts; we normalise to a flat list.
# ---------------------------------------------------------------------------


def parse_interface_counters_json(output: str) -> List[Dict[str, Any]]:
    """
    Parse `show interface counters --json` output.
    Format: {"Ethernet0": {"rx_ok": "0", "rx_err": "0", "tx_ok": "0", ...}, ...}
    Returns one row per interface with all counters as columns.
    """
    if not output or output.strip().startswith(("Error", "Usage", "ERROR", "ABORT")):
        return []
    try:
        stripped = output.strip()
        for ch in ('{', '['):
            idx = stripped.find(ch)
            if idx != -1:
                stripped = stripped[idx:]
                break
        data = json.loads(stripped)
    except Exception as e:
        logger.warning(f"parse_interface_counters_json: parse failed: {e!r}")
        return []
    if not isinstance(data, dict):
        return []
    rows = []
    for intf, counters in data.items():
        if isinstance(counters, dict):
            row = {"interface": intf}
            row.update(counters)
            rows.append(row)
    return rows


def parse_counters_json(output: str, cmd_name: str = "counters") -> List[Dict[str, Any]]:
    """
    Generic parser for SONiC counter JSON outputs.
    Handles nested format: {interface: {queue_id: {counter: value, ...}}}
    """
    if not output or output.strip().startswith(("Error", "Usage", "ERROR", "ABORT")):
        return []
    try:
        # Direct json.loads — more reliable than _parse_json_safe for counter output
        stripped = output.strip()
        # Find JSON start
        for ch in ('{', '['):
            idx = stripped.find(ch)
            if idx != -1:
                stripped = stripped[idx:]
                break
        data = json.loads(stripped)
    except Exception as e:
        logger.warning(f"parse_counters_json({cmd_name}): JSON parse failed: {e!r} | output[:200]={output[:200]!r}")
        return []

    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []

    # Flatten nested {intf: {queue_id: {counter: val}}} format
    rows = []
    for intf, val in data.items():
        if isinstance(val, dict):
            for sub_key, counters in val.items():
                if isinstance(counters, dict):
                    row = {"interface": intf, "queue_or_pg": sub_key}
                    row.update(counters)
                    rows.append(row)
                else:
                    rows.append({"interface": intf, "queue_or_pg": sub_key, "value": counters})
        else:
            rows.append({"interface": intf, "value": val})
    return rows


# ---------------------------------------------------------------------------
# Generic text table parser — for SONiC commands that don't support --json
# ---------------------------------------------------------------------------


def parse_text_table(output: str) -> List[Dict[str, str]]:
    """
    Parse SONiC fixed-width text table output.

    Strategy: find the dash separator line (e.g. "------  ------  ------").
    The runs of dashes define exact column start/end positions.
    The header line immediately before the dashes supplies column names.
    All subsequent lines are sliced by those boundaries.

    Also handles pipe-delimited tables (show vlan brief style).
    """
    if not output or output.strip().startswith(("Error", "Usage", "ERROR", "ABORT")):
        return []

    raw_lines = output.splitlines()

    # ── Pipe-delimited table (show vlan brief) ─────────────────────────────
    if any("|" in line for line in raw_lines[:8]):
        border_re = re.compile(r"^[\+\-\=\| ]+$")
        hdrs: List[str] = []
        rows: List[Dict[str, str]] = []
        for line in raw_lines:
            content = line.replace("|", "").replace("+", "").replace("-", "").replace("=", "").replace(" ", "")
            if border_re.match(line) and not content:
                continue
            if "|" in line:
                cells = [c.strip() for c in line.split("|")[1:-1]]
                if not hdrs:
                    hdrs = [re.sub(r"\s+", "_", c).lower().strip("_") or f"col{i}" for i, c in enumerate(cells)]
                elif cells:
                    rows.append(dict(zip(hdrs, cells)))
        return rows

    # ── Fixed-width table: find dash separator line ────────────────────────
    # Dash line: only dashes and spaces, length > 4, has at least one dash run
    dash_re = re.compile(r"^[\-\s]+$")
    title_re = re.compile(r"^[A-Za-z].*:$")  # e.g. "Egress shared pool...:"

    dash_idx = -1
    for i, line in enumerate(raw_lines):
        stripped = line.rstrip()
        if stripped and dash_re.match(stripped) and "-" in stripped and len(stripped) > 4:
            dash_idx = i
            break

    if dash_idx <= 0:
        return []  # no recognisable table structure

    dash_line = raw_lines[dash_idx]

    # Derive column boundaries from runs of '-' in dash_line
    col_ranges: List[tuple] = []  # (start, end) character positions
    in_dash = False
    col_start = 0
    for i, ch in enumerate(dash_line):
        if ch == "-" and not in_dash:
            col_start = i
            in_dash = True
        elif ch != "-" and in_dash:
            col_ranges.append((col_start, i))
            in_dash = False
    if in_dash:
        col_ranges.append((col_start, len(dash_line) + 1))

    if not col_ranges:
        return []

    # Header is the line immediately before the dashes
    # (skip any title/section lines that look like "Egress shared ...: ")
    header_line = raw_lines[dash_idx - 1]

    # Extract column names using the same boundaries
    def _cell(line: str, start: int, end: int) -> str:
        return line[start : min(end, len(line))].strip() if start < len(line) else ""

    headers = []
    for start, end in col_ranges:
        h = _cell(header_line, start, end)
        name = re.sub(r"[^a-z0-9]+", "_", h.lower()).strip("_") or f"col{len(headers)}"
        headers.append(name)

    # Parse data rows
    rows: List[Dict[str, str]] = []
    for line in raw_lines[dash_idx + 1 :]:
        stripped = line.rstrip()
        if not stripped:
            continue
        if dash_re.match(stripped):
            continue
        if title_re.match(stripped):
            continue
        if re.match(r"^\s*(total|summary|count)\b", stripped, re.I):
            continue
        row = {}
        for k, (start, end) in enumerate(col_ranges):
            # Last column gets everything to end of line
            eff_end = end if k < len(col_ranges) - 1 else len(line) + 1
            row[headers[k]] = _cell(line, start, eff_end)
        if any(row.values()):
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# show arp --json
# ---------------------------------------------------------------------------


def parse_arp(output: str) -> List[Dict[str, Any]]:
    data = _parse_json_safe(output)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [{"ip": k, **v} if isinstance(v, dict) else {"ip": k, "mac": v} for k, v in data.items()]
    # Text fallback for SONiC versions without --json
    return parse_text_table(output)


# ---------------------------------------------------------------------------
# show fdb --json  (FDB = Forwarding Database, similar to MAC table)
# ---------------------------------------------------------------------------


def parse_fdb(output: str) -> List[Dict[str, Any]]:
    data = _parse_json_safe(output)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [{"key": k, **v} if isinstance(v, dict) else {"key": k, "value": v} for k, v in data.items()]
    return []


# ---------------------------------------------------------------------------
# show vlan brief  (pipe-delimited, multi-row per VLAN)
# ---------------------------------------------------------------------------


def _pipe_split(line: str) -> List[str]:
    """Split a pipe-delimited table row into cells (strip whitespace)."""
    parts = line.split("|")
    # Leading/trailing empty strings from surrounding pipes
    return [p.strip() for p in parts[1:-1]] if len(parts) > 2 else []


def parse_show_vlan_brief(output: str) -> List[Dict[str, Any]]:
    """
    Parse `show vlan brief [--verbose]` pipe-delimited table output.

    Returns one dict per VLAN with keys:
      vlan_id       — integer VLAN ID
      ip_address    — IP address string (usually empty)
      ports         — comma-separated list of member ports
      port_tagging  — tagging mode of first port (usually "priority_tagged")
      proxy_arp     — "disabled" / "enabled"
      dhcp_helper   — DHCP helper address(es)

    Multi-row VLANs (where the VLAN ID cell is blank in continuation rows)
    are collapsed into a single entry; the port list is aggregated.
    """
    if not output or output.strip().startswith("ERROR:"):
        return []

    vlans: List[Dict[str, Any]] = []
    col_names: List[str] = []
    current: Dict[str, Any] = {}
    current_ports: List[str] = []

    # A row is a data row if it starts with | and the first cell is non-empty
    # OR is a continuation (first cell empty = more ports for the same VLAN)
    border_re = re.compile(r"^[+\-=|]+$")

    def _flush():
        if current:
            entry = dict(current)
            entry["ports"] = ", ".join(current_ports)
            vlans.append(entry)

    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Border/separator lines — skip
        if border_re.match(stripped.replace(" ", "")):
            continue

        # Must be a pipe-delimited data or header line
        if "|" not in line:
            continue

        cells = _pipe_split(line)
        if not cells:
            continue

        # Header row detection: first non-border pipe row
        if not col_names:
            col_names = [re.sub(r"\s+", "_", c.lower()).strip("_") for c in cells]
            # Normalise known column names
            col_names = [
                "vlan_id"
                if c in ("vlan_id", "vlan id")
                else "ip_address"
                if "ip" in c
                else "ports"
                if c == "ports"
                else "port_tagging"
                if "tagging" in c
                else "proxy_arp"
                if "proxy" in c
                else "dhcp_helper"
                if "dhcp" in c
                else c
                for c in col_names
            ]
            continue

        # Pad cells to number of columns
        while len(cells) < len(col_names):
            cells.append("")

        row = dict(zip(col_names, cells))
        vlan_cell = row.get("vlan_id", "").strip()

        if vlan_cell:
            # New VLAN — flush previous
            _flush()
            current = {k: v for k, v in row.items() if k != "ports"}
            current_ports = [row.get("ports", "").strip()] if row.get("ports", "").strip() else []
        else:
            # Continuation row — accumulate ports
            extra_port = row.get("ports", "").strip()
            if extra_port:
                current_ports.append(extra_port)

    _flush()

    logger.debug(f"parse_show_vlan_brief: parsed {len(vlans)} VLANs")
    return vlans


# ---------------------------------------------------------------------------
# show mac  (dash-separated header, one MAC per row)
# ---------------------------------------------------------------------------


def parse_show_mac(output: str) -> List[Dict[str, Any]]:
    """
    Parse `show mac` output from a SONiC switch ASIC namespace.

    Format:
      No.    Vlan  MacAddress         Port         Type
    -----  ------  -----------------  -----------  ------
        1     101  02:00:00:10:01:01  Ethernet294  Static
    ...
    Total number of entries 48

    Returns list of dicts with keys:
      no, vlan, mac_address, port, type

    mac_address is normalised to lowercase colon-delimited hex.
    """
    if not output or output.strip().startswith("ERROR:"):
        return []

    lines = output.splitlines()
    col_names: List[str] = []
    # Column start positions derived from the dash separator line
    col_starts: List[int] = []
    header_found = False
    rows: List[Dict[str, Any]] = []

    # Dash separator: "-----  ------  -----------------  -----------  ------"
    dash_re = re.compile(r"^[\s\-]+$")
    # Skip summary line
    summary_re = re.compile(r"^\s*Total\s+number", re.IGNORECASE)

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if summary_re.match(stripped):
            continue

        # Detect dash separator — use it to find column boundaries
        if dash_re.match(line) and not header_found:
            # Derive column starts from runs of dashes
            col_starts = []
            in_dash = False
            for i, ch in enumerate(line):
                if ch == "-" and not in_dash:
                    col_starts.append(i)
                    in_dash = True
                elif ch != "-":
                    in_dash = False
            header_found = True
            continue

        # Header line is the first non-empty line before the separator
        if not header_found and not col_names and stripped:
            # Record the raw header line — we'll parse after finding col_starts
            # Store it for now
            _header_line = line
            # Detect column positions from the header word starts
            col_starts_tmp: List[int] = []
            in_word = False
            for i, ch in enumerate(line):
                if ch not in (" ", "\t"):
                    if not in_word:
                        col_starts_tmp.append(i)
                        in_word = True
                else:
                    in_word = False
            col_names = [
                re.sub(r"[^a-z0-9]+", "_", line[col_starts_tmp[j] : col_starts_tmp[j + 1]].strip().lower()).strip("_")
                if j + 1 < len(col_starts_tmp)
                else re.sub(r"[^a-z0-9]+", "_", line[col_starts_tmp[j] :].strip().lower()).strip("_")
                for j in range(len(col_starts_tmp))
            ]
            # Normalise: "macaddress" → "mac_address", "no_" → "no"
            col_names = ["mac_address" if "mac" in c else "no" if c.startswith("no") else c for c in col_names]
            col_starts = col_starts_tmp
            continue

        # Data row — slice by column starts
        if col_starts and col_names:
            row: Dict[str, Any] = {}
            for j, (name, start) in enumerate(zip(col_names, col_starts)):
                end = col_starts[j + 1] if j + 1 < len(col_starts) else len(line) + 1
                cell = line[start:end].strip() if start < len(line) else ""
                row[name] = cell

            # Normalise MAC to lowercase colon-delimited
            mac_raw = row.get("mac_address", "")
            if mac_raw:
                hex_only = re.sub(r"[^0-9a-fA-F]", "", mac_raw).lower()
                if len(hex_only) == 12:
                    row["mac_address"] = ":".join(hex_only[i : i + 2] for i in range(0, 12, 2))

            if any(v for v in row.values()):
                rows.append(row)

    logger.debug(f"parse_show_mac: parsed {len(rows)} MAC rows")
    return rows
