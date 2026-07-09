"""
Parsers for CPU and memory data collected from Linux hosts.

Handles:
  lscpu          — key: value format, colon-separated
  lsmem          — table format with summary lines
  /proc/meminfo  — key: value kB format
  /proc/stat     — CPU time counters for utilization %
  /proc/loadavg  — load averages
  free -b        — memory in bytes
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# lscpu
# ---------------------------------------------------------------------------


def parse_lscpu(output: str) -> Dict[str, str]:
    """
    Parse `lscpu` output into a flat dict.
    Keys are normalised to lowercase with underscores.

    Example:
      Architecture:    x86_64
      CPU(s):          128
      Model name:      Intel(R) Xeon(R) Platinum ...
    """
    if not output or output.startswith("ERROR") or output.startswith("ABORT"):
        return {}
    result: Dict[str, str] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if key and val:
            result[key] = val
    return result


def lscpu_to_rows(data: Dict[str, str]) -> List[Dict[str, str]]:
    """Convert lscpu dict into a list of {field, value} rows for DataTable."""
    return [{"Field": k, "Value": v} for k, v in data.items()]


# ---------------------------------------------------------------------------
# lsmem
# ---------------------------------------------------------------------------


def parse_lsmem(output: str) -> Dict[str, Any]:
    """
    Parse `lsmem --summary=always` output.

    Returns:
      {
        "summary": {"total_online": "2048G", "total_offline": "0B", "block_size": "128M"},
        "ranges": [{"range": "...", "size": "...", "state": "...", "removable": "...", "block": "..."}]
      }
    """
    if not output or output.startswith("ERROR") or output.startswith("ABORT"):
        return {}

    summary: Dict[str, str] = {}
    ranges: List[Dict[str, str]] = []
    header: Optional[List[str]] = None

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue

        # Summary lines: "Memory block size:  128M"
        if ":" in line and not line.startswith("RANGE") and not re.match(r"0x", line):
            key, _, val = line.partition(":")
            summary[key.strip()] = val.strip()
            continue

        # Header line
        if line.startswith("RANGE"):
            header = line.split()
            continue

        # Data lines
        if header and re.match(r"0x", line):
            parts = line.split()
            row: Dict[str, str] = {}
            for i, col in enumerate(header):
                row[col] = parts[i] if i < len(parts) else ""
            ranges.append(row)

    return {"summary": summary, "ranges": ranges}


# ---------------------------------------------------------------------------
# /proc/meminfo
# ---------------------------------------------------------------------------


def parse_meminfo(output: str) -> Dict[str, int]:
    """
    Parse /proc/meminfo into a dict of {key: value_in_kB}.

    Example:
      MemTotal:       131906640 kB
      MemFree:        123737628 kB
    """
    if not output or output.startswith("ERROR") or output.startswith("ABORT"):
        return {}
    result: Dict[str, int] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        # Remove "kB" suffix and parse as int
        numeric = re.sub(r"\s*kB\s*$", "", val)
        try:
            result[key] = int(numeric)
        except ValueError:
            pass
    return result


def meminfo_summary(mem: Dict[str, int]) -> Dict[str, Any]:
    """Return high-level memory summary with human-readable sizes."""

    def kb_to_human(kb: int) -> str:
        if kb >= 1024 * 1024:
            return f"{kb / 1024 / 1024:.1f} GiB"
        if kb >= 1024:
            return f"{kb / 1024:.1f} MiB"
        return f"{kb} KiB"

    total = mem.get("MemTotal", 0)
    free = mem.get("MemFree", 0)
    available = mem.get("MemAvailable", 0)
    cached = mem.get("Cached", 0) + mem.get("Buffers", 0)
    used = total - available

    swap_total = mem.get("SwapTotal", 0)
    swap_free = mem.get("SwapFree", 0)
    swap_used = swap_total - swap_free

    used_pct = round(used / total * 100, 1) if total > 0 else 0

    return {
        "total": kb_to_human(total),
        "used": kb_to_human(used),
        "available": kb_to_human(available),
        "cached": kb_to_human(cached),
        "swap_total": kb_to_human(swap_total),
        "swap_used": kb_to_human(swap_used),
        "used_pct": used_pct,
        # raw kB for charting
        "total_kb": total,
        "used_kb": used,
        "free_kb": free,
        "available_kb": available,
    }


# ---------------------------------------------------------------------------
# /proc/stat — CPU utilization
# ---------------------------------------------------------------------------


def parse_proc_stat(output: str) -> Dict[str, Dict[str, float]]:
    """
    Parse /proc/stat and compute per-CPU utilization percentages.

    Returns {
      "cpu":  {"user": X, "system": X, "idle": X, "iowait": X, "total_pct": X},
      "cpu0": {...},
      ...
    }

    Fields in /proc/stat cpu line:
      user nice system idle iowait irq softirq steal guest guest_nice
    """
    if not output or output.startswith("ERROR") or output.startswith("ABORT"):
        return {}

    result: Dict[str, Dict[str, float]] = {}
    for line in output.splitlines():
        if not line.startswith("cpu"):
            break
        parts = line.split()
        if len(parts) < 5:
            continue
        name = parts[0]
        try:
            vals = [int(x) for x in parts[1:]]
        except ValueError:
            continue

        user = vals[0] + vals[1]  # user + nice
        system = vals[2] + vals[5] + vals[6]  # system + irq + softirq
        idle = vals[3]
        iowait = vals[4] if len(vals) > 4 else 0
        steal = vals[7] if len(vals) > 7 else 0
        total = sum(vals)
        used = total - idle - iowait

        if total == 0:
            continue

        result[name] = {
            "user_pct": round(user / total * 100, 1),
            "system_pct": round(system / total * 100, 1),
            "idle_pct": round(idle / total * 100, 1),
            "iowait_pct": round(iowait / total * 100, 1),
            "steal_pct": round(steal / total * 100, 1),
            "used_pct": round(used / total * 100, 1),
        }

    return result


# ---------------------------------------------------------------------------
# /proc/loadavg
# ---------------------------------------------------------------------------


def parse_loadavg(output: str) -> Dict[str, float]:
    """
    Parse /proc/loadavg.
    Format: 0.52 0.65 0.74 2/1234 5678
    """
    if not output or output.startswith("ERROR") or output.startswith("ABORT"):
        return {}
    parts = output.strip().split()
    try:
        return {
            "load1": float(parts[0]),
            "load5": float(parts[1]),
            "load15": float(parts[2]),
        }
    except (IndexError, ValueError):
        return {}
