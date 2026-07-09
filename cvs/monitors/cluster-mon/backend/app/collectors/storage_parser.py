"""
Parsers for storage-related command outputs.

Handles:
  lsblk -J        — block device JSON
  df -h            — filesystem usage
  iostat -x        — extended disk IO stats
  nvme list -o json — NVMe device list
  /proc/meminfo    — page cache, dirty pages, etc.
  /proc/vmstat     — VM counters
  /proc/*/io       — per-process IO (top IO consumers)
"""

from __future__ import annotations

import json as _json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# lsblk --json
# ---------------------------------------------------------------------------


def parse_lsblk(output: str) -> List[Dict[str, Any]]:
    """
    Parse `lsblk -J -o NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE,UUID,RO,RM,MODEL,TRAN`.
    Returns a flat list of all devices (disks, partitions, LVM volumes).
    """
    if not output or output.startswith(("ERROR", "ABORT")):
        return []
    clean = output.strip()
    idx = clean.find('{')
    if idx == -1:
        return []
    try:
        data = _json.loads(clean[idx:])
    except _json.JSONDecodeError as e:
        logger.warning(f"lsblk JSON parse error: {e}")
        return []

    rows: List[Dict[str, Any]] = []

    def _flatten(dev: Dict, depth: int = 0) -> None:
        row = {
            "name": dev.get("name", ""),
            "size": dev.get("size", ""),
            "type": dev.get("type", ""),
            "mountpoint": dev.get("mountpoint") or dev.get("mountpoints", [""])[0]
            if isinstance(dev.get("mountpoints"), list)
            else dev.get("mountpoint", ""),
            "fstype": dev.get("fstype", "") or "",
            "uuid": (dev.get("uuid", "") or "")[:8],  # truncate for display
            "ro": dev.get("ro", "0"),
            "rm": dev.get("rm", "0"),
            "model": dev.get("model", "") or "",
            "tran": dev.get("tran", "") or "",
            "depth": depth,
        }
        rows.append(row)
        for child in dev.get("children", []):
            _flatten(child, depth + 1)

    for dev in data.get("blockdevices", []):
        _flatten(dev)

    return rows


# ---------------------------------------------------------------------------
# df -h
# ---------------------------------------------------------------------------


def parse_df(output: str) -> List[Dict[str, Any]]:
    """
    Parse `df -h --output=source,fstype,size,used,avail,pcent,target`.
    Returns one dict per mounted filesystem.
    """
    if not output or output.startswith(("ERROR", "ABORT")):
        return []

    rows: List[Dict[str, Any]] = []
    lines = output.strip().splitlines()
    # Skip header line; handle both space-separated and field-output formats
    data_lines = [
        text_line
        for text_line in lines
        if text_line and not text_line.startswith("Filesystem") and not text_line.startswith("Source")
    ]
    for line in data_lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        # Try to parse the Use% field
        if len(parts) >= 6:
            try:
                pct = int(parts[5].replace('%', '')) if '%' in parts[5] else int(parts[4].replace('%', ''))
            except ValueError:
                pct = 0
            rows.append(
                {
                    "source": parts[0],
                    "fstype": parts[1] if len(parts) >= 7 else "",
                    "size": parts[2] if len(parts) >= 7 else parts[1],
                    "used": parts[3] if len(parts) >= 7 else parts[2],
                    "avail": parts[4] if len(parts) >= 7 else parts[3],
                    "use_pct": pct,
                    "mountpoint": parts[6] if len(parts) >= 7 else parts[5],
                }
            )
        elif len(parts) == 5:
            try:
                pct = int(parts[4].replace('%', ''))
            except ValueError:
                pct = 0
            rows.append(
                {
                    "source": parts[0],
                    "fstype": "",
                    "size": parts[1],
                    "used": parts[2],
                    "avail": parts[3],
                    "use_pct": pct,
                    "mountpoint": parts[4],
                }
            )

    # Filter out pseudo/special filesystems
    skip_types = {
        "tmpfs",
        "devtmpfs",
        "devpts",
        "sysfs",
        "proc",
        "cgroup",
        "cgroup2",
        "pstore",
        "bpf",
        "hugetlbfs",
        "mqueue",
        "debugfs",
        "tracefs",
        "securityfs",
        "fusectl",
        "configfs",
        "efivarfs",
        "squashfs",
    }
    return [
        r for r in rows if r.get("fstype", "") not in skip_types and not r["source"].startswith(("tmpfs", "devtmpfs"))
    ]


# ---------------------------------------------------------------------------
# iostat -d -x -k 1 1
# ---------------------------------------------------------------------------


def parse_iostat(output: str) -> List[Dict[str, Any]]:
    """
    Parse `iostat -d -x -k 1 1` text output.
    Returns one dict per block device with read/write KB/s and %util.
    """
    if not output or output.startswith(("ERROR", "ABORT")):
        return []

    rows: List[Dict[str, Any]] = []
    header: Optional[List[str]] = None
    in_device_section = False

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Device"):
            # iostat header line
            header = line.split()
            in_device_section = True
            continue
        if in_device_section and header and not line.startswith(("avg-cpu", "Linux", "iostat")):
            parts = line.split()
            if len(parts) != len(header):
                continue
            try:
                row: Dict[str, Any] = {}
                for col, val in zip(header, parts):
                    col_clean = col.replace('%', 'pct_').replace('/', '_').lower()
                    try:
                        row[col_clean] = float(val)
                    except ValueError:
                        row[col_clean] = val
                rows.append(row)
            except Exception:
                pass

    return rows


# ---------------------------------------------------------------------------
# nvme list -o json
# ---------------------------------------------------------------------------


def parse_nvme_list(output: str) -> List[Dict[str, Any]]:
    """
    Parse `nvme list -o json`.
    Returns one dict per NVMe device.
    """
    if not output or output.startswith(("ERROR", "ABORT")):
        return []
    clean = output.strip()
    idx = clean.find('{')
    if idx == -1:
        return []
    try:
        data = _json.loads(clean[idx:])
    except _json.JSONDecodeError:
        return []

    devices = data.get("Devices", [])
    rows = []
    for d in devices:
        size_bytes = d.get("PhysicalSize", 0)
        size_gb = f"{size_bytes / 1e9:.1f} GB" if size_bytes else "?"
        rows.append(
            {
                "device": d.get("DevicePath", ""),
                "model": d.get("ModelNumber", "").strip(),
                "serial": d.get("SerialNumber", "").strip(),
                "firmware": d.get("Firmware", "").strip(),
                "size": size_gb,
                "namespace": d.get("NameSpace", ""),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# /proc/meminfo (storage-relevant subset)
# ---------------------------------------------------------------------------


def parse_meminfo_storage(output: str) -> Dict[str, Any]:
    """
    Parse /proc/meminfo for storage-relevant fields:
    Dirty, Writeback, PageTables, Slab, Cached, Buffers, NFS_Unstable.
    Returns values in MiB.
    """
    if not output or output.startswith(("ERROR", "ABORT")):
        return {}

    result: Dict[str, Any] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        # Parse kB value
        m = re.match(r"(\d+)", val)
        if m:
            kb = int(m.group(1))
            result[key] = round(kb / 1024, 1)  # convert kB → MiB

    # Build human-friendly summary
    dirty = result.get("Dirty", 0)
    writeback = result.get("Writeback", 0)
    cached = result.get("Cached", 0)
    buffers = result.get("Buffers", 0)
    slab = result.get("Slab", 0)

    return {
        "dirty_mib": dirty,
        "writeback_mib": writeback,
        "cached_mib": cached,
        "buffers_mib": buffers,
        "slab_mib": slab,
        "page_cache_mib": round(cached + buffers, 1),
        "_raw": {
            k: v
            for k, v in result.items()
            if k
            in (
                "Dirty",
                "Writeback",
                "Cached",
                "Buffers",
                "Slab",
                "PageTables",
                "NFS_Unstable",
                "KReclaimable",
            )
        },
    }


# ---------------------------------------------------------------------------
# /proc/vmstat (key IO counters)
# ---------------------------------------------------------------------------


def parse_vmstat(output: str) -> Dict[str, Any]:
    """
    Parse /proc/vmstat for IO-relevant counters.
    """
    if not output or output.startswith(("ERROR", "ABORT")):
        return {}

    kv: Dict[str, int] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                kv[parts[0]] = int(parts[1])
            except ValueError:
                pass

    return {
        "pgpgin": kv.get("pgpgin", 0),  # pages paged in from disk
        "pgpgout": kv.get("pgpgout", 0),  # pages paged out to disk
        "pswpin": kv.get("pswpin", 0),  # swap pages read in
        "pswpout": kv.get("pswpout", 0),  # swap pages written out
        "pgfault": kv.get("pgfault", 0),  # page faults (minor+major)
        "pgmajfault": kv.get("pgmajfault", 0),  # major page faults (disk read)
        "nr_dirty": kv.get("nr_dirty", 0),  # dirty pages count
        "nr_writeback": kv.get("nr_writeback", 0),  # pages under writeback
        "nr_file_pages": kv.get("nr_file_pages", 0),  # file pages in page cache
        "nr_slab_reclaimable": kv.get("nr_slab_reclaimable", 0),
    }


# ---------------------------------------------------------------------------
# Top IO processes (from /proc/*/io scrape)
# ---------------------------------------------------------------------------


def parse_diskstats(output: str) -> List[Dict[str, Any]]:
    """
    Parse /proc/diskstats.

    Format (fields):
      major minor name
      reads_completed reads_merged sectors_read  ms_reading
      writes_completed writes_merged sectors_written ms_writing
      io_in_progress ms_doing_io weighted_ms_io
      [discards: discard_completed discard_merged sectors_discarded ms_discarding]
      [flushes: flush_completed ms_flushing]

    Returns one dict per real disk (skips loop, ram, sr devices).
    """
    if not output or output.startswith(("ERROR", "ABORT")):
        return []

    rows: List[Dict[str, Any]] = []
    skip_re = re.compile(r"^(loop|ram|sr|fd)\d+$")

    for line in output.strip().splitlines():
        parts = line.split()
        if len(parts) < 14:
            continue
        name = parts[2]
        if skip_re.match(name):
            continue
        try:
            reads = int(parts[3])
            reads_merged = int(parts[4])
            sectors_r = int(parts[5])
            ms_read = int(parts[6])
            writes = int(parts[7])
            writes_merged = int(parts[8])
            sectors_w = int(parts[9])
            ms_write = int(parts[10])
            io_in_prog = int(parts[11])
            ms_io = int(parts[12])
            rows.append(
                {
                    "device": name,
                    "reads": reads,
                    "reads_merged": reads_merged,
                    "read_kb": round(sectors_r * 0.5, 1),  # sectors × 512B ÷ 1024
                    "ms_reading": ms_read,
                    "writes": writes,
                    "writes_merged": writes_merged,
                    "write_kb": round(sectors_w * 0.5, 1),
                    "ms_writing": ms_write,
                    "io_in_progress": io_in_prog,
                    "ms_doing_io": ms_io,
                    # derived
                    "total_kb": round((sectors_r + sectors_w) * 0.5, 1),
                    "read_avg_ms": round(ms_read / reads, 2) if reads > 0 else 0,
                    "write_avg_ms": round(ms_write / writes, 2) if writes > 0 else 0,
                }
            )
        except (ValueError, IndexError):
            continue

    return sorted(rows, key=lambda r: r["total_kb"], reverse=True)


def parse_top_io_processes(output: str) -> List[Dict[str, Any]]:
    """
    Parse output of a bash loop that reads /proc/*/io:
      bash -c 'for f in /proc/[0-9]*/io; do
        pid=${f%/io}; pid=${pid##*/}
        comm=$(cat /proc/$pid/comm 2>/dev/null || echo ?)
        rb=$(awk "/^read_bytes/{print \$2}" $f 2>/dev/null || echo 0)
        wb=$(awk "/^write_bytes/{print \$2}" $f 2>/dev/null || echo 0)
        echo "$rb $wb $pid $comm"
      done | sort -rn | head -20'

    Returns top 20 processes sorted by total IO bytes.
    """
    if not output or output.startswith(("ERROR", "ABORT")):
        return []

    rows = []
    for line in output.strip().splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        try:
            rb = int(parts[0])
            wb = int(parts[1])
            pid = parts[2]
            comm = parts[3].strip()
            total = rb + wb
            rows.append(
                {
                    "pid": pid,
                    "command": comm,
                    "read_bytes": rb,
                    "write_bytes": wb,
                    "total_bytes": total,
                    "read_mb": round(rb / 1024 / 1024, 2),
                    "write_mb": round(wb / 1024 / 1024, 2),
                    "total_mb": round(total / 1024 / 1024, 2),
                }
            )
        except (ValueError, IndexError):
            continue

    return sorted(rows, key=lambda r: r["total_bytes"], reverse=True)[:20]
