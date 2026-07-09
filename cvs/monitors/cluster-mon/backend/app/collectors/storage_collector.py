"""
StorageCollector — collects block device, filesystem, IO and page-cache data.

Commands run via Go daemon on all cluster nodes:
  lsblk -J -o NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE,UUID,RO,RM,MODEL,TRAN
  df -Th
  iostat -d -x -k 1 1
  nvme list -o json  (graceful fallback if not installed)
  cat /proc/meminfo
  cat /proc/vmstat
  bash -c '...top IO processes loop...'

All data cached in app_state.latest_storage_data.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.collectors.base import BaseCollector, CollectorResult, CollectorState
from app.collectors.storage_parser import (
    parse_lsblk,
    parse_df,
    parse_iostat,
    parse_diskstats,
    parse_nvme_list,
    parse_meminfo_storage,
    parse_vmstat,
    parse_top_io_processes,
)
from app.core import go_collector

logger = logging.getLogger(__name__)

# bash one-liner to get top IO consumers from /proc/*/io
_TOP_IO_CMD = (
    "bash -c '"
    "for f in /proc/[0-9]*/io; do "
    "  pid=${f%/io}; pid=${pid##*/}; "
    "  comm=$(cat /proc/$pid/comm 2>/dev/null | head -c 20 || echo ?); "
    "  rb=$(awk \"/^read_bytes/{print \\$2}\" $f 2>/dev/null || echo 0); "
    "  wb=$(awk \"/^write_bytes/{print \\$2}\" $f 2>/dev/null || echo 0); "
    "  echo \"$rb $wb $pid $comm\"; "
    "done 2>/dev/null | sort -rn | head -20'"
)


async def _exec(hosts: List[str], cmd: str, timeout: int = 30) -> Dict[str, str]:
    if not hosts:
        return {}
    return await asyncio.to_thread(
        go_collector._exec_on_hosts,
        hosts,
        cmd,
        timeout,
        go_collector.CLUSTER_SOCKET,
    )


def _ok(out: str) -> bool:
    return bool(out) and not out.startswith(("ABORT", "ERROR"))


class StorageCollector(BaseCollector):
    name = "storage"
    poll_interval = 300
    collect_timeout = 90.0
    critical = False

    async def collect(self, ssh_manager) -> CollectorResult:
        if not go_collector.is_daemon_ready(go_collector.CLUSTER_SOCKET):
            return CollectorResult(
                collector_name=self.name,
                timestamp=CollectorResult.now_iso(),
                state=CollectorState.UNREACHABLE,
                data={},
                error="Go daemon not ready",
            )

        app_state = getattr(self, "_app_state", None)
        hosts: List[str] = []
        if ssh_manager and hasattr(ssh_manager, "_host_list"):
            hosts = list(ssh_manager._host_list)
        if not hosts and app_state:
            ng = getattr(app_state, "node_groups", None)
            if ng and ng.gpu_nodes.hosts:
                hosts = list(ng.gpu_nodes.hosts)
        if not hosts:
            return CollectorResult(
                collector_name=self.name,
                timestamp=CollectorResult.now_iso(),
                state=CollectorState.NO_SERVICE,
                data={},
                error="No hosts configured",
            )

        # Run commands sequentially — the Go daemon uses a single Unix socket
        # and serialises requests internally; concurrent socket calls from the
        # same Python process collide and cause spurious ABORT responses.
        lsblk_out = await _exec(
            hosts, "lsblk -J -o NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE,UUID,RO,RM,MODEL,TRAN 2>/dev/null || lsblk -J", 20
        )
        df_out = await _exec(hosts, "df -Th 2>/dev/null || df -h", 15)
        iostat_out = await _exec(hosts, "iostat -d -x -k 1 1 2>/dev/null || iostat -d 1 1", 20)
        diskstats_out = await _exec(hosts, "cat /proc/diskstats", 10)
        nvme_out = await _exec(hosts, "nvme list -o json 2>/dev/null || echo '{\"Devices\":[]}'", 15)
        meminfo_out = await _exec(hosts, "cat /proc/meminfo", 10)
        vmstat_out = await _exec(hosts, "cat /proc/vmstat", 10)
        topio_out = await _exec(hosts, _TOP_IO_CMD, 20)

        errors: Dict[str, str] = {}
        block_devices: Dict[str, List[dict]] = {}
        filesystems: Dict[str, List[dict]] = {}
        io_stats: Dict[str, List[dict]] = {}
        disk_stats: Dict[str, List[dict]] = {}
        nvme_devices: Dict[str, List[dict]] = {}
        mem_cache: Dict[str, dict] = {}
        vm_stats: Dict[str, dict] = {}
        top_io_procs: Dict[str, List[dict]] = {}

        for host in hosts:
            if not _ok(lsblk_out.get(host, "")):
                errors[host] = lsblk_out.get(host, "ABORT")
            else:
                block_devices[host] = parse_lsblk(lsblk_out.get(host, ""))
            filesystems[host] = parse_df(df_out.get(host, ""))
            io_stats[host] = parse_iostat(iostat_out.get(host, ""))
            disk_stats[host] = parse_diskstats(diskstats_out.get(host, ""))
            nvme_devices[host] = parse_nvme_list(nvme_out.get(host, ""))
            mem_cache[host] = parse_meminfo_storage(meminfo_out.get(host, ""))
            vm_stats[host] = parse_vmstat(vmstat_out.get(host, ""))
            top_io_procs[host] = parse_top_io_processes(topio_out.get(host, ""))

            logger.debug(
                f"Storage {host}: {len(block_devices.get(host, []))} block devs, "
                f"{len(filesystems.get(host, []))} filesystems, "
                f"{len(io_stats.get(host, []))} IO devices"
            )

        data = {
            "block_devices": block_devices,
            "filesystems": filesystems,
            "io_stats": io_stats,
            "disk_stats": disk_stats,
            "nvme_devices": nvme_devices,
            "mem_cache": mem_cache,
            "vm_stats": vm_stats,
            "top_io_procs": top_io_procs,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "errors": errors,
        }

        if app_state and hasattr(app_state, "latest_storage_data"):
            app_state.latest_storage_data = data

        state = CollectorState.OK if not errors else CollectorState.ERROR
        return CollectorResult(
            collector_name=self.name,
            timestamp=CollectorResult.now_iso(),
            state=state,
            data=data,
            error=f"{len(errors)} host(s) had errors" if errors else None,
        )

    async def run(self, ssh_manager, app_state: Any) -> None:
        self._app_state = app_state
        logger.info("StorageCollector started")

        while app_state.is_collecting:
            try:
                result = await asyncio.wait_for(self.collect(ssh_manager), timeout=self.collect_timeout)
            except asyncio.TimeoutError:
                result = CollectorResult(
                    collector_name=self.name,
                    timestamp=CollectorResult.now_iso(),
                    state=CollectorState.ERROR,
                    data={},
                    error=f"collect() timed out after {self.collect_timeout}s",
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"StorageCollector error: {e}", exc_info=True)
                result = CollectorResult(
                    collector_name=self.name,
                    timestamp=CollectorResult.now_iso(),
                    state=CollectorState.ERROR,
                    data={},
                    error=str(e),
                )

            if hasattr(app_state, "collector_results"):
                app_state.collector_results[self.name] = result

            await asyncio.sleep(self.poll_interval)

        logger.info("StorageCollector stopped")
