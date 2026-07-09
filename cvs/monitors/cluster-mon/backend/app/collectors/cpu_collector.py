"""
CPUCollector — collects CPU and memory hardware info and metrics.

Commands run on all cluster nodes via the Go daemon:
  Summary (hardware, collected every poll_interval):
    lscpu                           — CPU architecture, cores, sockets, speeds
    lsmem --summary=always          — memory block layout and totals
    cat /proc/meminfo               — detailed memory breakdown

  Metrics (live, collected every poll_interval):
    cat /proc/stat                  — CPU time counters → utilization %
    cat /proc/loadavg               — 1/5/15-minute load averages
    free -b                         — memory usage in bytes (alternative to meminfo)
    cat /proc/meminfo               — shared with summary for freshness

Data structure stored in app_state.latest_cpu_data:
  {
    "summary": {
      "<host>": {
        "lscpu":   {key: value, ...},
        "lsmem":   {"summary": {...}, "ranges": [...]},
        "meminfo": {key: kb_value, ...},
        "mem_summary": {...},
      }
    },
    "metrics": {
      "<host>": {
        "cpu_stat":  {"cpu": {user_pct, system_pct, idle_pct, ...}, "cpu0": {...}, ...},
        "loadavg":   {load1, load5, load15},
        "meminfo":   {key: kb_value, ...},
        "mem_summary": {...},
      }
    },
    "last_updated": "...",
    "errors": {host: msg},
  }
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.collectors.base import BaseCollector, CollectorResult, CollectorState
from app.collectors.cpu_parser import (
    parse_lscpu,
    lscpu_to_rows,
    parse_lsmem,
    parse_meminfo,
    meminfo_summary,
    parse_proc_stat,
    parse_loadavg,
)
from app.core import go_collector

logger = logging.getLogger(__name__)


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


def _parse_who(output: str) -> List[Dict[str, str]]:
    """Parse `who` text output into a list of session dicts."""
    rows = []
    for line in (output or "").splitlines():
        parts = line.split()
        if len(parts) >= 4:
            rows.append(
                {
                    "user": parts[0],
                    "tty": parts[1],
                    "date": parts[2],
                    "time": parts[3],
                    "from": parts[4].strip("()") if len(parts) > 4 else "",
                }
            )
    return rows


def _parse_ps(output: str) -> List[Dict[str, str]]:
    """
    Parse `ps -eo pid,user,pcpu,pmem,vsz,rss,stat,start,time,args --no-headers` output.
    Fields: pid user %cpu %mem vsz rss stat start time args...
    """
    rows = []
    for line in (output or "").splitlines():
        line = line.strip()
        if not line or line.startswith("PID"):
            continue
        parts = line.split(None, 9)  # split into max 10 fields (args kept together)
        if len(parts) < 5:
            continue
        rows.append(
            {
                "pid": parts[0] if len(parts) > 0 else "",
                "user": parts[1] if len(parts) > 1 else "",
                "cpu": parts[2] if len(parts) > 2 else "",
                "mem": parts[3] if len(parts) > 3 else "",
                "vsz": parts[4] if len(parts) > 4 else "",
                "rss": parts[5] if len(parts) > 5 else "",
                "stat": parts[6] if len(parts) > 6 else "",
                "start": parts[7] if len(parts) > 7 else "",
                "time": parts[8] if len(parts) > 8 else "",
                "args": parts[9] if len(parts) > 9 else "",
            }
        )
    return rows


class CPUCollector(BaseCollector):
    name = "cpu_info"
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

        # Get host list from ssh_manager or node_groups
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

        errors: Dict[str, str] = {}

        # Run commands sequentially — the Go daemon uses a single Unix socket
        # and serialises requests internally; concurrent socket calls from the
        # same Python process collide and cause spurious ABORT responses.
        lscpu_out = await _exec(hosts, "lscpu", 30)
        lsmem_out = await _exec(hosts, "lsmem --summary=always", 20)
        meminfo_out = await _exec(hosts, "cat /proc/meminfo", 10)
        stat_out = await _exec(hosts, "cat /proc/stat", 10)
        loadavg_out = await _exec(hosts, "cat /proc/loadavg", 10)

        # Logged-in users
        who_out = await _exec(hosts, "who", 10)

        # Non-root user processes (full args for identification)
        ps_out = await _exec(
            hosts,
            "ps -eo pid,user,pcpu,pmem,vsz,rss,stat,start,time,args --no-headers 2>/dev/null"
            " | awk '$2 != \"root\"' | head -200",
            20,
        )

        # KFD processes — PIDs in /sys/class/kfd/kfd/proc/ are processes using AMD GPU KFD
        kfd_cmd = (
            "bash -c '"
            "kfd_dir=/sys/class/kfd/kfd/proc; "
            "if [ -d $kfd_dir ]; then "
            "  for pid in $(ls $kfd_dir 2>/dev/null); do "
            "    ps -p $pid -o pid=,user=,pcpu=,pmem=,args= --no-headers 2>/dev/null; "
            "  done; "
            "fi'"
        )
        kfd_out = await _exec(hosts, kfd_cmd, 20)

        summary: Dict[str, Any] = {}
        metrics: Dict[str, Any] = {}

        for host in hosts:
            # Collect errors
            for label, out in [
                ("lscpu", lscpu_out.get(host, "")),
                ("stat", stat_out.get(host, "")),
            ]:
                if not _ok(out):
                    errors[host] = out or "ABORT: no response"
                    break

            # Summary
            lscpu_data = parse_lscpu(lscpu_out.get(host, ""))
            lsmem_data = parse_lsmem(lsmem_out.get(host, ""))
            mem_data = parse_meminfo(meminfo_out.get(host, ""))
            summary[host] = {
                "lscpu": lscpu_data,
                "lscpu_rows": lscpu_to_rows(lscpu_data),
                "lsmem": lsmem_data,
                "meminfo": mem_data,
                "mem_summary": meminfo_summary(mem_data),
                "logged_in": _parse_who(who_out.get(host, "")),
                "user_procs": _parse_ps(ps_out.get(host, "")),
                "kfd_procs": _parse_ps(kfd_out.get(host, "")),
            }

            # Metrics
            metrics[host] = {
                "cpu_stat": parse_proc_stat(stat_out.get(host, "")),
                "loadavg": parse_loadavg(loadavg_out.get(host, "")),
                "meminfo": mem_data,
                "mem_summary": meminfo_summary(mem_data),
            }

        data = {
            "summary": summary,
            "metrics": metrics,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "errors": errors,
        }

        if app_state and hasattr(app_state, "latest_cpu_data"):
            app_state.latest_cpu_data = data

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
        logger.info("CPUCollector started")

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
                logger.error(f"CPUCollector error: {e}", exc_info=True)
                result = CollectorResult(
                    collector_name=self.name,
                    timestamp=CollectorResult.now_iso(),
                    state=CollectorState.ERROR,
                    data={},
                    error=str(e),
                )

            if hasattr(app_state, "collector_results"):
                app_state.collector_results[self.name] = result

            try:
                from app.main import broadcast_metrics

                if hasattr(app_state, "latest_metrics"):
                    app_state.latest_metrics[self.name] = result.data
                    app_state.latest_metrics["timestamp"] = result.timestamp
                    await broadcast_metrics(app_state.latest_metrics)
            except Exception:
                pass

            await asyncio.sleep(self.poll_interval)

        logger.info("CPUCollector stopped")
