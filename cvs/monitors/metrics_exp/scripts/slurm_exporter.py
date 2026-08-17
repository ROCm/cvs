#!/usr/bin/env python3
"""
Slurm Control Plane Metrics Exporter
Collects metrics from a Slurm head node via CLI commands (sinfo, squeue, sdiag, sacct)
and exposes them in Prometheus format on a configurable HTTP port.

Usage:
    python3 slurm_exporter.py [--port 9418] [--interval 30]
"""

import argparse
import http.server
import json
import logging
import re
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("slurm_exporter")

# Global state
metrics_output = "# Slurm metrics not yet collected\n"
metrics_lock = threading.Lock()

METRIC_PREFIX = "slurm"

# Default ports for common Slurm services
DEFAULT_PORT = 9418
DEFAULT_INTERVAL = 30


def _run(cmd: List[str], timeout: int = 30) -> Optional[str]:
    """Run a shell command and return stdout, or None on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return result.stdout
        logger.warning(f"Command {cmd[0]} exited {result.returncode}: {result.stderr[:200]}")
        return None
    except subprocess.TimeoutExpired:
        logger.warning(f"Command {cmd[0]} timed out after {timeout}s")
        return None
    except FileNotFoundError:
        logger.warning(f"Command not found: {cmd[0]}")
        return None
    except Exception as e:
        logger.warning(f"Command {cmd[0]} failed: {e}")
        return None


def _process_up(service_name: str) -> int:
    """
    Check if a Slurm daemon is reachable.
    Works correctly from login nodes (where the daemons don't run locally)
    by using Slurm's own RPC ping commands instead of systemctl.
    """
    if service_name == "slurmctld":
        # scontrol ping checks slurmctld reachability from any node in the cluster
        out = _run(["scontrol", "ping"], timeout=10)
        if out:
            out_lower = out.lower()
            if "is up" in out_lower or "responding" in out_lower:
                return 1
            # Some versions print "Slurmctld(primary) is UP" or just list the hostname
            if "slurmctld" in out_lower and "down" not in out_lower and "not" not in out_lower:
                return 1
        return 0
    if service_name == "slurmdbd":
        # sacctmgr ping checks slurmdbd reachability
        out = _run(["sacctmgr", "-n", "show", "stats"], timeout=10)
        if out and out.strip():
            return 1
        return 0
    # Fallback for local process checks
    out = _run(["systemctl", "is-active", service_name])
    if out and out.strip() == "active":
        return 1
    out2 = _run(["pgrep", "-x", service_name])
    return 1 if (out2 and out2.strip()) else 0


def _state_bucket(state: str) -> str:
    """Normalise a Slurm node state string to a display bucket."""
    s = state.lower()
    if "allocated" in s or s == "alloc":
        return "allocated"
    if "idle" in s and "drain" not in s and "down" not in s:
        return "idle"
    if "mixed" in s or "mix" in s:
        return "mixed"
    if "drain" in s:
        return "drained"
    if "down" in s:
        return "down"
    if "planned" in s or "plnd" in s:
        return "planned"
    return "other"


def collect_node_metrics() -> Tuple[Dict, List[Dict]]:
    """
    Collect per-node state counts, CPU, memory, and per-node info rows.

    Handles three sinfo JSON layouts:
      1. Old format:    {"nodes": [{hostname, state, cpus, ...}]}
      2. Modern format: {"sinfo": [{node: {hostname, state, cpus,...}, partition: {name,...}}]}
      3. Text fallback: sinfo -N (one line per node)
    """
    state_counts: Dict[str, int] = {}
    node_rows: List[Dict] = []
    cpu_total = cpu_alloc = cpu_idle = mem_total = mem_alloc = 0
    seen_nodes: set = set()

    # ---- Attempt 1: sinfo --json ----
    out = _run(["sinfo", "--json"])
    if out:
        try:
            data = json.loads(out)
            top_keys = list(data.keys())
            logger.debug(f"sinfo --json top-level keys: {top_keys}")

            # Modern Slurm (20.11+): top key is "sinfo", each entry has node + partition sub-objects
            if "sinfo" in data:
                for entry in data["sinfo"]:
                    node = entry.get("node", {})
                    node_name = (node.get("hostname") or node.get("name") or "").strip()
                    if not node_name or node_name in seen_nodes:
                        continue
                    seen_nodes.add(node_name)

                    raw_state = node.get("state", "unknown")
                    state_str = (
                        "+".join(s.lower() for s in raw_state)
                        if isinstance(raw_state, list)
                        else str(raw_state).lower()
                    )
                    bucket = _state_bucket(state_str)
                    state_counts[bucket] = state_counts.get(bucket, 0) + 1

                    total_cpus = node.get("cpus", 0) or 0
                    alloc_cpus = node.get("alloc_cpus", 0) or 0
                    real_mem = node.get("real_memory", 0) or 0
                    alloc_mem = node.get("alloc_mem", 0) or 0
                    cpu_total += total_cpus
                    mem_total += real_mem
                    # Only count CPUs as truly allocated on actively running nodes.
                    # "planned" and "reserved" nodes have CPUs reserved but no jobs
                    # running yet — counting them inflates utilization incorrectly.
                    if bucket in ("allocated", "mixed"):
                        cpu_alloc += alloc_cpus
                        cpu_idle += max(0, total_cpus - alloc_cpus)
                    else:
                        cpu_idle += total_cpus
                    if bucket in ("allocated", "mixed"):
                        mem_alloc += alloc_mem

                    partition_name = entry.get("partition", {}).get("name", "unknown")
                    reason = (node.get("reason") or "").replace('"', "'")
                    node_rows.append(
                        {
                            "node": node_name,
                            "partition": partition_name,
                            "state": bucket,
                            "cpus": str(total_cpus),
                            "memory": str(real_mem),
                            "reason": reason,
                        }
                    )

            # Old Slurm format: top key is "nodes", flat list of node objects
            elif "nodes" in data:
                for node in data["nodes"]:
                    node_name = (node.get("hostname") or node.get("name") or "").strip()
                    if not node_name or node_name in seen_nodes:
                        continue
                    seen_nodes.add(node_name)

                    raw_state = node.get("state", "unknown")
                    state_str = (
                        "+".join(s.lower() for s in raw_state)
                        if isinstance(raw_state, list)
                        else str(raw_state).lower()
                    )
                    bucket = _state_bucket(state_str)
                    state_counts[bucket] = state_counts.get(bucket, 0) + 1

                    total_cpus = node.get("cpus", 0) or 0
                    alloc_cpus = node.get("alloc_cpus", 0) or 0
                    real_mem = node.get("real_memory", 0) or 0
                    alloc_mem = node.get("alloc_mem", 0) or 0
                    cpu_total += total_cpus
                    mem_total += real_mem
                    if bucket in ("allocated", "mixed"):
                        cpu_alloc += alloc_cpus
                        cpu_idle += max(0, total_cpus - alloc_cpus)
                        mem_alloc += alloc_mem
                    else:
                        cpu_idle += total_cpus

                    partitions = node.get("partitions", [])
                    partition_str = ",".join(partitions) if partitions else "unknown"
                    reason = (node.get("reason") or "").replace('"', "'")
                    node_rows.append(
                        {
                            "node": node_name,
                            "partition": partition_str,
                            "state": bucket,
                            "cpus": str(total_cpus),
                            "memory": str(real_mem),
                            "reason": reason,
                        }
                    )

            if seen_nodes:
                logger.debug(f"sinfo JSON: collected {len(seen_nodes)} nodes")
                return state_counts, node_rows, cpu_total, cpu_alloc, cpu_idle, mem_total, mem_alloc

            logger.warning(f"sinfo --json parsed but found 0 nodes (keys={top_keys}). Falling back to text.")
        except Exception as e:
            logger.warning(f"sinfo --json parse failed: {e}. Falling back to text.")

    # ---- Attempt 2: plain text with -N (one line per node) ----
    # -N forces one row per node; %n=hostname, %P=partition, %T=state, %C=A/I/O/T, %m=mem, %E=reason
    out = _run(["sinfo", "-N", "--noheader", "-o", "%n %P %T %C %m %E"])
    if out:
        for line in out.splitlines():
            # split on whitespace with max 5 splits so reason (col 6+) stays together
            parts = line.strip().split(None, 5)
            if len(parts) < 5:
                continue
            node_name = parts[0]
            if node_name in seen_nodes:
                continue  # node appears in multiple partitions — count it once
            seen_nodes.add(node_name)

            partition = parts[1]
            state = parts[2].lower()
            cpu_str = parts[3]  # format: A/I/O/T
            mem = parts[4]
            reason = parts[5].replace('"', "'") if len(parts) > 5 else ""

            bucket = _state_bucket(state)
            state_counts[bucket] = state_counts.get(bucket, 0) + 1

            node_cpus = 0
            cpu_parts = cpu_str.split("/")
            if len(cpu_parts) == 4:
                try:
                    a, i, o, t = (int(x) for x in cpu_parts)
                    node_cpus = t
                    cpu_total += t
                    # Only count allocated CPUs from truly running nodes
                    if bucket in ("allocated", "mixed"):
                        cpu_alloc += a
                        cpu_idle += i
                    else:
                        cpu_idle += t
                except ValueError:
                    pass

            try:
                mem_mb = int(mem)
            except ValueError:
                mem_mb = 0
            mem_total += mem_mb

            node_rows.append(
                {
                    "node": node_name,
                    "partition": partition,
                    "state": bucket,
                    "cpus": str(node_cpus),
                    "memory": str(mem_mb),
                    "reason": reason,
                }
            )

    return state_counts, node_rows, cpu_total, cpu_alloc, cpu_idle, mem_total, mem_alloc


def collect_job_metrics() -> Tuple[Dict, List[Dict]]:
    """
    Parse squeue JSON for job state counts and per-job info.
    Returns (state_counts, job_rows).
    """
    state_counts: Dict[str, int] = {}
    job_rows: List[Dict] = []

    out = _run(["squeue", "--json"])
    if out:
        try:
            data = json.loads(out)
            jobs = data.get("jobs", [])
            for job in jobs:
                raw_state = job.get("job_state", "unknown")
                if isinstance(raw_state, list):
                    state = raw_state[0].lower() if raw_state else "unknown"
                else:
                    state = str(raw_state).lower()

                state_counts[state] = state_counts.get(state, 0) + 1

                reason = job.get("state_reason", "") or ""
                time_limit = job.get("time_limit", {})
                if isinstance(time_limit, dict):
                    tl_val = time_limit.get("number", 0)
                    tl_str = str(tl_val) if tl_val else "unlimited"
                else:
                    tl_str = str(time_limit) if time_limit else "unlimited"

                cpus = job.get("cpus", {})
                if isinstance(cpus, dict):
                    cpus_num = cpus.get("number", 0)
                else:
                    cpus_num = cpus or 0

                partition = job.get("partition", "unknown") or "unknown"
                user = job.get("user_name", "unknown") or "unknown"
                name = (job.get("name", "unknown") or "unknown").replace('"', "'")

                job_rows.append(
                    {
                        "job_id": str(job.get("job_id", "unknown")),
                        "name": name,
                        "user": user,
                        "partition": partition,
                        "state": state,
                        "cpus": str(cpus_num),
                        "time_limit": tl_str,
                        "reason": reason.replace('"', "'"),
                    }
                )
            return state_counts, job_rows
        except Exception as e:
            logger.warning(f"Failed to parse squeue JSON: {e}")

    # Fallback: plain text squeue
    out = _run(["squeue", "-h", "-o", "%i %j %u %P %T %C %l %r"])
    if out:
        for line in out.splitlines():
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            job_id, name, user, partition, state, cpus, time_limit = parts[:7]
            reason = parts[7] if len(parts) > 7 else ""
            state = state.lower()
            state_counts[state] = state_counts.get(state, 0) + 1
            job_rows.append(
                {
                    "job_id": job_id,
                    "name": name.replace('"', "'"),
                    "user": user,
                    "partition": partition,
                    "state": state,
                    "cpus": cpus,
                    "time_limit": time_limit,
                    "reason": reason.replace('"', "'"),
                }
            )

    return state_counts, job_rows


def collect_partition_metrics() -> List[Dict]:
    """Collect per-partition summary info."""
    rows: List[Dict] = []
    out = _run(["sinfo", "-h", "-o", "%P %a %D %C %m", "--noheader"])
    if not out:
        return rows
    for line in out.splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        partition = parts[0].rstrip("*")
        state = parts[1]
        nodes = parts[2]
        cpu_str = parts[3]  # A/I/O/T
        cpu_parts = cpu_str.split("/")
        cpus_total = cpu_parts[3] if len(cpu_parts) == 4 else "0"
        cpus_alloc = cpu_parts[0] if len(cpu_parts) == 4 else "0"
        rows.append(
            {
                "partition": partition,
                "state": state,
                "nodes": nodes,
                "cpus_total": cpus_total,
                "cpus_alloc": cpus_alloc,
            }
        )
    return rows


def collect_recent_jobs() -> List[Dict]:
    """Collect recently completed jobs via sacct (last 24 hours)."""
    rows: List[Dict] = []
    out = _run(
        [
            "sacct",
            "-X",
            "--starttime=now-24hours",
            "--format=JobID,JobName,User,State,Elapsed,ExitCode",
            "--noheader",
            "--parsable2",
        ],
        timeout=60,
    )
    if not out:
        return rows
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 6:
            continue
        job_id, name, user, state, elapsed, exit_code = parts[:6]
        # Skip running/pending - we only want completed
        state_lower = state.lower()
        if "running" in state_lower or "pending" in state_lower:
            continue
        rows.append(
            {
                "job_id": job_id,
                "name": name.replace('"', "'"),
                "user": user,
                "state": state_lower,
                "elapsed": elapsed,
                "exit_code": exit_code,
            }
        )
    return rows[:200]  # Cap at 200 rows for metric cardinality safety


def collect_scheduler_metrics() -> Dict[str, float]:
    """Collect scheduler diagnostics via sdiag."""
    result: Dict[str, float] = {
        "backfill_cycle_last_seconds": 0.0,
        "backfill_cycle_mean_seconds": 0.0,
        "backfill_jobs_total": 0.0,
        "threads_active": 0.0,
        "dbd_agent_queue_size": 0.0,
    }
    out = _run(["sdiag"])
    if not out:
        return result

    for line in out.splitlines():
        line = line.strip()
        # Backfill last cycle time (microseconds)
        m = re.search(r"Last backfill cycle time\s*[:\(].*?(\d+)\s*(?:usec|microsec)?", line, re.IGNORECASE)
        if m:
            result["backfill_cycle_last_seconds"] = int(m.group(1)) / 1_000_000
            continue
        # Mean backfill cycle
        m = re.search(r"Mean backfill cycle time\s*[:\(].*?(\d+)\s*(?:usec|microsec)?", line, re.IGNORECASE)
        if m:
            result["backfill_cycle_mean_seconds"] = int(m.group(1)) / 1_000_000
            continue
        # Jobs started via backfill
        m = re.search(r"Total backfilled jobs.*?:\s*(\d+)", line, re.IGNORECASE)
        if m:
            result["backfill_jobs_total"] = float(m.group(1))
            continue
        # Active threads
        m = re.search(r"Server thread count\s*:\s*(\d+)", line, re.IGNORECASE)
        if m:
            result["threads_active"] = float(m.group(1))
            continue
        # DBD agent queue size
        m = re.search(r"Agent queue size\s*:\s*(\d+)", line, re.IGNORECASE)
        if m:
            result["dbd_agent_queue_size"] = float(m.group(1))
            continue

    return result


def _escape_label(v: str) -> str:
    """Escape backslash and double-quote for Prometheus label values."""
    return v.replace("\\", "\\\\").replace('"', '\\"')


def collect() -> str:
    """Run all collectors and return Prometheus text format."""
    lines = []

    # ---- Process health ----
    slurmctld_up = _process_up("slurmctld")
    slurmdbd_up = _process_up("slurmdbd")
    lines.append(f"# HELP {METRIC_PREFIX}_slurmctld_up slurmctld process health (1=up)\n")
    lines.append(f"# TYPE {METRIC_PREFIX}_slurmctld_up gauge\n")
    lines.append(f"{METRIC_PREFIX}_slurmctld_up {slurmctld_up}\n")
    lines.append(f"# HELP {METRIC_PREFIX}_slurmdbd_up slurmdbd process health (1=up)\n")
    lines.append(f"# TYPE {METRIC_PREFIX}_slurmdbd_up gauge\n")
    lines.append(f"{METRIC_PREFIX}_slurmdbd_up {slurmdbd_up}\n")

    # ---- Node metrics ----
    try:
        state_counts, node_rows, cpu_total, cpu_alloc, cpu_idle, mem_total, mem_alloc = collect_node_metrics()
    except Exception as e:
        logger.error(f"Node metrics collection failed: {e}")
        state_counts, node_rows = {}, []
        cpu_total = cpu_alloc = cpu_idle = mem_total = mem_alloc = 0

    lines.append(f"# HELP {METRIC_PREFIX}_nodes_total Total nodes in cluster\n")
    lines.append(f"# TYPE {METRIC_PREFIX}_nodes_total gauge\n")
    total_nodes = sum(state_counts.values())
    lines.append(f"{METRIC_PREFIX}_nodes_total {total_nodes}\n")

    lines.append(f"# HELP {METRIC_PREFIX}_nodes_state Number of nodes in each state\n")
    lines.append(f"# TYPE {METRIC_PREFIX}_nodes_state gauge\n")
    for state, count in state_counts.items():
        lines.append(f'{METRIC_PREFIX}_nodes_state{{state="{_escape_label(state)}"}} {count}\n')

    lines.append(f"# HELP {METRIC_PREFIX}_cpus_total Total CPUs in cluster\n")
    lines.append(f"# TYPE {METRIC_PREFIX}_cpus_total gauge\n")
    lines.append(f"{METRIC_PREFIX}_cpus_total {cpu_total}\n")

    lines.append(f"# HELP {METRIC_PREFIX}_cpus_allocated Allocated CPUs\n")
    lines.append(f"# TYPE {METRIC_PREFIX}_cpus_allocated gauge\n")
    lines.append(f"{METRIC_PREFIX}_cpus_allocated {cpu_alloc}\n")

    lines.append(f"# HELP {METRIC_PREFIX}_cpus_idle Idle CPUs\n")
    lines.append(f"# TYPE {METRIC_PREFIX}_cpus_idle gauge\n")
    lines.append(f"{METRIC_PREFIX}_cpus_idle {cpu_idle}\n")

    lines.append(f"# HELP {METRIC_PREFIX}_memory_total_mb Total memory across cluster in MB\n")
    lines.append(f"# TYPE {METRIC_PREFIX}_memory_total_mb gauge\n")
    lines.append(f"{METRIC_PREFIX}_memory_total_mb {mem_total}\n")

    lines.append(f"# HELP {METRIC_PREFIX}_memory_allocated_mb Allocated memory in MB\n")
    lines.append(f"# TYPE {METRIC_PREFIX}_memory_allocated_mb gauge\n")
    lines.append(f"{METRIC_PREFIX}_memory_allocated_mb {mem_alloc}\n")

    # Node info table metrics
    lines.append(f"# HELP {METRIC_PREFIX}_node_info Node info for table display (always 1)\n")
    lines.append(f"# TYPE {METRIC_PREFIX}_node_info gauge\n")
    for row in node_rows:
        n = _escape_label(row["node"])
        p = _escape_label(row["partition"])
        s = _escape_label(row["state"])
        c = _escape_label(row["cpus"])
        m = _escape_label(row["memory"])
        r = _escape_label(row["reason"])
        lines.append(
            f'{METRIC_PREFIX}_node_info{{node="{n}",partition="{p}",state="{s}",'
            f'cpus="{c}",memory="{m}",reason="{r}"}} 1\n'
        )

    # ---- Job metrics ----
    try:
        job_state_counts, job_rows = collect_job_metrics()
    except Exception as e:
        logger.error(f"Job metrics collection failed: {e}")
        job_state_counts, job_rows = {}, []

    lines.append(f"# HELP {METRIC_PREFIX}_jobs_state Number of jobs in each state\n")
    lines.append(f"# TYPE {METRIC_PREFIX}_jobs_state gauge\n")
    for state, count in job_state_counts.items():
        lines.append(f'{METRIC_PREFIX}_jobs_state{{state="{_escape_label(state)}"}} {count}\n')

    # Compute running CPUs from squeue data — most accurate utilization signal.
    # Uses only RUNNING jobs so planned/pending nodes don't inflate the number.
    running_cpus = sum(int(row["cpus"]) for row in job_rows if row["state"] == "running" and row["cpus"].isdigit())
    lines.append(f"# HELP {METRIC_PREFIX}_running_cpus Total CPUs used by actively running jobs\n")
    lines.append(f"# TYPE {METRIC_PREFIX}_running_cpus gauge\n")
    lines.append(f"{METRIC_PREFIX}_running_cpus {running_cpus}\n")

    # Job info table metrics (label-based for Grafana table panels)
    lines.append(f"# HELP {METRIC_PREFIX}_job_info Job info for table display (always 1)\n")
    lines.append(f"# TYPE {METRIC_PREFIX}_job_info gauge\n")
    # Numeric CPU count per job — allows Grafana to sort top consumers by value
    lines.append(f"# HELP {METRIC_PREFIX}_job_cpu_count CPUs allocated per job (numeric)\n")
    lines.append(f"# TYPE {METRIC_PREFIX}_job_cpu_count gauge\n")
    for row in job_rows:
        jid = _escape_label(row["job_id"])
        nm = _escape_label(row["name"])
        usr = _escape_label(row["user"])
        pt = _escape_label(row["partition"])
        st = _escape_label(row["state"])
        cp = _escape_label(row["cpus"])
        tl = _escape_label(row["time_limit"])
        rs = _escape_label(row["reason"])
        lines.append(
            f'{METRIC_PREFIX}_job_info{{job_id="{jid}",name="{nm}",user="{usr}",'
            f'partition="{pt}",state="{st}",cpus="{cp}",time_limit="{tl}",reason="{rs}"}} 1\n'
        )
        try:
            cpu_num = int(row["cpus"])
            lines.append(
                f'{METRIC_PREFIX}_job_cpu_count{{job_id="{jid}",name="{nm}",user="{usr}",'
                f'partition="{pt}",state="{st}"}} {cpu_num}\n'
            )
        except ValueError:
            pass

    # ---- Partition metrics ----
    try:
        partition_rows = collect_partition_metrics()
    except Exception as e:
        logger.error(f"Partition metrics collection failed: {e}")
        partition_rows = []

    lines.append(f"# HELP {METRIC_PREFIX}_partition_info Partition info for table display (always 1)\n")
    lines.append(f"# TYPE {METRIC_PREFIX}_partition_info gauge\n")
    for row in partition_rows:
        pt = _escape_label(row["partition"])
        st = _escape_label(row["state"])
        nd = _escape_label(row["nodes"])
        ct = _escape_label(row["cpus_total"])
        ca = _escape_label(row["cpus_alloc"])
        lines.append(
            f'{METRIC_PREFIX}_partition_info{{partition="{pt}",state="{st}",'
            f'nodes="{nd}",cpus_total="{ct}",cpus_alloc="{ca}"}} 1\n'
        )

    # ---- Recent completed jobs ----
    try:
        recent_rows = collect_recent_jobs()
    except Exception as e:
        logger.error(f"Recent jobs collection failed: {e}")
        recent_rows = []

    lines.append(f"# HELP {METRIC_PREFIX}_recent_job_info Recent completed job info (always 1)\n")
    lines.append(f"# TYPE {METRIC_PREFIX}_recent_job_info gauge\n")
    for row in recent_rows:
        jid = _escape_label(row["job_id"])
        nm = _escape_label(row["name"])
        usr = _escape_label(row["user"])
        st = _escape_label(row["state"])
        el = _escape_label(row["elapsed"])
        ec = _escape_label(row["exit_code"])
        lines.append(
            f'{METRIC_PREFIX}_recent_job_info{{job_id="{jid}",name="{nm}",user="{usr}",'
            f'state="{st}",elapsed="{el}",exit_code="{ec}"}} 1\n'
        )

    # ---- Scheduler metrics ----
    try:
        sched = collect_scheduler_metrics()
    except Exception as e:
        logger.error(f"Scheduler metrics collection failed: {e}")
        sched = {}

    lines.append(f"# HELP {METRIC_PREFIX}_scheduler_backfill_cycle_last_seconds Last backfill cycle time in seconds\n")
    lines.append(f"# TYPE {METRIC_PREFIX}_scheduler_backfill_cycle_last_seconds gauge\n")
    lines.append(
        f"{METRIC_PREFIX}_scheduler_backfill_cycle_last_seconds {sched.get('backfill_cycle_last_seconds', 0)}\n"
    )

    lines.append(f"# HELP {METRIC_PREFIX}_scheduler_backfill_cycle_mean_seconds Mean backfill cycle time in seconds\n")
    lines.append(f"# TYPE {METRIC_PREFIX}_scheduler_backfill_cycle_mean_seconds gauge\n")
    lines.append(
        f"{METRIC_PREFIX}_scheduler_backfill_cycle_mean_seconds {sched.get('backfill_cycle_mean_seconds', 0)}\n"
    )

    lines.append(f"# HELP {METRIC_PREFIX}_scheduler_backfill_jobs_total Jobs started via backfill scheduler\n")
    lines.append(f"# TYPE {METRIC_PREFIX}_scheduler_backfill_jobs_total counter\n")
    lines.append(f"{METRIC_PREFIX}_scheduler_backfill_jobs_total {sched.get('backfill_jobs_total', 0)}\n")

    lines.append(f"# HELP {METRIC_PREFIX}_scheduler_threads_active Active scheduler threads\n")
    lines.append(f"# TYPE {METRIC_PREFIX}_scheduler_threads_active gauge\n")
    lines.append(f"{METRIC_PREFIX}_scheduler_threads_active {sched.get('threads_active', 0)}\n")

    lines.append(f"# HELP {METRIC_PREFIX}_scheduler_dbd_agent_queue_size DBD agent message queue size\n")
    lines.append(f"# TYPE {METRIC_PREFIX}_scheduler_dbd_agent_queue_size gauge\n")
    lines.append(f"{METRIC_PREFIX}_scheduler_dbd_agent_queue_size {sched.get('dbd_agent_queue_size', 0)}\n")

    return "".join(lines)


def collector_loop(interval: int):
    global metrics_output
    while True:
        try:
            data = collect()
            with metrics_lock:
                metrics_output = data
            logger.debug("Metrics collected successfully")
        except Exception as e:
            logger.error(f"Collection error: {e}")
        time.sleep(interval)


class MetricsHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # Suppress access logs

    def do_GET(self):
        if self.path == "/metrics":
            with metrics_lock:
                data = metrics_output
            body = data.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slurm Control Plane Metrics Exporter")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"HTTP port (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Collection interval in seconds (default: {DEFAULT_INTERVAL})",
    )
    args = parser.parse_args()

    logger.info(f"Starting Slurm exporter on port {args.port}, interval {args.interval}s")

    # Initial collection in background before server starts
    t = threading.Thread(target=collector_loop, args=(args.interval,), daemon=True)
    t.start()
    time.sleep(2)  # Brief wait for first collection

    server = http.server.HTTPServer(("", args.port), MetricsHandler)
    logger.info(f"Slurm exporter listening on :{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.shutdown()
