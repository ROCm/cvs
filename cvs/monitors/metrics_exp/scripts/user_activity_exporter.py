#!/usr/bin/env python3
"""
User Activity Metrics Exporter for GPU Nodes

Collects without root access:
  - Currently logged-in users       via 'who'
  - Recent login history            via 'last'
  - Process count per user          via /proc
  - Notable command per user        via /proc (prefers GPU/ML workloads)
  - KFD GPU processes               via /sys/class/kfd/kfd/proc/ + /proc
    (count=0 and empty table if KFD not available or no active GPU processes)

All sources (/proc, /sys, who, last) are world-readable; no root required.
The systemd service runs as root (same as all other exporters in this stack),
which additionally allows reading /var/log paths if needed in future.

Usage:
    python3 user_activity_exporter.py [--port 9420] [--interval 30]
"""

import argparse
import http.server
import logging
import os
import pwd
import re
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("user_activity_exporter")

metrics_output = "# User activity metrics not yet collected\n"
metrics_lock = threading.Lock()

METRIC_PREFIX = "node"
KFD_PROC_DIR = "/sys/class/kfd/kfd/proc"
DEFAULT_PORT = 9420
DEFAULT_INTERVAL = 30
MAX_CMD_LEN = 150
MAX_RECENT_LOGINS = 30

# Users whose processes are not interesting to track
SYSTEM_USERS = frozenset(
    {
        "root",
        "daemon",
        "bin",
        "sys",
        "sync",
        "games",
        "man",
        "lp",
        "mail",
        "news",
        "uucp",
        "proxy",
        "www-data",
        "backup",
        "list",
        "irc",
        "gnats",
        "nobody",
        "systemd-timesync",
        "systemd-network",
        "systemd-resolve",
        "syslog",
        "messagebus",
        "uuidd",
        "dnsmasq",
        "usbmux",
        "rtkit",
        "cups-pk-helper",
        "speech-dispatcher",
        "saned",
        "colord",
        "hplip",
        "whoopsie",
        "avahi",
        "avahi-autoipd",
        "kernoops",
        "pulse",
        "gdm",
        "sshd",
        "ntp",
        "chrony",
        "polkitd",
        "dbus",
        "prometheus",
        "node_exporter",
        "grafana",
        "loki",
        "promtail",
    }
)

# Keywords indicating GPU/ML workloads (used to rank top processes)
GPU_KEYWORDS = {
    "python",
    "pytorch",
    "torch",
    "tensorflow",
    "keras",
    "jax",
    "cuda",
    "rocm",
    "hip",
    "train",
    "nccl",
    "mpi",
    "horovod",
    "triton",
    "vllm",
    "llama",
    "transformers",
    "deepspeed",
    "srun",
    "mpirun",
    "mpiexec",
}


# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────


def _escape_label(v: str) -> str:
    """Escape a string for use as a Prometheus label value."""
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").strip()


def _uid_to_username(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except (KeyError, TypeError):
        return str(uid)


def _get_pid_user(pid: str) -> Optional[str]:
    """Return the username owning a PID, or None if the process is gone."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("Uid:"):
                    uid = int(line.split()[1])
                    return _uid_to_username(uid)
    except FileNotFoundError:
        return None  # process exited between ls and open
    except Exception:
        return None
    return None


def _get_pid_cmdline(pid: str) -> str:
    """Return the command line for a PID (max MAX_CMD_LEN chars)."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read(512)
        cmd = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        return cmd[:MAX_CMD_LEN]
    except Exception:
        return ""


# ─────────────────────────────────────────────────────
# Collectors
# ─────────────────────────────────────────────────────


def collect_logged_in_users() -> List[Dict]:
    """
    Current interactive sessions from 'who'.
    Returns list of {user, tty, from_host, login_time}.
    """
    users = []
    try:
        result = subprocess.run(["who"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0 or not result.stdout.strip():
            return users
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            user = parts[0]
            tty = parts[1]
            login_time = f"{parts[2]} {parts[3]}"
            # who prints the originating host/IP in parentheses at the end
            m = re.search(r'\(([^)]+)\)', line)
            from_host = m.group(1) if m else "local"
            users.append(
                {
                    "user": user,
                    "tty": tty,
                    "login_time": login_time,
                    "from_host": from_host,
                }
            )
    except FileNotFoundError:
        logger.debug("'who' command not found")
    except Exception as e:
        logger.warning(f"collect_logged_in_users error: {e}")
    return users


def collect_recent_logins(max_entries: int = MAX_RECENT_LOGINS) -> List[Dict]:
    """
    Recent login history from 'last -n N'.
    Returns list of {user, tty, from_host, date}.
    Skips system entries (reboot, shutdown, etc.).
    """
    logins = []
    skip_users = {"reboot", "shutdown", "runlevel", "crash", "wtmp", "btmp", ""}
    try:
        result = subprocess.run(
            ["last", "-n", str(max_entries + 10), "-w"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return logins
        for line in result.stdout.splitlines():
            parts = line.split()
            if not parts or parts[0] in skip_users:
                continue
            user = parts[0]
            tty = parts[1] if len(parts) > 1 else "-"
            from_host = parts[2] if len(parts) > 2 else "local"
            from_host = from_host.replace("(", "").replace(")", "")
            # Date/time info: everything after from_host, up to the login/logout marker
            date_info = " ".join(parts[3 : min(9, len(parts))]) if len(parts) > 3 else ""
            logins.append(
                {
                    "user": user,
                    "tty": tty,
                    "from_host": from_host,
                    "date": date_info[:50],
                }
            )
            if len(logins) >= max_entries:
                break
    except FileNotFoundError:
        logger.debug("'last' command not found")
    except Exception as e:
        logger.warning(f"collect_recent_logins error: {e}")
    return logins


def collect_kfd_processes() -> Tuple[List[Dict], int]:
    """
    Collect GPU process info from KFD sysfs (/sys/class/kfd/kfd/proc/).

    Returns (rows, kfd_count) where:
      - rows   = list of {pid, user, cmd, gpu_mem_mb}
      - kfd_count = 0 if KFD unavailable or no active GPU processes

    Handles all cases gracefully:
      - KFD directory missing:  returns ([], 0)
      - KFD directory empty:    returns ([], 0)
      - Process exited mid-read: skips that entry
    """
    if not os.path.isdir(KFD_PROC_DIR):
        logger.debug("KFD proc dir not found — KFD module not loaded or no GPU")
        return [], 0

    try:
        pid_entries = [e for e in os.listdir(KFD_PROC_DIR) if e.isdigit()]
    except Exception as e:
        logger.warning(f"Cannot list {KFD_PROC_DIR}: {e}")
        return [], 0

    if not pid_entries:
        logger.debug("KFD proc dir empty — no active GPU processes")
        return [], 0

    rows = []
    for pid in pid_entries:
        pid_path = os.path.join(KFD_PROC_DIR, pid)

        # Read GPU memory from gpumem file
        gpu_mem_bytes = 0
        gpumem_path = os.path.join(pid_path, "gpumem")
        if os.path.exists(gpumem_path):
            try:
                with open(gpumem_path) as f:
                    content = f.read()
                # The gpumem file contains lines with memory allocation info.
                # Extract all numeric values and sum them as bytes.
                for mem_line in content.splitlines():
                    nums = re.findall(r'\b(\d{4,})\b', mem_line)  # 4+ digit = likely bytes
                    for n in nums:
                        try:
                            gpu_mem_bytes += int(n)
                        except ValueError:
                            pass
            except Exception:
                pass

        # Get owning user — skip if process already exited
        user = _get_pid_user(pid)
        if user is None:
            continue  # process exited between listing and reading

        cmd = _get_pid_cmdline(pid)

        # Double-check process still exists (race condition guard)
        if not os.path.exists(f"/proc/{pid}"):
            continue

        rows.append(
            {
                "pid": pid,
                "user": user,
                "cmd": cmd or "unknown",
                "gpu_mem_mb": str(gpu_mem_bytes // (1024 * 1024)),
            }
        )

    return rows, len(rows)


def collect_user_processes() -> Tuple[Dict[str, int], List[Dict]]:
    """
    Scan /proc to count processes per non-system user and identify
    the most notable command per user (GPU/ML workloads preferred).

    Returns:
      user_counts  - {username: process_count}
      top_rows     - [{user, cmd}] — one notable process per user
    """
    user_counts: Dict[str, int] = {}
    user_top: Dict[str, str] = {}  # user -> best cmd seen so far
    user_top_is_gpu: Dict[str, bool] = {}

    try:
        for pid_entry in os.listdir("/proc"):
            if not pid_entry.isdigit():
                continue
            user = _get_pid_user(pid_entry)
            if user is None or user in SYSTEM_USERS:
                continue

            user_counts[user] = user_counts.get(user, 0) + 1

            # Pick the most interesting command for this user
            cmd = _get_pid_cmdline(pid_entry)
            if not cmd:
                continue
            cmd_lower = cmd.lower()
            is_gpu = any(kw in cmd_lower for kw in GPU_KEYWORDS)

            if user not in user_top:
                user_top[user] = cmd
                user_top_is_gpu[user] = is_gpu
            elif is_gpu and not user_top_is_gpu.get(user, False):
                # Upgrade to GPU workload
                user_top[user] = cmd
                user_top_is_gpu[user] = True
    except Exception as e:
        logger.warning(f"collect_user_processes error: {e}")

    top_rows = [{"user": u, "cmd": cmd} for u, cmd in user_top.items()]
    return user_counts, top_rows


# ─────────────────────────────────────────────────────
# Main collection function
# ─────────────────────────────────────────────────────


def collect() -> str:
    lines = []

    def h(name: str, help_text: str, metric_type: str = "gauge") -> None:
        lines.append(f"# HELP {name} {help_text}\n")
        lines.append(f"# TYPE {name} {metric_type}\n")

    # ── Currently logged-in users ──────────────────────────
    logged_in = collect_logged_in_users()

    h(f"{METRIC_PREFIX}_logged_in_users_count", "Number of users currently logged in (interactive sessions)")
    lines.append(f"{METRIC_PREFIX}_logged_in_users_count {len(logged_in)}\n")

    h(f"{METRIC_PREFIX}_logged_in_user_info", "Currently logged-in user session details (value always 1)")
    for u in logged_in:
        user = _escape_label(u["user"])
        tty = _escape_label(u["tty"])
        from_host = _escape_label(u["from_host"])
        login_time = _escape_label(u["login_time"])
        lines.append(
            f'{METRIC_PREFIX}_logged_in_user_info{{user="{user}",tty="{tty}",'
            f'from_host="{from_host}",login_time="{login_time}"}} 1\n'
        )

    # ── Recent login history ───────────────────────────────
    recent_logins = collect_recent_logins()

    h(f"{METRIC_PREFIX}_recent_login_info", "Recent login history entries (value always 1)")
    for login in recent_logins:
        user = _escape_label(login["user"])
        tty = _escape_label(login["tty"])
        from_host = _escape_label(login["from_host"])
        date = _escape_label(login["date"])
        lines.append(
            f'{METRIC_PREFIX}_recent_login_info{{user="{user}",tty="{tty}",from_host="{from_host}",date="{date}"}} 1\n'
        )

    # ── KFD GPU processes ──────────────────────────────────
    kfd_rows, kfd_count = collect_kfd_processes()

    h(
        f"{METRIC_PREFIX}_kfd_process_count",
        "Number of processes actively using GPU via KFD (0 if KFD unavailable or no GPU processes)",
    )
    lines.append(f"{METRIC_PREFIX}_kfd_process_count {kfd_count}\n")

    h(
        f"{METRIC_PREFIX}_kfd_process_info",
        "Per-process GPU usage details via KFD sysfs (value always 1; gpu_mem_mb in label)",
    )
    for row in kfd_rows:
        user = _escape_label(row["user"])
        pid = _escape_label(row["pid"])
        cmd = _escape_label(row["cmd"])
        gpu_mem_mb = _escape_label(row["gpu_mem_mb"])
        lines.append(
            f'{METRIC_PREFIX}_kfd_process_info{{user="{user}",pid="{pid}",cmd="{cmd}",gpu_mem_mb="{gpu_mem_mb}"}} 1\n'
        )

    # ── Processes per user ─────────────────────────────────
    user_counts, top_rows = collect_user_processes()

    h(f"{METRIC_PREFIX}_user_process_count", "Number of running processes per non-system user")
    for user, count in sorted(user_counts.items(), key=lambda x: -x[1]):
        lines.append(f'{METRIC_PREFIX}_user_process_count{{user="{_escape_label(user)}"}} {count}\n')

    h(
        f"{METRIC_PREFIX}_user_top_process_info",
        "Most notable running process per user, GPU/ML workloads prioritised (value always 1)",
    )
    for row in top_rows:
        user = _escape_label(row["user"])
        cmd = _escape_label(row["cmd"])
        lines.append(f'{METRIC_PREFIX}_user_top_process_info{{user="{user}",cmd="{cmd}"}} 1\n')

    return "".join(lines) if lines else "# no user activity metrics\n"


# ─────────────────────────────────────────────────────
# HTTP server
# ─────────────────────────────────────────────────────


def collector_loop(interval: int) -> None:
    global metrics_output
    while True:
        try:
            data = collect()
            with metrics_lock:
                metrics_output = data
            logger.debug("User activity metrics collected")
        except Exception as e:
            logger.error(f"Collection error: {e}")
        time.sleep(interval)


class MetricsHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request access logs

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
    parser = argparse.ArgumentParser(description="User Activity Metrics Exporter")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"HTTP port (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Collection interval in seconds (default: {DEFAULT_INTERVAL})",
    )
    args = parser.parse_args()

    logger.info(f"Starting user activity exporter on port {args.port}, interval {args.interval}s")
    logger.info(
        f"KFD proc dir: {KFD_PROC_DIR} "
        f"({'found' if os.path.isdir(KFD_PROC_DIR) else 'not present — will check each cycle'})"
    )

    t = threading.Thread(target=collector_loop, args=(args.interval,), daemon=True)
    t.start()
    time.sleep(2)  # brief wait for first collection

    server = http.server.HTTPServer(("", args.port), MetricsHandler)
    logger.info(f"User activity exporter listening on :{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.shutdown()
