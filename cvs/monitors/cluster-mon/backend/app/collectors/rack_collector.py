"""
RackCollector — collects from all three node groups via the Go daemon.

  gpu_nodes        → afmctl show device/port + amd-smi list (compute tray detection)
  scale_up_switches  → SONiC show vlan brief + show mac (asic0/1)
  scale_out_switches → SONiC show vlan brief + show mac (asic0/1)

Compute tray auto-detection:
  1. Run afmctl show device on ALL gpu_nodes via Go daemon.
  2. Valid output  → compute tray (IFoE-capable GPU node).
  3. "not found"  → check /opt/amd/afm/ directory.
  4. Dir present  → compute tray (retry afmctl with full path).
  5. Dir absent   → regular GPU node (run amd-smi list --json instead).

All SSH is via the Go daemon — NO paramiko for node commands.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import re as _re
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.collectors.base import BaseCollector, CollectorResult, CollectorState
from app.collectors.rack_topology import build_topology
from app.core import go_collector
from app.collectors.afmctl_parser import (
    parse_afmctl_show_device,
    parse_afmctl_show_port,
    parse_afmctl_stats_gzip_b64,
)

from app.collectors.sonic_parser import (
    parse_show_vlan_brief,
    parse_show_mac,
    parse_platform_summary,
    parse_platform_psu,
    parse_platform_fan,
    parse_platform_temperature,
    parse_show_version,
    parse_system_health_summary,
    parse_docker_ps,
    parse_docker_stats,
    parse_free,
    parse_interfaces_status,
    parse_counters_json,
    parse_interface_counters_json,
    parse_text_table,
)

# Feature flag: enable PPOD/VPOD topology view (can be disabled for rollback)
ENABLE_PPOD_VPOD_VIEW = os.environ.get("ENABLE_PPOD_VPOD_VIEW", "true").lower() == "true"
logger = logging.getLogger(__name__)

_AFMCTL_MISSING_RE = _re.compile(
    r"command not found|No such file or directory|not found",
    _re.IGNORECASE,
)
_HOST_UNREACHABLE_RE = _re.compile(r"^ABORT", _re.IGNORECASE)


def _afmctl_missing(out: str) -> bool:
    return bool(out) and bool(_AFMCTL_MISSING_RE.search(out[:300]))


def _host_unreachable(out: str) -> bool:
    return not out or bool(_HOST_UNREACHABLE_RE.match(out[:50]))


# ---------------------------------------------------------------------------
# PPOD/VPOD sysfs data collection (only for AFM-admitted compute trays)
# ---------------------------------------------------------------------------

PPOD_VPOD_CMD = """
echo "===HOSTNAME==="; hostname 2>/dev/null || echo "N/A";
echo "===PPOD_ID==="; sudo cat /sys/class/drm/card*/device/ualink/ppod_id 2>/dev/null || echo "N/A";
echo "===VPOD_ID==="; sudo cat /sys/class/drm/card*/device/ualink/vpod_id 2>/dev/null || echo "N/A";
echo "===LOCAL_ACCELS==="; sudo cat /sys/class/drm/card*/device/ualink/local_accels 2>/dev/null || echo "N/A";
echo "===VPOD_ACTIVE_ACCELS==="; sudo cat /sys/class/drm/card*/device/ualink/config/vpod_active_accels 2>/dev/null || echo "N/A";
echo "===LANE_EN_BITMAP==="; sudo cat /sys/class/drm/card*/device/ualink/stations/lane_en_bitmap 2>/dev/null || echo "N/A";
echo "===ACCEL_ID==="; sudo cat /sys/class/drm/card*/device/ualink/accel_id 2>/dev/null || echo "N/A";
echo "===ACCEL_STATE==="; cat /sys/class/drm/card*/device/ualink/accel_state 2>/dev/null || echo "N/A"
"""


def parse_ppod_vpod_output(output: str) -> Dict[str, Any]:
    """
    Parse combined PPOD/VPOD sysfs output.

    Returns:
        {
            'hostname': str or None,     # Hostname of the compute tray
            'ppod_id': str or None,
            'vpod_ids': list[int],       # VPOD IDs for each GPU (4 entries typically)
            'local_accels': list[int],   # Accelerator IDs for local GPUs
            'vpod_active_accels': list[int],  # All active accels in the VPOD
            'lane_en_bitmaps': list[str],     # Hex bitmap per GPU (one per card)
            'accel_ids': list[int],           # Accelerator IDs per GPU
            'accel_states': list[str],        # State per GPU (ready, unconfigured, etc.)
        }
    """
    result: Dict[str, Any] = {
        'hostname': None,
        'ppod_id': None,
        'vpod_ids': [],
        'local_accels': [],
        'vpod_active_accels': [],
        'lane_en_bitmaps': [],
        'accel_ids': [],
        'accel_states': [],
    }

    if not output or output.startswith(("ABORT", "ERROR")):
        return result

    current_section = None
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith('===') and line.endswith('==='):
            current_section = line.strip('=')
            continue
        if not line or line == 'N/A':
            continue

        if current_section == 'HOSTNAME':
            # Hostname, take the first non-empty line
            if not result['hostname']:
                result['hostname'] = line
        elif current_section == 'PPOD_ID':
            # UUID format, take the first non-empty line
            if not result['ppod_id']:
                result['ppod_id'] = line
        elif current_section == 'VPOD_ID':
            # Integer VPOD ID, may have multiple (one per card)
            try:
                result['vpod_ids'].append(int(line))
            except ValueError:
                pass
        elif current_section == 'LOCAL_ACCELS':
            # Comma-separated accelerator IDs or individual lines
            if ',' in line:
                for part in line.split(','):
                    try:
                        result['local_accels'].append(int(part.strip()))
                    except ValueError:
                        pass
            else:
                try:
                    result['local_accels'].append(int(line))
                except ValueError:
                    pass
        elif current_section == 'VPOD_ACTIVE_ACCELS':
            # Comma-separated or space-separated
            for sep in [',', ' ']:
                if sep in line:
                    for part in line.split(sep):
                        try:
                            result['vpod_active_accels'].append(int(part.strip()))
                        except ValueError:
                            pass
                    break
            else:
                try:
                    result['vpod_active_accels'].append(int(line))
                except ValueError:
                    pass
        elif current_section == 'LANE_EN_BITMAP':
            # Hex string bitmap, one per GPU/card
            if line.startswith('0x') or all(c in '0123456789abcdefABCDEF' for c in line):
                result['lane_en_bitmaps'].append(line.upper().replace('0X', ''))
        elif current_section == 'ACCEL_ID':
            # Accelerator ID per GPU (integer)
            try:
                result['accel_ids'].append(int(line))
            except ValueError:
                pass
        elif current_section == 'ACCEL_STATE':
            # State per GPU (ready, unconfigured, etc.)
            result['accel_states'].append(line)

    # Deduplicate lists while preserving order
    result['local_accels'] = list(dict.fromkeys(result['local_accels']))
    result['vpod_active_accels'] = list(dict.fromkeys(result['vpod_active_accels']))

    return result


def _is_error(out: str) -> bool:
    return _afmctl_missing(out) or _host_unreachable(out)


async def _exec_group(
    hosts: List[str],
    cmd: str,
    timeout: int = 60,
    socket_path: str = None,
) -> Dict[str, str]:
    if not hosts:
        return {}
    sp = socket_path or go_collector.CLUSTER_SOCKET
    return await asyncio.to_thread(go_collector._exec_on_hosts, hosts, cmd, timeout, sp)


async def _sonic_group(hosts: List[str]) -> Dict[str, Any]:
    """Run VLAN/MAC commands on switch trays via the Go switch daemon (sequentially)."""
    if not hosts:
        return {"vlan0": {}, "vlan1": {}, "mac0": {}, "mac1": {}}
    v0 = await _exec_group(
        hosts, "sudo ip netns exec asic0 show vlan brief --verbose", socket_path=go_collector.SWITCH_SOCKET
    )
    v1 = await _exec_group(
        hosts, "sudo ip netns exec asic1 show vlan brief --verbose", socket_path=go_collector.SWITCH_SOCKET
    )
    m0 = await _exec_group(hosts, "sudo ip netns exec asic0 show mac", socket_path=go_collector.SWITCH_SOCKET)
    m1 = await _exec_group(hosts, "sudo ip netns exec asic1 show mac", socket_path=go_collector.SWITCH_SOCKET)
    return {"vlan0": v0, "vlan1": v1, "mac0": m0, "mac1": m1}


def _log_cmd_result(cmd: str, result: Dict[str, str]) -> None:
    """Log the first 300 chars of each host's output for debugging."""
    for host, out in result.items():
        preview = (out or '').strip()[:300].replace('\n', '\\n')
        logger.info(f"  CMD [{cmd[:60]}] host={host}: {preview!r}")


async def _sonic_overview_group(hosts: List[str]) -> Dict[str, Any]:
    """
    Collect platform/system overview data from SONiC switch trays.
    Commands run without asic namespace (switch-wide) via SWITCH_SOCKET.
    All are sequential to avoid concurrent socket collisions.
    """
    if not hosts:
        return {}

    async def sw(cmd: str, timeout: int = 30) -> Dict[str, str]:
        logger.info(f"[sonic_overview] running: {cmd}")
        result = await _exec_group(hosts, cmd, timeout=timeout, socket_path=go_collector.SWITCH_SOCKET)
        _log_cmd_result(cmd, result)
        return result

    logger.info(f"[sonic_overview] starting for {hosts}")
    platform_sum = await sw("show platform summary --json")
    psu_status = await sw("show platform psustatus --json")
    fan_status = await sw("show platform fan")  # no --json on AMD-Anacapa
    temperature = await sw("show platform temperature")  # no --json on AMD-Anacapa
    sw_version = await sw("show version")  # SONiC version / build info
    system_status = await sw("sudo show system-health summary")  # health LED + services + hw
    docker_ps = await sw("sudo docker ps --format '{{json .}}'")
    docker_stats = await sw("sudo docker stats --no-stream --format '{{json .}}'", timeout=45)
    free_mem = await sw("free -m")
    # show arp and show mac (global) both exceed 30s timeout on this platform.
    # MAC data is already collected per-ASIC via _sonic_group. Skipping here.
    logger.info("[sonic_overview] done")

    return {
        "platform_sum": platform_sum,
        "psu_status": psu_status,
        "fan_status": fan_status,
        "temperature": temperature,
        "sw_version": sw_version,
        "system_status": system_status,
        "docker_ps": docker_ps,
        "docker_stats": docker_stats,
        "free_mem": free_mem,
    }


async def _sonic_metrics_group(hosts: List[str]) -> Dict[str, Any]:
    """
    Collect interface metrics / counter data from SONiC switch trays.
    Sequential to avoid concurrent SWITCH_SOCKET collisions.
    """
    if not hosts:
        return {}

    async def sw(cmd: str, timeout: int = 30) -> Dict[str, str]:
        logger.info(f"[sonic_metrics] running: {cmd}")
        result = await _exec_group(hosts, cmd, timeout=timeout, socket_path=go_collector.SWITCH_SOCKET)
        _log_cmd_result(cmd, result)
        return result

    logger.info(f"[sonic_metrics] starting for {hosts}")
    intf_status = await sw("show interfaces status")  # no --json on AMD-Anacapa
    intf_counters = await sw("show interface counters --json", timeout=60)  # JSON supported
    pfc_counters = await sw("show pfc counters")  # no --json on AMD-Anacapa
    queue_counters = await sw("show queue counters --json", timeout=120)  # large output
    queue_wm = await sw("show queue watermark unicast")  # watermark subcommand
    logger.info("[sonic_metrics] done")

    return {
        "intf_status": intf_status,
        "intf_counters": intf_counters,
        "pfc_counters": pfc_counters,
        "queue_counters": queue_counters,
        "queue_wm": queue_wm,
    }


class RackCollector(BaseCollector):
    name = "rack"
    poll_interval = 300
    collect_timeout = 360.0  # increased: show queue counters --json takes up to 120s on large switches
    critical = False

    def __init__(self) -> None:
        self._refresh_event: asyncio.Event = asyncio.Event()

    def _node_groups(self, app_state: Any):
        return getattr(app_state, "node_groups", None)

    async def collect(self, ssh_manager) -> CollectorResult:
        app_state = self._app_state
        ng = self._node_groups(app_state)

        if ng is None:
            try:
                from app.core.node_groups import load_node_groups

                ng = load_node_groups()
                app_state.node_groups = ng
            except Exception as e:
                logger.warning(f"RackCollector: cannot load node_groups: {e}")

        if ng is None or not any(
            [
                ng.gpu_nodes.hosts,
                ng.scale_up_switches.hosts,
                ng.scale_out_switches.hosts,
            ]
        ):
            return CollectorResult(
                collector_name=self.name,
                timestamp=CollectorResult.now_iso(),
                state=CollectorState.NO_SERVICE,
                data={},
                error="No hosts configured in node groups",
            )

        if not go_collector.is_daemon_ready():
            return CollectorResult(
                collector_name=self.name,
                timestamp=CollectorResult.now_iso(),
                state=CollectorState.UNREACHABLE,
                data={},
                error="Go SSH daemon not ready — ensure credentials are registered via Configuration page",
            )

        gpu_hosts = ng.gpu_nodes.hosts
        scaleup_hosts = ng.scale_up_switches.hosts
        scaleout_hosts = ng.scale_out_switches.hosts
        errors: Dict[str, str] = {}

        logger.info(
            f"RackCollector: {len(gpu_hosts)} GPU nodes (cluster daemon), "
            f"{len(scaleup_hosts)} scale-up switches, "
            f"{len(scaleout_hosts)} scale-out switches (switch daemon, "
            f"ready={go_collector.is_daemon_ready(go_collector.SWITCH_SOCKET)})"
        )

        # ------------------------------------------------------------------
        # Run all commands concurrently via Go daemon instances:
        #   GPU nodes    → CLUSTER_SOCKET (afmctl show device/port)
        #   Switch trays → SWITCH_SOCKET  (separate daemon, separate creds)
        #
        # NOTE: afmctl port statistics are NOT collected here — the full-fleet
        # JSON output (~668 KB per host) overwhelms the SSH session buffer.
        # Stats are collected per-BDF after compute trays are identified.
        # ------------------------------------------------------------------
        # All SWITCH_SOCKET calls must be fully sequential — the Go daemon's Unix
        # socket is single-stream; concurrent connections collide and return ABORT.
        # CLUSTER_SOCKET (afmctl) and SWITCH_SOCKET are independent, so afmctl
        # commands run concurrently with switch commands via asyncio.gather.
        async def _all_switch_commands():
            su_data = await _sonic_group(scaleup_hosts)
            so_data = await _sonic_group(scaleout_hosts)
            su_overview = await _sonic_overview_group(scaleup_hosts)
            so_overview = await _sonic_overview_group(scaleout_hosts)
            su_metrics = await _sonic_metrics_group(scaleup_hosts)
            so_metrics = await _sonic_metrics_group(scaleout_hosts)
            return su_data, so_data, su_overview, so_overview, su_metrics, so_metrics

        async def _all_cluster_commands():
            rd = await _exec_group(gpu_hosts, "sudo afmctl show device", 60)
            rp = await _exec_group(gpu_hosts, "sudo afmctl show port", 60)
            return rd, rp

        (
            (scaleup_data, scaleout_data, scaleup_overview, scaleout_overview, scaleup_metrics, scaleout_metrics),
            (raw_device, raw_port),
        ) = await asyncio.gather(_all_switch_commands(), _all_cluster_commands())

        # ------------------------------------------------------------------
        # Auto-detect compute trays vs regular GPU nodes
        # ------------------------------------------------------------------
        compute_tray_hosts: List[str] = []
        regular_node_hosts: List[str] = []
        needs_dir_check: List[str] = []

        for host in gpu_hosts:
            out = raw_device.get(host, "")
            if _host_unreachable(out):
                errors[host] = out or "ABORT: no response"
            elif _afmctl_missing(out):
                needs_dir_check.append(host)
            else:
                compute_tray_hosts.append(host)

        # Fallback: check /opt/amd/afm/ for hosts where afmctl not in PATH
        if needs_dir_check:
            logger.info(f"RackCollector: checking /opt/amd/afm/ on {needs_dir_check}")
            dir_results = await _exec_group(
                needs_dir_check,
                "test -d /opt/amd/afm && echo YES || echo NO",
                timeout=20,
            )
            for host in needs_dir_check:
                result = dir_results.get(host, "NO").strip()
                if result == "YES":
                    logger.info(f"RackCollector: {host} is a compute tray (/opt/amd/afm/ found)")
                    compute_tray_hosts.append(host)
                else:
                    regular_node_hosts.append(host)

        logger.info(
            f"RackCollector: {len(compute_tray_hosts)} compute trays, "
            f"{len(regular_node_hosts)} regular GPU nodes, "
            f"{len(errors)} unreachable"
        )

        # For compute trays detected via dir check, retry afmctl with full path
        dir_check_trays = [h for h in compute_tray_hosts if h in needs_dir_check]
        if dir_check_trays:
            retry_device = await _exec_group(dir_check_trays, "sudo /opt/amd/afm/bin/afmctl show device", 60)
            retry_port = await _exec_group(dir_check_trays, "sudo /opt/amd/afm/bin/afmctl show port", 60)
            for host in dir_check_trays:
                raw_device[host] = retry_device.get(host, raw_device.get(host, ""))
                raw_port[host] = retry_port.get(host, raw_port.get(host, ""))

        # ------------------------------------------------------------------
        # Collect GPU list from regular nodes via amd-smi
        # ------------------------------------------------------------------
        raw_gpu_list: Dict[str, str] = {}
        if regular_node_hosts:
            raw_gpu_list = await _exec_group(
                regular_node_hosts,
                "bash -c 'amd-smi list --json 2>/dev/null"
                " || /opt/rocm/bin/amd-smi list --json 2>/dev/null"
                " || cd /opt/rocm/bin && ./amd-smi list --json'",
                timeout=30,
            )

        # ------------------------------------------------------------------
        # Parse compute tray afmctl data (device, port, statistics)
        # ------------------------------------------------------------------
        compute_devices: Dict[str, List[dict]] = {}
        compute_ports: Dict[str, List[dict]] = {}

        for host in compute_tray_hosts:
            out = raw_device.get(host, "")
            compute_devices[host] = parse_afmctl_show_device(out) if not _is_error(out) else []
            out_p = raw_port.get(host, "")
            compute_ports[host] = parse_afmctl_show_port(out_p) if not _is_error(out_p) else []

        # ------------------------------------------------------------------
        # Port statistics — gzip+base64 encoding to keep SSH transfer small.
        #
        # Raw afmctl stats JSON = 668 KB → SSH session drops mid-transfer.
        # Piping through gzip|base64 = ~27 KB → reliable SSH transfer.
        # The --bdf flag is NOT supported for the statistics subcommand
        # (tested: produces empty output). Run all BDFs at once per type.
        #
        # Command:
        #   sudo afmctl show port statistics <type> --json 2>/dev/null | gzip -c | base64 -w0
        #
        # JSON structure returned: {"devices": [{"bdf":"...", "ports":[{...}]}]}
        # Parsed by parse_afmctl_stats_gzip_b64 into flat row list.
        # IFCP skipped: firmware returns "Function not implemented" on all ports.
        # ------------------------------------------------------------------
        compute_port_stats: Dict[str, Dict[str, List[dict]]] = {
            h: {"mac": [], "fec": [], "ifcp": [], "pfc": []} for h in compute_tray_hosts
        }

        if compute_tray_hosts:
            logger.info(
                f"RackCollector stats: collecting mac/fec/pfc for "
                f"{len(compute_tray_hosts)} compute trays via gzip+base64"
            )
            # Sequential — concurrent calls to the same CLUSTER_SOCKET collide.
            mac_out = await _exec_group(
                compute_tray_hosts,
                "sudo afmctl show port statistics mac --json 2>/dev/null | gzip -c | base64 -w0",
                90,
            )
            fec_out = await _exec_group(
                compute_tray_hosts,
                "sudo afmctl show port statistics fec --json 2>/dev/null | gzip -c | base64 -w0",
                90,
            )
            pfc_out = await _exec_group(
                compute_tray_hosts,
                "sudo afmctl show port statistics pfc --json 2>/dev/null | gzip -c | base64 -w0",
                90,
            )

            for host in compute_tray_hosts:
                for stat_type, raw_out in [
                    ("mac", mac_out),
                    ("fec", fec_out),
                    ("pfc", pfc_out),
                ]:
                    raw = raw_out.get(host, "")
                    if raw.startswith(("ABORT", "ERROR")):
                        logger.warning(f"Stats {stat_type} for {host}: {raw[:120]}")
                        continue
                    rows = parse_afmctl_stats_gzip_b64(raw, stat_type)
                    compute_port_stats[host][stat_type] = rows

            for host in compute_tray_hosts:
                logger.info(
                    f"Stats for {host}: "
                    f"mac={len(compute_port_stats[host]['mac'])}, "
                    f"fec={len(compute_port_stats[host]['fec'])}, "
                    f"pfc={len(compute_port_stats[host]['pfc'])} rows"
                )

        # ------------------------------------------------------------------
        # Collect PPOD/VPOD topology data (sysfs files, AFM-admitted only)
        # ------------------------------------------------------------------
        ppod_vpod_data: Dict[str, Dict[str, Any]] = {}

        if ENABLE_PPOD_VPOD_VIEW and compute_tray_hosts:
            logger.info(f"RackCollector: collecting PPOD/VPOD data from {len(compute_tray_hosts)} compute trays")
            ppod_vpod_raw = await _exec_group(compute_tray_hosts, PPOD_VPOD_CMD, 30)

            for host in compute_tray_hosts:
                raw_out = ppod_vpod_raw.get(host, "")
                parsed = parse_ppod_vpod_output(raw_out)
                ppod_vpod_data[host] = parsed
                logger.info(
                    f"PPOD/VPOD for {host}: ppod_id={parsed['ppod_id']}, "
                    f"vpod_ids={parsed['vpod_ids']}, local_accels={parsed['local_accels']}, "
                    f"bitmaps={len(parsed['lane_en_bitmaps'])}"
                )

        # Parse amd-smi list for regular nodes
        regular_node_gpus: Dict[str, list] = {}
        for host in regular_node_hosts:
            out = raw_gpu_list.get(host, "")
            if out and not _host_unreachable(out):
                clean = out.strip()
                for ch in ("[", "{"):
                    idx = clean.find(ch)
                    if idx != -1:
                        clean = clean[idx:]
                        break
                try:
                    parsed = _json.loads(clean)
                    regular_node_gpus[host] = parsed if isinstance(parsed, list) else [parsed]
                except _json.JSONDecodeError:
                    regular_node_gpus[host] = []
            else:
                regular_node_gpus[host] = []

        # ------------------------------------------------------------------
        # Parse switch data
        # ------------------------------------------------------------------
        def _parse_switch(hosts: List[str], sw_data: dict) -> tuple:
            vlan_out: Dict[str, Dict[str, List[dict]]] = {}
            mac_out: Dict[str, Dict[str, List[dict]]] = {}
            for h in hosts:
                vlan_out[h] = {
                    "asic0": parse_show_vlan_brief(sw_data["vlan0"].get(h, ""))
                    if not _is_error(sw_data["vlan0"].get(h, ""))
                    else [],
                    "asic1": parse_show_vlan_brief(sw_data["vlan1"].get(h, ""))
                    if not _is_error(sw_data["vlan1"].get(h, ""))
                    else [],
                }
                mac_out[h] = {
                    "asic0": parse_show_mac(sw_data["mac0"].get(h, ""))
                    if not _is_error(sw_data["mac0"].get(h, ""))
                    else [],
                    "asic1": parse_show_mac(sw_data["mac1"].get(h, ""))
                    if not _is_error(sw_data["mac1"].get(h, ""))
                    else [],
                }
                v0 = sw_data["vlan0"].get(h, "")
                if v0.startswith("ERROR") or v0.startswith("ABORT"):
                    errors[h] = v0
            return vlan_out, mac_out

        scaleup_vlan, scaleup_mac = _parse_switch(scaleup_hosts, scaleup_data)
        scaleout_vlan, scaleout_mac = _parse_switch(scaleout_hosts, scaleout_data)

        all_switch_mac = {**scaleup_mac, **scaleout_mac}
        topology = build_topology(compute_ports, all_switch_mac)

        # Parse switch overview data for each host
        def _parse_overview(hosts: List[str], raw: Dict[str, Any]) -> Dict[str, Any]:
            result: Dict[str, Any] = {}
            for h in hosts:
                parsed = {
                    "platform_summary": parse_platform_summary(raw["platform_sum"].get(h, "")),
                    "psu_status": parse_platform_psu(raw["psu_status"].get(h, "")),
                    "fan_status": parse_platform_fan(raw["fan_status"].get(h, "")),
                    "temperature": parse_platform_temperature(raw["temperature"].get(h, "")),
                    "sw_version": parse_show_version(raw["sw_version"].get(h, "")),
                    "system_status": parse_system_health_summary(raw["system_status"].get(h, "")),
                    "docker_ps": parse_docker_ps(raw["docker_ps"].get(h, "")),
                    "docker_stats": parse_docker_stats(raw["docker_stats"].get(h, "")),
                    "memory": parse_free(raw["free_mem"].get(h, "")),
                    "arp_table": [],
                    "fdb_table": [],
                }
                result[h] = parsed
                logger.info(
                    f"[parse_overview] {h}: platform={bool(parsed['platform_summary'])}, "
                    f"psu={len(parsed['psu_status'])}, fan={len(parsed['fan_status'])}, "
                    f"temp={len(parsed['temperature'])}, docker_ps={len(parsed['docker_ps'])}, "
                    f"docker_stats={len(parsed['docker_stats'])}, "
                    f"arp={len(parsed['arp_table'])}, fdb={len(parsed['fdb_table'])}, "
                    f"mem={bool(parsed['memory'])}"
                )
            return result

        def _parse_metrics(hosts: List[str], raw: Dict[str, Any]) -> Dict[str, Any]:
            result: Dict[str, Any] = {}
            for h in hosts:
                parsed = {
                    "interfaces": parse_interfaces_status(raw["intf_status"].get(h, "")),
                    "intf_counters": parse_interface_counters_json(raw["intf_counters"].get(h, "")),
                    "pfc_counters": parse_text_table(raw["pfc_counters"].get(h, "")),
                    "queue_counters": parse_counters_json(raw["queue_counters"].get(h, ""), "queue"),
                    "queue_wm": parse_text_table(raw["queue_wm"].get(h, "")),
                }
                result[h] = parsed
                logger.info(
                    f"[parse_metrics] {h}: interfaces={len(parsed['interfaces'])}, "
                    f"intf_counters={len(parsed['intf_counters'])}, "
                    f"pfc={len(parsed['pfc_counters'])}, "
                    f"queue={len(parsed['queue_counters'])}, queue_wm={len(parsed['queue_wm'])}"
                )
            return result

        switch_overview = {
            **_parse_overview(scaleup_hosts, scaleup_overview),
            **_parse_overview(scaleout_hosts, scaleout_overview),
        }
        switch_metrics = {
            **_parse_metrics(scaleup_hosts, scaleup_metrics),
            **_parse_metrics(scaleout_hosts, scaleout_metrics),
        }

        data = {
            "compute_trays": compute_devices,
            "compute_ports": compute_ports,
            "compute_port_stats": compute_port_stats,
            "ppod_vpod": ppod_vpod_data,  # PPOD/VPOD topology (AFM-admitted trays only)
            "regular_nodes": regular_node_hosts,
            "regular_node_gpus": regular_node_gpus,
            "scale_up_vlan": scaleup_vlan,
            "scale_up_mac": scaleup_mac,
            "scale_out_vlan": scaleout_vlan,
            "scale_out_mac": scaleout_mac,
            "topology": topology,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "errors": errors,
            # switch overview and metrics
            "switch_overview": switch_overview,
            "switch_metrics": switch_metrics,
            # backward-compat keys used by IFoEDetailsPage
            "compute_devices": compute_devices,
            "switch_vlan": {**scaleup_vlan, **scaleout_vlan},
            "switch_mac": {**scaleup_mac, **scaleout_mac},
        }

        if hasattr(app_state, "latest_rack_data"):
            app_state.latest_rack_data = data

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
        logger.info("RackCollector started")

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
                logger.error(f"RackCollector error: {e}", exc_info=True)
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

            ng = self._node_groups(app_state)
            interval = ng.poll_interval if ng else 300

            try:
                await asyncio.wait_for(self._refresh_event.wait(), timeout=interval)
                self._refresh_event.clear()
                logger.info("RackCollector: manual refresh triggered")
            except asyncio.TimeoutError:
                pass

        logger.info("RackCollector stopped")

    def trigger_refresh(self) -> None:
        self._refresh_event.set()
