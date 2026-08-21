"""
Primus preflight Tier 3 info checks.

Launches ``primus-cli direct -- preflight --host --gpu --network`` across
reachable cluster nodes in parallel via SSH (no Slurm).  Uses torchrun on each
node with a shared ``MASTER_ADDR`` / ``NODE_RANK`` rendezvous so Primus can
aggregate the Host / GPU / Network info report on rank 0.

Reference: Primus ``docs/02-user-guide/preflight.md`` on branch
``dev/preflight-direct-test``.
"""

from __future__ import annotations

import re
import shlex
from typing import Any, Dict, List, Optional

from cvs.lib.preflight.base import PreflightCheck
from cvs.lib.preflight.node_smoke import (
    DEFAULT_ARTIFACTS_ROOT_DIR,
    _config_flag_enabled,
    _normalize_mode,
    get_nested_config,
    resolve_rdma_gid_index,
    resolve_rdma_interfaces,
)

_REPORT_BEGIN = "---CVS_TIER3_REPORT_BEGIN---"
_REPORT_END = "---CVS_TIER3_REPORT_END---"
_STATUS_RE = re.compile(
    r"\[Primus:Preflight\]\s+checks=(?P<checks>[^\s]+)\s+host=(?P<host>\S+)\s+status=(?P<status>PASS|FAIL|WARN)",
    re.IGNORECASE,
)
_FINDINGS_RE = re.compile(
    r"\[Primus:Preflight\]\s+(?P<level>FAIL|WARN):\s+(?P<message>.+)$",
    re.IGNORECASE,
)

# Only shared Primus install paths may inherit from node_smoke. Operational knobs
# (connectivity_mode, timeouts, dump_path, NCCL overrides, etc.) must stay
# tier3_info-local so enabling node_smoke does not silently launch Tier 3.
_TIER3_NODE_SMOKE_FALLBACK_KEYS = frozenset({"primus_dir", "venv_activate"})


def resolve_tier3_setting(cfg: dict, key: str, default=None):
    """Read ``tier3_info.<key>``, with a narrow ``node_smoke`` fallback for Primus paths."""
    value = get_nested_config(cfg, "tier3_info", key, None)
    if value not in (None, ""):
        return value
    if key in _TIER3_NODE_SMOKE_FALLBACK_KEYS:
        fallback = get_nested_config(cfg, "node_smoke", key, None)
        if fallback not in (None, ""):
            return fallback
    return default


def _resolve_dump_path(cfg: dict) -> str:
    """Return Tier 3 dump directory; empty config uses reporting artifacts root."""
    artifacts_root = get_nested_config(cfg, "reporting", "artifacts_root_dir", DEFAULT_ARTIFACTS_ROOT_DIR)
    default_dump = f"{str(artifacts_root).rstrip('/')}/tier3_info"
    configured = resolve_tier3_setting(cfg, "dump_path", default_dump)
    configured_s = str(configured or "").strip()
    return configured_s if configured_s else default_dump


def build_preflight_info_flags(
    *,
    dump_path: str = "output/preflight",
    report_file_name: str = "tier3_info",
    dist_timeout_sec: Optional[int] = None,
    save_pdf: bool = False,
    extra_args: Optional[List[str]] = None,
) -> str:
    """Build ``primus-cli preflight --host --gpu --network`` CLI flags."""
    effective_dump = str(dump_path or "").strip() or "output/preflight"
    flags: List[str] = [
        "--host",
        "--gpu",
        "--network",
        f"--dump-path {shlex.quote(effective_dump)}",
        f"--report-file-name {shlex.quote(str(report_file_name or 'tier3_info'))}",
    ]

    if dist_timeout_sec is not None and int(dist_timeout_sec) > 0:
        flags.append(f"--dist-timeout-sec {int(dist_timeout_sec)}")

    if not save_pdf:
        flags.append("--disable-pdf")

    if extra_args:
        for arg in extra_args:
            if arg:
                flags.append(str(arg))

    return " ".join(flags)


def build_remote_preflight_info_command(
    *,
    primus_dir: str,
    venv_activate: str,
    node_rank: int,
    nnodes: int,
    master_addr: str,
    master_port: int,
    gpus_per_node: int,
    dump_path: str,
    preflight_flags: str,
    report_file_name: str,
    nccl_socket_ifname: Optional[str] = None,
    gloo_socket_ifname: Optional[str] = None,
    nccl_ib_hca: Optional[str] = None,
    nccl_ib_gid_index: Optional[int] = None,
) -> str:
    """Build the remote shell command for one node's Tier 3 preflight run."""
    primus_q = shlex.quote(primus_dir)
    venv_q = shlex.quote(venv_activate)
    effective_dump = str(dump_path or "").strip() or "output/preflight"
    report_md_q = shlex.quote(f"{effective_dump.rstrip('/')}/{report_file_name}.md")

    env_lines = [
        f"export VENV_ACTIVATE={venv_q}",
        f"export NNODES={nnodes}",
        f"export NODE_RANK={node_rank}",
        f"export MASTER_ADDR={shlex.quote(master_addr)}",
        f"export MASTER_PORT={master_port}",
        f"export GPUS_PER_NODE={gpus_per_node}",
    ]
    if nccl_socket_ifname:
        env_lines.append(f"export NCCL_SOCKET_IFNAME={shlex.quote(nccl_socket_ifname)}")
    if gloo_socket_ifname:
        env_lines.append(f"export GLOO_SOCKET_IFNAME={shlex.quote(gloo_socket_ifname)}")
    if nccl_ib_hca:
        env_lines.append(f"export NCCL_IB_HCA={shlex.quote(nccl_ib_hca)}")
    if nccl_ib_gid_index is not None:
        env_lines.append(f"export NCCL_IB_GID_INDEX={int(nccl_ib_gid_index)}")

    primus_cli = f"{primus_q}/runner/primus-cli"
    run_cmd = f"{primus_cli} direct -- preflight {preflight_flags}"

    report_cat = f"echo '{_REPORT_BEGIN}'; if [ -f {report_md_q} ]; then cat {report_md_q}; fi; echo '{_REPORT_END}'"

    if node_rank == 0:
        return f"cd {primus_q} && {' && '.join(env_lines)} && {run_cmd}; rc=$?; {report_cat}; exit $rc"

    return f"cd {primus_q} && {' && '.join(env_lines)} && {run_cmd}"


def _worst_status(current: str, new: str) -> str:
    order = {"FAIL": 3, "WARN": 2, "PASS": 1, "UNKNOWN": 0}
    cur = str(current or "UNKNOWN").upper()
    nxt = str(new or "UNKNOWN").upper()
    return nxt if order.get(nxt, 0) > order.get(cur, 0) else cur


def parse_preflight_info_output(output: str) -> Dict[str, Any]:
    """Parse Tier 3 preflight stdout/stderr for per-host status and report text."""
    result: Dict[str, Any] = {
        "status": "UNKNOWN",
        "checks": [],
        "fail_reasons": [],
        "host_statuses": {},
        "report_markdown": None,
    }

    if not output or not str(output).strip():
        result["fail_reasons"].append("empty output from preflight --host --gpu --network")
        result["status"] = "FAIL"
        return result

    text = str(output)

    begin = text.find(_REPORT_BEGIN)
    end = text.find(_REPORT_END)
    if begin != -1 and end != -1 and end > begin:
        report_blob = text[begin + len(_REPORT_BEGIN) : end].strip()
        if report_blob:
            result["report_markdown"] = report_blob

    host_statuses: Dict[str, str] = {}
    checks_seen: List[str] = []
    for match in _STATUS_RE.finditer(text):
        host = match.group("host")
        status = match.group("status").upper()
        checks = match.group("checks")
        if checks and checks not in checks_seen:
            checks_seen.append(checks)
        host_statuses[host] = _worst_status(host_statuses.get(host, "UNKNOWN"), status)

    for match in _FINDINGS_RE.finditer(text):
        level = match.group("level").upper()
        message = match.group("message").strip()
        if level == "FAIL":
            result["fail_reasons"].append(message)

    if "ABORT: Host Unreachable Error" in text:
        result["status"] = "FAIL"
        result["fail_reasons"].append("SSH unreachable")
        return result

    if "[Primus:Preflight] ERROR: distributed init failed" in text:
        result["status"] = "FAIL"
        result["fail_reasons"].append("distributed init failed during Tier 3 preflight")
        return result

    result["host_statuses"] = host_statuses
    result["checks"] = checks_seen

    if host_statuses:
        statuses = list(host_statuses.values())
        if any(status == "FAIL" for status in statuses):
            result["status"] = "FAIL"
        elif any(status == "WARN" for status in statuses):
            result["status"] = "WARN"
        else:
            result["status"] = "PASS"
    elif result["fail_reasons"]:
        result["status"] = "FAIL"
    else:
        result["status"] = "FAIL"
        result["fail_reasons"].append("could not determine Tier 3 preflight status from output")

    return result


class Tier3InfoCheck(PreflightCheck):
    """Run Primus preflight Host/GPU/Network info checks across cluster nodes via SSH."""

    CONFIG_SECTION = "tier3_info"

    def __init__(self, phdl, node_list: List[str], config_dict=None):
        super().__init__(phdl, config_dict)
        self.node_list = list(node_list)
        self._load_settings()

    def _load_settings(self):
        cfg = self.config_dict or {}

        self.mode = _normalize_mode(get_nested_config(cfg, "tier3_info", "connectivity_mode", "skip"))
        self.primus_dir = str(resolve_tier3_setting(cfg, "primus_dir", "") or "")
        self.venv_activate = str(resolve_tier3_setting(cfg, "venv_activate", "") or "")
        self.gpus_per_node = int(resolve_tier3_setting(cfg, "gpus_per_node", 8))
        self.master_port = int(resolve_tier3_setting(cfg, "master_port", 1234))
        self.ssh_timeout = int(resolve_tier3_setting(cfg, "ssh_timeout", 600))
        self.dist_timeout_sec = int(resolve_tier3_setting(cfg, "dist_timeout_sec", 120))
        self.report_file_name = str(resolve_tier3_setting(cfg, "report_file_name", "tier3_info") or "tier3_info")
        self.save_pdf = _config_flag_enabled(resolve_tier3_setting(cfg, "save_pdf", False))
        self.dump_path = _resolve_dump_path(cfg)

        rdma_ifaces = resolve_rdma_interfaces(cfg)
        self.nccl_socket_ifname = resolve_tier3_setting(cfg, "nccl_socket_ifname", "") or None
        self.gloo_socket_ifname = resolve_tier3_setting(cfg, "gloo_socket_ifname", self.nccl_socket_ifname) or None

        nccl_ib_hca = resolve_tier3_setting(cfg, "nccl_ib_hca", None)
        if not nccl_ib_hca and rdma_ifaces:
            nccl_ib_hca = ",".join(rdma_ifaces)
        self.nccl_ib_hca = nccl_ib_hca or None

        gid_index = resolve_tier3_setting(cfg, "nccl_ib_gid_index", None)
        if gid_index is None:
            gid_index = resolve_rdma_gid_index(cfg)
        self.nccl_ib_gid_index = int(gid_index) if gid_index not in (None, "") else None

        extra = resolve_tier3_setting(cfg, "extra_args", [])
        self.extra_args = [str(arg) for arg in extra if arg] if isinstance(extra, (list, tuple)) else []
        self.auto_setup = _config_flag_enabled(resolve_tier3_setting(cfg, "auto_setup", True), default=True)

    def _validate_prerequisites(self) -> Optional[str]:
        if not self.primus_dir:
            return "tier3_info.primus_dir (or node_smoke.primus_dir) is required when connectivity_mode is 'run'"
        if not self.venv_activate:
            return "tier3_info.venv_activate (or node_smoke.venv_activate) is required when connectivity_mode is 'run'"
        if not self.node_list:
            return "no reachable nodes available for Tier 3 preflight info"
        return None

    def _preflight_flags(self) -> str:
        return build_preflight_info_flags(
            dump_path=self.dump_path,
            report_file_name=self.report_file_name,
            dist_timeout_sec=self.dist_timeout_sec,
            save_pdf=self.save_pdf,
            extra_args=self.extra_args,
        )

    def run(self) -> Dict[str, Any]:
        if self.mode in ("skip", "off", "disabled", "false", "0"):
            return {
                "mode": self.mode,
                "skipped": True,
                "message": "Primus Tier 3 preflight info check skipped by configuration",
                "node_results": {},
            }

        err = self._validate_prerequisites()
        if err:
            return {
                "mode": self.mode,
                "skipped": True,
                "message": err,
                "node_results": {},
            }

        hosts = [h for h in self.node_list if h in self.phdl.reachable_hosts]
        if not hosts:
            return {
                "mode": self.mode,
                "skipped": True,
                "message": "no reachable hosts remain for Tier 3 preflight info",
                "node_results": {},
            }

        setup_results = None
        if self.auto_setup:
            from cvs.lib.preflight.primus_setup import PrimusSetup

            setup = PrimusSetup(
                self.phdl,
                hosts,
                self.config_dict,
                config_section=self.CONFIG_SECTION,
                fallback_section="node_smoke",
            )
            setup_results = setup.run()
            if setup_results.get("status") == "FAIL":
                return {
                    "mode": self.mode,
                    "skipped": True,
                    "status": "FAIL",
                    "message": "Primus auto_setup failed — fix setup errors before Tier 3 preflight info",
                    "setup_results": setup_results,
                    "node_results": {},
                }

        nnodes = len(hosts)
        master_addr = hosts[0]
        preflight_flags = self._preflight_flags()
        hosts_set = set(hosts)
        host_ranks = {host: rank for rank, host in enumerate(hosts)}

        self.log_info(
            f"Launching Primus preflight --host --gpu --network on {nnodes} node(s) "
            f"(primus_dir={self.primus_dir}, dump_path={self.dump_path}, "
            f"dist_timeout_sec={self.dist_timeout_sec})"
        )

        commands: List[str] = []
        for h in self.phdl.reachable_hosts:
            if h not in hosts_set:
                commands.append("true")
            else:
                commands.append(
                    build_remote_preflight_info_command(
                        primus_dir=self.primus_dir,
                        venv_activate=self.venv_activate,
                        node_rank=host_ranks[h],
                        nnodes=nnodes,
                        master_addr=master_addr,
                        master_port=self.master_port,
                        gpus_per_node=self.gpus_per_node,
                        dump_path=self.dump_path,
                        preflight_flags=preflight_flags,
                        report_file_name=self.report_file_name,
                        nccl_socket_ifname=self.nccl_socket_ifname,
                        gloo_socket_ifname=self.gloo_socket_ifname,
                        nccl_ib_hca=self.nccl_ib_hca,
                        nccl_ib_gid_index=self.nccl_ib_gid_index,
                    )
                )

        out_dict = self.phdl.exec_cmd_list(commands, timeout=self.ssh_timeout)

        node_results: Dict[str, Any] = {}
        cluster_report = None
        for host, output in out_dict.items():
            if host not in hosts_set:
                continue
            parsed = parse_preflight_info_output(output)
            node_results[host] = {
                "status": parsed["status"],
                "fail_reasons": parsed["fail_reasons"],
                "node_rank": host_ranks[host],
                "host_statuses": parsed.get("host_statuses") or {},
                "checks": parsed.get("checks") or [],
            }
            if parsed.get("report_markdown"):
                cluster_report = parsed["report_markdown"]
            if parsed["status"] == "FAIL":
                for reason in parsed["fail_reasons"]:
                    self.log_error(f"Node {host} Tier 3 preflight info: {reason}")

        failed_nodes = [n for n, r in node_results.items() if r.get("status") == "FAIL"]
        passing_nodes = [n for n, r in node_results.items() if r.get("status") == "PASS"]
        unknown_nodes = [n for n, r in node_results.items() if r.get("status") not in ("PASS", "FAIL", "WARN")]

        summary_status = "FAIL" if failed_nodes or unknown_nodes else "PASS"
        self.results = {
            "mode": self.mode,
            "skipped": False,
            "status": summary_status,
            "total_nodes": len(node_results),
            "passing_nodes": passing_nodes,
            "failed_nodes": failed_nodes,
            "unknown_nodes": unknown_nodes,
            "node_results": node_results,
            "dump_path": self.dump_path,
            "report_file_name": self.report_file_name,
            "report_markdown": cluster_report,
            "primus_dir": self.primus_dir,
            "dist_timeout_sec": self.dist_timeout_sec,
        }
        if setup_results is not None:
            self.results["setup_results"] = setup_results
        return self.results
