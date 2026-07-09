"""
Primus node_smoke preflight check.

Launches ``primus-cli direct -- node_smoke`` on each reachable cluster node in
parallel (no Slurm required) for host / GPU / RDMA roll-call screening.

Reference: Primus ``docs/node-smoke-test-instruction.md`` on branch
``dev/preflight-direct-test``.
"""

from __future__ import annotations

import json
import re
import shlex
from typing import Any, Dict, List, Optional, Tuple

from cvs.lib.preflight.base import PreflightCheck

_JSON_BEGIN = "---CVS_NODE_SMOKE_JSON_BEGIN---"
_JSON_END = "---CVS_NODE_SMOKE_JSON_END---"
_STATUS_RE = re.compile(r"\bstatus=(PASS|FAIL)\b", re.IGNORECASE)


def get_nested_config(config_dict, section, key, default):
    """Read a nested preflight config value (``section.key`` or dotted section)."""
    if not config_dict:
        return default

    sections = section.split(".")
    current = config_dict
    for sec in sections:
        if isinstance(current, dict) and sec in current:
            current = current[sec]
        else:
            return default

    if isinstance(current, dict) and key in current:
        return current[key]
    return default


def _config_flag_enabled(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _normalize_mode(mode) -> str:
    if isinstance(mode, str):
        return mode.strip().lower()
    return "skip" if not mode else "run"


def _resolve_dump_path(cfg: dict) -> str:
    """Return node_smoke dump directory; empty config uses reporting artifacts root."""
    artifacts_root = get_nested_config(cfg, "reporting", "artifacts_root_dir", "/tmp/preflight")
    default_dump = f"{str(artifacts_root).rstrip('/')}/node_smoke"
    configured = get_nested_config(cfg, "node_smoke", "dump_path", default_dump)
    configured_s = str(configured or "").strip()
    return configured_s if configured_s else default_dump


def build_node_smoke_flags(
    *,
    dump_path: str = "output/preflight",
    expected_gpus: Optional[int] = None,
    expected_rdma_nics: Optional[int] = None,
    rdma_nic_allowlist: Optional[str] = None,
    ulimit_l_min_gb: Optional[float] = None,
    shm_min_gb: Optional[float] = None,
    skip_dmesg: bool = False,
    allow_foreign_procs: bool = False,
    allowed_procs: Optional[str] = None,
    require_tools: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> str:
    """Build primus-cli ``node_smoke`` CLI flags."""
    effective_dump = str(dump_path or "").strip() or "output/preflight"
    flags: List[str] = [f"--dump-path {shlex.quote(effective_dump)}"]

    if expected_gpus is not None and int(expected_gpus) > 0:
        flags.append(f"--expected-gpus {int(expected_gpus)}")

    if expected_rdma_nics is not None and int(expected_rdma_nics) > 0:
        flags.append(f"--expected-rdma-nics {int(expected_rdma_nics)}")

    if rdma_nic_allowlist:
        flags.append(f"--rdma-nic-allowlist {shlex.quote(str(rdma_nic_allowlist))}")

    if ulimit_l_min_gb is not None:
        flags.append(f"--ulimit-l-min-gb {float(ulimit_l_min_gb)}")

    if shm_min_gb is not None:
        flags.append(f"--shm-min-gb {float(shm_min_gb)}")

    if skip_dmesg:
        flags.append("--skip-dmesg")

    if allow_foreign_procs:
        flags.append("--allow-foreign-procs")

    if allowed_procs is not None:
        flags.append(f"--allowed-procs {shlex.quote(str(allowed_procs))}")

    if require_tools:
        flags.append(f"--require-tools {shlex.quote(str(require_tools))}")

    if extra_args:
        for arg in extra_args:
            if arg:
                flags.append(str(arg))

    return " ".join(flags)


def build_remote_node_smoke_command(
    *,
    primus_dir: str,
    venv_activate: str,
    node_rank: int,
    nnodes: int,
    master_addr: str,
    master_port: int,
    gpus_per_node: int,
    dump_path: str,
    smoke_flags: str,
    nccl_socket_ifname: Optional[str] = None,
    gloo_socket_ifname: Optional[str] = None,
    nccl_ib_hca: Optional[str] = None,
    nccl_ib_gid_index: Optional[int] = None,
) -> str:
    """Build the remote shell command for one node's node_smoke run."""
    primus_q = shlex.quote(primus_dir)
    venv_q = shlex.quote(venv_activate)
    effective_dump = str(dump_path or "").strip() or "output/preflight"
    dump_q = shlex.quote(effective_dump)

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
    json_cat = (
        f"echo '{_JSON_BEGIN}'; "
        f"json=$(ls -1 {dump_q}/smoke/*.json 2>/dev/null | head -1); "
        f'if [ -n "$json" ]; then cat "$json"; fi; '
        f"echo '{_JSON_END}'"
    )

    return (
        f"cd {primus_q} && "
        f"{' && '.join(env_lines)} && "
        f"{primus_cli} direct --single -- node_smoke {smoke_flags}; "
        f"rc=$?; {json_cat}; exit $rc"
    )


def parse_node_smoke_output(output: str) -> Dict[str, Any]:
    """Parse node_smoke stdout for status and optional embedded JSON payload."""
    result: Dict[str, Any] = {
        "status": "UNKNOWN",
        "fail_reasons": [],
        "node_payload": None,
        "raw_status_line": None,
    }

    if not output or not str(output).strip():
        result["fail_reasons"].append("empty output from node_smoke")
        result["status"] = "FAIL"
        return result

    text = str(output)

    begin = text.find(_JSON_BEGIN)
    end = text.find(_JSON_END)
    if begin != -1 and end != -1 and end > begin:
        json_blob = text[begin + len(_JSON_BEGIN) : end].strip()
        if json_blob:
            try:
                payload = json.loads(json_blob)
                result["node_payload"] = payload
                result["status"] = str(payload.get("status", "UNKNOWN")).upper()
                result["fail_reasons"] = list(payload.get("fail_reasons") or [])
            except json.JSONDecodeError as exc:
                result["fail_reasons"].append(f"failed to parse node_smoke JSON: {exc}")

    if result["status"] == "UNKNOWN":
        matches = _STATUS_RE.findall(text)
        if matches:
            result["raw_status_line"] = matches[-1]
            result["status"] = matches[-1].upper()
        elif "ABORT: Host Unreachable Error" in text:
            result["status"] = "FAIL"
            result["fail_reasons"].append("SSH unreachable")
        elif "argument --dump-path: expected one argument" in text:
            result["status"] = "FAIL"
            result["fail_reasons"].append(
                "invalid --dump-path (set node_smoke.dump_path or leave it empty to use artifacts_root_dir/node_smoke)"
            )
        else:
            result["status"] = "FAIL"
            result["fail_reasons"].append("could not determine node_smoke status from output")

    return result


class NodeSmokeCheck(PreflightCheck):
    """Run Primus node_smoke checks across cluster nodes via parallel SSH."""

    def __init__(self, phdl, node_list: List[str], config_dict=None):
        super().__init__(phdl, config_dict)
        self.node_list = list(node_list)
        self._load_settings()

    def _load_settings(self):
        cfg = self.config_dict or {}
        node_check = cfg.get("node_check") or {}

        self.mode = _normalize_mode(get_nested_config(cfg, "node_smoke", "connectivity_mode", "skip"))
        self.primus_dir = get_nested_config(cfg, "node_smoke", "primus_dir", "")
        self.venv_activate = get_nested_config(cfg, "node_smoke", "venv_activate", "")
        self.gpus_per_node = int(get_nested_config(cfg, "node_smoke", "gpus_per_node", 8))
        self.master_port = int(get_nested_config(cfg, "node_smoke", "master_port", 1234))
        self.ssh_timeout = int(get_nested_config(cfg, "node_smoke", "ssh_timeout", 300))

        artifacts_root = get_nested_config(cfg, "reporting", "artifacts_root_dir", "/tmp/preflight")
        self.dump_path = _resolve_dump_path(cfg)

        rdma_ifaces = node_check.get("rdma_interfaces") or []
        default_rdma_nics = len(rdma_ifaces) if rdma_ifaces else None
        expected_rdma = get_nested_config(cfg, "node_smoke", "expected_rdma_nics", default_rdma_nics)
        self.expected_rdma_nics = int(expected_rdma) if expected_rdma not in (None, "", 0) else None

        self.ulimit_l_min_gb = float(get_nested_config(cfg, "node_smoke", "ulimit_l_min_gb", 32.0))
        self.shm_min_gb = float(get_nested_config(cfg, "node_smoke", "shm_min_gb", 8.0))
        self.skip_dmesg = _config_flag_enabled(get_nested_config(cfg, "node_smoke", "skip_dmesg", False))
        self.allow_foreign_procs = _config_flag_enabled(
            get_nested_config(cfg, "node_smoke", "allow_foreign_procs", False)
        )
        self.allowed_procs = get_nested_config(
            cfg,
            "node_smoke",
            "allowed_procs",
            "gpuagent,rocm-smi-daemon,amd-smi,dcgm-exporter",
        )
        self.require_tools = get_nested_config(cfg, "node_smoke", "require_tools", "")

        self.nccl_socket_ifname = get_nested_config(cfg, "node_smoke", "nccl_socket_ifname", "") or None
        self.gloo_socket_ifname = get_nested_config(
            cfg, "node_smoke", "gloo_socket_ifname", self.nccl_socket_ifname
        ) or None

        rdma_allowlist = get_nested_config(cfg, "node_smoke", "rdma_nic_allowlist", None)
        if not rdma_allowlist and rdma_ifaces:
            rdma_allowlist = ",".join(rdma_ifaces)
        self.rdma_nic_allowlist = rdma_allowlist or None

        nccl_ib_hca = get_nested_config(cfg, "node_smoke", "nccl_ib_hca", None)
        if not nccl_ib_hca and rdma_ifaces:
            nccl_ib_hca = ",".join(rdma_ifaces)
        self.nccl_ib_hca = nccl_ib_hca or None

        gid_index = get_nested_config(cfg, "node_smoke", "nccl_ib_gid_index", None)
        if gid_index is None:
            gid_index = node_check.get("gid_index")
        self.nccl_ib_gid_index = int(gid_index) if gid_index not in (None, "") else None

        extra = get_nested_config(cfg, "node_smoke", "extra_args", [])
        self.extra_args = [str(arg) for arg in extra if arg] if isinstance(extra, (list, tuple)) else []
        self.auto_setup = _config_flag_enabled(get_nested_config(cfg, "node_smoke", "auto_setup", True), default=True)

    def _validate_prerequisites(self) -> Optional[str]:
        if not self.primus_dir:
            return "node_smoke.primus_dir is required when connectivity_mode is 'run'"
        if not self.venv_activate:
            return "node_smoke.venv_activate is required when connectivity_mode is 'run'"
        if not self.node_list:
            return "no reachable nodes available for node_smoke"
        return None

    def _smoke_flags(self) -> str:
        return build_node_smoke_flags(
            dump_path=self.dump_path,
            expected_gpus=self.gpus_per_node,
            expected_rdma_nics=self.expected_rdma_nics,
            rdma_nic_allowlist=self.rdma_nic_allowlist,
            ulimit_l_min_gb=self.ulimit_l_min_gb,
            shm_min_gb=self.shm_min_gb,
            skip_dmesg=self.skip_dmesg,
            allow_foreign_procs=self.allow_foreign_procs,
            allowed_procs=self.allowed_procs,
            require_tools=self.require_tools,
            extra_args=self.extra_args,
        )

    def run(self) -> Dict[str, Any]:
        if self.mode in ("skip", "off", "disabled", "false", "0"):
            return {
                "mode": self.mode,
                "skipped": True,
                "message": "Primus node_smoke check skipped by configuration",
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
                "message": "no reachable hosts remain for node_smoke",
                "node_results": {},
            }

        setup_results = None
        if self.auto_setup:
            from cvs.lib.preflight.primus_setup import PrimusSetup

            setup = PrimusSetup(self.phdl, hosts, self.config_dict)
            setup_results = setup.run()
            if setup_results.get("status") == "FAIL":
                return {
                    "mode": self.mode,
                    "skipped": True,
                    "status": "FAIL",
                    "message": "Primus auto_setup failed — fix setup errors before node_smoke",
                    "setup_results": setup_results,
                    "node_results": {},
                }

        nnodes = len(hosts)
        master_addr = hosts[0]
        smoke_flags = self._smoke_flags()

        self.log_info(
            f"Launching Primus node_smoke on {nnodes} node(s) "
            f"(primus_dir={self.primus_dir}, dump_path={self.dump_path})"
        )

        commands: List[str] = []
        for rank, host in enumerate(hosts):
            commands.append(
                build_remote_node_smoke_command(
                    primus_dir=self.primus_dir,
                    venv_activate=self.venv_activate,
                    node_rank=rank,
                    nnodes=nnodes,
                    master_addr=master_addr,
                    master_port=self.master_port,
                    gpus_per_node=self.gpus_per_node,
                    dump_path=self.dump_path,
                    smoke_flags=smoke_flags,
                    nccl_socket_ifname=self.nccl_socket_ifname,
                    gloo_socket_ifname=self.gloo_socket_ifname,
                    nccl_ib_hca=self.nccl_ib_hca,
                    nccl_ib_gid_index=self.nccl_ib_gid_index,
                )
            )

        out_dict = self.phdl.exec_cmd_list(commands, timeout=self.ssh_timeout)

        node_results: Dict[str, Any] = {}
        for host, output in out_dict.items():
            parsed = parse_node_smoke_output(output)
            node_results[host] = {
                "status": parsed["status"],
                "fail_reasons": parsed["fail_reasons"],
                "node_rank": hosts.index(host) if host in hosts else -1,
                "node_payload": parsed.get("node_payload"),
            }
            if parsed["status"] == "FAIL":
                for reason in parsed["fail_reasons"]:
                    self.log_error(f"Node {host} node_smoke: {reason}")

        failed_nodes = [n for n, r in node_results.items() if r.get("status") == "FAIL"]
        passing_nodes = [n for n, r in node_results.items() if r.get("status") == "PASS"]
        unknown_nodes = [n for n, r in node_results.items() if r.get("status") not in ("PASS", "FAIL")]

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
            "primus_dir": self.primus_dir,
        }
        if setup_results is not None:
            self.results["setup_results"] = setup_results
        return self.results
