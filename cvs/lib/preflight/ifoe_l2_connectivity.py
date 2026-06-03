"""
IFoE L2 Connectivity Check (AIMVT-180).

Validates L2 connectivity by invoking ``afmctl test ping`` on each
reachable node and parsing the per-port pass/fail counts and the
aggregate ``Summary:`` section that ``afmctl`` emits.

This is a *per-node* preflight check: each node runs one
``afmctl test ping`` invocation per configured ``(bdf, dst_accelerator)``
pairing. The check requires no pairwise SSH coordination because
``afmctl`` drives the request/response state machine in the device and
reports a per-port pass/fail table plus an aggregate Summary that we
surface to the operator.

Example command issued on each node::

    afmctl test ping -b 0001:01:00.1 -c 1 --dst-accelerator 0

Example output parsed by :class:`AfmctlPingParser`::

    0001:01:00.1                   : Ping test results (1 pings per port pair)
    Accel ID    Port#     IFoE Req        IFoE Rsp        Non-IFoE
    --------    -----     --------        ---------       --------
    0           0         1/1 PASS        1/1 PASS        1/1 PASS

    Summary:
      IFoE Request    : 1/1 PASS, 0/1 fail (0.00% loss)
      IFoE Response   : 1/1 PASS, 0/1 fail (0.00% loss)
      Non-IFoE        : 1/1 PASS, 0/1 fail (0.00% loss)
"""

from __future__ import annotations

import re
import shlex
from typing import Dict, Iterable, List, Optional, Tuple

from cvs.lib.preflight.base import PreflightCheck

TRAFFIC_TYPES: Tuple[str, str, str] = ("ifoe_req", "ifoe_resp", "non_ifoe")

TRAFFIC_LABELS: Dict[str, str] = {
    "ifoe_req": "IFoE Request",
    "ifoe_resp": "IFoE Response",
    "non_ifoe": "Non-IFoE",
}

_BDF_PATTERN = re.compile(r"^([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])\s*:")

_PER_PORT_PATTERN = re.compile(
    r"""^\s*
        (?P<accel>\d+)\s+
        (?P<port>\d+)\s+
        (?P<req_pass>\d+)/(?P<req_total>\d+)\s+(?P<req_status>PASS|FAIL)\s+
        (?P<resp_pass>\d+)/(?P<resp_total>\d+)\s+(?P<resp_status>PASS|FAIL)\s+
        (?P<non_pass>\d+)/(?P<non_total>\d+)\s+(?P<non_status>PASS|FAIL)\s*$
    """,
    re.VERBOSE,
)

_SUMMARY_LINE_PATTERN = re.compile(
    r"""^\s*
        (?P<label>IFoE\s+Request|IFoE\s+Response|Non-IFoE)\s*:\s*
        (?P<pass>\d+)/(?P<total>\d+)\s+PASS\s*,\s*
        (?P<fail>\d+)/(?P<total2>\d+)\s+fail\s*
        \(\s*(?P<loss>[0-9]+(?:\.[0-9]+)?)\s*%\s+loss\s*\)\s*$
    """,
    re.VERBOSE | re.IGNORECASE,
)

_LABEL_TO_KEY: Dict[str, str] = {
    "ifoe request": "ifoe_req",
    "ifoe response": "ifoe_resp",
    "non-ifoe": "non_ifoe",
}


def _normalize_label(label: str) -> str:
    """Map a Summary section label to our canonical traffic-type key."""
    key = " ".join(label.strip().split()).lower()
    return _LABEL_TO_KEY.get(key, key)


class AfmctlPingParser:
    """Parse the output of ``afmctl test ping`` into structured results.

    The parser is intentionally tolerant of extra log/banner lines that
    afmctl may emit before/after the table (it locates the header by the
    ``Accel ID  Port#`` line and the Summary section by the literal
    ``Summary:`` marker).
    """

    HEADER_RE = re.compile(r"^\s*Accel\s*ID\s+Port#\s+IFoE\s+Req", re.IGNORECASE)

    @classmethod
    def parse(cls, output: str) -> Dict:
        """Parse one ``afmctl test ping`` invocation's stdout.

        Args:
            output: Full stdout (and stderr if merged) from one afmctl run.

        Returns:
            dict with keys:
              * ``bdf``: BDF reported on the first results line (or ``None``).
              * ``pings_per_port``: declared count from the banner (or ``None``).
              * ``ports``: ``{port_str: {traffic_key: {pass, total, status}}}``.
              * ``summary``: ``{traffic_key: {pass, total, fail, loss_pct, status}}``.
              * ``parse_errors``: list of human-readable parse error strings.
        """
        result: Dict = {
            "bdf": None,
            "pings_per_port": None,
            "ports": {},
            "summary": {},
            "parse_errors": [],
        }
        if not output:
            result["parse_errors"].append("Empty afmctl output")
            return result

        lines = output.splitlines()

        for line in lines:
            m = _BDF_PATTERN.match(line)
            if m and "Ping test results" in line:
                result["bdf"] = m.group(1)
                count_match = re.search(r"\((\d+)\s+pings?\s+per\s+port\s+pair\)", line, re.IGNORECASE)
                if count_match:
                    result["pings_per_port"] = int(count_match.group(1))
                break

        in_table = False
        in_summary = False
        for line in lines:
            stripped = line.rstrip()
            if cls.HEADER_RE.match(stripped):
                in_table = True
                continue
            if in_table and re.match(r"^\s*-{3,}\s+-{3,}", stripped):
                continue
            if stripped.strip().startswith("Summary"):
                in_table = False
                in_summary = True
                continue

            if in_table:
                if not stripped.strip():
                    in_table = False
                    continue
                pm = _PER_PORT_PATTERN.match(stripped)
                if pm:
                    port = pm.group("port")
                    result["ports"][port] = {
                        "accelerator_id": int(pm.group("accel")),
                        "ifoe_req": {
                            "pass": int(pm.group("req_pass")),
                            "total": int(pm.group("req_total")),
                            "status": pm.group("req_status").upper(),
                        },
                        "ifoe_resp": {
                            "pass": int(pm.group("resp_pass")),
                            "total": int(pm.group("resp_total")),
                            "status": pm.group("resp_status").upper(),
                        },
                        "non_ifoe": {
                            "pass": int(pm.group("non_pass")),
                            "total": int(pm.group("non_total")),
                            "status": pm.group("non_status").upper(),
                        },
                    }

            if in_summary:
                sm = _SUMMARY_LINE_PATTERN.match(stripped)
                if sm:
                    key = _normalize_label(sm.group("label"))
                    p = int(sm.group("pass"))
                    t = int(sm.group("total"))
                    f = int(sm.group("fail"))
                    loss = float(sm.group("loss"))
                    result["summary"][key] = {
                        "pass": p,
                        "total": t,
                        "fail": f,
                        "loss_pct": loss,
                        "status": "PASS" if f == 0 and p == t and t > 0 else "FAIL",
                    }

        if not result["ports"] and not result["summary"]:
            result["parse_errors"].append("Could not locate afmctl ping result table or Summary section in output")

        return result


def parse_afmctl_show_device(output: str) -> List[Dict]:
    """Parse one or more ``afmctl show device`` blocks into device descriptors.

    The output of ``afmctl show device`` is a multi-line block per device::

        BDF                              : 0001:01:00.1
        Spec:
          Accelerator id                 : 0
          Local accelerators             : 0, 1
          ...
            No. of network ports         : 72

    Args:
        output: Combined stdout from running ``afmctl show device``.

    Returns:
        List of dicts with keys ``bdf``, ``accelerator_id``,
        ``local_accelerators`` (list[int]), ``num_network_ports`` (int|None).
    """
    devices: List[Dict] = []
    cur: Optional[Dict] = None
    for raw in output.splitlines():
        line = raw.strip()
        m = re.match(r"^BDF\s*:\s*([0-9a-fA-F:.\-]+)\s*$", line)
        if m:
            if cur:
                devices.append(cur)
            cur = {
                "bdf": m.group(1),
                "accelerator_id": None,
                "local_accelerators": [],
                "num_network_ports": None,
            }
            continue
        if cur is None:
            continue
        am = re.match(r"^Accelerator\s+id\s*:\s*(\d+)\s*$", line)
        if am:
            cur["accelerator_id"] = int(am.group(1))
            continue
        lm = re.match(r"^Local\s+accelerators\s*:\s*(.+)$", line)
        if lm:
            raw_list = lm.group(1).strip()
            if raw_list and raw_list != "-":
                cur["local_accelerators"] = [int(tok) for tok in re.split(r"[,\s]+", raw_list) if tok.isdigit()]
            continue
        nm = re.match(r"^No\.\s*of\s*network\s*ports\s*:\s*(\d+)\s*$", line, re.IGNORECASE)
        if nm:
            cur["num_network_ports"] = int(nm.group(1))
            continue
    if cur:
        devices.append(cur)
    return devices


def _coerce_int_list(value) -> List[int]:
    """Best-effort conversion of config values to a list of ints."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        return [int(t) for t in re.split(r"[,\s]+", value.strip()) if t.strip().isdigit()]
    if isinstance(value, (list, tuple)):
        out: List[int] = []
        for item in value:
            if isinstance(item, bool):
                continue
            if isinstance(item, int):
                out.append(item)
            elif isinstance(item, str) and item.strip().isdigit():
                out.append(int(item.strip()))
        return out
    return []


def _coerce_str_list(value) -> List[str]:
    """Best-effort conversion of config values to a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [tok.strip() for tok in re.split(r"[,\s]+", value.strip()) if tok.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _format_ports_arg(ports) -> Optional[str]:
    """Render a ports spec for the ``-p`` flag.

    Accepts ``None``/``"all"`` (returns ``None`` meaning *all ports*), a list
    of port numbers, or a pre-formatted string like ``"0,1,2"`` or ``"0-7"``.
    """
    if ports is None:
        return None
    if isinstance(ports, str):
        normalized = ports.strip()
        if not normalized or normalized.lower() == "all":
            return None
        return normalized
    if isinstance(ports, (list, tuple)):
        if not ports:
            return None
        return ",".join(str(p) for p in ports)
    return str(ports)


class IfoeL2ConnectivityCheck(PreflightCheck):
    """Validate IFoE L2 connectivity via ``afmctl test ping`` (AIMVT-180).

    Each reachable cluster node runs one ``afmctl test ping`` invocation per
    configured ``(bdf, dst_accelerator)`` pairing. The check reports a node
    as PASS only when **every** invocation completes and every enabled
    traffic type meets the configured ``loss_threshold_percent``.
    """

    DEFAULT_AFMCTL_PATH = "afmctl"
    DEFAULT_PINGS_PER_PORT = 1
    DEFAULT_PER_PING_TIMEOUT_SEC = 10
    DEFAULT_SSH_TIMEOUT_SEC = 180
    DEFAULT_LOSS_THRESHOLD_PCT = 0.0
    DEFAULT_TRAFFIC_TYPES: Tuple[str, ...] = TRAFFIC_TYPES

    def __init__(
        self,
        phdl,
        afmctl_path: Optional[str] = None,
        bdfs: Optional[Iterable[str]] = None,
        dst_accelerators: Optional[Iterable[int]] = None,
        ports=None,
        pings_per_port: Optional[int] = None,
        per_ping_timeout: Optional[int] = None,
        traffic_types: Optional[Iterable[str]] = None,
        loss_threshold_pct: Optional[float] = None,
        ssh_timeout: Optional[int] = None,
        use_sudo: bool = False,
        bdf_discovery: str = "auto",
        config_dict: Optional[Dict] = None,
    ):
        """Initialize the IFoE L2 connectivity check.

        Args:
            phdl: Parallel SSH handle.
            afmctl_path: Absolute path or command name for the ``afmctl``
                binary. Defaults to ``"afmctl"`` (PATH lookup).
            bdfs: Iterable of accelerator BDFs (e.g. ``"0001:01:00.1"``) to
                test on each node. If empty and ``bdf_discovery == "auto"``,
                BDFs are auto-discovered per-node via ``afmctl show device``.
            dst_accelerators: Iterable of destination accelerator IDs passed
                to ``--dst-accelerator``. Defaults to ``[0]``.
            ports: Port spec for ``-p``: ``"all"``/``None`` for all ports,
                a list ``[0,1,2]``, or a string ``"0,1,2"`` / ``"0-7"``.
            pings_per_port: Value for ``-c`` (per-port-pair ping count).
            per_ping_timeout: Value for ``-t`` (per-ping timeout). Omitted
                from the command line if ``None``.
            traffic_types: Subset of ``("ifoe_req", "ifoe_resp", "non_ifoe")``
                used both to filter ``--traffic-type`` and to gate pass/fail
                evaluation. Defaults to all three.
            loss_threshold_pct: Maximum acceptable loss percentage for any
                enabled traffic type (default ``0.0``).
            ssh_timeout: Overall SSH timeout for each ``afmctl`` invocation.
            use_sudo: Prepend ``sudo`` when calling ``afmctl``.
            bdf_discovery: ``"auto"`` (run ``afmctl show device`` if ``bdfs``
                is empty) or ``"config"`` (require ``bdfs`` to be supplied).
            config_dict: Optional full preflight config block (passed through
                to the base class for reporting purposes).
        """
        super().__init__(phdl, config_dict)
        self.afmctl_path = afmctl_path or self.DEFAULT_AFMCTL_PATH
        self.bdfs: List[str] = _coerce_str_list(bdfs)
        self.dst_accelerators: List[int] = _coerce_int_list(dst_accelerators) or [0]
        self.ports = ports if ports not in ("", None) else "all"
        self.pings_per_port: int = int(pings_per_port) if pings_per_port else self.DEFAULT_PINGS_PER_PORT
        self.per_ping_timeout: Optional[int] = int(per_ping_timeout) if per_ping_timeout not in (None, "", 0) else None

        tt = _coerce_str_list(traffic_types) or list(self.DEFAULT_TRAFFIC_TYPES)
        canonical: List[str] = []
        for raw in tt:
            t = raw.strip().lower().replace("-", "_")
            if t in ("request", "ifoe_request"):
                t = "ifoe_req"
            elif t in ("response", "ifoe_response"):
                t = "ifoe_resp"
            elif t in ("non_ifoe", "nonifoe"):
                t = "non_ifoe"
            if t in TRAFFIC_TYPES:
                canonical.append(t)
        self.traffic_types: Tuple[str, ...] = tuple(canonical) if canonical else self.DEFAULT_TRAFFIC_TYPES

        self.loss_threshold_pct: float = (
            float(loss_threshold_pct) if loss_threshold_pct is not None else self.DEFAULT_LOSS_THRESHOLD_PCT
        )
        self.ssh_timeout: int = int(ssh_timeout) if ssh_timeout else self.DEFAULT_SSH_TIMEOUT_SEC
        self.use_sudo: bool = bool(use_sudo)
        self.bdf_discovery: str = (bdf_discovery or "auto").strip().lower()

    def _traffic_type_cli(self) -> Optional[str]:
        """Render ``--traffic-type`` argument or ``None`` if all are enabled."""
        if not self.traffic_types or set(self.traffic_types) == set(TRAFFIC_TYPES):
            return None
        afmctl_names = {
            "ifoe_req": "request",
            "ifoe_resp": "response",
            "non_ifoe": "non-ifoe",
        }
        return ",".join(afmctl_names[t] for t in self.traffic_types)

    def build_ping_command(self, bdf: str, dst_accelerator: int) -> str:
        """Render the ``afmctl test ping`` command line for one invocation."""
        parts: List[str] = []
        if self.use_sudo:
            parts.append("sudo")
        parts.extend([self.afmctl_path, "test", "ping"])
        parts.extend(["-b", bdf])
        parts.extend(["-c", str(self.pings_per_port)])
        port_spec = _format_ports_arg(self.ports)
        if port_spec:
            parts.extend(["-p", port_spec])
        parts.extend(["--dst-accelerator", str(dst_accelerator)])
        if self.per_ping_timeout:
            parts.extend(["-t", str(self.per_ping_timeout)])
        ttype = self._traffic_type_cli()
        if ttype:
            parts.extend(["--traffic-type", ttype])
        return " ".join(shlex.quote(p) for p in parts)

    def _discover_bdfs_per_node(self) -> Dict[str, List[str]]:
        """Run ``afmctl show device`` on each reachable host and parse BDFs."""
        cmd = self.afmctl_path + " show device"
        if self.use_sudo:
            cmd = "sudo " + cmd
        cmd = f"{cmd} 2>&1 || true"
        out_dict = self.phdl.exec(cmd, timeout=self.ssh_timeout, print_console=False)
        per_node: Dict[str, List[str]] = {}
        for node, output in out_dict.items():
            devices = parse_afmctl_show_device(output or "")
            per_node[node] = [d["bdf"] for d in devices if d.get("bdf")]
        return per_node

    def _evaluate_summary(self, parsed: Dict) -> Tuple[str, List[str]]:
        """Decide PASS/FAIL for one parsed afmctl ping output.

        Returns:
            Tuple of (status_string, list_of_human_readable_errors).
        """
        summary = parsed.get("summary") or {}
        errors: List[str] = []
        status = "PASS"
        for ttype in self.traffic_types:
            if ttype not in summary:
                errors.append(f"Missing {TRAFFIC_LABELS.get(ttype, ttype)} summary line in afmctl output")
                status = "FAIL"
                continue
            entry = summary[ttype]
            if entry.get("total", 0) == 0:
                errors.append(f"{TRAFFIC_LABELS[ttype]}: zero pings reported")
                status = "FAIL"
                continue
            loss = float(entry.get("loss_pct", 0.0))
            if loss > self.loss_threshold_pct + 1e-9:
                errors.append(
                    f"{TRAFFIC_LABELS[ttype]}: {entry['fail']}/{entry['total']} failed "
                    f"({loss:.2f}% loss > {self.loss_threshold_pct:.2f}% threshold)"
                )
                status = "FAIL"
        for ttype in self.traffic_types:
            for port, port_result in (parsed.get("ports") or {}).items():
                rr = port_result.get(ttype)
                if rr and rr.get("status") == "FAIL":
                    errors.append(f"Port {port} {TRAFFIC_LABELS[ttype]}: {rr['pass']}/{rr['total']} (FAIL)")
                    status = "FAIL"
        return status, errors

    def _resolve_bdfs_for_node(self, node: str, discovered: Dict[str, List[str]]) -> List[str]:
        """Return the BDFs that should be exercised on a single node."""
        if self.bdfs:
            return list(self.bdfs)
        if self.bdf_discovery == "auto":
            return list(discovered.get(node, []))
        return []

    def run(self) -> Dict:
        """Execute IFoE L2 connectivity check across all reachable nodes.

        Returns:
            ``{node: {status, errors, accelerators: {bdf: {dst_accelerator:
            {command, raw_output, parsed, status, errors}}}, ...}}``.
        """
        self.log_info(
            f"Running IFoE L2 connectivity check (afmctl={self.afmctl_path}, "
            f"dst_accelerators={self.dst_accelerators}, ports={self.ports}, "
            f"pings_per_port={self.pings_per_port}, "
            f"traffic_types={list(self.traffic_types)}, "
            f"loss_threshold_pct={self.loss_threshold_pct})"
        )

        discovered: Dict[str, List[str]] = {}
        if not self.bdfs and self.bdf_discovery == "auto":
            self.log_info("Auto-discovering accelerator BDFs via 'afmctl show device'")
            discovered = self._discover_bdfs_per_node()

        self.results = {}
        for node in self.phdl.reachable_hosts:
            self.results[node] = {
                "status": "PASS",
                "errors": [],
                "accelerators": {},
            }
            node_bdfs = self._resolve_bdfs_for_node(node, discovered)
            if not node_bdfs:
                msg = (
                    "No IFoE BDFs configured and auto-discovery returned no devices"
                    if self.bdf_discovery == "auto"
                    else "No IFoE BDFs configured (bdf_discovery=config)"
                )
                self.results[node]["status"] = "FAIL"
                self.results[node]["errors"].append(msg)
                continue
            self.results[node]["bdfs_under_test"] = node_bdfs

        for bdf in self._all_unique_bdfs(discovered):
            for dst in self.dst_accelerators:
                cmd = self.build_ping_command(bdf, dst)
                self.log_info(f"Executing on cluster: {cmd}")
                out_dict = self.phdl.exec(cmd, timeout=self.ssh_timeout, print_console=False)
                for node, output in out_dict.items():
                    if node not in self.results:
                        continue
                    accel_block = self.results[node]["accelerators"].setdefault(bdf, {})
                    if bdf not in self.results[node].get("bdfs_under_test", []):
                        accel_block[str(dst)] = {
                            "command": cmd,
                            "status": "SKIPPED",
                            "errors": [f"BDF {bdf} not present on this node"],
                            "raw_output": "",
                            "parsed": {},
                        }
                        continue
                    parsed = AfmctlPingParser.parse(output or "")
                    status, errs = self._evaluate_summary(parsed)
                    if parsed.get("parse_errors"):
                        status = "FAIL"
                        errs.extend(parsed["parse_errors"])
                    accel_block[str(dst)] = {
                        "command": cmd,
                        "dst_accelerator": dst,
                        "status": status,
                        "errors": errs,
                        "raw_output": output or "",
                        "parsed": parsed,
                    }
                    if status == "FAIL":
                        self.results[node]["status"] = "FAIL"
                        for err in errs:
                            self.results[node]["errors"].append(f"{bdf} -> accel {dst}: {err}")

        return self.results

    def _all_unique_bdfs(self, discovered: Dict[str, List[str]]) -> List[str]:
        """Union of explicitly configured BDFs and per-node discovered BDFs."""
        seen: List[str] = []
        for b in self.bdfs:
            if b not in seen:
                seen.append(b)
        if not self.bdfs and self.bdf_discovery == "auto":
            for blist in discovered.values():
                for b in blist:
                    if b not in seen:
                        seen.append(b)
        return seen
