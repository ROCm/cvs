"""
IFoE L2 Connectivity Check (AIMVT-180).

Validates L2 connectivity by invoking ``afmctl test ping`` on each
reachable node and parsing its aggregate ``Summary:`` section. Current MI4XX
AFM images expose JSON for inventory commands but not for ``test ping``; the
ping parser therefore treats its documented tabular text format as the
primary protocol rather than a JSON compatibility fallback. By default the
check requests ``--skip-pass``: successful per-port rows are suppressed while
summary totals preserve selected-port coverage and failing rows remain useful
diagnostics.

This is a *per-node* preflight check: each node runs one
``afmctl test ping`` invocation per configured ``(bdf, dst_accelerator)``
pairing. The check requires no pairwise SSH coordination because
``afmctl`` drives the request/response state machine in the device and
reports an aggregate Summary that we surface to the operator.

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

import json
import os
import re
import shlex
import tempfile
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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

_BDF_VALUE_PATTERN = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]$")


def _normalize_bdf(value) -> Optional[str]:
    """Return a canonical BDF, or ``None`` when *value* is not a BDF."""
    candidate = str(value).strip()
    return candidate.lower() if _BDF_VALUE_PATTERN.match(candidate) else None


def _json_normalise_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _json_flatten_values(value: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten JSON dictionaries while retaining both leaf and path keys."""
    flattened: Dict[str, Any] = {}
    if not isinstance(value, Mapping):
        return flattened
    for key, child in value.items():
        normalised = _json_normalise_key(key)
        path = f"{prefix}_{normalised}" if prefix else normalised
        if isinstance(child, Mapping):
            flattened.update(_json_flatten_values(child, path))
        else:
            flattened.setdefault(normalised, child)
            flattened[path] = child
    return flattened


def _json_first_value(flattened: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = flattened.get(_json_normalise_key(key))
        if value is not None:
            return value
    return None


def _walk_json_dicts(value: Any, inherited_bdf: Optional[str] = None):
    """Yield all JSON dictionaries and BDF inherited from a BDF-keyed envelope."""
    if isinstance(value, Mapping):
        direct_bdf = _json_first_value(_json_flatten_values(value), ("bdf", "device_bdf", "pci_bdf"))
        current_bdf = _normalize_bdf(direct_bdf) or inherited_bdf
        yield dict(value), current_bdf
        for key, child in value.items():
            yield from _walk_json_dicts(child, _normalize_bdf(key) or current_bdf)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json_dicts(child, inherited_bdf)


def parse_accelerator_ranges(value) -> Tuple[List[int], List[str]]:
    """Parse accelerator IDs and inclusive ranges without silently guessing.

    AFM device output uses both ``24, 25`` and ``24-27`` forms.  This helper
    accepts strings, integers, and nested lists/tuples, preserves first-seen
    order, and returns malformed tokens separately so discovery can fail
    closed when an AFM CLI format changes.
    """
    values: List[int] = []
    errors: List[str] = []

    def add(number: int) -> None:
        if number not in values:
            values.append(number)

    def parse_item(item) -> None:
        if item is None or isinstance(item, bool):
            return
        if isinstance(item, int):
            if item < 0:
                errors.append(f"Negative accelerator ID {item}")
            else:
                add(item)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                parse_item(child)
            return
        if not isinstance(item, str):
            errors.append(f"Unsupported accelerator ID value {item!r}")
            return
        for token in re.split(r"[,\s]+", item.strip()):
            if not token:
                continue
            match = re.match(r"^(\d+)(?:-(\d+))?$", token)
            if not match:
                errors.append(f"Malformed accelerator ID/range {token!r}")
                continue
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) is not None else start
            if end < start:
                errors.append(f"Descending accelerator range {token!r}")
                continue
            for number in range(start, end + 1):
                add(number)

    parse_item(value)
    return values, errors


def expand_accelerator_ranges(value) -> List[int]:
    """Best-effort public shorthand for :func:`parse_accelerator_ranges`."""
    return parse_accelerator_ranges(value)[0]


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

    @staticmethod
    def _traffic_key(value: Any) -> Optional[str]:
        key = _json_normalise_key(value)
        aliases = {
            "ifoe_req": "ifoe_req",
            "ifoe_request": "ifoe_req",
            "request": "ifoe_req",
            "ifoe_resp": "ifoe_resp",
            "ifoe_response": "ifoe_resp",
            "response": "ifoe_resp",
            "non_ifoe": "non_ifoe",
            "nonifoe": "non_ifoe",
        }
        return aliases.get(key)

    @staticmethod
    def _as_int(value: Any) -> Optional[int]:
        if isinstance(value, bool) or value is None:
            return None
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

    @classmethod
    def _metric_from_json(cls, value: Any) -> Optional[Dict[str, Any]]:
        """Normalize a JSON traffic-counter object without assuming AFM's wrapper schema."""
        if isinstance(value, Mapping):
            flattened = _json_flatten_values(value)
            passed = cls._as_int(_json_first_value(flattened, ("pass", "passed", "pass_count", "success")))
            total = cls._as_int(_json_first_value(flattened, ("total", "count", "num_pings", "pings")))
            failed = cls._as_int(_json_first_value(flattened, ("fail", "failed", "fail_count", "errors")))
            raw_status = _json_first_value(flattened, ("status", "state", "result"))
            raw_loss = _json_first_value(flattened, ("loss_pct", "loss_percent", "loss"))
        elif isinstance(value, str):
            pass_match = re.search(r"(\d+)\s*/\s*(\d+)\s+PASS", value, re.IGNORECASE)
            fail_match = re.search(r"(\d+)\s*/\s*(\d+)\s+fail", value, re.IGNORECASE)
            if not pass_match:
                return None
            passed, total = int(pass_match.group(1)), int(pass_match.group(2))
            failed = int(fail_match.group(1)) if fail_match else max(0, total - passed)
            raw_status = None
            loss_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%\s*loss", value, re.IGNORECASE)
            raw_loss = loss_match.group(1) if loss_match else None
        else:
            return None
        if passed is None or total is None:
            return None
        if failed is None:
            failed = max(0, total - passed)
        try:
            loss_pct = float(str(raw_loss).strip().rstrip("%")) if raw_loss is not None else (
                (failed / total) * 100.0 if total else 100.0
            )
        except (TypeError, ValueError):
            return None
        status = str(raw_status or "").strip().upper()
        if status not in ("PASS", "FAIL"):
            status = "PASS" if failed == 0 and passed == total and total > 0 else "FAIL"
        return {
            "pass": passed,
            "total": total,
            "fail": failed,
            "fail_total": total,
            "loss_pct": loss_pct,
            "status": status,
        }

    @classmethod
    def _parse_json(cls, document: Any, result: Dict) -> None:
        if not isinstance(document, (Mapping, list)):
            result["parse_errors"].append("afmctl ping JSON must be an object or array")
            return
        root_flattened = _json_flatten_values(document) if isinstance(document, Mapping) else {}
        root_bdf = _normalize_bdf(_json_first_value(root_flattened, ("bdf", "device_bdf", "pci_bdf")))
        result["bdf"] = root_bdf
        result["pings_per_port"] = cls._as_int(
            _json_first_value(root_flattened, ("pings_per_port", "ping_count", "num_pings"))
        )

        def record_port(port: Any, accelerator: Any, traffic: Dict[str, Dict[str, Any]]) -> None:
            port_number = cls._as_int(port)
            if port_number is None or port_number < 0:
                result["parse_errors"].append(f"Malformed port ID {port!r} in afmctl ping JSON")
                return
            key = str(port_number)
            if key in result["ports"]:
                result["parse_errors"].append(f"Duplicate afmctl ping result row for port {key}")
                return
            entry = {"accelerator_id": cls._as_int(accelerator)}
            entry.update({name: {k: v for k, v in metric.items() if k not in ("fail", "fail_total", "loss_pct")} for name, metric in traffic.items()})
            result["ports"][key] = entry

        def walk(value: Any, inherited_bdf: Optional[str] = None, in_summary: bool = False) -> None:
            if isinstance(value, Mapping):
                flattened = _json_flatten_values(value)
                bdf = _normalize_bdf(_json_first_value(flattened, ("bdf", "device_bdf", "pci_bdf"))) or inherited_bdf
                if result["bdf"] is None and bdf:
                    result["bdf"] = bdf
                port = _json_first_value(flattened, ("port", "port_id", "port_number", "port_num", "port#"))
                accelerator = _json_first_value(flattened, ("accelerator_id", "accel_id", "destination_accelerator"))
                traffic: Dict[str, Dict[str, Any]] = {}
                for key, child in value.items():
                    traffic_key = cls._traffic_key(key)
                    if traffic_key:
                        metric = cls._metric_from_json(child)
                        if metric is not None:
                            traffic[traffic_key] = metric
                if port is not None and traffic:
                    record_port(port, accelerator, traffic)
                elif in_summary and traffic:
                    for traffic_key, metric in traffic.items():
                        existing = result["summary"].get(traffic_key)
                        if existing and existing != metric:
                            result["parse_errors"].append(
                                f"Conflicting afmctl ping summary values for {traffic_key}"
                            )
                        else:
                            result["summary"][traffic_key] = metric
                for key, child in value.items():
                    walk(
                        child,
                        _normalize_bdf(key) or bdf,
                        in_summary or "summary" in _json_normalise_key(key),
                    )
            elif isinstance(value, list):
                for child in value:
                    walk(child, inherited_bdf, in_summary)

        walk(document)
        if not result["ports"] and not result["summary"]:
            result["parse_errors"].append("Could not locate port results or Summary data in afmctl ping JSON")

    @classmethod
    def parse(cls, output: str, *, allow_text_fallback: bool = True) -> Dict:
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
            "format": None,
        }
        if not output:
            result["parse_errors"].append("Empty afmctl output")
            return result

        stripped_output = output.lstrip()
        if stripped_output.startswith("{") or stripped_output.startswith("["):
            result["format"] = "json"
            try:
                cls._parse_json(json.loads(output), result)
            except json.JSONDecodeError as exc:
                result["parse_errors"].append(f"Invalid afmctl ping JSON: {exc.msg}")
            return result
        result["format"] = "text-v1"
        if not allow_text_fallback:
            result["parse_errors"].append("Expected JSON from afmctl test ping; legacy text fallback is disabled")
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
                    if port in result["ports"]:
                        result["parse_errors"].append(f"Duplicate afmctl ping result row for port {port}")
                        continue
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
                        "fail_total": int(sm.group("total2")),
                        "loss_pct": loss,
                        "status": "PASS" if f == 0 and p == t and t > 0 else "FAIL",
                    }

        if not result["ports"] and not result["summary"]:
            result["parse_errors"].append("Could not locate afmctl ping result table or Summary section in output")

        return result


def _parse_afmctl_show_device_text(output: str) -> List[Dict]:
    """Parse legacy text ``afmctl show device`` blocks into device descriptors.

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
        ``local_accelerators`` (list[int]), ``vpod_accelerators`` (list[int]),
        ``num_network_ports`` (int|None), and ``parse_errors``.  Range-valued
        accelerator fields are expanded, e.g. ``24-27`` becomes
        ``[24, 25, 26, 27]``.
    """
    devices: List[Dict] = []
    cur: Optional[Dict] = None
    for raw in output.splitlines():
        line = raw.strip()
        m = re.match(r"^BDF\s*:\s*([^\s]+)\s*$", line, re.IGNORECASE)
        if m:
            if cur:
                devices.append(cur)
            bdf = _normalize_bdf(m.group(1))
            cur = {
                "bdf": bdf or m.group(1),
                "accelerator_id": None,
                "local_accelerators": [],
                "vpod_accelerators": [],
                "num_network_ports": None,
                "parse_errors": ([] if bdf else [f"Malformed BDF {m.group(1)!r}"]),
            }
            continue
        if cur is None:
            continue
        am = re.match(r"^Accelerator\s+id\s*:\s*(.+)$", line, re.IGNORECASE)
        if am:
            ids, errors = parse_accelerator_ranges(am.group(1).strip())
            if len(ids) == 1:
                cur["accelerator_id"] = ids[0]
            elif not ids:
                errors.append(f"Missing accelerator ID in {am.group(0)!r}")
            else:
                errors.append(f"Expected one accelerator ID, found {ids!r}")
            cur["parse_errors"].extend(errors)
            continue
        lm = re.match(r"^Local\s+accelerators\s*:\s*(.+)$", line, re.IGNORECASE)
        if lm:
            raw_list = lm.group(1).strip()
            if raw_list and raw_list != "-":
                values, errors = parse_accelerator_ranges(raw_list)
                cur["local_accelerators"] = values
                cur["parse_errors"].extend(errors)
            continue
        vm = re.match(r"^v\s*pod\s+accelerators\s*:\s*(.+)$", line, re.IGNORECASE)
        if vm:
            raw_list = vm.group(1).strip()
            if raw_list and raw_list != "-":
                values, errors = parse_accelerator_ranges(raw_list)
                cur["vpod_accelerators"] = values
                cur["parse_errors"].extend(errors)
            continue
        nm = re.match(r"^No\.\s*of\s*network\s*ports\s*:\s*(\d+)\s*$", line, re.IGNORECASE)
        if nm:
            cur["num_network_ports"] = int(nm.group(1))
            continue
    if cur:
        devices.append(cur)
    return devices


def parse_afmctl_show_device_json(output: str) -> Tuple[List[Dict], List[str]]:
    """Parse semantic device records from ``afmctl show device --json``.

    AFM JSON wrapper keys have changed between platform releases, so this
    parser deliberately identifies a device by its BDF plus accelerator/vPOD
    fields instead of relying on a fixed top-level list name.
    """
    try:
        document = json.loads(output)
    except json.JSONDecodeError as exc:
        return [], [f"Invalid afmctl show device JSON: {exc.msg}"]

    devices: List[Dict] = []
    seen_bdfs = set()
    for record, inherited_bdf in _walk_json_dicts(document):
        flattened = _json_flatten_values(record)
        bdf = _normalize_bdf(_json_first_value(flattened, ("bdf", "device_bdf", "pci_bdf"))) or inherited_bdf
        if not bdf or bdf in seen_bdfs:
            continue
        if not any(
            _json_normalise_key(key) in flattened
            for key in ("accelerator_id", "accel_id", "local_accelerators", "vpod_accelerators")
        ):
            continue
        seen_bdfs.add(bdf)
        accelerator_ids, accelerator_errors = parse_accelerator_ranges(
            _json_first_value(flattened, ("accelerator_id", "accel_id", "accelerator"))
        )
        local_accelerators, local_errors = parse_accelerator_ranges(
            _json_first_value(flattened, ("local_accelerators", "local_accels"))
        )
        vpod_accelerators, vpod_errors = parse_accelerator_ranges(
            _json_first_value(flattened, ("vpod_accelerators", "v_pod_accelerators", "vpod_accels"))
        )
        errors = accelerator_errors + local_errors + vpod_errors
        if len(accelerator_ids) != 1:
            errors.append("Expected one accelerator ID")
        port_count = _json_first_value(
            flattened, ("num_network_port", "num_network_ports", "network_ports", "number_of_network_ports")
        )
        try:
            port_count = int(port_count)
        except (TypeError, ValueError):
            port_count = None
        devices.append(
            {
                "bdf": bdf,
                "accelerator_id": accelerator_ids[0] if len(accelerator_ids) == 1 else None,
                "local_accelerators": local_accelerators,
                "vpod_accelerators": vpod_accelerators,
                "num_network_ports": port_count,
                # Retain generic AFM state fields for callers that enforce
                # additional admission policy beyond topology discovery.
                "config_phase": _json_first_value(
                    flattened, ("config_phase", "configuration_phase", "phase")
                ),
                "virtualization_mode": _json_first_value(
                    flattened, ("virtualization_mode", "virtualisation_mode", "virt_mode", "virtualization")
                ),
                "parse_errors": errors,
            }
        )
    if not devices:
        return [], ["afmctl show device JSON contained no accelerator descriptors"]
    return devices, []


def parse_afmctl_show_device_output(output: str, *, allow_text_fallback: bool = True) -> Tuple[List[Dict], List[str], str]:
    """Parse AFM device output, treating text as an opt-in compatibility mode."""
    raw = (output or "").strip()
    if not raw:
        return [], ["Empty afmctl show device output"], "empty"
    if raw.startswith("{") or raw.startswith("["):
        devices, errors = parse_afmctl_show_device_json(raw)
        return devices, errors, "json"
    if not allow_text_fallback:
        return [], ["Expected JSON from afmctl show device; legacy text fallback is disabled"], "text-v1"
    devices = _parse_afmctl_show_device_text(output)
    errors = [] if devices else ["afmctl show device text output contained no accelerator descriptors"]
    return devices, errors, "text-v1"


def parse_afmctl_show_device(output: str) -> List[Dict]:
    """Compatibility convenience wrapper around :func:`parse_afmctl_show_device_output`."""
    return parse_afmctl_show_device_output(output, allow_text_fallback=True)[0]


class AfmctlPortParser:
    """Parse ``afmctl show port --json`` without assuming a MI4XX schema.

    The AFM JSON schema has not yet been captured from MI4XX hardware.  JSON
    is therefore parsed by field semantics (``port`` + ``state`` and optional
    ``bdf``) rather than by a guessed fixed tree.  The text parser is a
    deliberately narrow ``text-v1`` fallback: it only accepts an explicit
    ``Port``/``State`` table or labelled records.  Unknown formats return
    parse errors rather than classifying all ports as DOWN.
    """

    _PORT_KEYS = ("port", "port_id", "port_number", "port_num", "port#", "id")
    _STATION_KEYS = ("station_id", "station")
    _STATE_KEYS = ("state", "status", "port_state", "link_state", "link_status", "link_up")
    _BDF_KEYS = ("bdf", "device_bdf", "pci_bdf")
    _KNOWN_STATES = {
        "UP",
        "DOWN",
        "UNKNOWN",
        "DISABLED",
        "ERROR",
        "FAILED",
        "INACTIVE",
        "ACTIVE",
        "LINK_DOWN",
        "NONE",
    }

    @classmethod
    def parse(cls, output: str, *, allow_text_fallback: bool = True) -> Dict:
        """Return structured port inventory from JSON or supported text.

        The result has ``ports_by_bdf`` and ``unscoped_ports`` maps whose
        values are ``{port_id: {port, state, is_up}}``.  A port is considered
        UP only for the literal AFM state ``UP``; ``ACTIVE`` deliberately does
        not become an optimistic synonym until a hardware fixture confirms it.
        """
        result: Dict = {
            "format": None,
            "ports_by_bdf": {},
            "unscoped_ports": {},
            "parse_errors": [],
        }
        if not output or not output.strip():
            result["parse_errors"].append("Empty afmctl show port output")
            return result

        stripped = output.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                document = json.loads(output)
            except json.JSONDecodeError as exc:
                result["format"] = "json"
                result["parse_errors"].append(f"Invalid afmctl port JSON: {exc.msg}")
                return result
            result["format"] = "json"
            cls._parse_json(document, result)
        else:
            result["format"] = "text-v1"
            if allow_text_fallback:
                cls._parse_text(output, result)
            else:
                result["parse_errors"].append(
                    "Expected JSON from afmctl show port; legacy text fallback is disabled"
                )

        if not result["ports_by_bdf"] and not result["unscoped_ports"]:
            result["parse_errors"].append("Could not locate port/state records in afmctl show port output")
        return result

    @staticmethod
    def _norm_key(key) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")

    @classmethod
    def _value_for(cls, item: Dict, keys: Tuple[str, ...]):
        normalised = {cls._norm_key(key): value for key, value in item.items()}
        for key in keys:
            value = normalised.get(cls._norm_key(key))
            if value is not None and not isinstance(value, Mapping):
                return value
        # MI4XX AFM JSON records the physical state under ``status`` (for
        # example ``status.link_status: LINK_UP``). Inspect that bounded
        # nested block without flattening the whole parent record, which could
        # otherwise mistake a child port ID for a device-level port.
        for container_name in ("status", "link", "spec"):
            nested = normalised.get(container_name)
            if not isinstance(nested, Mapping):
                continue
            nested_normalised = {cls._norm_key(key): value for key, value in nested.items()}
            for key in keys:
                value = nested_normalised.get(cls._norm_key(key))
                if value is not None and not isinstance(value, Mapping):
                    return value
        return None

    @classmethod
    def _record(cls, result: Dict, port, state, bdf=None, station=None) -> None:
        try:
            if isinstance(port, bool):
                raise ValueError
            port_number = int(str(port).strip())
            if port_number < 0:
                raise ValueError
        except (TypeError, ValueError):
            result["parse_errors"].append(f"Malformed port ID {port!r}")
            return
        state_string = re.sub(r"[^A-Z0-9]+", "_", str(state).strip().upper()).strip("_")
        if state_string == "LINK_UP":
            state_string = "UP"
        elif state_string.startswith("NO_PHY_LINK"):
            state_string = "DOWN"
        if state_string not in cls._KNOWN_STATES:
            result["parse_errors"].append(f"Unknown port state {state!r} for port {port_number}")
            return
        entry = {"port": port_number, "state": state_string, "is_up": state_string == "UP"}
        if station is not None:
            try:
                if isinstance(station, bool):
                    raise ValueError
                station_number = int(str(station).strip())
                if station_number < 0:
                    raise ValueError
            except (TypeError, ValueError):
                result["parse_errors"].append(
                    f"Malformed station ID {station!r} for port {port_number}"
                )
                return
            entry["station_id"] = station_number
        normalized_bdf = _normalize_bdf(bdf) if bdf is not None else None
        if bdf is not None and not normalized_bdf:
            result["parse_errors"].append(f"Malformed BDF {bdf!r} for port {port_number}")
            return
        target = result["ports_by_bdf"].setdefault(normalized_bdf, {}) if normalized_bdf else result["unscoped_ports"]
        existing = target.get(str(port_number))
        if existing and existing != entry:
            result["parse_errors"].append(f"Conflicting states for port {port_number}")
            return
        target[str(port_number)] = entry

    @classmethod
    def _parse_json(cls, document, result: Dict) -> None:
        def walk(value, inherited_bdf=None, candidate_port=None, allow_port_record=True) -> None:
            if isinstance(value, dict):
                bdf = cls._value_for(value, cls._BDF_KEYS)
                current_bdf = bdf if bdf is not None else inherited_bdf
                port = cls._value_for(value, cls._PORT_KEYS)
                if port is None:
                    port = candidate_port
                state = cls._value_for(value, cls._STATE_KEYS)
                station = cls._value_for(value, cls._STATION_KEYS)
                if allow_port_record and port is not None and state is not None:
                    cls._record(result, port, state, current_bdf, station)
                for key, child in value.items():
                    key_bdf = _normalize_bdf(key)
                    child_bdf = key_bdf or current_bdf
                    child_port = int(key) if str(key).isdigit() else None
                    child_allow_port_record = allow_port_record and cls._norm_key(key) not in {
                        "status",
                        "spec",
                        "ifcp",
                    }
                    walk(child, child_bdf, child_port, child_allow_port_record)
            elif isinstance(value, list):
                for child in value:
                    walk(child, inherited_bdf, candidate_port, allow_port_record)

        walk(document)

    @classmethod
    def _parse_text(cls, output: str, result: Dict) -> None:
        current_bdf: Optional[str] = None
        table_mode = False
        for raw in output.splitlines():
            line = raw.strip()
            if not line:
                continue
            bdf_match = re.search(r"\bBDF\s*[:=]\s*([^\s,]+)", line, re.IGNORECASE)
            if bdf_match:
                current_bdf = _normalize_bdf(bdf_match.group(1))
                if current_bdf is None:
                    result["parse_errors"].append(f"Malformed BDF {bdf_match.group(1)!r} in port output")
                # A BDF line may also contain a port record, so do not continue.

            if re.search(r"\bport(?:\s*#|\s+id|\s+number)?\b", line, re.IGNORECASE) and re.search(
                r"\b(?:state|status)\b", line, re.IGNORECASE
            ):
                table_mode = True

            labelled = re.search(
                r"\bport(?:\s*#|\s+id|\s+number)?\s*[:=]\s*(\d+).*?\b(?:state|status)\s*[:=]\s*([A-Za-z_-]+)",
                line,
                re.IGNORECASE,
            )
            if labelled:
                cls._record(result, labelled.group(1), labelled.group(2), current_bdf)
                continue

            inline = re.search(
                r"\b([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])\s+(\d+)\s+(UP|DOWN|UNKNOWN|DISABLED|ERROR|FAILED|INACTIVE|ACTIVE)\b",
                line,
                re.IGNORECASE,
            )
            if inline:
                cls._record(result, inline.group(2), inline.group(3), inline.group(1))
                continue

            if table_mode:
                row = re.match(
                    r"^(\d+)\s+(UP|DOWN|UNKNOWN|DISABLED|ERROR|FAILED|INACTIVE|ACTIVE)\b", line, re.IGNORECASE
                )
                if row:
                    cls._record(result, row.group(1), row.group(2), current_bdf)


def _coerce_int_list(value) -> List[int]:
    """Best-effort conversion of config values to a list of ints."""
    return expand_accelerator_ranges(value)


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
    DEFAULT_SSH_TIMEOUT_SEC = 180
    DEFAULT_LOSS_THRESHOLD_PCT = 0.0
    DEFAULT_TRAFFIC_TYPES: Tuple[str, ...] = TRAFFIC_TYPES

    def __init__(
        self,
        phdl,
        afmctl_path: Optional[str] = None,
        bdfs: Optional[Iterable[str]] = None,
        dst_accelerators: Optional[Iterable[int]] = None,
        mesh_mode: str = "config",
        ports=None,
        port_discovery: str = "auto",
        pings_per_port: Optional[int] = None,
        per_ping_timeout: Optional[int] = None,
        traffic_types: Optional[Iterable[str]] = None,
        loss_threshold_pct: Optional[float] = None,
        ssh_timeout: Optional[int] = None,
        use_sudo: bool = False,
        json_args: Optional[Sequence[str]] = None,
        allow_text_fallback: bool = True,
        bdf_discovery: str = "auto",
        require_complete_coverage: bool = True,
        strict_discovery: bool = True,
        admitted_port_ids_by_node: Optional[Mapping[str, Mapping[str, Iterable[int]]]] = None,
        skip_pass: bool = True,
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
                to ``--dst-accelerator`` in ``mesh_mode="config"``. Defaults
                to ``[0]``.
            mesh_mode: ``"config"`` for legacy static destinations or
                ``"full_mesh"`` to build ordered non-self pairs from the
                discovered vPOD topology.
            ports: Port spec for ``-p``: ``"all"``/``None`` for all ports,
                ``"up"`` for discovered operational ports, a list
                ``[0,1,2]``, or a string ``"0,1,2"`` / ``"0-7"``.
            port_discovery: ``"auto"`` uses ``afmctl show port -b <BDF>
                --json`` for ``ports="up"``; ``"config"`` rejects automatic
                discovery.
            pings_per_port: Value for ``-c`` (per-port-pair ping count).
            per_ping_timeout: Value for ``-t`` in minutes (per-ping timeout).
                Omitted from the command line if ``None``.
            traffic_types: Subset of ``("ifoe_req", "ifoe_resp", "non_ifoe")``
                used both to filter ``--traffic-type`` and to gate pass/fail
                evaluation. Defaults to all three.
            loss_threshold_pct: Maximum acceptable loss percentage for any
                enabled traffic type (default ``0.0``).
            ssh_timeout: Overall SSH timeout for each ``afmctl`` invocation.
            use_sudo: Prepend ``sudo`` when calling ``afmctl``.
            json_args: JSON-output option(s) appended to AFM discovery
                commands (``show device`` and ``show port``); AFM ping
                output is structured text on current MI4XX images.
            allow_text_fallback: Permit legacy text parsing for the AFM
                discovery commands when they cannot honour the requested JSON
                option. The typed preflight config disables this compatibility
                fallback by default.
            bdf_discovery: ``"auto"`` (run ``afmctl show device`` if ``bdfs``
                is empty) or ``"config"`` (require ``bdfs`` to be supplied).
            require_complete_coverage: Fail a node when planned pairs were not
                all invoked and parsed.
            strict_discovery: Treat malformed topology/port discovery output as
                a failure instead of a warning.
            admitted_port_ids_by_node: Optional node/BDF-specific port IDs
                admitted by MI4XX node health. When supplied with
                ``ports="up"``, physical UP ports outside this set (for
                example, intentionally station-masked ports) are excluded.
            skip_pass: Pass AFM ``--skip-pass`` to emit only failed port rows
                plus the aggregate Summary. Summary totals still prove
                selected-port coverage; failed rows remain diagnostics.
            config_dict: Optional full preflight config block (passed through
                to the base class for reporting purposes).
        """
        super().__init__(phdl, config_dict)
        self.afmctl_path = afmctl_path or self.DEFAULT_AFMCTL_PATH
        self.bdfs = []
        for raw_bdf in _coerce_str_list(bdfs):
            bdf = _normalize_bdf(raw_bdf)
            if not bdf:
                raise ValueError(f"Invalid IFoE source BDF {raw_bdf!r}")
            if bdf not in self.bdfs:
                self.bdfs.append(bdf)
        self.dst_accelerators: List[int] = _coerce_int_list(dst_accelerators) or [0]
        raw_mesh_mode = str(mesh_mode or "config").strip().lower()
        self.mesh_mode: str = "full_mesh" if raw_mesh_mode in ("full_mesh", "full", "auto") else "config"
        if raw_mesh_mode not in ("config", "full_mesh", "full", "auto"):
            raise ValueError("mesh_mode must be 'config' or 'full_mesh'")
        self.ports = ports if ports not in ("", None) else "all"
        self.port_discovery: str = str(port_discovery or "auto").strip().lower()
        if self.port_discovery not in ("auto", "config"):
            raise ValueError("port_discovery must be 'auto' or 'config'")
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
        self.json_args = [str(arg) for arg in (json_args or ["--json"])]
        if not self.json_args or not any(arg in ("--json", "-j") for arg in self.json_args):
            raise ValueError("json_args must request JSON output")
        self.allow_text_fallback = bool(allow_text_fallback)
        self.bdf_discovery: str = (bdf_discovery or "auto").strip().lower()
        if self.bdf_discovery not in ("auto", "config"):
            raise ValueError("bdf_discovery must be 'auto' or 'config'")
        self.require_complete_coverage: bool = bool(require_complete_coverage)
        self.strict_discovery: bool = bool(strict_discovery)
        self.skip_pass: bool = bool(skip_pass)
        self.admitted_port_ids_by_node: Dict[str, Dict[str, List[int]]] = {}
        for node, per_bdf in (admitted_port_ids_by_node or {}).items():
            if not isinstance(per_bdf, Mapping):
                raise ValueError(f"Admitted IFoE ports for {node!r} must be a BDF mapping")
            normalized_bdfs: Dict[str, List[int]] = {}
            for raw_bdf, raw_ports in per_bdf.items():
                bdf = _normalize_bdf(raw_bdf)
                if not bdf:
                    raise ValueError(f"Invalid admitted IFoE BDF {raw_bdf!r} for {node!r}")
                values, errors = parse_accelerator_ranges(raw_ports)
                if errors:
                    raise ValueError(f"Invalid admitted IFoE port IDs for {node!r}/{bdf}: {errors[0]}")
                normalized_bdfs[bdf] = values
            self.admitted_port_ids_by_node[str(node)] = normalized_bdfs
        # Backends without ``exec_cmd_list`` can only broadcast a command to
        # every host.  Cache each broadcast so consuming its result for node A
        # does not repeat the same source command on node B (and, importantly,
        # preserves the one-broadcast-per-command behaviour of the historical
        # implementation).
        self._broadcast_result_cache: Dict[str, Dict[str, Dict]] = {}

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

    def _afmctl_parts(self) -> List[str]:
        """Return the privilege-safe executable prefix for an afmctl command."""
        return (["sudo", "-n"] if self.use_sudo else []) + [self.afmctl_path]

    def build_ping_command(self, bdf: str, dst_accelerator: int, ports=None) -> str:
        """Render the ``afmctl test ping`` command line for one invocation."""
        parts = self._afmctl_parts() + ["test", "ping"]
        parts.extend(["-b", bdf])
        parts.extend(["-c", str(self.pings_per_port)])
        port_spec = _format_ports_arg(self.ports if ports is None else ports)
        if port_spec:
            parts.extend(["-p", port_spec])
        parts.extend(["--dst-accelerator", str(dst_accelerator)])
        if self.per_ping_timeout:
            parts.extend(["-t", str(self.per_ping_timeout)])
        ttype = self._traffic_type_cli()
        if ttype:
            parts.extend(["--traffic-type", ttype])
        if self.skip_pass:
            parts.append("--skip-pass")
        return " ".join(shlex.quote(p) for p in parts)

    def build_show_port_command(self, bdf: str) -> str:
        """Render the source-BDF-scoped port discovery command.

        The command is explicit about the source BDF and requests JSON. AFM
        does not permit ``--brief`` and ``--json`` in the same invocation.
        """
        parts = self._afmctl_parts() + ["show", "port", "-b", bdf, *self.json_args]
        return " ".join(shlex.quote(p) for p in parts)

    @staticmethod
    def _port_artifact_path(bdf: str) -> str:
        """Return a unique, shell-safe remote path for one AFM JSON inventory."""
        safe_bdf = bdf.replace(":", "_").replace(".", "_")
        return f"/tmp/cvs-ifoe-port-{safe_bdf}-{uuid.uuid4().hex}.json"

    def build_show_port_artifact_command(self, bdf: str, artifact_path: str) -> str:
        """Write a source-BDF AFM JSON inventory to a private remote file."""
        return f"umask 077; {self.build_show_port_command(bdf)} > {shlex.quote(artifact_path)}"

    @staticmethod
    def _build_remove_artifact_command(artifact_path: str) -> str:
        """Render a safe cleanup command for a generated AFM JSON artifact."""
        return f"rm -f -- {shlex.quote(artifact_path)}"

    @staticmethod
    def _normalise_exec_results(raw_results) -> Dict[str, Dict]:
        """Normalize Pssh's string and detailed result shapes."""
        normalized: Dict[str, Dict] = {}
        if not isinstance(raw_results, dict):
            return normalized
        for node, value in raw_results.items():
            if isinstance(value, dict):
                normalized[node] = {
                    "output": value.get("output", "") or "",
                    "exit_status": value.get("exit_code", value.get("exit_status")),
                }
            else:
                normalized[node] = {"output": value or "", "exit_status": None}
        return normalized

    def _exec_all(self, command: str) -> Dict[str, Dict]:
        """Run one discovery command across all reachable nodes with exit data."""
        try:
            raw_results = self.phdl.exec(command, timeout=self.ssh_timeout, print_console=False, detailed=True)
        except TypeError:
            raw_results = self.phdl.exec(command, timeout=self.ssh_timeout, print_console=False)
        return self._normalise_exec_results(raw_results)

    @staticmethod
    def _extract_exit_sentinel(output: str) -> Tuple[str, Optional[int]]:
        """Remove the exit sentinel used by ``exec_cmd_list`` target calls."""
        match = re.search(r"(?:^|\n)__CVS_AFMCTL_EXIT_STATUS__=(-?\d+)\s*$", output or "")
        if not match:
            return output or "", None
        return (output or "")[: match.start()].rstrip("\n"), int(match.group(1))

    def _exec_on_node(self, node: str, command: str) -> Dict:
        """Run a command only on ``node`` when the Pssh backend supports it.

        ``MultiProcessPssh.exec_cmd_list`` schedules one command per host.  A
        no-op is sent to non-target hosts so a BDF is never accidentally
        exercised on a host where it is absent.  Lightweight/mock backends
        without that API retain the older broadcast fallback for compatibility.
        """
        command_list_fn = getattr(self.phdl, "exec_cmd_list", None)
        is_mock = type(command_list_fn).__module__.startswith("unittest.mock")
        if callable(command_list_fn) and not is_mock:
            target_command = (
                f"{command}; _cvs_afmctl_rc=$?; "
                "printf '\\n__CVS_AFMCTL_EXIT_STATUS__=%s\\n' \"$_cvs_afmctl_rc\"; "
                "exit \"$_cvs_afmctl_rc\""
            )
            commands = [target_command if host == node else "true" for host in self.phdl.reachable_hosts]
            raw_results = command_list_fn(commands, timeout=self.ssh_timeout, print_console=False)
            result = self._normalise_exec_results(raw_results).get(node)
            if result is not None:
                output, exit_status = self._extract_exit_sentinel(result["output"])
                result["output"] = output
                result["exit_status"] = exit_status if exit_status is not None else result.get("exit_status")
                return result

        # Compatibility fallback for tests and third-party Pssh wrappers that
        # do not expose exec_cmd_list.  The output is still evaluated only for
        # the intended source node.
        if command not in self._broadcast_result_cache:
            self._broadcast_result_cache[command] = self._exec_all(command)
        result = self._broadcast_result_cache[command].get(node)
        return result or {"output": "", "exit_status": -1}

    def _add_node_issue(self, node: str, category: str, message: str, fatal: bool = True) -> None:
        """Append a classified issue and, when appropriate, fail the node."""
        block = self.results[node]
        block["error_details"].append({"category": category, "message": message})
        block["errors"].append(message)
        if fatal:
            block["status"] = "FAIL"

    def _discover_topology(self) -> None:
        """Populate ``topology`` for every reachable node via ``show device``."""
        command = " ".join(
            shlex.quote(p) for p in self._afmctl_parts() + ["show", "device", *self.json_args]
        )
        for node, result in self._exec_all(command).items():
            if node not in self.results:
                continue
            exit_status = result.get("exit_status")
            if exit_status not in (None, 0):
                self._add_node_issue(node, "DISCOVERY_ERROR", f"afmctl show device exited with status {exit_status}")
            devices, parse_errors, output_format = parse_afmctl_show_device_output(
                result.get("output", ""), allow_text_fallback=self.allow_text_fallback
            )
            self.results[node]["topology"] = devices
            self.results[node]["topology_format"] = output_format
            for error in parse_errors:
                self._add_node_issue(node, "DISCOVERY_ERROR", error, self.strict_discovery)
            if not devices:
                self._add_node_issue(node, "DISCOVERY_ERROR", "afmctl show device returned no accelerator descriptors")
            for device in devices:
                for error in device.get("parse_errors") or []:
                    self._add_node_issue(
                        node, "DISCOVERY_ERROR", f"{device.get('bdf', '<unknown BDF>')}: {error}", self.strict_discovery
                    )

    def _discover_up_ports(self, node: str, bdf: str) -> Optional[List[int]]:
        """Discover operational ports from an AFM JSON artifact retrieved over SFTP."""
        artifact_path = self._port_artifact_path(bdf)
        command = self.build_show_port_artifact_command(bdf, artifact_path)
        result = self._exec_on_node(node, command)
        exit_status = result.get("exit_status")
        parsed: Dict[str, Any] = {
            "format": None,
            "ports_by_bdf": {},
            "unscoped_ports": {},
            "parse_errors": [],
        }
        transfer_error = None
        cleanup_error = None
        cleanup_status = None
        try:
            if exit_status in (None, 0):
                with tempfile.TemporaryDirectory(prefix="cvs_ifoe_port_") as temp_dir:
                    local_prefix = os.path.join(temp_dir, "port.json")
                    downloaded_paths = self.phdl.download_file(artifact_path, local_prefix, hosts=[node])
                    local_path = downloaded_paths.get(node)
                    if not local_path:
                        raise IOError(f"SFTP did not return an artifact path for {node}")
                    with open(local_path, "r", encoding="utf-8") as artifact_file:
                        parsed = AfmctlPortParser.parse(
                            artifact_file.read(), allow_text_fallback=self.allow_text_fallback
                        )
        except (OSError, ValueError, TypeError) as exc:
            transfer_error = str(exc)
        finally:
            try:
                cleanup_result = self._exec_on_node(node, self._build_remove_artifact_command(artifact_path))
                cleanup_status = cleanup_result.get("exit_status")
            except (OSError, ValueError, TypeError) as exc:
                cleanup_error = str(exc)

        bdf_key = _normalize_bdf(bdf) or bdf
        scoped = (parsed.get("ports_by_bdf") or {}).get(bdf_key)
        ports = scoped if scoped is not None else parsed.get("unscoped_ports") or {}
        physical_up_ports = sorted(entry["port"] for entry in ports.values() if entry.get("is_up"))
        admitted_ports = self.admitted_port_ids_by_node.get(node, {}).get(bdf_key)
        if admitted_ports is None:
            up_ports = physical_up_ports
            excluded_masked_ports: List[int] = []
        else:
            admitted_set = set(admitted_ports)
            up_ports = [port for port in physical_up_ports if port in admitted_set]
            excluded_masked_ports = [port for port in physical_up_ports if port not in admitted_set]
        inventory = {
            "command": command,
            "artifact_transport": "sftp",
            "exit_status": exit_status,
            "format": parsed.get("format"),
            "ports": list(ports.values()),
            "physical_up_ports": physical_up_ports,
            "mask_enabled_port_ids": admitted_ports,
            "excluded_masked_up_ports": excluded_masked_ports,
            "up_ports": up_ports,
            "parse_errors": list(parsed.get("parse_errors") or []),
        }
        self.results[node]["port_inventory"][bdf] = inventory
        if exit_status not in (None, 0):
            self._add_node_issue(node, "PORT_DISCOVERY_ERROR", f"{bdf}: show port exited with status {exit_status}")
            return None
        if cleanup_status not in (None, 0) or cleanup_error:
            self._add_node_issue(node, "PORT_DISCOVERY_ERROR", f"{bdf}: failed to remove temporary port artifact")
            return None
        if transfer_error:
            self._add_node_issue(
                node, "PORT_DISCOVERY_ERROR", f"{bdf}: SFTP port artifact retrieval failed: {transfer_error}"
            )
            return None
        if inventory["parse_errors"]:
            for error in inventory["parse_errors"]:
                self._add_node_issue(node, "PORT_DISCOVERY_ERROR", f"{bdf}: {error}", self.strict_discovery)
            if self.strict_discovery:
                return None
        if not inventory["up_ports"]:
            self._add_node_issue(node, "PORT_DISCOVERY_ERROR", f"{bdf}: no UP ports discovered")
            return None
        return inventory["up_ports"]

    @staticmethod
    def _explicit_port_list(ports) -> Optional[List[int]]:
        """Expand an explicit numeric port selection; ``None`` means all ports."""
        if ports is None or (isinstance(ports, str) and ports.strip().lower() == "all"):
            return None
        if isinstance(ports, str) and ports.strip().lower() == "up":
            return []
        values, errors = parse_accelerator_ranges(ports)
        return values if not errors else []

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
            if entry.get("fail_total", entry.get("total")) != entry.get("total"):
                errors.append(
                    f"{TRAFFIC_LABELS[ttype]}: inconsistent summary denominators "
                    f"{entry.get('total')} pass-total vs {entry.get('fail_total')} fail-total"
                )
                status = "FAIL"
            if entry.get("pass", 0) + entry.get("fail", 0) != entry.get("total", 0):
                errors.append(
                    f"{TRAFFIC_LABELS[ttype]}: inconsistent summary counts "
                    f"{entry.get('pass')} pass + {entry.get('fail')} fail != {entry.get('total')} total"
                )
                status = "FAIL"
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

    def _evaluate_invocation(
        self, parsed: Dict, bdf: str, dst_accelerator: int, selected_ports: Optional[List[int]], exit_status
    ) -> Tuple[str, List[str], Optional[str]]:
        """Apply command, parser, and exact-coverage invariants to one ping."""
        status, errors = self._evaluate_summary(parsed)
        failure_category: Optional[str] = None
        if exit_status not in (None, 0):
            errors.append(f"afmctl test ping exited with status {exit_status}")
            status, failure_category = "FAIL", "COMMAND_ERROR"
        if parsed.get("parse_errors"):
            errors.extend(parsed["parse_errors"])
            status, failure_category = "FAIL", "PARSE_ERROR"
        actual_bdf = parsed.get("bdf")
        if not actual_bdf:
            errors.append(f"Result output did not identify the requested source BDF {bdf}")
            status, failure_category = "FAIL", "PARSE_ERROR"
        elif _normalize_bdf(actual_bdf) != _normalize_bdf(bdf):
            errors.append(f"Result BDF {actual_bdf} does not match requested source BDF {bdf}")
            status, failure_category = "FAIL", "PARSE_ERROR"
        reported_pings_per_port = parsed.get("pings_per_port")
        if reported_pings_per_port is not None and reported_pings_per_port != self.pings_per_port:
            errors.append(
                f"afmctl reported {reported_pings_per_port} pings per port pair; "
                f"requested {self.pings_per_port}"
            )
            status, failure_category = "FAIL", "COVERAGE_ERROR"
        if selected_ports is not None:
            expected_ports = set(selected_ports)
            if not self.skip_pass:
                actual_ports = {int(port) for port in (parsed.get("ports") or {}) if str(port).isdigit()}
                missing_ports = sorted(expected_ports - actual_ports)
                unexpected_ports = sorted(actual_ports - expected_ports)
                if missing_ports or unexpected_ports:
                    errors.append(
                        f"Port coverage mismatch: missing={missing_ports or 'none'}, unexpected={unexpected_ports or 'none'}"
                    )
                    status, failure_category = "FAIL", "COVERAGE_ERROR"
            expected_total = len(expected_ports) * self.pings_per_port
            for ttype in self.traffic_types:
                entry = (parsed.get("summary") or {}).get(ttype)
                if entry is None:
                    errors.append(f"Missing {TRAFFIC_LABELS[ttype]} summary line for selected-port coverage")
                    status, failure_category = "FAIL", "COVERAGE_ERROR"
                elif entry.get("total") != expected_total:
                    errors.append(
                        f"{TRAFFIC_LABELS[ttype]} summary total {entry.get('total')} does not match "
                        f"{len(expected_ports)} selected ports x {self.pings_per_port} pings"
                    )
                    status, failure_category = "FAIL", "COVERAGE_ERROR"
        returned_destinations = {
            value.get("accelerator_id")
            for value in (parsed.get("ports") or {}).values()
            if isinstance(value, dict) and value.get("accelerator_id") is not None
        }
        if returned_destinations and returned_destinations != {dst_accelerator}:
            errors.append(
                f"Result destination accelerator(s) {sorted(returned_destinations)} do not match requested {dst_accelerator}"
            )
            status, failure_category = "FAIL", "PARSE_ERROR"
        return status, errors, failure_category

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
            f"Running IFoE L2 connectivity check (afmctl={self.afmctl_path}, mesh_mode={self.mesh_mode}, "
            f"ports={self.ports}, pings_per_port={self.pings_per_port}, "
            f"traffic_types={list(self.traffic_types)}, loss_threshold_pct={self.loss_threshold_pct})"
        )

        self.results = {
            node: {
                "status": "PASS",
                "errors": [],
                "error_details": [],
                "accelerators": {},
                "topology": [],
                "port_inventory": {},
                "plan": [],
                "coverage": {
                    "expected_pairs": 0,
                    "planned_pairs": 0,
                    "expected_invocations": 0,
                    "completed_invocations": 0,
                    "complete": False,
                },
            }
            for node in self.phdl.reachable_hosts
        }

        needs_topology = self.bdf_discovery == "auto" or self.mesh_mode == "full_mesh" or self.ports == "up"
        if needs_topology:
            self.log_info("Discovering IFoE accelerator topology via 'afmctl show device'")
            self._discover_topology()

        for node, node_result in self.results.items():
            discovered_devices = node_result.get("topology") or []
            if self.bdfs:
                source_devices = [device for device in discovered_devices if device.get("bdf") in self.bdfs]
                if not source_devices and self.mesh_mode == "config":
                    source_devices = [{"bdf": bdf, "accelerator_id": None} for bdf in self.bdfs]
            else:
                source_devices = list(discovered_devices)
            if not source_devices:
                self._add_node_issue(node, "DISCOVERY_ERROR", "No IFoE source BDFs available for testing")
                continue
            node_result["bdfs_under_test"] = [device.get("bdf") for device in source_devices if device.get("bdf")]

            local_ids = [
                device.get("accelerator_id") for device in discovered_devices if device.get("accelerator_id") is not None
            ]
            for source in source_devices:
                bdf = source.get("bdf")
                source_accelerator = source.get("accelerator_id")
                if not bdf:
                    self._add_node_issue(node, "DISCOVERY_ERROR", "Topology descriptor has no BDF")
                    continue

                if self.mesh_mode == "full_mesh":
                    if source_accelerator is None:
                        self._add_node_issue(node, "DISCOVERY_ERROR", f"{bdf}: missing source accelerator ID")
                        continue
                    vpod_accelerators = list(source.get("vpod_accelerators") or [])
                    if vpod_accelerators:
                        peers = vpod_accelerators
                    else:
                        fallback_peers = list(source.get("local_accelerators") or local_ids)
                        self._add_node_issue(
                            node,
                            "DISCOVERY_ERROR",
                            f"{bdf}: vPOD accelerator membership is unavailable; cannot prove full-mesh coverage",
                            self.strict_discovery,
                        )
                        if self.strict_discovery:
                            continue
                        peers = fallback_peers
                else:
                    peers = list(self.dst_accelerators)
                peers = [peer for peer in peers if peer != source_accelerator]
                if not peers:
                    self._add_node_issue(
                        node,
                        "COVERAGE_ERROR",
                        f"{bdf}: no non-self destination accelerators available",
                    )
                    continue

                coverage = node_result["coverage"]
                coverage["expected_pairs"] += len(peers)
                if self.ports == "up":
                    if self.port_discovery != "auto":
                        self._add_node_issue(node, "PORT_DISCOVERY_ERROR", f"{bdf}: ports='up' requires port_discovery='auto'")
                        continue
                    selected_ports = self._discover_up_ports(node, bdf)
                    if selected_ports is None:
                        continue
                else:
                    selected_ports = self._explicit_port_list(self.ports)
                    if selected_ports == [] and self.ports not in (None, "", "all"):
                        self._add_node_issue(node, "PORT_DISCOVERY_ERROR", f"{bdf}: invalid explicit port selection {self.ports!r}")
                        continue

                for dst in peers:
                    cell = {
                        "source_bdf": bdf,
                        "source_accelerator": source_accelerator,
                        "dst_accelerator": dst,
                        "selected_ports": selected_ports,
                    }
                    node_result["plan"].append(cell)
                    coverage["planned_pairs"] += 1
                    coverage["expected_invocations"] += 1

        for node, node_result in self.results.items():
            for cell in node_result["plan"]:
                bdf = cell["source_bdf"]
                dst = cell["dst_accelerator"]
                selected_ports = cell["selected_ports"]
                command = self.build_ping_command(bdf, dst, ports=selected_ports)
                self.log_info(f"Executing on {node}: {command}")
                execution = self._exec_on_node(node, command)
                parsed = AfmctlPingParser.parse(execution.get("output", ""), allow_text_fallback=True)
                status, errors, failure_category = self._evaluate_invocation(
                    parsed, bdf, dst, selected_ports, execution.get("exit_status")
                )
                node_result["coverage"]["completed_invocations"] += 1
                node_result["accelerators"].setdefault(bdf, {})[str(dst)] = {
                    "command": command,
                    "dst_accelerator": dst,
                    "source_accelerator": cell.get("source_accelerator"),
                    "selected_ports": selected_ports,
                    "exit_status": execution.get("exit_status"),
                    "status": status,
                    "failure_category": failure_category,
                    "errors": errors,
                    "raw_output": execution.get("output", ""),
                    "parsed": parsed,
                }
                if status == "FAIL":
                    node_result["status"] = "FAIL"
                    category = failure_category or "PING_FAILURE"
                    for error in errors:
                        self._add_node_issue(node, category, f"{bdf} -> accel {dst}: {error}", fatal=False)

            coverage = node_result["coverage"]
            coverage["complete"] = (
                coverage["expected_pairs"] > 0
                and coverage["expected_pairs"] == coverage["planned_pairs"]
                and coverage["expected_invocations"] == coverage["completed_invocations"]
            )
            if self.require_complete_coverage and not coverage["complete"]:
                self._add_node_issue(
                    node,
                    "COVERAGE_ERROR",
                    "Incomplete IFoE mesh coverage: "
                    f"expected_pairs={coverage['expected_pairs']}, planned_pairs={coverage['planned_pairs']}, "
                    f"expected_invocations={coverage['expected_invocations']}, "
                    f"completed_invocations={coverage['completed_invocations']}",
                )

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
