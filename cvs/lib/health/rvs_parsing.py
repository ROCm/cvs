'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Parse ROCm Validation Suite stdout into Run Deck records.

Numeric lines follow the RVS user-guide RESULT formats (GST GFLOPS, PEBB/PBQT
GBps duration, IET Power(W), BABEL kernel tables). Interval GST samples without
a Target GFLOPS line are ignored.
'''

import re

_TEST_MODULE = {
    "gst_single": "gst",
    "pebb_single": "pebb",
    "pbqt_single": "pbqt",
    "babel_stream": "babel",
    "iet_stress": "iet",
    "mem_test": "mem",
    "level_config": "level",
    "gpu_enumeration": "gpup",
}

_GST_FINAL = re.compile(
    r"\[(?P<action>[^\]]+)\]\s*\[GPU::\s*(?P<gpu>\S+)\]\s*"
    r"GFLOPS\s+(?P<value>[\d.]+)\s+Target GFLOPS:\s+(?P<target>[\d.]+)\s+"
    r"met:\s*(?P<met>TRUE|FALSE)",
    re.I,
)
_PEBB_DONE = re.compile(
    r"\[(?P<action>[^\]]+)\]\s*pcie-bandwidth.*?\[GPU::\s*\S+\s*-\s*(?P<gpu>\S+)\s*-"
    r".*?h2d::(?P<h2d>\S+)\s+d2h::(?P<d2h>\S+)\s+(?P<value>[\d.]+)\s+GBps\s+duration:",
    re.I,
)
_PBQT_DONE = re.compile(
    r"\[(?P<action>[^\]]+)\]\s*p2p-bandwidth.*?\[GPU::\s*\S+\s*-\s*(?P<src>\S+)\s*-"
    r".*?\[GPU::\s*\S+\s*-\s*(?P<dst>\S+)\s*-.*?bidirectional:\s*(?P<bidi>\S+)\s+"
    r"(?P<value>[\d.]+)\s+GBps\s+duration:",
    re.I,
)
_IET_POWER = re.compile(
    r"\[(?P<action>[^\]]+)\]\s*\[GPU::\s*(?P<gpu>\S+)\]\s*Power\(W\)\s+(?P<value>[\d.]+)",
    re.I,
)
_IET_PASS = re.compile(
    r"\[(?P<action>[^\]]+)\]\s*\[GPU::\s*(?P<gpu>\S+)\]\s*pass:\s*(?P<met>TRUE|FALSE)",
    re.I,
)
_BABEL_ROW = re.compile(
    r"^\s*(?P<gpu>\d+)\s+(?P<kernel>Read|Write|Copy|Mul|Add|Triad|Dot)\s+"
    r"(?P<mbytes>[\d.]+)\s+(?P<max_mb>[\d.]+)\s+(?P<min_mb>[\d.]+)\s+(?P<avg_mb>[\d.]+)\s*$",
    re.I,
)
_MODULE_NAME = re.compile(r"Module name\s*:\s*(?P<module>\S+)", re.I)
_ACTION_NAME = re.compile(r"Action name\s*:\s*(?P<action>\S+)", re.I)
_NO_GPUS = re.compile(r"No supported GPUs available", re.I)

_BABEL_KERNELS = frozenset({"read", "write", "copy", "mul", "add", "triad", "dot"})


def normalize_module(name):
    if not name:
        return ""
    return _TEST_MODULE.get(name, name)


def _row(node, gpu, module, action, metric, value, unit, target=None, passed=None, extra=None):
    rec = {
        "node": node,
        "gpu": str(gpu) if gpu is not None else "",
        "module": module or "",
        "action": action or "",
        "metric": metric,
        "value": value,
        "unit": unit,
        "target": target,
        "passed": passed,
    }
    if extra:
        rec.update(extra)
    return rec


def _truthy(met):
    return str(met).upper() == "TRUE"


def parse_rvs_output(text, node, module=None):
    """Return a list of metric records for one node's RVS stdout."""
    module = normalize_module(module)
    records = []
    current_module = module
    current_action = ""
    iet_peak = {}

    for raw in (text or "").splitlines():
        line = raw.strip()
        m = _MODULE_NAME.search(line)
        if m:
            current_module = normalize_module(m.group("module"))
            continue
        m = _ACTION_NAME.search(line)
        if m:
            current_action = m.group("action")
            continue

        m = _GST_FINAL.search(line)
        if m:
            records.append(
                _row(
                    node,
                    m.group("gpu"),
                    current_module or "gst",
                    m.group("action").strip(),
                    "gflops",
                    float(m.group("value")),
                    "GFLOPS",
                    target=float(m.group("target")),
                    passed=_truthy(m.group("met")),
                )
            )
            continue

        m = _PEBB_DONE.search(line)
        if m:
            records.append(
                _row(
                    node,
                    m.group("gpu"),
                    current_module or "pebb",
                    m.group("action").strip(),
                    "pcie_gbps",
                    float(m.group("value")),
                    "GB/s",
                    extra={"h2d": m.group("h2d"), "d2h": m.group("d2h")},
                )
            )
            continue

        m = _PBQT_DONE.search(line)
        if m:
            src, dst = m.group("src"), m.group("dst")
            records.append(
                _row(
                    node,
                    src,
                    current_module or "pbqt",
                    m.group("action").strip(),
                    "p2p_gbps",
                    float(m.group("value")),
                    "GB/s",
                    extra={"dst": dst, "bidirectional": m.group("bidi")},
                )
            )
            continue

        m = _IET_POWER.search(line)
        if m:
            key = (m.group("gpu"), m.group("action").strip())
            val = float(m.group("value"))
            prev = iet_peak.get(key)
            if prev is None or val > prev[0]:
                iet_peak[key] = (val, current_module or "iet")
            continue

        m = _IET_PASS.search(line)
        if m:
            records.append(
                _row(
                    node,
                    m.group("gpu"),
                    current_module or "iet",
                    m.group("action").strip(),
                    "status",
                    None,
                    "",
                    passed=_truthy(m.group("met")),
                )
            )
            continue

        m = _BABEL_ROW.match(line)
        if m and m.group("kernel").lower() in _BABEL_KERNELS:
            kernel = m.group("kernel")
            records.append(
                _row(
                    node,
                    m.group("gpu"),
                    current_module or "babel",
                    current_action or kernel,
                    "babel_mbytes_s",
                    float(m.group("avg_mb")),
                    "MB/s",
                    extra={"kernel": kernel},
                )
            )

    for (gpu, action), (peak, iet_module) in sorted(iet_peak.items()):
        records.append(_row(node, gpu, iet_module, action, "power_w", peak, "W"))

    if _NO_GPUS.search(text or ""):
        records.append(_row(node, "", current_module or "gpup", "enumerate", "status", None, "", passed=False))
    elif (current_module or module) == "gpup" and not records:
        records.append(_row(node, "", "gpup", "enumerate", "status", None, "", passed=True))

    return records


def append_rvs_records(store, text, node, module=None, failed=None):
    """Append parsed records onto store['records']. Adds a status row when empty."""
    recs = parse_rvs_output(text, node, module=module)
    if not recs:
        recs = [
            _row(
                node,
                "",
                normalize_module(module) or "rvs",
                module or "rvs",
                "status",
                None,
                "",
                passed=None if failed is None else (not failed),
            )
        ]
    elif failed is True:
        for rec in recs:
            if rec.get("passed") is None:
                rec["passed"] = False
    elif failed is False:
        for rec in recs:
            if rec.get("passed") is None:
                rec["passed"] = True
    store.setdefault("records", []).extend(recs)
    return recs
