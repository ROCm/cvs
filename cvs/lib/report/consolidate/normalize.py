'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Summarize one Run Deck for the consolidated report: counts in the suite's result
unit, the rows behind its detail panel, and the failures to inspect. Follows the
AI Lab CVS dashboard mockup's adapter so both read decks the same way.
'''

STATUSES = ("fail", "pass", "record", "na", "unknown")
UNIT_CHECKS = "node/check groups"
UNIT_CONFIGURATIONS = "configurations"
UNIT_ROWS = "recorded result rows"
UNIT_RESULTS = "results"
NOT_AVAILABLE = "\u2014"
# Status-matrix suites whose rows are test cases rather than hosts.
NON_HOST_SUITES = frozenset({"ib_perf_bw_test", "mori_benchmark_test"})
MAX_ITEMS_IN_MESSAGE = 5
_MIN_KINDS = ("min", "min_tok_s", "min_ratio")
_MAX_KINDS = ("max", "max_ms")


def text(value):
    '''Display text for a JSON scalar; anything else, including None, shows as a dash.'''
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else ("%.4f" % value).rstrip("0").rstrip(".")
    if isinstance(value, (str, int)):
        return str(value)
    return NOT_AVAILABLE


def status_of(value):
    if value in ("pass", "fail", "record", "na"):
        return value
    if value in ("n/a", "n-a"):
        return "na"
    return "unknown"


def combined(statuses):
    '''One status for several gates: any failure fails, then unknown, pass, N/A, and record only.'''
    if "fail" in statuses:
        return "fail"
    if "unknown" in statuses or not statuses:
        return "unknown"
    if "pass" in statuses:
        return "pass"
    if "na" in statuses:
        return "na"
    return "record"


def empty_counts():
    return {status: 0 for status in STATUSES}


def metric(raw):
    '''A measured metric as name, actual, target, unit, and status strings.'''
    spec = raw.get("spec")
    if isinstance(spec, dict) and spec.get("value") is not None:
        kind = spec.get("kind")
        sign = "\u2265" if kind in _MIN_KINDS else "\u2264" if kind in _MAX_KINDS else text(kind)
        target = "%s %s" % (sign, text(spec.get("value")))
    elif raw.get("threshold") is not None:
        direction = raw.get("direction")
        sign = "\u2265 " if direction == "higher" else "\u2264 " if direction == "lower" else ""
        target = sign + text(raw.get("threshold"))
    else:
        target = "Not gated"
    return {
        "name": text(raw.get("label") or raw.get("name") or raw.get("metric")),
        "actual": text(raw.get("actual", raw.get("value"))),
        "target": target,
        "unit": text(raw.get("unit") or ""),
        "status": status_of(raw.get("status")),
    }


def run_card_metadata(data):
    '''Run card rows as ``[label, value]``. Link rows are left out; their targets are relative to the deck.'''
    rows = []
    for row in data.get("run_card_display") or []:
        if isinstance(row, (list, tuple)) and len(row) >= 2 and not (len(row) > 2 and row[2]):
            rows.append([text(row[0]), text(row[1])])
    return rows


def empty_result():
    return {
        "status": "unknown",
        "counts": empty_counts(),
        "total": 0,
        "unit": UNIT_RESULTS,
        "partial": False,
        "notes": [],
        "failed_nodes": [],
        "nodes": [],
        "host_rows": False,
        "scope": None,
        "cluster": None,
    }


def unavailable(note):
    '''Result and detail for a deck that could not be read.'''
    result = empty_result()
    result["partial"] = True
    result["notes"].append(note)
    return result, {"metadata": [], "evidence": []}


def normalize(data):
    '''Return ``(result, detail)`` for one valid deck.

    ``result`` holds the suite's verdict, counts in its result unit, and the rows that
    failed. ``detail`` holds the run card, failures to inspect, and the node-by-check
    grid, configurations, or recorded rows behind the suite's detail panel.
    '''
    metadata = run_card_metadata(data)
    result = empty_result()
    result["status"] = status_of(data.get("overall_status"))
    result["cluster"] = next(
        (value for label, value in metadata if label.lower() == "cluster" and value not in ("", NOT_AVAILABLE)), None
    )
    detail = {"metadata": metadata, "evidence": []}
    datasets = data.get("datasets") or {}
    if isinstance(datasets.get("status_matrix"), dict):
        _status_matrix(data, datasets["status_matrix"], result, detail)
    elif isinstance(datasets.get("sweep"), dict):
        _sweep(data, datasets["sweep"], result, detail)
    elif isinstance(datasets.get("training_sweep"), dict):
        _sweep(data, datasets["training_sweep"], result, detail)
    elif isinstance(datasets.get("series"), dict):
        _series(data, datasets["series"], result, detail)
    else:
        _mark_partial(result, "Detailed results use an unsupported dataset format.")
    if result["counts"]["unknown"]:
        _mark_partial(result, "Some result outcomes are unavailable.")
    if result["status"] == "unknown":
        _mark_partial(result, "The report has an unrecognized overall status.")
    return result, detail


def _mark_partial(result, note):
    result["partial"] = True
    result["notes"].append(note)


def _note_row(result, name, failed):
    if name not in result["nodes"]:
        result["nodes"].append(name)
    if failed and name not in result["failed_nodes"]:
        result["failed_nodes"].append(name)


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _count(number, noun):
    return "%d %s%s" % (number, noun, "" if number == 1 else "s")


def _status_matrix(data, matrix, result, detail):
    nodes = [text(node) for node in matrix.get("nodes") or []]
    groups = [text(group) for group in matrix.get("groups") or []]
    grid = _as_dict(matrix.get("grid"))
    result["unit"] = UNIT_CHECKS
    result["host_rows"] = data.get("suite_id") not in NON_HOST_SUITES
    rows = {}
    for node in nodes:
        row, cells = {}, _as_dict(grid.get(node))
        for group in groups:
            cell = _as_dict(cells.get(group))
            status = status_of(cell.get("status"))
            row[group] = status
            result["counts"][status] += 1
            _note_row(result, node, status == "fail")
            if status == "fail":
                detail["evidence"].append(_check_evidence(node, group, cell))
        rows[node] = row
    detail["matrix"] = {"nodes": nodes, "groups": groups, "grid": rows}
    if nodes:
        result["scope"] = _count(len(nodes), "node" if result["host_rows"] else "test case")
    result["total"] = len(nodes) * len(groups)
    if not result["total"]:
        _mark_partial(result, "Node/check details are unavailable.")


def _check_evidence(node, group, cell):
    failed = [item for item in cell.get("items") or [] if isinstance(item, dict) and item.get("status") == "fail"]
    names = [text(item.get("message") or item.get("name")) for item in failed[:MAX_ITEMS_IN_MESSAGE]]
    if len(failed) > MAX_ITEMS_IN_MESSAGE:
        names.append("and %d more" % (len(failed) - MAX_ITEMS_IN_MESSAGE))
    summary = cell.get("items_summary")
    return {
        "subject": node,
        "check": group,
        "status": "fail",
        "message": "; ".join(names) or (summary if isinstance(summary, str) else ""),
        "metrics": [metric(raw) for raw in cell.get("metrics") or [] if isinstance(raw, dict)],
    }


def _sweep(data, sweep, result, detail):
    cells = sweep.get("all_cells")
    if not isinstance(cells, list):
        cells = sweep.get("cells") if isinstance(sweep.get("cells"), list) else data.get("cells")
    cells = [cell for cell in cells or [] if isinstance(cell, dict)]
    declared = _as_dict(data.get("summary")).get("total_cells")
    result["unit"] = UNIT_CONFIGURATIONS
    result["total"] = max(len(cells), declared if isinstance(declared, int) else 0)
    if len(cells) < result["total"]:
        _mark_partial(result, "Only %d of %d configurations have details." % (len(cells), result["total"]))
    configurations = []
    for index, cell in enumerate(cells):
        tiers = {str(name): status_of(value) for name, value in _as_dict(cell.get("tiers")).items()}
        metrics = [metric(raw) for raw in cell.get("metrics") or [] if isinstance(raw, dict)]
        status = combined(list(tiers.values()) or [item["status"] for item in metrics])
        result["counts"][status] += 1
        host = cell.get("host")
        if isinstance(host, str) and host.strip():
            _note_row(result, host.strip(), status == "fail")
        label = text(cell.get("label") or cell.get("cell_id") or "Configuration %d" % (index + 1))
        if status == "fail":
            failed_gates = [name.replace("_", " ") for name, value in tiers.items() if value == "fail"]
            detail["evidence"].append(
                {
                    "subject": label,
                    "check": "Configuration gates",
                    "status": "fail",
                    "message": (
                        "Failed gates: " + ", ".join(failed_gates)
                        if failed_gates
                        else "One or more measured metrics failed."
                    ),
                    "metrics": [item for item in metrics if item["status"] == "fail"],
                }
            )
        configurations.append({"label": label, "status": status, "tiers": tiers, "metrics": metrics})
    result["counts"]["unknown"] += result["total"] - len(cells)
    hosts = result["nodes"]
    if hosts:
        result["scope"] = ", ".join(hosts) if len(hosts) <= 2 else _count(len(hosts), "host")
    detail["configurations"] = configurations


def _series(data, series, result, detail):
    table = series.get("results_table")
    if not isinstance(table, dict):
        table = _as_dict(data.get("results_table"))
    headers = [text(header) for header in table.get("headers") or []]
    rows = [[text(value) for value in row] for row in table.get("rows") or [] if isinstance(row, (list, tuple))]
    result["unit"] = UNIT_ROWS
    result["total"] = result["counts"]["record"] = len(rows)
    if result["status"] == "record":
        result["notes"].append("Measurements are recorded without a pass/fail verdict.")
    detail["table"] = {"headers": headers, "rows": rows}
