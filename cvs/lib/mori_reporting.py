'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Record Mori benchmark results in the status_matrix shape the Run Deck renders.

The store holds ``groups`` (one per operation, titled with the data direction), whose
``nodes`` are the IO cases or the IBGDA nodes, each carrying its curves as ``series``.
An extra "all cases" (IO) or "both nodes" (IBGDA) cell carries the bandwidth and latency
heatmaps over message sizes. The per-rank heatmap lives in ``_meta["rank_grid"]`` and the
full results in ``_meta["results_table"]``, which the builder passes through, so the
profile can render them as their own sections. Raw per-size
results live under ``_mori`` so every record call can rebuild the groups in a fixed order.
'''

KEY_MSG_SIZE = 1 << 20
# Latency spans three decades on a linear axis; splitting at 32 KiB keeps small messages readable.
LATENCY_SPLIT = 32 << 10

_IO_SHARED_FIELDS = ("MsgSize_B", "BatchSize", "TotalSize_MB")


def size_text(size):
    for base, suffix in ((1 << 30, "GiB"), (1 << 20, "MiB"), (1 << 10, "KiB")):
        if size >= base and size % base == 0:
            return f"{size // base} {suffix}"
    return f"{size} B"


def _tick(size):
    return size_text(size).replace(" KiB", "K").replace(" MiB", "M").replace(" GiB", "G").replace(" ", "")


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def node_names(hosts):
    """{host: "node00", ...} in cluster order, the names the run card and section titles use."""
    return {str(host): f"node{index:02d}" for index, host in enumerate(hosts or [])}


def _rank_values(rank_rows, field):
    values = []
    for rank, row in rank_rows:
        try:
            values.append((rank, float(row[field])))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def _mean(values):
    return round(sum(value for _rank, value in values) / len(values), 2)


def _aggregate_rank_rows(rank_rows):
    entry = {field: rank_rows[0][1][field] for field in _IO_SHARED_FIELDS if field in rank_rows[0][1]}
    entry["ranks"] = len(rank_rows)
    bandwidth = _rank_values(rank_rows, "Avg_BW_GBps")
    if bandwidth:
        slowest_rank, slowest_bw = min(bandwidth, key=lambda item: item[1])
        entry.update(
            {
                "Avg_BW_GBps": _mean(bandwidth),
                "Slowest_rank": slowest_rank,
                "Slowest_rank_BW_GBps": slowest_bw,
                "Rank_BW_GBps": {str(rank): value for rank, value in bandwidth},
            }
        )
    latency = _rank_values(rank_rows, "Avg_Lat_us")
    if latency:
        entry.update({"Avg_Lat_us": _mean(latency), "Worst_rank_Lat_us": max(value for _rank, value in latency)})
    return entry


def _titles(meta):
    names = node_names(meta.get("hosts"))
    initiator = names.get(str(meta.get("initiator")))
    target = next((name for name in names.values() if name != initiator), None)
    titles = {"io_read": "MORI-IO READ", "io_write": "MORI-IO WRITE", "ibgda_write": "IBGDA WRITE inside each node"}
    if initiator and target:
        titles["io_write"] = f"MORI-IO WRITE · {initiator} → {target}"
        titles["io_read"] = f"MORI-IO READ · {target} → {initiator}"
    devices = [d.strip() for d in str(meta.get("mori_device_list") or "").split(",") if d.strip()]
    if len(devices) >= 2 and not devices[0].startswith("^"):
        titles["ibgda_write"] = f"IBGDA WRITE · {devices[0]} → {devices[1]} inside each node"
    return titles, names, (initiator or "initiator", target or "target")


def _remember_variant(store, variant):
    meta = store.setdefault("_meta", {"suite": "mori_benchmark_test"})
    if isinstance(variant, dict):
        for key in ("hosts", "initiator", "mori_device_list"):
            if variant.get(key) is not None:
                meta[key] = variant[key]


def _series(name, unit, points):
    return {"name": name, "unit": unit, "x_label": "message size", "points": points}


def _io_cell(case, status):
    ordered = sorted(case.items(), key=lambda item: _int(item[0]))
    bw = [[_tick(_int(s)), e["Avg_BW_GBps"]] for s, e in ordered if e.get("Avg_BW_GBps") is not None]
    lat = [(_int(s), e["Avg_Lat_us"]) for s, e in ordered if e.get("Avg_Lat_us") is not None]
    small = [[_tick(s), v] for s, v in lat if s <= LATENCY_SPLIT]
    large = [[_tick(s), v] for s, v in lat if s >= LATENCY_SPLIT]
    return {
        "status": status,
        "series": [
            _series("Bandwidth", "GB/s", bw),
            _series(f"Latency per batch, up to {size_text(LATENCY_SPLIT)}", "µs", small),
            _series(f"Latency per batch, from {size_text(LATENCY_SPLIT)}", "µs", large),
        ],
    }


def _rank_heatmap(io_cases, directions):
    """Per-rank bandwidth at KEY_MSG_SIZE for the largest case both operations ran, or None."""
    by_op = {}
    for (op, buffer_size, batch, qp), case in io_cases.items():
        ranks = (case.get(str(KEY_MSG_SIZE)) or {}).get("Rank_BW_GBps")
        if ranks:
            by_op.setdefault(op, {})[(buffer_size, batch, qp)] = ranks
    common = set(by_op.get("io_write", {})) & set(by_op.get("io_read", {}))
    if not common:
        return None
    shape = max(common)
    write, read = by_op["io_write"][shape], by_op["io_read"][shape]
    ranks = sorted(set(write) | set(read), key=_int)
    values = sorted(v for column in (write, read) for v in column.values())
    median = values[len(values) // 2]
    initiator, target = directions
    return {
        "name": f"MORI-IO per rank at {size_text(KEY_MSG_SIZE)}, both directions · {_case_label(*shape)}",
        "unit": "GB/s",
        "rows": [f"{initiator} → {target}", f"{target} → {initiator}"],
        "cols": [f"rank {_int(r)} · GPU {_int(r)}" for r in ranks],
        "values": [[write.get(r) for r in ranks], [read.get(r) for r in ranks]],
        # The run has no line-rate threshold; flag ranks more than 3% below the grid median.
        "threshold": round(0.97 * median, 2),
        "direction": "higher",
    }


def _key_sizes(sizes):
    """All sizes up to eight, else every third doubling counted back from the largest, so a heatmap stays readable."""
    sizes = sorted(sizes)
    return sizes if len(sizes) <= 8 else sizes[::-1][::3][::-1]


def _size_heatmap(name, unit, rows):
    """Heatmap of ``rows`` ([(label, {size: value})]) over key message sizes, coloured by value."""
    sizes = _key_sizes({size for _label, values in rows for size in values})
    return {
        "name": name,
        "unit": unit,
        "rows": [label for label, _values in rows],
        "cols": [_tick(size) for size in sizes],
        "values": [[values.get(size) for size in sizes] for _label, values in rows],
    }


def _case_label(buffer_size, batch, qp):
    return f"buffer {size_text(buffer_size)} · batch {batch} · QP {qp}"


def _results_table(io_cases, statuses, ibgda, titles, names):
    """One row per IO case x message size, then per IBGDA node x message size."""
    rows = []
    for op in ("io_read", "io_write"):
        for case_key in sorted(k for k in io_cases if k[0] == op):
            status = statuses["|".join(str(part) for part in case_key)]
            for size, entry in sorted(io_cases[case_key].items(), key=lambda item: _int(item[0])):
                rows.append(
                    [
                        titles[op].split(" · ")[0],
                        _case_label(*case_key[1:]),
                        _int(size),
                        entry.get("Avg_BW_GBps", "—"),
                        entry.get("Slowest_rank", "—"),
                        entry.get("Slowest_rank_BW_GBps", "—"),
                        entry.get("Avg_Lat_us", "—"),
                        entry.get("Worst_rank_Lat_us", "—"),
                        "—",
                        status,
                    ]
                )
    for host in sorted(ibgda, key=lambda h: (h not in names, names.get(h, h))):
        for row in sorted(ibgda[host]["rows"], key=lambda r: _int(r.get("size_bytes"))):
            rows.append(
                [titles["ibgda_write"].split(" · ")[0], names.get(host, host), _int(row["size_bytes"])]
                + [row.get("bandwidth_gb", "—"), "—", "—", "—", "—", row.get("rate_mpps", "—"), ibgda[host]["status"]]
            )
    headers = [
        "Operation",
        "Case or node",
        "Message size (B)",
        "Bandwidth (GB/s)",
        "Slowest rank",
        "Slowest rank BW (GB/s)",
        "Latency per batch (usec)",
        "Worst rank latency (usec)",
        "Message rate (Mpps)",
        "Status",
    ]
    return {"headers": headers, "rows": rows}


def _rebuild(store):
    titles, names, directions = _titles(store.get("_meta") or {})
    raw = store.get("_mori") or {}
    io_cases = {}
    for key, value in (raw.get("io") or {}).items():
        op, buffer_size, batch, qp = key.split("|")
        io_cases[(op, _int(buffer_size), _int(batch), _int(qp))] = value["sizes"]
    statuses = {key: value["status"] for key, value in (raw.get("io") or {}).items()}
    groups = {}
    for op in ("io_read", "io_write"):
        nodes = {}
        for case_key in sorted(k for k in io_cases if k[0] == op):
            _op, buffer_size, batch, qp = case_key
            cell = _io_cell(io_cases[case_key], statuses["|".join(str(part) for part in case_key)])
            nodes[_case_label(buffer_size, batch, qp)] = cell
        if nodes:
            cases = [(label, case_key) for label, case_key in zip(nodes, sorted(k for k in io_cases if k[0] == op))]
            heatmaps = []
            for field, name, unit in (
                ("Avg_BW_GBps", "bandwidth", "GB/s"),
                ("Avg_Lat_us", "latency per batch", "µs"),
            ):
                rows = [
                    (label, {_int(s): e.get(field) for s, e in io_cases[case_key].items()}) for label, case_key in cases
                ]
                heatmaps.append(_size_heatmap(f"{titles[op].split(' · ')[0]} {name}", unit, rows))
            nodes["all cases"] = {"status": "na", "heatmaps": heatmaps}
            groups[titles[op]] = {"nodes": nodes}
    ibgda = raw.get("ibgda") or {}
    if ibgda:
        nodes = {}
        for host in sorted(ibgda, key=lambda h: (h not in names, names.get(h, h))):
            rows = sorted(ibgda[host]["rows"], key=lambda row: _int(row.get("size_bytes")))
            nodes[names.get(host, host)] = {
                "status": ibgda[host]["status"],
                "series": [
                    _series("Bandwidth", "GB/s", [[_tick(_int(r["size_bytes"])), r["bandwidth_gb"]] for r in rows]),
                    _series("Message rate", "Mpps", [[_tick(_int(r["size_bytes"])), r["rate_mpps"]] for r in rows]),
                ],
            }
        rows = [
            (names.get(host, host), {_int(r["size_bytes"]): r.get("bandwidth_gb") for r in ibgda[host]["rows"]})
            for host in sorted(ibgda, key=lambda h: (h not in names, names.get(h, h)))
        ]
        nodes["both nodes"] = {
            "status": "na",
            "heatmaps": [_size_heatmap("IBGDA WRITE bandwidth", "GB/s", rows)],
        }
        groups[titles["ibgda_write"]] = {"nodes": nodes}
    store["groups"] = groups
    meta = store.setdefault("_meta", {"suite": "mori_benchmark_test"})
    rank_heatmap = _rank_heatmap(io_cases, directions)
    if rank_heatmap:
        meta["rank_grid"] = {"heatmaps": [rank_heatmap]}
    else:
        meta.pop("rank_grid", None)
    meta["results_table"] = _results_table(io_cases, statuses, ibgda, titles, names)
    return store


def record_io_results(results_store, parsed_results, case_qp_count=None, status="pass", variant=None):
    """Record one IO case (all ranks, every message size) and rebuild the deck groups.

    Each message size keeps the rank mean of Avg_BW_GBps and Avg_Lat_us, the slowest
    rank and its bandwidth, the worst-rank latency, and every rank's bandwidth.
    """
    # run_mori_torch_io_test returns None when the run produced no result table.
    if not parsed_results:
        return results_store
    operation = parsed_results.get("operation", "io")
    qp_count = parsed_results.get("qp_count")
    case_qp = qp_count if case_qp_count is None else case_qp_count

    rows_by_size = {}
    ranks = parsed_results.get("ranks") or {}
    for rank in sorted(ranks, key=int):
        for row in (ranks[rank] or {}).get("rows", []):
            message_size = row.get("MsgSize_B")
            if message_size is None:
                continue
            rows_by_size.setdefault(str(message_size), []).append((rank, row))
    if not rows_by_size:
        return results_store

    _remember_variant(results_store, variant)
    key = "|".join(
        str(part)
        for part in (
            operation,
            _int(parsed_results.get("buffer_size")),
            _int(parsed_results.get("transfer_batch_size")),
            _int(case_qp),
        )
    )
    sizes = {size: _aggregate_rank_rows(rank_rows) for size, rank_rows in rows_by_size.items()}
    results_store.setdefault("_mori", {}).setdefault("io", {})[key] = {"sizes": sizes, "status": status}
    return _rebuild(results_store)


def record_ibgda_results(results_store, parsed_results, status="pass", variant=None):
    """Record each node's dist_write rows and rebuild the deck groups."""
    nodes = parsed_results.get("nodes") or {}
    if not any((node or {}).get("rows") for node in nodes.values()):
        return results_store
    _remember_variant(results_store, variant)
    store = results_store.setdefault("_mori", {}).setdefault("ibgda", {})
    for host, node_results in nodes.items():
        rows = [row for row in (node_results or {}).get("rows", []) if row.get("size_bytes") is not None]
        if rows:
            store[str(host)] = {"rows": rows, "status": status}
    return _rebuild(results_store)


__all__ = ["node_names", "record_ibgda_results", "record_io_results", "size_text"]
