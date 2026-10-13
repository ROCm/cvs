'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Normalize Mori benchmark parser output for Run Deck series reporting.
'''

_IO_SHARED_FIELDS = ("MsgSize_B", "BatchSize", "TotalSize_MB")


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
            {"Avg_BW_GBps": _mean(bandwidth), "Slowest_rank": slowest_rank, "Slowest_rank_BW_GBps": slowest_bw}
        )
    latency = _rank_values(rank_rows, "Avg_Lat_us")
    if latency:
        entry.update({"Avg_Lat_us": _mean(latency), "Worst_rank_Lat_us": max(value for _rank, value in latency)})
    max_bw = _rank_values(rank_rows, "Max_BW_GBps")
    if max_bw:
        entry["Max_BW_GBps"] = max(value for _rank, value in max_bw)
    min_lat = _rank_values(rank_rows, "Min_Lat_us")
    if min_lat:
        entry["Min_Lat_us"] = min(value for _rank, value in min_lat)
    return entry


def record_io_results(results_store, parsed_results, case_qp_count=None, status="pass"):
    """Record one series per IO case, aggregating every rank at each message size.

    Avg_BW_GBps and Avg_Lat_us are rank means; Slowest_rank, Slowest_rank_BW_GBps and
    Worst_rank_Lat_us keep the outlier ranks visible; Max_BW_GBps and Min_Lat_us are
    the best rank's values.
    """
    # run_mori_torch_io_test returns None when the run produced no result table.
    if not parsed_results:
        return results_store
    operation = parsed_results.get("operation", "io")
    buffer_size = parsed_results.get("buffer_size")
    batch_size = parsed_results.get("transfer_batch_size")
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

    label = f"{operation} · buffer {buffer_size} · batch {batch_size} · QP {qp_count}"
    if str(case_qp) != str(qp_count):
        label += f" · case QP {case_qp}"
    points = results_store.setdefault(label, {})
    for message_size, rank_rows in rows_by_size.items():
        entry = _aggregate_rank_rows(rank_rows)
        entry.update(
            {
                "test": operation,
                "node": "—",
                "buffer_size": buffer_size,
                "transfer_batch_size": batch_size,
                "qp_count": qp_count,
                "case_qp_count": case_qp,
                "processes": "—",
                "ctas": "—",
                "threads": "—",
                "iterations": "—",
                "status": status,
            }
        )
        points[message_size] = entry
    return results_store


def record_ibgda_results(results_store, parsed_results, status="pass"):
    operation = parsed_results.get("operation", "ibgda_write")
    for node, node_results in (parsed_results.get("nodes") or {}).items():
        label = (
            f"{operation} · procs {parsed_results.get('processes')} "
            f"· CTAs {parsed_results.get('ctas')} · QP {parsed_results.get('qp_count')} "
            f"· {node}"
        )
        points = results_store.setdefault(label, {})
        for row in (node_results or {}).get("rows", []):
            message_size = row.get("size_bytes")
            if message_size is None:
                continue
            entry = dict(row)
            entry.update(
                {
                    "test": operation,
                    "node": node,
                    "ranks": "—",
                    "buffer_size": "—",
                    "transfer_batch_size": "—",
                    "qp_count": parsed_results.get("qp_count"),
                    "case_qp_count": parsed_results.get("qp_count"),
                    "processes": parsed_results.get("processes"),
                    "ctas": parsed_results.get("ctas"),
                    "threads": parsed_results.get("threads"),
                    "iterations": parsed_results.get("iterations"),
                    "status": status,
                }
            )
            points[str(message_size)] = entry
    return results_store


__all__ = ["record_ibgda_results", "record_io_results"]
