'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Normalize Mori benchmark parser output for Run Deck series reporting.
'''


def record_io_results(results_store, parsed_results, case_qp_count=None, status="pass"):
    operation = parsed_results.get("operation", "io")
    buffer_size = parsed_results.get("buffer_size")
    batch_size = parsed_results.get("transfer_batch_size")
    qp_count = parsed_results.get("qp_count")
    case_qp = qp_count if case_qp_count is None else case_qp_count

    for rank, rank_results in (parsed_results.get("ranks") or {}).items():
        label = f"{operation} · buffer {buffer_size} · batch {batch_size} · QP {qp_count} · rank {rank}"
        if str(case_qp) != str(qp_count):
            label += f" · case QP {case_qp}"

        points = results_store.setdefault(label, {})
        for row in (rank_results or {}).get("rows", []):
            message_size = row.get("MsgSize_B")
            if message_size is None:
                continue
            entry = dict(row)
            entry.update(
                {
                    "test": operation,
                    "node": "—",
                    "rank": rank,
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
            points[str(message_size)] = entry
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
                    "rank": "—",
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
