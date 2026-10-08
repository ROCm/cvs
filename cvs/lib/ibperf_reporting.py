'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Normalize ibperf bandwidth and latency results for Run Deck series reporting.
'''

from statistics import mean


def _samples(per_node, field):
    samples = []
    for node, instances in (per_node or {}).items():
        for metrics in (instances or {}).values():
            try:
                samples.append((node, float((metrics or {})[field])))
            except (KeyError, TypeError, ValueError):
                continue
    return samples


def _values(per_node, field):
    return [value for _node, value in _samples(per_node, field)]


def _counts(samples):
    return {"nics": len(samples), "nodes": len({node for node, _value in samples})}


def record_bw_results(results_store, bw_test, msg_size, qp_count, per_node):
    bw = _samples(per_node, "bw")
    # A run whose instances all failed collection has nothing to plot.
    if not bw:
        return results_store
    values = [value for _node, value in bw]
    entry = {
        "test": bw_test,
        "qp_count": qp_count,
        "bw_mean": round(mean(values), 3),
        "bw_min": round(min(values), 3),
        **_counts(bw),
    }
    pps = _values(per_node, "pps")
    if pps:
        entry["pps_mean"] = round(mean(pps), 3)
    results_store.setdefault(f"{bw_test} · QP {qp_count}", {})[str(msg_size)] = entry
    return results_store


def record_lat_results(results_store, lat_test, msg_size, per_node):
    avg = _samples(per_node, "t_avg")
    if not avg:
        return results_store
    entry = {"test": lat_test, "t_avg": round(mean(value for _node, value in avg), 3), **_counts(avg)}
    for field in ("t_99_pct", "t_max"):
        values = _values(per_node, field)
        if values:
            entry[field] = round(max(values), 3)
    results_store.setdefault(lat_test, {})[str(msg_size)] = entry
    return results_store


__all__ = ["record_bw_results", "record_lat_results"]
