'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Run-card hook for RCCL perf, regression, and pairwise suite stems.
'''

from cvs.lib.report.rundeck.config_builder import provenance_link_rows, thresholds_run_card_row


def _mpi_field(variant, key, default="\u2014"):
    mpi = getattr(variant, "mpi_params", None) or {}
    if isinstance(mpi, dict) and mpi.get(key) not in (None, ""):
        return str(mpi[key])
    return default


def _test_field(variant, key, default="\u2014"):
    params = getattr(variant, "rccl_test_params", None) or {}
    if isinstance(params, dict) and params.get(key) not in (None, ""):
        value = params[key]
        if isinstance(value, list):
            return ", ".join(str(item) for item in value)
        return str(value)
    return default


def _measured_fields(variant):
    raw_results = getattr(variant, "raw_results", None)
    if raw_results is None:
        return (
            _test_field(variant, "rccl_collective"),
            f"{_test_field(variant, 'start_msg_size')} .. {_test_field(variant, 'end_msg_size')}",
        )
    collectives = set()
    sizes = set()
    for results in raw_results.values():
        for row in results or []:
            if row.get("name"):
                collectives.add(str(row["name"]))
            if row.get("size") is not None:
                sizes.add(int(row["size"]))
    return (
        ", ".join(sorted(collectives)) or "\u2014",
        f"{min(sizes)} .. {max(sizes)}" if sizes else "\u2014",
    )


def rccl_run_card_display(variant, provenance):
    nnodes = getattr(variant, "nnodes", None)
    mpi_nodes = _mpi_field(variant, "no_of_nodes")
    local_ranks = _mpi_field(variant, "no_of_local_ranks")
    run_nodes = getattr(variant, "run_nodes", None) or {}
    if run_nodes:
        nnodes = len({node for nodes in run_nodes.values() for node in nodes})
        mpi_nodes = ", ".join(str(count) for count in sorted({len(nodes) for nodes in run_nodes.values()}))
    try:
        total_ranks = ", ".join(str(int(count) * int(local_ranks)) for count in mpi_nodes.split(", "))
    except (TypeError, ValueError):
        total_ranks = "\u2014"
    collectives, msg_size = _measured_fields(variant)
    rows = [
        ("Framework", getattr(variant, "framework", None) or "rccl", False),
        ("Nodes", str(nnodes) if nnodes not in (None, "") else "\u2014", False),
        ("MPI nodes", mpi_nodes, False),
        ("Local ranks", local_ranks, False),
        ("MPI ranks", total_ranks, False),
        ("Collectives", collectives, False),
        ("Msg size (bytes)", msg_size, False),
        thresholds_run_card_row(variant),
    ]
    nic = None
    cvs_params = getattr(variant, "cvs_params", None) or {}
    if isinstance(cvs_params, dict):
        nic = cvs_params.get("nic_model")
    if nic:
        rows.insert(-1, ("NIC model", str(nic), False))
    rows.extend(provenance_link_rows(provenance))
    return rows


__all__ = ["rccl_run_card_display"]
