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


def rccl_run_card_display(variant, provenance):
    nnodes = getattr(variant, "nnodes", None)
    rows = [
        ("Framework", getattr(variant, "framework", None) or "rccl", False),
        ("Nodes", str(nnodes) if nnodes not in (None, "") else "\u2014", False),
        ("MPI nodes", _mpi_field(variant, "no_of_nodes"), False),
        ("Local ranks", _mpi_field(variant, "no_of_local_ranks"), False),
        ("Collectives", _test_field(variant, "rccl_collective"), False),
        ("Msg size", f"{_test_field(variant, 'start_msg_size')} .. {_test_field(variant, 'end_msg_size')}", False),
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
