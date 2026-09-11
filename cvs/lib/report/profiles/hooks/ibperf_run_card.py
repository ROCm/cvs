'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

Run-card fields for the ibperf bandwidth suite.
'''


def _format_values(values):
    if isinstance(values, (list, tuple)):
        return ", ".join(str(value) for value in values)
    if values in (None, ""):
        return "—"
    return str(values)


def ibperf_run_card_display(variant, _provenance):
    config = variant if isinstance(variant, dict) else {}
    return [
        ("GID index", _format_values(config.get("gid_index")), False),
        ("Duration (s)", _format_values(config.get("duration")), False),
        ("Message sizes", _format_values(config.get("msg_size_list")), False),
        ("QP counts", _format_values(config.get("qp_count_list")), False),
    ]


__all__ = ["ibperf_run_card_display"]
