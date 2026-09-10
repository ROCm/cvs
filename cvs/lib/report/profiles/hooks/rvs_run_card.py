'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Run-card rows for the RVS health suite.
'''

from cvs.lib.report.rundeck.config_builder import provenance_link_rows


def rvs_run_card_display(variant, provenance):
    meta = variant if isinstance(variant, dict) else {}
    rows = [
        ("Suite", "RVS", False),
        ("RVS version", str(meta.get("rvs_version") or "—"), False),
        ("Test level", str(meta.get("rvs_test_level") if meta.get("rvs_test_level") is not None else "—"), False),
    ]
    rows.extend(provenance_link_rows(provenance or {}))
    return rows
