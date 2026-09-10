'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Run-card rows for the RVS health suite.
'''

from cvs.lib.report.rundeck.config_builder import provenance_link_rows


def rvs_run_card_display(variant, provenance):
    rows = [
        ("Suite", "RVS", False),
        ("Test level", str(getattr(variant, "rvs_test_level", "—")), False),
        ("RVS path", str(getattr(variant, "rvs_path", None) or "—"), False),
    ]
    rows.extend(provenance_link_rows(provenance))
    return rows
