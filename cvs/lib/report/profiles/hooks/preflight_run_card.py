'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Preflight run-card hook for the JSON deck profile.
'''


def preflight_run_card_display(context, _provenance):
    context = context if isinstance(context, dict) else {}
    enabled_checks = context.get('enabled_checks') or []
    enabled_display = ', '.join(str(check) for check in enabled_checks) or 'None'
    rows = [
        ('Cluster size', str(context.get('cluster_size', 0)), False),
        ('Checks enabled', enabled_display, False),
    ]
    detailed_report_path = context.get('detailed_report_path')
    if detailed_report_path:
        rows.append(('Detailed preflight report', str(detailed_report_path), True))
    return rows


__all__ = ['preflight_run_card_display']
