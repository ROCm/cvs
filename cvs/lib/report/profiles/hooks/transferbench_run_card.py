'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

from cvs.lib.report.rundeck.config_builder import provenance_link_rows


def transferbench_run_card_display(variant, provenance):
    variant = variant or {}
    tests = variant.get('tests_enabled') or []
    duration = float(variant.get('duration_seconds') or 0.0)
    rows = [
        ('ROCm path', variant.get('rocm_path') or '—', False),
        ('Tests enabled', ', '.join(tests) if tests else '—', False),
        ('Duration', f'{duration:.1f} s', False),
    ]
    rows.extend(provenance_link_rows(provenance))
    return rows


__all__ = ['transferbench_run_card_display']
