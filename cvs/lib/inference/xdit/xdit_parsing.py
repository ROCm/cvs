"""
xDiT metric vocabulary for Run Deck reporting.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

XDIT_METRIC_UNITS = {
    "avg_pipe_time_s": "s",
    "avg_total_time_s": "s",
    "clip_score": "score",
    "mean_frame_clip_score": "score",
    "sample_count": "count",
}

XDIT_RESULTS_COLUMNS = (
    ("Model", None),
    ("GPU", None),
    ("Shape", None),
    ("Steps/frames", None),
    ("Cell", None),
    ("Workers", None),
    ("Host", None),
    ("Avg pipe time (s)", "avg_pipe_time_s"),
    ("Avg total time (s)", "avg_total_time_s"),
    ("CLIP score", "clip_score"),
    ("Mean frame CLIP", "mean_frame_clip_score"),
    ("Samples", "sample_count"),
    ("Backend", "backend"),
)

METRIC_TIERS = {
    "latency": ("avg_pipe_time_s", "avg_total_time_s"),
    "quality": ("clip_score", "mean_frame_clip_score"),
    "record": ("sample_count",),
}


def tier_metric_specs(thresholds_cell, tier):
    return {metric: thresholds_cell[metric] for metric in METRIC_TIERS.get(tier, ()) if metric in thresholds_cell}


__all__ = [
    "METRIC_TIERS",
    "XDIT_METRIC_UNITS",
    "XDIT_RESULTS_COLUMNS",
    "tier_metric_specs",
]
