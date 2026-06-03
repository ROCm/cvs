"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import re
from datetime import timedelta

from cvs.cli_plugins.base import SubcommandPlugin

_DURATION_RE = re.compile(r"^\s*(\d+)\s*([smhdw])\s*$")
_UNIT_SECONDS = {
    "s": 1,
    "m": 60,
    "h": 60 * 60,
    "d": 60 * 60 * 24,
    "w": 60 * 60 * 24 * 7,
}


def _parse_duration(text):
    """Parse '1h', '7d', '30m', '45s', '2w' into a timedelta. Raises ValueError on garbage."""
    if text is None:
        return None
    m = _DURATION_RE.match(text)
    if not m:
        raise ValueError(f"--since must be a duration like '1h', '7d', '30m', got {text!r}")
    n, unit = int(m.group(1)), m.group(2)
    return timedelta(seconds=n * _UNIT_SECONDS[unit])


class ExportPlugin(SubcommandPlugin):
    """`cvs export` -- flatten N DTNI run directories into one Parquet fact table."""

    def get_name(self):
        return "export"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser(
            "export",
            help="Flatten DTNI run manifests under a directory into one Parquet fact table",
        )
        parser.add_argument(
            "--artifact-dir",
            required=True,
            dest="artifact_dir",
            help="Root directory containing DTNI run manifests",
        )
        parser.add_argument(
            "-o",
            "--out",
            required=True,
            dest="out",
            help="Output Parquet file path",
        )
        parser.add_argument(
            "--since",
            default=None,
            help="Only include runs whose started_at is within this duration (e.g. '1h', '7d').",
        )
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
Export Commands:
  cvs export --artifact-dir /tmp/cvs/artifacts -o runs.parquet      Build a fact table from all runs
  cvs export --artifact-dir /tmp/cvs/artifacts --since 7d -o w.parquet  Only runs from the last 7 days"""

    def run(self, args):
        # Lazy import: keep the engine (pandas/pyarrow) out of the CLI startup path.
        from cvs.lib.manifest.export import export_runs

        # Validate --since BEFORE doing filesystem work so garbage values fail loudly.
        window = _parse_duration(getattr(args, "since", None))
        out = export_runs(args.artifact_dir, args.out, since=window)
        print(f"Wrote fact table: {out}")
        return 0
