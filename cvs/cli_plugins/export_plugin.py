"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

from cvs.cli_plugins.base import SubcommandPlugin


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
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
Export Commands:
  cvs export --artifact-dir /tmp/cvs/artifacts -o runs.parquet      Build a fact table from all runs"""

    def run(self, args):
        # Lazy import: keep the engine (pandas/pyarrow) out of the CLI startup path.
        from cvs.lib.manifest.export import export_runs

        out = export_runs(args.artifact_dir, args.out)
        print(f"Wrote fact table: {out}")
        return 0
