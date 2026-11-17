'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.
'''

import logging
import sys

from cvs.lib.report.consolidate import ConsolidationError, consolidate

from .base import SubcommandPlugin


class ConsolidateReportPlugin(SubcommandPlugin):
    def get_name(self):
        return "consolidate-report"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser(
            "consolidate-report",
            help="Combine the Run Decks under one or more run directories into one report zip",
        )
        parser.add_argument(
            "--log-dir",
            required=True,
            help="Directory to search recursively for Run Decks, in run folders and zip bundles. "
            "Separate several directories with commas.",
        )
        parser.add_argument(
            "--output",
            help="Directory for the report zip. Default: the --log-dir directory, "
            "or the current directory when several are given.",
        )
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
Consolidated Report Commands:
  cvs consolidate-report --log-dir <dir>                    Report on every Run Deck under <dir>; zip written to <dir>
  cvs consolidate-report --log-dir <dir1>,<dir2>            Report on several directories; zip written to the current directory
  cvs consolidate-report --log-dir <dir> --output <outDir>  Write the zip to <outDir>"""

    def run(self, args):
        # A CLI run has no log handler yet; without one, INFO progress lines would be dropped.
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        try:
            consolidate(args.log_dir, output_dir=args.output)
        except ConsolidationError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
