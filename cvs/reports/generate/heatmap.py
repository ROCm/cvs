#!/usr/bin/env python3
"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

import argparse
import os
import json
import traceback

from datetime import datetime
from cvs.cli_plugins.generate_plugin import GeneratorPlugin
from cvs.lib import html_lib


class HeatmapGenerator(GeneratorPlugin):
    """Generator plugin for creating RCCL performance heatmap from JSON files"""

    def get_name(self):
        return "heatmap"

    def get_description(self):
        return "Generate RCCL performance heatmap from actual and reference JSON files"

    def get_parser(self):
        parser = argparse.ArgumentParser(
            description="Generate RCCL performance heatmap HTML report",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic heatmap generation
  cvs generate heatmap -a actual.json -r reference.json -o output.html

  # With custom title and metadata
  cvs generate heatmap -a results.json -r golden.json \\
      -o heatmap.html -t "Multi-Node RCCL Performance" --metadata

  # Auto-named output in /tmp
  cvs generate heatmap -a actual.json -r reference.json

Notes:
  - Both JSON files must be in RCCL graph format (from convert_to_graph_dict)
  - Reference JSON contains baseline/golden performance metrics
  - Actual JSON contains test results to compare against reference

JSON Format:
  Basic format (results only):
    {
      "all_reduce_perf-float-8": {
        "1048576": {"bus_bw": "245.5", "alg_bw": "122.75", "time": "42.3"}
      }
    }

  With metadata (use --metadata flag):
    {
      "metadata": {
        "date": "2026-02-09 10:30:00",
        "rocm_version": "7.0.2",
        "gpu_model": "MI325X",
        "kernel": "6.8.0-49-generic"
      },
      "result": {
        "all_reduce_perf-float-8": {
          "1048576": {"bus_bw": "245.5", "alg_bw": "122.75", "time": "42.3"}
        }
      }
    }
            """,
        )

        parser.add_argument("-a", "--actual", required=True, help="Path to actual results JSON file")

        parser.add_argument("-r", "--reference", required=True, help="Path to golden reference JSON file")

        parser.add_argument(
            "-o", "--output", help="Output HTML file path (default: /tmp/rccl_heatmap_<timestamp>.html)"
        )

        parser.add_argument(
            "-t",
            "--title",
            default="RCCL Performance Heatmap",
            help="Heatmap title (default: 'RCCL Performance Heatmap')",
        )

        parser.add_argument(
            "--metadata",
            action="store_true",
            help="Include metadata table (requires actual JSON to have 'metadata' key)",
        )

        parser.add_argument("--no-data-table", action="store_true", help="Exclude data table from output")

        return parser

    def generate(self, args):
        """Generate the RCCL performance heatmap HTML report"""

        # Validate input files
        if not os.path.exists(args.actual):
            print(f"Error: Actual results file not found: {args.actual}")
            return 1

        if not os.path.exists(args.reference):
            print(f"Error: Reference file not found: {args.reference}")
            return 1

        # Determine output file
        if args.output:
            heatmap_file = args.output
        else:
            time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            heatmap_file = f'/tmp/rccl_heatmap_{time_stamp}.html'

        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.abspath(heatmap_file))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print("Generating RCCL performance heatmap...")
        print(f"  Actual results:    {args.actual}")
        print(f"  Reference results: {args.reference}")
        print(f"  Output file:       {heatmap_file}")
        print(f"  Title:             {args.title}")

        try:
            # Validate JSON files can be loaded
            with open(args.actual, 'r') as f:
                actual_data = json.load(f)

            # Check if metadata exists in actual data
            has_metadata = 'metadata' in actual_data if isinstance(actual_data, dict) else False

            # Generate HTML heatmap
            html_lib.add_html_begin(heatmap_file)

            # Main heatmap visualization
            html_lib.build_rccl_heatmap(heatmap_file, 'heatmapdiv', args.title, args.actual, args.reference)

            # Optionally add metadata table
            if args.metadata:
                if has_metadata:
                    print("  Including metadata table...")
                    html_lib.build_rccl_heatmap_metadata_table(heatmap_file, args.actual, args.reference)
                else:
                    print("  Warning: --metadata specified but actual JSON has no 'metadata' key")

            # Optionally add data table
            if not args.no_data_table:
                print("  Including data table...")
                html_lib.build_rccl_heatmap_table(heatmap_file, 'Heatmap Data Table', args.actual, args.reference)

            html_lib.add_html_end(heatmap_file)

            print("\n✓ Heatmap generated successfully!")
            print("\nOpen in browser:")
            print(f"  file://{os.path.abspath(heatmap_file)}")

            return 0

        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format - {e}")
            return 1
        except Exception as e:
            print(f"Error generating heatmap: {e}")
            traceback.print_exc()
            return 1
