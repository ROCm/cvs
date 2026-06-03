"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

`cvs migrate-config` -- thin CLI shell over `cvs.lib.config.migrate.migrate_vllm_megaconfig`.

The plugin reads a v1 mega-config YAML, asks the engine to split it into
one v2 config per model, self-checks each emitted v2 via `parse_config`
(the runtime path -- resolves ${env:...} placeholders), and writes the
result(s) to a single output YAML.

Failure modes are translated to non-zero `SystemExit` so users never see a
raw traceback. `ConfigError` from the self-check exits with code 2
specifically -- a v2 file the loader would reject must never reach disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from cvs.cli_plugins.base import SubcommandPlugin
from cvs.lib.config.loader import ConfigError, parse_config
from cvs.lib.config.migrate import migrate_vllm_megaconfig


class MigrateConfigPlugin(SubcommandPlugin):
    """`cvs migrate-config <v1-yaml> -o <v2-yaml>` -- rewrite v1 to v2."""

    def get_name(self):
        return "migrate-config"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser(
            "migrate-config",
            help="Migrate a v1 mega-config YAML to a v2 YAML",
        )
        parser.add_argument(
            "input_file",
            help="Path to the v1 mega-config YAML to migrate",
        )
        parser.add_argument(
            "-o",
            "--output",
            required=True,
            help="Path to write the v2 YAML output",
        )
        parser.add_argument(
            "-t",
            "--target-gpu",
            dest="target_gpu",
            default="mi355x",
            help="Target GPU label stamped onto every emitted config (default: mi355x)",
        )
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
Migrate Commands:
  cvs migrate-config v1_mega.yaml -o v2.yaml
  cvs migrate-config v1_mega.yaml -o v2.yaml -t mi300x"""

    def run(self, args):
        input_path = Path(args.input_file)
        output_path = Path(args.output)
        target_gpu = args.target_gpu

        # --- Read v1 ---------------------------------------------------------
        if not input_path.exists():
            print(f"Error: input file does not exist: {input_path}", file=sys.stderr)
            sys.exit(1)
        try:
            raw_text = input_path.read_text()
        except OSError as exc:
            print(f"Error: cannot read input file {input_path}: {exc}", file=sys.stderr)
            sys.exit(1)
        try:
            raw = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            print(f"Error: malformed YAML in {input_path}: {exc}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(raw, dict) or not raw.get("benchmark_params"):
            print(
                f"Error: input {input_path} has no 'benchmark_params' to migrate",
                file=sys.stderr,
            )
            sys.exit(1)

        # --- Engine ----------------------------------------------------------
        try:
            migrated = migrate_vllm_megaconfig(raw, target_gpu)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        if not migrated:
            print("Error: migration produced no configs", file=sys.stderr)
            sys.exit(1)

        # --- Self-check via parse_config (runtime path, resolves env refs) ---
        # This is the fail-fast: a v2 dict the loader would reject must never
        # be written to disk. ConfigError -> exit 2 specifically.
        try:
            for v2 in migrated.values():
                parse_config(v2)
        except ConfigError as exc:
            print(f"Error: migrated config failed self-check: {exc}", file=sys.stderr)
            sys.exit(2)

        # --- Write -----------------------------------------------------------
        # Serialize the PRE-parse_config dicts, so deferred placeholders like
        # ${env:HF_TOKEN} survive verbatim (parse_config resolves them in
        # memory but never mutates the source dict).
        configs = list(migrated.values())
        try:
            if len(configs) == 1:
                output_path.write_text(yaml.safe_dump(configs[0], sort_keys=False))
            else:
                # Multi-doc YAML keeps every model present and individually loadable.
                output_path.write_text(yaml.safe_dump_all(configs, sort_keys=False))
        except OSError as exc:
            print(f"Error: cannot write output file {output_path}: {exc}", file=sys.stderr)
            sys.exit(1)

        print(f"Migrated {len(configs)} config(s) from {input_path.name} -> {output_path}")
