#!/usr/bin/env python3
'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication
and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import argparse
import re
import sys
from dataclasses import dataclass, field
from importlib import resources
from typing import Dict, List, Optional

from jinja2 import Template

from cvs.input.generate.cluster_json import ClusterJsonGenerator


# ---------------------------------------------------------------------------
# Data model for a single parsed rack group
# ---------------------------------------------------------------------------


@dataclass
class RackEntry:
    rack_id: str
    compute_nodes: List[str] = field(default_factory=list)
    switch_trays: List[str] = field(default_factory=list)
    platform: str = "HeliosP"


# ---------------------------------------------------------------------------
# argv pre-processor: split on --rack0 / --rack1 / ... --rackN markers
# ---------------------------------------------------------------------------


def split_rack_groups(argv: List[str]):
    """
    Split argv into global args and per-rack arg groups.

    Everything before the first --rackN marker is treated as a global arg.
    Each --rackN marker starts a new rack group; args that follow (until the
    next --rackN marker or end of argv) belong to that rack.

    Returns:
        global_args  (list[str])  – args before the first --rackN
        rack_groups  (dict[str, list[str]])  – {"--rack0": [...], "--rack1": [...], ...}
                     ordered by marker name
    """
    global_args: List[str] = []
    rack_groups: Dict[str, List[str]] = {}
    current: Optional[str] = None

    for arg in argv:
        if re.match(r'^--rack\d+$', arg):
            current = arg
            rack_groups[current] = []
        elif current is not None:
            rack_groups[current].append(arg)
        else:
            global_args.append(arg)

    return global_args, rack_groups


def _rack_sort_key(marker: str) -> int:
    """Sort --rackN markers numerically."""
    m = re.match(r'^--rack(\d+)$', marker)
    return int(m.group(1)) if m else 0


def build_rack_parser() -> argparse.ArgumentParser:
    """Return an ArgumentParser for per-rack sub-args (used after split_rack_groups)."""
    p = argparse.ArgumentParser(prog='--rackN', add_help=False)
    p.add_argument(
        '--id',
        dest='rack_id',
        default=None,
        help="Rack identifier, e.g. 'rack-01'. Defaults to rack-00, rack-01, ... by index.",
    )
    p.add_argument(
        '--computes',
        required=True,
        help="Comma-separated compute node IPs/hostnames; supports ranges like 10.0.0.1-5 and node[1-10]",
    )
    p.add_argument(
        '--switches', default='', help="Comma-separated switch tray IPs; supports same range syntax as --computes"
    )
    p.add_argument('--platform', default='HeliosP', help="ARC platform name (default: HeliosP)")
    return p


# ---------------------------------------------------------------------------
# Generator plugin
# ---------------------------------------------------------------------------


class RackClusterJsonGenerator(ClusterJsonGenerator):
    """
    Generator plugin for rack-aware cluster.json files.

    Inherits IP/hostname range expansion from ClusterJsonGenerator so that
    --computes and --switches accept the same range notation as --hosts:
      10.0.0.1,10.0.0.2-7     (IP range with dash)
      node[01-10]                       (bracket hostname range)

    CLI usage (single rack):
      cvs generate rack_cluster_json \\
        --username ichristo --key_file ~/.ssh/id_rsa \\
        --switch_ssh_user admin --switch_ssh_password password \\
        --output_json_file cluster.json \\
        --rack0 --id rack-01 --computes 10.0.0.1,10.0.0.2 \\
                --switches 10.0.2.1,10.0.2.2,10.0.2.3

    CLI usage (multi-rack):
      cvs generate rack_cluster_json \\
        --username ichristo --key_file ~/.ssh/id_rsa \\
        --switch_ssh_user admin --switch_ssh_password password \\
        --output_json_file cluster.json \\
        --rack0 --id rack-01 --computes 10.0.0.1-5 --switches 10.0.1.1,10.0.1.2 \\
        --rack1 --id rack-02 --computes 10.0.0.6-10 --switches 10.0.1.3
    """

    def supports_raw_argv(self) -> bool:
        return True

    def get_name(self) -> str:
        return "rack_cluster_json"

    def get_description(self) -> str:
        return "Generate rack-aware cluster JSON with compute and switch tray topology"

    def get_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Generate rack-aware cluster.json from --rackN argument groups",
            epilog=(
                "Each --rackN group accepts: --id --computes --switches --platform. "
                "N is zero-based and must be contiguous (--rack0, --rack1, ...). "
                "Global switch credentials (--switch_ssh_user, --switch_ssh_password, "
                "--switch_ssh_key_file) apply uniformly to all racks."
            ),
        )
        parser.add_argument('--username', required=True, help="SSH username for compute nodes")
        parser.add_argument('--key_file', required=True, help="Path to SSH private key for compute nodes")
        parser.add_argument('--switch_ssh_user', default=None, help="SSH username for all switch trays")
        parser.add_argument('--switch_ssh_password', default=None, help="SSH password for all switch trays")
        parser.add_argument(
            '--switch_ssh_key_file',
            default=None,
            help="Global SSH private key path for all switch trays; takes priority over --switch_ssh_password",
        )
        parser.add_argument(
            '--head_node', default=None, help="IP of the head node (defaults to first compute node across all racks)"
        )
        parser.add_argument('--output_json_file', required=True, help="Output cluster JSON file path")
        return parser

    def generate(self, args) -> None:
        # 1. Split raw generator args on --rackN boundaries.
        #    _raw_args is injected by _run_generator when supports_raw_argv() is True.
        #    Fall back to sys.argv[3:] when invoked as a standalone script.
        raw_argv = getattr(args, '_raw_args', None) or sys.argv[3:]
        global_argv, rack_groups_raw = split_rack_groups(raw_argv)

        if not rack_groups_raw:
            print("ERROR: No rack groups found. Provide at least --rack0 with --computes and --switches.")
            sys.exit(1)

        # 2. Parse global args
        main_args = self.get_parser().parse_args(global_argv)

        # 3. Parse each rack group in numeric order
        rack_parser = build_rack_parser()
        racks: Dict[str, RackEntry] = {}
        all_compute_nodes: List[str] = []

        for idx, marker in enumerate(sorted(rack_groups_raw.keys(), key=_rack_sort_key)):
            rack_argv = rack_groups_raw[marker]
            try:
                rack_args = rack_parser.parse_args(rack_argv)
            except SystemExit:
                print(f"ERROR: Failed to parse args for {marker}.")
                sys.exit(1)

            rack_id = rack_args.rack_id or f"rack-{idx:02d}"

            compute_nodes = self.parse_hosts_list(rack_args.computes)
            if not compute_nodes:
                print(f"ERROR: {marker} --computes produced no hosts.")
                sys.exit(1)

            switch_trays = self.parse_hosts_list(rack_args.switches) if rack_args.switches else []

            entry = RackEntry(
                rack_id=rack_id,
                compute_nodes=compute_nodes,
                switch_trays=switch_trays,
                platform=rack_args.platform,
            )
            racks[rack_id] = entry
            all_compute_nodes.extend(compute_nodes)

        # 4. Determine head node
        head_node_ip = main_args.head_node or (all_compute_nodes[0] if all_compute_nodes else "")

        # 5. Render Jinja2 template
        template_content = (
            resources.files('cvs.input.templates.cluster_file').joinpath('rack_cluster_json.template').read_text()
        )
        template = Template(template_content)
        rendered = template.render(
            username=main_args.username,
            priv_key_file=main_args.key_file,
            head_node_ip=head_node_ip,
            switch_ssh_user=main_args.switch_ssh_user or "",
            switch_ssh_password=main_args.switch_ssh_password or "",
            switch_ssh_key_file=main_args.switch_ssh_key_file or "",
            racks=racks,
        )

        # 6. Write output
        with open(main_args.output_json_file, 'w') as fp:
            fp.write(rendered)

        print(f"Generated rack cluster JSON: {main_args.output_json_file}")
        print(f"Head node: {head_node_ip}")
        print(f"Total compute nodes: {len(all_compute_nodes)}")
        for rack_id, entry in racks.items():
            print(
                f"  {rack_id}: {len(entry.compute_nodes)} compute, {len(entry.switch_trays)} switches "
                f"(platform={entry.platform})"
            )


def main():
    generator = RackClusterJsonGenerator()
    parser = generator.get_parser()
    args = parser.parse_args()
    generator.generate(args)


if __name__ == "__main__":
    main()
