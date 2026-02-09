#!/usr/bin/env python3
'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import argparse
import sys
from jinja2 import Template
from importlib import resources
from cvs.cli_plugins.generate_plugin import GeneratorPlugin


class ClusterJsonGenerator(GeneratorPlugin):
    """Generator plugin for creating cluster JSON configuration files"""

    def get_name(self):
        return "cluster_json"

    def get_description(self):
        return "Generate cluster JSON configuration file from hosts list"

    def get_parser(self):
        parser = argparse.ArgumentParser(description="Generate cluster json file")

        # Create mutually exclusive group for input sources
        input_group = parser.add_mutually_exclusive_group(required=True)
        input_group.add_argument(
            "--input_hosts_file",
            help="Input file with host IPs - one address per line, supports ranges like 192.168.1.10-20 and hostname[1-10]",
        )
        input_group.add_argument(
            "--hosts",
            help="Comma-separated list of host IPs or hostnames, supports ranges like 192.168.1.10-20 and hostname[1-10]",
        )

        parser.add_argument("--output_json_file", required=True, help="Output cluster file in JSON format")
        parser.add_argument("--username", required=True, help="Username to ssh to the hosts")
        parser.add_argument("--key_file", required=True, help="keyfile with private keys")
        parser.add_argument("--head_node", help="IP of the head node (optional, defaults to first host in hosts file)")
        return parser

    def expand_ip_range(self, ip_range):
        """
        Expand IP range like '192.168.1.10-20' to list of IPs
        """
        if '-' not in ip_range:
            return [ip_range]

        # Split into base and range
        parts = ip_range.rsplit('-', 1)
        if len(parts) != 2:
            return [ip_range]

        base_ip = parts[0]
        end_octet = parts[1]

        # Extract the last octet from base_ip
        base_parts = base_ip.split('.')
        if len(base_parts) != 4:
            return [ip_range]

        try:
            start_octet = int(base_parts[3])
            end_octet = int(end_octet)
        except ValueError:
            return [ip_range]

        # Generate IP list
        ips = []
        for octet in range(start_octet, end_octet + 1):
            ip = f"{base_parts[0]}.{base_parts[1]}.{base_parts[2]}.{octet}"
            ips.append(ip)

        return ips

    def expand_hostname_bracket_range(self, hostname):
        """
        Expand hostname range with bracket notation like 'mia1-p01-g[24-30]' to list of hostnames
        """
        import re

        # Check for bracket notation pattern
        match = re.match(r'^(.+)\[(\d+)-(\d+)\](.*)$', hostname)
        if not match:
            return [hostname]

        prefix = match.group(1)
        start_num = int(match.group(2))
        end_num = int(match.group(3))
        suffix = match.group(4)

        # Determine padding (preserve leading zeros)
        start_str = match.group(2)
        padding = len(start_str) if start_str.startswith('0') else 0

        # Generate hostname list
        hostnames = []
        for num in range(start_num, end_num + 1):
            if padding > 0:
                num_str = str(num).zfill(padding)
            else:
                num_str = str(num)
            hostname = f"{prefix}{num_str}{suffix}"
            hostnames.append(hostname)

        return hostnames

    def expand_range(self, entry):
        """
        Unified method to expand both IP ranges and hostname bracket ranges
        """
        # First try bracket notation (hostname ranges)
        if '[' in entry:
            return self.expand_hostname_bracket_range(entry)

        # Then try IP range notation
        if '-' in entry and '.' in entry:
            return self.expand_ip_range(entry)

        # No range, return as-is
        return [entry]

    def parse_hosts_file(self, filename):
        """
        Parse hosts file with support for IP ranges and hostname bracket ranges
        """
        node_list = []
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Expand ranges (both IP and hostname bracket notation)
                expanded = self.expand_range(line)
                node_list.extend(expanded)

        return node_list

    def parse_hosts_list(self, hosts_string):
        """
        Parse comma-separated hosts string with support for IP ranges and hostname bracket ranges
        """
        node_list = []
        hosts = [h.strip() for h in hosts_string.split(',')]
        for host in hosts:
            if not host:
                continue
            # Expand ranges (both IP and hostname bracket notation)
            expanded = self.expand_range(host)
            node_list.extend(expanded)

        return node_list

    def determine_head_node(self, node_list, specified_head_node=None):
        """Determine the head node IP from node list and optional specification"""
        if specified_head_node:
            head_node_ip = specified_head_node
            # If head node is in the node list, move it to the front
            if head_node_ip in node_list:
                node_list.remove(head_node_ip)
                node_list.insert(0, head_node_ip)
            # If head node is not in the list, keep it separate (don't add to node_list)
        else:
            head_node_ip = node_list[0]

        return head_node_ip

    def generate(self, args):
        # Parse hosts from file or comma-separated list
        if args.input_hosts_file:
            node_list = self.parse_hosts_file(args.input_hosts_file)
        else:
            node_list = self.parse_hosts_list(args.hosts)

        if not node_list:
            print("ERROR !! No hosts provided, this is mandatory, aborting !!")
            sys.exit(1)

        # Determine head node
        head_node_ip = self.determine_head_node(node_list, getattr(args, 'head_node', None))

        # Load and render template using importlib.resources
        template_content = (
            resources.files('cvs.input.templates.cluster_file').joinpath('cluster_json.template').read_text()
        )

        template = Template(template_content)
        rendered_json = template.render(
            username=args.username, priv_key_file=args.key_file, head_node_ip=head_node_ip, node_list=node_list
        )

        # Write the rendered JSON to output file
        with open(args.output_json_file, "w") as fp:
            fp.write(rendered_json)

        print(f"Generated cluster JSON file: {args.output_json_file}")
        print(f"Head node: {head_node_ip}")
        print(f"Total nodes: {len(node_list)}")


def main():
    generator = ClusterJsonGenerator()
    parser = generator.get_parser()
    args = parser.parse_args()
    generator.generate(args)


if __name__ == "__main__":
    main()
