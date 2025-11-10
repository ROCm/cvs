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
from cvs.input.generate.base import GeneratorPlugin

class ClusterJsonGenerator(GeneratorPlugin):
    """Generator plugin for creating cluster JSON configuration files"""
    
    def get_name(self):
        return "cluster_json"
    
    def get_description(self):
        return "Generate cluster JSON configuration file from hosts list"
    
    def get_parser(self):
        parser = argparse.ArgumentParser(description="Generate cluster json file")
        parser.add_argument("--input_hosts_file", required=True,
               help="Input file with host IPs - one address per line, supports ranges like 192.168.1.10-20")
        parser.add_argument("--output_json_file", required=True,
               help="Output cluster file in JSON format")
        parser.add_argument("--username", required=True,
               help="Username to ssh to the hosts")
        parser.add_argument("--key_file", required=True,
               help="keyfile with private keys")
        parser.add_argument("--head_node", 
               help="IP of the head node (optional, defaults to first host in hosts file)")
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

    def parse_hosts_file(self, filename):
        """
        Parse hosts file with support for IP ranges
        """
        node_list = []
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Expand ranges
                expanded = self.expand_ip_range(line)
                node_list.extend(expanded)
        
        return node_list

    def determine_head_node(self, node_list, specified_head_node=None):
        """Determine the head node IP from node list and optional specification"""
        if specified_head_node:
            head_node_ip = specified_head_node
            # Ensure head node is in the node list
            if head_node_ip not in node_list:
                print(f"WARNING: Head node {head_node_ip} not found in hosts file, adding it to node list")
                node_list.insert(0, head_node_ip)
        else:
            head_node_ip = node_list[0]
        
        return head_node_ip
    
    def generate(self, args):
        # Parse hosts file with range expansion
        node_list = self.parse_hosts_file(args.input_hosts_file)
        if not node_list:
            print("ERROR !! No hosts in the file, this is mandatory, aborting !!")
            sys.exit(1)

        # Determine head node
        head_node_ip = self.determine_head_node(node_list, getattr(args, 'head_node', None))

        # Load and render template using importlib.resources
        template_content = resources.files('cvs.input.templates.cluster_file').joinpath('cluster_json.template').read_text()
        
        template = Template(template_content)
        rendered_json = template.render(
            username=args.username,
            priv_key_file=args.key_file,
            head_node_ip=head_node_ip,
            node_list=node_list
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
