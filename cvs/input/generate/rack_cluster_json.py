#!/usr/bin/env python3
'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication
and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import argparse
import copy
import json
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass, field

from cvs.input.generate.cluster_json import ClusterJsonGenerator
from cvs.lib.arc_rack_lib import CREDENTIAL_KEYS, PLACEHOLDER

# --rackN flag -> racks.<rack_id> field; each takes a comma-separated host list in tray order.
RACK_LIST_FLAGS = (
    ("switches", "switch_trays"),
    ("switch_bmc", "switch_bmc"),
    ("rmc", "rmc"),
    ("front_end_switches", "front_end_switches"),
    ("back_end_switches", "back_end_switches"),
    ("power_shelves", "power_shelves"),
)


DEFAULT_PLATFORM = "HeliosP"
NO_BMC = ("", "NA", PLACEHOLDER)


@dataclass
class RackEntry:
    '''One --rackN group. Unset fields stay empty so --amend can tell what was passed.'''

    rack_id: str
    compute_nodes: list = field(default_factory=list)
    compute_bmcs: list = field(default_factory=list)
    platform: str = ""
    switch_model: str = ""
    arc_controller: str = ""
    devices: dict = field(default_factory=dict)

    def fields(self):
        '''Return the racks.<rack_id> fields this group sets.'''
        entry = {"platform": self.platform, "switch_model": self.switch_model, "arc_controller": self.arc_controller}
        entry.update(self.devices)
        return {key: value for key, value in entry.items() if value}

    def node_entries(self, previous=None):
        '''Return node_dict entries for compute_nodes, reusing a previous entry for a host already listed.'''
        previous = previous or {}
        bmcs = self.compute_bmcs or [None] * len(self.compute_nodes)
        nodes = {}
        for host, bmc in zip(self.compute_nodes, bmcs):
            info = dict(previous.get(host) or {"bmc_ip": "NA", "vpc_ip": host})
            info["rack_id"] = self.rack_id
            if bmc:
                info["bmc_ip"] = bmc
            nodes[host] = info
        return nodes


def check_rack(rack_id, rack, hosts):
    '''Raise ValueError when a rack's combined fields disagree with each other or with its computes.'''
    trays, bmcs = rack.get("switch_trays") or [], rack.get("switch_bmc") or []
    if bmcs and len(bmcs) != len(trays):
        raise ValueError(f"racks.{rack_id}.switch_bmc has {len(bmcs)} entries; switch_trays has {len(trays)}.")
    controller = rack.get("arc_controller")
    if controller and controller not in hosts:
        raise ValueError(f"racks.{rack_id}.arc_controller {controller} is not one of its computes.")


def load_cluster_file(path):
    '''Read the cluster file --amend extends, raising ValueError when it is missing or not a JSON object.'''
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise ValueError(f"--amend needs an existing {path}; generate it first without --amend.")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}")
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a cluster JSON object.")
    return data


def _merge_setting(target, key, value, where, replace=False, hint="edit the file to change it"):
    '''Set target[key] when unset, a placeholder, or replace is set; raise ValueError on any other difference.'''
    if value in (None, "", []):
        return
    current = target.get(key)
    if replace or current in (None, "", [], PLACEHOLDER):
        target[key] = value
    elif current != value:
        raise ValueError(f"{where} is {current!r} in the file, not {value!r}; {hint}.")


def _rack_hosts(nodes, rack_id):
    return [host for host, info in nodes.items() if (info or {}).get("rack_id") == rack_id]


def _splice_nodes(nodes, rack_id, new_nodes):
    '''Return nodes with rack_id's entries swapped for new_nodes at the first old entry's position.'''
    result = {}
    placed = False
    for host, info in nodes.items():
        if (info or {}).get("rack_id") != rack_id:
            result[host] = info
        elif not placed:
            result.update(new_nodes)
            placed = True
    if not placed:
        result.update(new_nodes)
    return result


def _update_rack(nodes, rack, entry, replace):
    '''Apply entry to an existing rack and its nodes; return the new node_dict.'''
    rack_id = entry.rack_id
    hosts = _rack_hosts(nodes, rack_id)
    if entry.compute_nodes and entry.compute_nodes != hosts:
        if not replace:
            raise ValueError(f"--computes for {rack_id} differs from the file's list; pass --replace to change it.")
        clash = sorted(set(entry.compute_nodes).intersection(set(nodes) - set(hosts)))
        if clash:
            raise ValueError(f"Computes already in the file under another rack: {', '.join(clash)}.")
        nodes = _splice_nodes(nodes, rack_id, entry.node_entries(previous=nodes))
        hosts = list(entry.compute_nodes)
    elif entry.compute_bmcs:
        if len(entry.compute_bmcs) != len(hosts):
            raise ValueError(
                f"--bmcs for {rack_id} has {len(entry.compute_bmcs)} entries; the rack has {len(hosts)} computes."
            )
        nodes = dict(nodes)
        for host, bmc in zip(hosts, entry.compute_bmcs):
            info = dict(nodes[host] or {})
            if str(info.get("bmc_ip") or "").strip() in NO_BMC:
                info["bmc_ip"] = ""
            _merge_setting(info, "bmc_ip", bmc, f"node_dict.{host}.bmc_ip", replace, "pass --replace to change it")
            nodes[host] = info
    for key, value in entry.fields().items():
        _merge_setting(rack, key, value, f"racks.{rack_id}.{key}", replace, "pass --replace to change it")
    check_rack(rack_id, rack, hosts)
    return nodes


def merge_cluster(existing, base, racks, replace=False, head_node=None):
    '''
    Return existing with racks added or updated, leaving every other field as it was.

    base carries the command's logins (username, priv_key_file, racks credentials).
    A new rack is added whole. For a rack already in the file, fields it lacks
    are filled in; with replace, the fields passed overwrite the file's.

    Raises:
        ValueError: on a login or rack field that differs from the file's
        (rack fields only without replace), or a compute in another rack.
    '''
    cluster = copy.deepcopy(existing)
    for key in ("username", "priv_key_file"):
        _merge_setting(cluster, key, base.get(key), key)
    block_key = "rack_groups" if "racks" not in cluster and "rack_groups" in cluster else "racks"
    block = cluster.get(block_key)
    if not isinstance(block, dict):
        block = cluster[block_key] = {}
    for key in CREDENTIAL_KEYS:
        value = base["racks"].get(key)
        if key == "bmc_password" and value == PLACEHOLDER:
            block.setdefault(key, PLACEHOLDER)
        else:
            _merge_setting(block, key, value, f"{block_key}.{key}")

    nodes = cluster.get("node_dict") or {}
    for rack_id, entry in racks.items():
        rack = block.get(rack_id)
        if isinstance(rack, dict) and not rack_id.startswith("_"):
            rack = copy.deepcopy(rack)
            nodes = _update_rack(nodes, rack, entry, replace)
            block[rack_id] = rack
            continue
        if not entry.compute_nodes:
            raise ValueError(f"Rack {rack_id} is not in the file; give its --computes to add it.")
        clash = sorted(set(entry.compute_nodes).intersection(nodes))
        if clash:
            raise ValueError(f"Computes already in the file under another rack: {', '.join(clash)}.")
        rack = {"platform": DEFAULT_PLATFORM, **entry.fields()}
        check_rack(rack_id, rack, entry.compute_nodes)
        nodes = {**nodes, **entry.node_entries()}
        block[rack_id] = rack
    cluster["node_dict"] = nodes

    head = cluster.get("head_node_dict")
    if not isinstance(head, dict):
        head = cluster["head_node_dict"] = {}
    _merge_setting(head, "mgmt_ip", head_node, "head_node_dict.mgmt_ip")
    if not head.get("mgmt_ip") and nodes:
        head["mgmt_ip"] = next(iter(nodes))
    return cluster


def write_json_atomically(path, data):
    '''Write data to path through a temporary file, keeping an existing file's permissions.'''
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".cluster_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fp:
            json.dump(data, fp, indent=2)
            fp.write("\n")
        mode = stat.S_IMODE(os.stat(path).st_mode) if os.path.exists(path) else 0o666 & ~_umask()
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _umask():
    mask = os.umask(0)
    os.umask(mask)
    return mask


def split_rack_groups(argv):
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
    global_args = []
    rack_groups = {}
    current = None

    for arg in argv:
        if re.match(r'^--rack\d+$', arg):
            current = arg
            rack_groups[current] = []
        elif current is not None:
            rack_groups[current].append(arg)
        else:
            global_args.append(arg)

    return global_args, rack_groups


def _rack_sort_key(marker):
    """Sort --rackN markers numerically."""
    m = re.match(r'^--rack(\d+)$', marker)
    return int(m.group(1)) if m else 0


def build_rack_parser():
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
        default='',
        help="Comma-separated compute node IPs/hostnames in tray order; supports ranges like 10.0.0.1-5 and node[1-10]. "
        "Required for a new rack; optional with --amend on a rack already in the file.",
    )
    p.add_argument(
        '--bmcs', default='', help="Compute BMC addresses, same order and count as --computes. Omit without BMC access."
    )
    p.add_argument(
        '--platform',
        default='',
        help=f"ARC platform: HeliosP, HeliosR, or HeliosM (default for a new rack: {DEFAULT_PLATFORM})",
    )
    p.add_argument('--switch_model', default='', help="Scale-up switch model, when the platform accepts more than one")
    p.add_argument('--arc_controller', default='', help="Compute node that runs ARC. Default: the first of --computes")
    p.add_argument('--switches', default='', help="Scale-up switch host addresses, in tray order")
    p.add_argument('--switch_bmc', default='', help="Scale-up switch BMC addresses, same order and count as --switches")
    p.add_argument('--rmc', default='', help="Rack management controller addresses, in tray order")
    p.add_argument('--front_end_switches', default='', help="Front-end switch host addresses, in tray order")
    p.add_argument('--back_end_switches', default='', help="Back-end switch host addresses, in tray order")
    p.add_argument('--power_shelves', default='', help="Power shelf addresses, in tray order (HeliosM)")
    return p


class RackClusterJsonGenerator(ClusterJsonGenerator):
    """
    Generator plugin for rack-aware cluster.json files.

    Inherits IP/hostname range expansion from ClusterJsonGenerator so that
    every host list accepts the same range notation as --hosts:
      10.0.0.1,10.0.0.2-7     (IP range with dash)
      node[01-10]             (bracket hostname range)

    CLI usage (single rack, compute only):
      cvs generate rack_cluster_json \\
        --username ichristo --key_file ~/.ssh/id_rsa \\
        --output_json_file cluster.json \\
        --rack0 --id rack-01 --platform HeliosP --computes 10.0.0.1-18

    CLI usage (multi-rack, with switches and BMCs):
      cvs generate rack_cluster_json \\
        --username ichristo --key_file ~/.ssh/id_rsa \\
        --switch_ssh_user admin --bmc_user root \\
        --output_json_file cluster.json \\
        --rack0 --id rack-01 --computes 10.0.0.1-18 --bmcs 10.0.1.1-18 \\
                --switches 10.0.2.1-6 --switch_bmc 10.0.3.1-6 --rmc 10.0.4.1 \\
        --rack1 --id rack-02 --platform HeliosR --computes 10.1.0.1-18

    CLI usage (add a rack to an existing file; logins come from the file):
      cvs generate rack_cluster_json --amend --output_json_file cluster.json \\
        --rack0 --id rack-03 --computes 10.2.0.1-18

    CLI usage (add switch details to a rack already in the file):
      cvs generate rack_cluster_json --amend --switch_ssh_user admin --output_json_file cluster.json \\
        --rack0 --id rack-01 --switches 10.0.2.1-6
    """

    def supports_raw_argv(self):
        return True

    def get_name(self):
        return "rack_cluster_json"

    def get_description(self):
        return "Generate rack-aware cluster JSON with compute, switch, and rack device topology"

    def get_parser(self):
        parser = argparse.ArgumentParser(
            description="Generate rack-aware cluster.json from --rackN argument groups",
            epilog=(
                "Each --rackN group accepts: --id --computes --bmcs --platform --switch_model --arc_controller "
                "--switches --switch_bmc --rmc --front_end_switches --back_end_switches --power_shelves. "
                "List every device in tray order (Helios compute: U9 first, U33 last); ARC derives tray numbers "
                "from that order. N is zero-based (--rack0, --rack1, ...). Switch and BMC logins apply to all racks; "
                f"without --bmc_password, the BMC password is written as {PLACEHOLDER} for you to fill in. "
                "--amend adds racks to an existing file, or fills in fields a rack already in it lacks, and keeps "
                "everything else. A value that differs from the file's is an error; with --replace, the rack fields "
                "you pass overwrite the file's. Logins are never overwritten; edit the file to change them."
            ),
        )
        parser.add_argument('--username', default=None, help="SSH username for compute nodes (required unless --amend)")
        parser.add_argument(
            '--key_file', default=None, help="Path to SSH private key for compute nodes (required unless --amend)"
        )
        parser.add_argument('--switch_ssh_user', default=None, help="SSH username for all switch trays")
        parser.add_argument('--switch_ssh_password', default=None, help="SSH password for all switch trays")
        parser.add_argument(
            '--switch_ssh_key_file',
            default=None,
            help="Global SSH private key path for all switch trays; takes priority over --switch_ssh_password",
        )
        parser.add_argument('--bmc_user', default=None, help="Login for every BMC: compute, switch, RMC, power shelf")
        parser.add_argument(
            '--bmc_password', default=None, help=f"Password for --bmc_user. Default: {PLACEHOLDER} in the output"
        )
        parser.add_argument(
            '--head_node', default=None, help="IP of the head node (defaults to first compute node across all racks)"
        )
        parser.add_argument('--output_json_file', required=True, help="Output cluster JSON file path")
        parser.add_argument(
            '--amend',
            action='store_true',
            help="Update the existing --output_json_file: add new racks and fill in fields that racks already "
            "in it lack. Each rack needs --id.",
        )
        parser.add_argument(
            '--replace',
            action='store_true',
            help="With --amend, overwrite the fields you pass for a rack already in the file; --computes "
            "replaces its compute nodes",
        )
        return parser

    def _parse_rack(self, marker, idx, rack_argv, require_id=False):
        '''Parse one --rackN group into a RackEntry, raising ValueError on inconsistent lists.'''
        try:
            args = build_rack_parser().parse_args(rack_argv)
        except SystemExit:
            raise ValueError(f"Failed to parse args for {marker}.")
        if require_id and not args.rack_id:
            raise ValueError(f"{marker} needs --id with --amend.")
        computes = self.parse_hosts_list(args.computes)
        bmcs = self.parse_hosts_list(args.bmcs)
        if computes and bmcs and len(bmcs) != len(computes):
            raise ValueError(f"{marker} --bmcs has {len(bmcs)} entries; --computes has {len(computes)}.")
        return RackEntry(
            rack_id=args.rack_id or f"rack-{idx:02d}",
            compute_nodes=computes,
            compute_bmcs=bmcs,
            platform=args.platform,
            switch_model=args.switch_model,
            arc_controller=args.arc_controller,
            devices={key: self.parse_hosts_list(getattr(args, flag)) for flag, key in RACK_LIST_FLAGS},
        )

    def build_base(self, main_args):
        '''Return a cluster with the command's logins and no racks, the starting point for a new file.'''
        block = {}
        if main_args.switch_ssh_user:
            block["switch_ssh_user"] = main_args.switch_ssh_user
        if main_args.switch_ssh_password:
            block["switch_ssh_password"] = main_args.switch_ssh_password
        if main_args.switch_ssh_key_file:
            block["switch_ssh_key_file"] = main_args.switch_ssh_key_file
        if main_args.bmc_user:
            block["bmc_user"] = main_args.bmc_user
        if main_args.bmc_user or main_args.bmc_password:
            block["bmc_password"] = main_args.bmc_password or PLACEHOLDER
        return {
            "username": main_args.username,
            "priv_key_file": main_args.key_file,
            "head_node_dict": {"mgmt_ip": ""},
            "node_dict": {},
            "racks": block,
        }

    def generate(self, args):
        # _raw_args is injected by _run_generator when supports_raw_argv() is True;
        # fall back to sys.argv[3:] when invoked as a standalone script.
        raw_argv = getattr(args, '_raw_args', None) or sys.argv[3:]
        global_argv, rack_groups_raw = split_rack_groups(raw_argv)

        if not rack_groups_raw:
            print("ERROR: No rack groups found. Provide at least --rack0 with --computes.")
            sys.exit(1)

        main_args = self.get_parser().parse_args(global_argv)
        path = main_args.output_json_file

        racks = {}
        seen_nodes = set()
        try:
            if main_args.replace and not main_args.amend:
                raise ValueError("--replace works only with --amend.")
            if not main_args.amend and not (main_args.username and main_args.key_file):
                raise ValueError("--username and --key_file are required unless --amend.")
            for idx, marker in enumerate(sorted(rack_groups_raw.keys(), key=_rack_sort_key)):
                entry = self._parse_rack(marker, idx, rack_groups_raw[marker], require_id=main_args.amend)
                if entry.rack_id in racks:
                    raise ValueError(f"{marker} repeats rack id {entry.rack_id}.")
                repeated = seen_nodes.intersection(entry.compute_nodes)
                if repeated:
                    raise ValueError(f"{marker} lists computes already in another rack: {', '.join(sorted(repeated))}.")
                seen_nodes.update(entry.compute_nodes)
                racks[entry.rack_id] = entry
            base = self.build_base(main_args)
            existing = load_cluster_file(path) if main_args.amend else base
            cluster = merge_cluster(existing, base, racks, main_args.replace, main_args.head_node)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)

        write_json_atomically(path, cluster)

        print(f"{'Amended' if main_args.amend else 'Generated'} rack cluster JSON: {path}")
        print(f"Head node: {cluster['head_node_dict']['mgmt_ip']}")
        print(f"Total compute nodes: {len(cluster['node_dict'])}")
        block = cluster.get("racks") or cluster.get("rack_groups") or {}
        for rack_id in racks:
            rack = block[rack_id]
            print(
                f"  {rack_id}: {len(_rack_hosts(cluster['node_dict'], rack_id))} compute, "
                f"{len(rack.get('switch_trays') or [])} switches (platform={rack.get('platform')})"
            )
        if PLACEHOLDER in json.dumps(cluster["racks"]):
            print(f"Replace every {PLACEHOLDER} in {main_args.output_json_file} before running ARC suites.")


def main():
    generator = RackClusterJsonGenerator()
    parser = generator.get_parser()
    args = parser.parse_args()
    generator.generate(args)


if __name__ == "__main__":
    main()
