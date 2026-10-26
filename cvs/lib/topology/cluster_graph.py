'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Stitch per-node topology records into one cluster graph.

The graph groups nodes by physical position parsed from the hostname
(rack/chassis/slot) and carries each node's intra-node GPU XGMI mesh as edges.
Inter-node fabric (which node connects to which switch over what cable) is NOT
represented here: that requires a fabric-aware source such as ARC, and inventing
those edges from a node list would be fiction. The cluster graph therefore
models what a fan-out of node probes can actually know -- every node and its
internal GPU mesh -- and leaves inter-node edges as an explicit empty list.

The dict shape from to_dict() is intentionally close to a generic
components+connections graph so a future ARC bridge can fold rack fabric in
without reshaping what this module already produces.
'''

import re

SCHEMA_VERSION = "1.0"

# Datacenter hostname convention, e.g. "ctheliosp-rck-g02-h21-7":
#   <prefix>-rck-<rack>-<chassis>-<slot>
# Captures rack (g02), chassis (h21), slot (7). Names that do not match fall
# back to a single synthetic rack so stitching never drops a node.
_HOSTNAME_RE = re.compile(
    r"rck-(?P<rack>[a-z]?\d+)-(?P<chassis>[a-z]\d+)-(?P<slot>\d+)",
    re.IGNORECASE,
)

UNKNOWN_RACK = "unknown-rack"
UNKNOWN_CHASSIS = "unknown-chassis"


def parse_location(host, hostname=None):
    '''Derive {"rack","chassis","slot"} from a node's host key or hostname.

    Tries the cluster_file key first (usually an FQDN), then the probed
    hostname. Returns the UNKNOWN_* sentinels when neither matches the
    convention, so a non-standard name groups under one fallback rack rather
    than being lost.
    '''
    for candidate in (host, hostname):
        if not candidate:
            continue
        match = _HOSTNAME_RE.search(candidate)
        if match:
            return {
                "rack": match.group("rack").lower(),
                "chassis": match.group("chassis").lower(),
                "slot": match.group("slot"),
            }
    return {"rack": UNKNOWN_RACK, "chassis": UNKNOWN_CHASSIS, "slot": None}


def _gpu_xgmi_edges(host, gpus):
    '''Intra-node GPU edges from the amd-smi link records of one node.

    Emits one undirected edge per connected GPU pair (deduped by ordering the
    endpoints), skipping SELF links and any link whose status is not ENABLED.
    Edge endpoints are namespaced by host so they are unique cluster-wide.
    '''
    edges = []
    seen = set()
    for gpu in gpus:
        src = gpu.get("bdf")
        if not src:
            continue
        for link in gpu.get("links", []):
            link_type = (link.get("link_type") or "").upper()
            if link_type in ("", "SELF"):
                continue
            if (link.get("link_status") or "").upper() not in ("", "ENABLED"):
                continue
            dst = link.get("peer_bdf")
            if not dst or dst == src:
                continue
            pair = tuple(sorted((src, dst)))
            key = (pair, link_type)
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                {
                    "host": host,
                    "src_bdf": pair[0],
                    "dst_bdf": pair[1],
                    "link_type": link_type,
                    "weight": link.get("weight"),
                    "num_hops": link.get("num_hops"),
                    # The GPU XGMI mesh IS the scale-up fabric (tensor-parallel
                    # shards stay on it).
                    "family": "scale_up",
                }
            )
    return edges


def _inter_node_edges(per_node):
    '''Build ARC-shaped inter-node (scale-out) edges from each node's LLDP map.

    One edge per interface that LLDP resolved to a switch, shaped to mirror ARC's
    Edge.to_dict (arc/taxonomy/graph.py): {src:{name,port}, dst:{name,port},
    role, via, speed, status, family}. The matching RDMA record (by netdev) adds
    the line rate and state. Returns (edges, switches) where switches is the set
    of discovered switch names (added to the graph as nodes). Empty when no node
    has LLDP neighbours -- inter-node wiring is then "not discovered" (ARC's job).
    '''
    edges = []
    switches = set()
    for host in sorted(per_node.keys()):
        record = per_node[host]
        lldp = record.get("lldp") or {}
        rdma_by_netdev = {r.get("netdev"): r for r in record.get("rdma", []) if r.get("netdev")}
        for ifname in sorted(lldp.keys()):
            neighbor = lldp[ifname]
            switch = neighbor.get("switch_name")
            if not switch:
                continue
            switches.add(switch)
            rdma = rdma_by_netdev.get(ifname, {})
            edges.append(
                {
                    "src": {"name": host, "port": ifname},
                    "dst": {"name": switch, "port": neighbor.get("switch_port") or ""},
                    "role": "data",
                    "via": switch,
                    "speed": rdma.get("rate"),
                    "status": rdma.get("state"),
                    "family": "scale_out",
                }
            )
    return edges, switches


class ClusterTopology:
    '''A stitched cluster topology: nodes grouped by rack/chassis plus edges.'''

    def __init__(
        self,
        nodes=None,
        racks=None,
        intra_node_edges=None,
        inter_node_edges=None,
        switches=None,
    ):
        self.nodes = nodes or []
        self.racks = racks or []
        self.intra_node_edges = intra_node_edges or []
        # Inter-node (scale-out) fabric. Populated from LLDP when available; when
        # no node has LLDP neighbours this stays empty and the wiring is "not
        # discovered" (needs LLDP or ARC). Edges are ARC-shaped so ARC's
        # RackGraph edges can fold in later without reshaping.
        self.inter_node_edges = inter_node_edges or []
        # Switches discovered via LLDP, carried as graph nodes.
        self.switches = switches or []

    def to_dict(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "racks": self.racks,
            "nodes": self.nodes,
            "switches": self.switches,
            "intra_node_edges": self.intra_node_edges,
            "inter_node_edges": self.inter_node_edges,
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            nodes=data.get("nodes", []),
            racks=data.get("racks", []),
            intra_node_edges=data.get("intra_node_edges", []),
            inter_node_edges=data.get("inter_node_edges", []),
            switches=data.get("switches", []),
        )

    def _scale_up_mode(self):
        '''The cluster-wide scale-up partition mode, or "mixed"/None.'''
        modes = {n.get("scale_up_mode") for n in self.nodes if n.get("scale_up_mode")}
        if not modes:
            return None
        return modes.pop() if len(modes) == 1 else "mixed"

    def summary(self):
        total_gpus = sum(n.get("gpu_count", 0) for n in self.nodes)
        reachable = sum(1 for n in self.nodes if n.get("reachable"))
        total_dimms = sum(len(n.get("dimms", [])) for n in self.nodes)
        total_nvme = sum(len(n.get("nvme", [])) for n in self.nodes)
        # Scale-out NICs = active RDMA ports that are not the front-end/mgmt NIC
        # (domain 0000). Those are the inter-node data-plane uplinks.
        scale_out_nics = sum(
            1
            for n in self.nodes
            for r in n.get("rdma", [])
            if r.get("active") and (r.get("bdf") or "").split(":")[0] != "0000"
        )
        return {
            "node_count": len(self.nodes),
            "reachable_node_count": reachable,
            "unreachable_node_count": len(self.nodes) - reachable,
            "rack_count": len(self.racks),
            "total_gpus": total_gpus,
            "xgmi_edge_count": len(self.intra_node_edges),
            "total_dimms": total_dimms,
            "total_nvme": total_nvme,
            "scale_up_mode": self._scale_up_mode(),
            "scale_out_nic_count": scale_out_nics,
            "switch_count": len(self.switches),
            "inter_node_edge_count": len(self.inter_node_edges),
        }


def stitch(per_node):
    '''Build a ClusterTopology from {host: node_topology_dict}.

    per_node values are the records produced by
    topology_probe.parse_node_topology. Nodes are annotated with their parsed
    location, grouped into a rack -> chassis -> nodes tree, and each node's GPU
    XGMI mesh becomes intra-node edges.
    '''
    nodes = []
    edges = []
    # rack -> chassis -> [host]
    rack_tree = {}

    for host in sorted(per_node.keys()):
        record = dict(per_node[host])
        record.setdefault("host", host)
        location = parse_location(host, record.get("hostname"))
        record["location"] = location
        nodes.append(record)

        rack_tree.setdefault(location["rack"], {}).setdefault(location["chassis"], []).append(host)

        edges.extend(_gpu_xgmi_edges(host, record.get("gpus", [])))

    racks = []
    for rack_id in sorted(rack_tree.keys()):
        chassis_list = []
        for chassis_id in sorted(rack_tree[rack_id].keys()):
            chassis_list.append(
                {
                    "chassis": chassis_id,
                    "nodes": sorted(rack_tree[rack_id][chassis_id]),
                }
            )
        racks.append({"rack": rack_id, "chassis": chassis_list})

    inter_edges, switches = _inter_node_edges(per_node)

    return ClusterTopology(
        nodes=nodes,
        racks=racks,
        intra_node_edges=edges,
        inter_node_edges=inter_edges,
        switches=sorted(switches),
    )
