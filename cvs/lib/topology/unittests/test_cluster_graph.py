'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/lib/topology/cluster_graph.py: hostname location parsing,
# rack/chassis grouping, GPU XGMI edge derivation, and JSON round-trip. No live
# cluster.

import json
import unittest

from cvs.lib.topology import cluster_graph as cg


def _gpu(bdf, index, peers):
    '''Build a GPU record with a SELF link plus one XGMI link per peer bdf.'''
    links = [
        {
            "peer_bdf": bdf,
            "peer_gpu": index,
            "link_type": "SELF",
            "link_status": "ENABLED",
        }
    ]
    for peer_index, peer_bdf in peers:
        links.append(
            {
                "peer_bdf": peer_bdf,
                "peer_gpu": peer_index,
                "link_type": "XGMI",
                "weight": 15,
                "num_hops": 1,
                "link_status": "ENABLED",
            }
        )
    return {"bdf": bdf, "gpu_index": index, "uuid": None, "links": links}


class TestParseLocation(unittest.TestCase):
    def test_standard_rck_hostname(self):
        loc = cg.parse_location("testnode-rck-g02-h21-7.example")
        self.assertEqual(loc["rack"], "g02")
        self.assertEqual(loc["chassis"], "h21")
        self.assertEqual(loc["slot"], "7")

    def test_falls_back_to_hostname_when_key_unmatched(self):
        loc = cg.parse_location("10.0.0.5", "testnode-rck-g05-j19-11")
        self.assertEqual(loc["rack"], "g05")
        self.assertEqual(loc["chassis"], "j19")
        self.assertEqual(loc["slot"], "11")

    def test_nonstandard_name_uses_unknown_sentinels(self):
        loc = cg.parse_location("testnode-1b112-b07-2.example")
        self.assertEqual(loc["rack"], cg.UNKNOWN_RACK)
        self.assertEqual(loc["chassis"], cg.UNKNOWN_CHASSIS)
        self.assertIsNone(loc["slot"])


class TestStitch(unittest.TestCase):
    def setUp(self):
        self.per_node = {
            "testnode-rck-g02-h21-7": {
                "host": "testnode-rck-g02-h21-7",
                "hostname": "h21-7",
                "reachable": True,
                "gpu_count": 2,
                "gpus": [
                    _gpu("0001:01:00.0", 0, [(1, "0002:01:00.0")]),
                    _gpu("0002:01:00.0", 1, [(0, "0001:01:00.0")]),
                ],
            },
            "testnode-rck-g02-h21-12": {
                "host": "testnode-rck-g02-h21-12",
                "hostname": "h21-12",
                "reachable": True,
                "gpu_count": 0,
                "gpus": [],
            },
            "testnode-rck-g05-j19-1": {
                "host": "testnode-rck-g05-j19-1",
                "hostname": "j19-1",
                "reachable": False,
                "gpu_count": 0,
                "gpus": [],
            },
        }
        self.graph = cg.stitch(self.per_node)

    def test_summary_counts(self):
        s = self.graph.summary()
        self.assertEqual(s["node_count"], 3)
        self.assertEqual(s["reachable_node_count"], 2)
        self.assertEqual(s["unreachable_node_count"], 1)
        self.assertEqual(s["total_gpus"], 2)
        self.assertEqual(s["rack_count"], 2)
        # DIMM/NVMe totals default to 0 when nodes carry no such records.
        self.assertEqual(s["total_dimms"], 0)
        self.assertEqual(s["total_nvme"], 0)

    def test_summary_totals_dimms_and_nvme(self):
        per_node = {
            "testnode-rck-g02-h21-7": {
                "host": "testnode-rck-g02-h21-7",
                "reachable": True,
                "gpu_count": 0,
                "gpus": [],
                "dimms": [{"size": "128 GB"}, {"size": "128 GB"}],
                "nvme": [{"name": "nvme0n1"}],
            },
        }
        summary = cg.stitch(per_node).summary()
        self.assertEqual(summary["total_dimms"], 2)
        self.assertEqual(summary["total_nvme"], 1)

    def test_xgmi_edge_deduped_to_single_undirected(self):
        # The pair appears from both GPUs' link lists; one undirected edge expected.
        self.assertEqual(self.graph.summary()["xgmi_edge_count"], 1)
        edge = self.graph.intra_node_edges[0]
        self.assertEqual(edge["src_bdf"], "0001:01:00.0")
        self.assertEqual(edge["dst_bdf"], "0002:01:00.0")
        self.assertEqual(edge["link_type"], "XGMI")
        self.assertEqual(edge["host"], "testnode-rck-g02-h21-7")

    def test_rack_chassis_grouping(self):
        racks = {r["rack"]: r for r in self.graph.racks}
        self.assertIn("g02", racks)
        self.assertIn("g05", racks)
        g02_chassis = {c["chassis"]: c for c in racks["g02"]["chassis"]}
        self.assertIn("h21", g02_chassis)
        self.assertEqual(
            sorted(g02_chassis["h21"]["nodes"]),
            ["testnode-rck-g02-h21-12", "testnode-rck-g02-h21-7"],
        )

    def test_inter_node_edges_empty_but_present(self):
        # Inter-node fabric is ARC territory; the key must exist and be empty.
        self.assertEqual(self.graph.to_dict()["inter_node_edges"], [])

    def test_json_round_trip_preserves_summary(self):
        as_json = json.dumps(self.graph.to_dict())
        restored = cg.ClusterTopology.from_dict(json.loads(as_json))
        self.assertEqual(restored.summary(), self.graph.summary())

    def test_location_annotated_on_nodes(self):
        node = next(n for n in self.graph.nodes if n["host"] == "testnode-rck-g02-h21-7")
        self.assertEqual(node["location"]["rack"], "g02")
        self.assertEqual(node["location"]["chassis"], "h21")


class TestXgmiEdgeFiltering(unittest.TestCase):
    def test_disabled_links_are_skipped(self):
        gpus = [
            {
                "bdf": "0001:01:00.0",
                "gpu_index": 0,
                "links": [
                    {
                        "peer_bdf": "0002:01:00.0",
                        "link_type": "XGMI",
                        "link_status": "DISABLED",
                    }
                ],
            }
        ]
        edges = cg._gpu_xgmi_edges("host", gpus)
        self.assertEqual(edges, [])

    def test_self_links_are_skipped(self):
        gpus = [
            {
                "bdf": "0001:01:00.0",
                "gpu_index": 0,
                "links": [
                    {
                        "peer_bdf": "0001:01:00.0",
                        "link_type": "SELF",
                        "link_status": "ENABLED",
                    }
                ],
            }
        ]
        self.assertEqual(cg._gpu_xgmi_edges("host", gpus), [])


class TestScaleUpScaleOut(unittest.TestCase):
    def _rdma(self, bdf, gbps, active=True):
        return {
            "bdf": bdf,
            "netdev": "eth-" + bdf,
            "gbps": gbps,
            "rate": f"{gbps} Gb/sec (4X NDR)",
            "fabric": "RoCE",
            "state": "4: ACTIVE" if active else "0: DOWN",
            "active": active,
        }

    def _node(self, host, lldp=None):
        return {
            "host": host,
            "hostname": host,
            "reachable": True,
            "gpu_count": 2,
            "scale_up_mode": "SPX/NPS1",
            "gpus": [
                _gpu("0001:01:00.0", 0, [(1, "0002:01:00.0")]),
                _gpu("0002:01:00.0", 1, [(0, "0001:01:00.0")]),
            ],
            # One front-end (domain 0000) + two backend scale-out NICs.
            "rdma": [self._rdma("0000:04:00.0", 100), self._rdma("0001:44:00.0", 400), self._rdma("0002:44:00.0", 400)],
            "lldp": lldp or {},
        }

    def test_xgmi_edges_tagged_scale_up(self):
        graph = cg.stitch({"testnode-rck-g02-h21-7": self._node("testnode-rck-g02-h21-7")})
        self.assertTrue(all(e["family"] == "scale_up" for e in graph.intra_node_edges))

    def test_summary_scale_fields(self):
        graph = cg.stitch({"testnode-rck-g02-h21-7": self._node("testnode-rck-g02-h21-7")})
        s = graph.summary()
        self.assertEqual(s["scale_up_mode"], "SPX/NPS1")
        # Front-end (domain 0000) excluded; two backend NICs counted.
        self.assertEqual(s["scale_out_nic_count"], 2)
        self.assertEqual(s["switch_count"], 0)
        self.assertEqual(s["inter_node_edge_count"], 0)

    def test_no_lldp_means_no_inter_node_edges(self):
        graph = cg.stitch({"testnode-rck-g02-h21-7": self._node("testnode-rck-g02-h21-7")})
        self.assertEqual(graph.inter_node_edges, [])
        self.assertEqual(graph.switches, [])

    def test_lldp_builds_arc_shaped_edges(self):
        node = self._node(
            "testnode-rck-g02-h21-7",
            lldp={"eth-0001:44:00.0": {"switch_name": "tor-g02-1", "switch_port": "Eth1/5"}},
        )
        graph = cg.stitch({"testnode-rck-g02-h21-7": node})
        self.assertEqual(len(graph.inter_node_edges), 1)
        edge = graph.inter_node_edges[0]
        # ARC-shaped: src/dst with name+port, role, via, speed, status, family.
        self.assertEqual(edge["src"]["name"], "testnode-rck-g02-h21-7")
        self.assertEqual(edge["dst"]["name"], "tor-g02-1")
        self.assertEqual(edge["dst"]["port"], "Eth1/5")
        self.assertEqual(edge["via"], "tor-g02-1")
        self.assertEqual(edge["family"], "scale_out")
        self.assertEqual(edge["speed"], "400 Gb/sec (4X NDR)")
        self.assertIn("tor-g02-1", graph.switches)

    def test_mixed_scale_up_mode(self):
        n1 = self._node("testnode-rck-g02-h21-7")
        n2 = self._node("testnode-rck-g02-h21-8")
        n2["scale_up_mode"] = "CPX/NPS4"
        graph = cg.stitch({"testnode-rck-g02-h21-7": n1, "testnode-rck-g02-h21-8": n2})
        self.assertEqual(graph.summary()["scale_up_mode"], "mixed")

    def test_scale_fields_survive_round_trip(self):
        node = self._node(
            "testnode-rck-g02-h21-7",
            lldp={"eth-0001:44:00.0": {"switch_name": "tor-g02-1", "switch_port": "Eth1/5"}},
        )
        graph = cg.stitch({"testnode-rck-g02-h21-7": node})
        restored = cg.ClusterTopology.from_dict(json.loads(json.dumps(graph.to_dict())))
        self.assertEqual(restored.inter_node_edges, graph.inter_node_edges)
        self.assertEqual(restored.switches, graph.switches)
        self.assertEqual(restored.summary(), graph.summary())


if __name__ == "__main__":
    unittest.main()
