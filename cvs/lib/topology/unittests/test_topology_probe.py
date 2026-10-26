'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/lib/topology/topology_probe.py. Parsing is exercised against
# REAL amd-smi JSON captured from a healthy MI-class node (fixtures/), plus the
# driver-down and unreachable degradation paths. No live cluster, no SSH.

import os
import unittest

from cvs.lib.topology import topology_probe as tp

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _fixture(name):
    with open(os.path.join(FIXTURE_DIR, name)) as fp:
        return fp.read()


def _full_probe_output(hostname, topology_json, list_json, lscpu, numa, nics):
    '''Assemble a complete marker-delimited probe blob like a node would emit.'''
    return f"{tp.SEC_HOSTNAME}\n{hostname}\n{tp.SEC_AMDSMI_TOPOLOGY}\n{topology_json}\n{tp.SEC_AMDSMI_LIST}\n{list_json}\n{tp.SEC_LSCPU}\n{lscpu}\n{tp.SEC_NUMA}\n{numa}\n{tp.SEC_NICS}\n{nics}\n{tp.SEC_END}"


class TestBuildProbeCmd(unittest.TestCase):
    def test_cmd_contains_all_sections_and_tools(self):
        cmd = tp.build_probe_cmd()
        for marker in (
            tp.SEC_HOSTNAME,
            tp.SEC_AMDSMI_TOPOLOGY,
            tp.SEC_AMDSMI_LIST,
            tp.SEC_LSCPU,
            tp.SEC_NUMA,
            tp.SEC_NICS,
            tp.SEC_DIMMS,
            tp.SEC_NVME,
            tp.SEC_PARTITION,
            tp.SEC_RDMA,
            tp.SEC_LLDP,
            tp.SEC_END,
        ):
            self.assertIn(marker, cmd)
        self.assertIn("amd-smi topology --json", cmd)
        self.assertIn("amd-smi list --json", cmd)
        self.assertIn("amd-smi partition --json", cmd)
        self.assertIn("/sys/class/infiniband", cmd)
        self.assertIn("lldpctl", cmd)
        # DIMM detail is attempted via sudo but must never prompt.
        self.assertIn("sudo -n", cmd)
        self.assertIn("dmidecode -q -t 17", cmd)
        self.assertIn("/sys/block/nvme", cmd)
        # Must not abort the whole probe on one missing tool.
        self.assertIn("|| true", cmd)


class TestLoadJsonDocuments(unittest.TestCase):
    def test_parses_multiple_concatenated_arrays(self):
        # amd-smi list emits two top-level arrays back-to-back.
        text = _fixture("b072_amdsmi_list.json")
        docs = tp._load_json_documents(text)
        self.assertEqual(len(docs), 2)
        self.assertTrue(all(isinstance(d, list) for d in docs))

    def test_empty_text_yields_no_documents(self):
        self.assertEqual(tp._load_json_documents(""), [])

    def test_skips_leading_noise(self):
        docs = tp._load_json_documents("banner junk\n[{\"gpu\": 0}]")
        self.assertEqual(docs, [[{"gpu": 0}]])


class TestParseHealthyGpuNode(unittest.TestCase):
    def setUp(self):
        self.raw = _full_probe_output(
            "testnode-1b112-b07-2",
            _fixture("b072_amdsmi_topology.json"),
            _fixture("b072_amdsmi_list.json"),
            "CPU(s):   256\nModel name:  AMD Eng Sample\nSocket(s):  1\n"
            "Core(s) per socket:  128\nThread(s) per core:  2\nNUMA node(s):  38",
            "node0: 0-63,128-191\nnode1: 64-127,192-254",
            "enP1p68s0 bdf=0001:44:00.0 numa=-1 oper=up\ndocker0 bdf= numa= oper=down",
        )
        self.record = tp.parse_node_topology(self.raw, host="b07-2-key")

    def test_identity_and_reachable(self):
        self.assertEqual(self.record["host"], "b07-2-key")
        self.assertEqual(self.record["hostname"], "testnode-1b112-b07-2")
        self.assertTrue(self.record["reachable"])

    def test_four_gpus_with_xgmi_links(self):
        self.assertEqual(self.record["gpu_count"], 4)
        gpu0 = self.record["gpus"][0]
        self.assertEqual(gpu0["bdf"], "0001:01:00.0")
        self.assertTrue(gpu0["uuid"].startswith("af0075c1"))
        # 4 link rows: self + 3 peers.
        self.assertEqual(len(gpu0["links"]), 4)
        xgmi = [link for link in gpu0["links"] if link["link_type"] == "XGMI"]
        self.assertEqual(len(xgmi), 3)
        self.assertEqual(xgmi[0]["weight"], 15)
        self.assertEqual(xgmi[0]["num_hops"], 1)

    def test_cpu_numa_nic_parsed(self):
        self.assertEqual(self.record["cpu"]["cpus"], "256")
        self.assertEqual(self.record["cpu"]["cores_per_socket"], "128")
        self.assertEqual(len(self.record["numa"]), 2)
        self.assertEqual(self.record["numa"][0]["node"], "node0")
        names = [n["name"] for n in self.record["nics"]]
        self.assertIn("enP1p68s0", names)
        enp = next(n for n in self.record["nics"] if n["name"] == "enP1p68s0")
        self.assertEqual(enp["bdf"], "0001:44:00.0")
        self.assertEqual(enp["operstate"], "up")

    def test_ai_nic_classified_separately(self):
        # The second array in list.json is an ai_nic, not a GPU.
        self.assertEqual(len(self.record["ai_nics"]), 1)
        self.assertEqual(self.record["gpu_count"], 4)


class TestParseDegradedNodes(unittest.TestCase):
    def test_driver_down_node_has_no_gpus_but_is_reachable(self):
        # amd-smi prints nothing when /dev/kfd is absent; sysfs still works.
        raw = _full_probe_output(
            "testnode-rck-g02-j17-7",
            "",
            "",
            "CPU(s):   64",
            "node0: 0-63",
            "nic0 bdf=0000:04:00.0 numa=0 oper=up",
        )
        record = tp.parse_node_topology(raw, host="j17-7")
        self.assertTrue(record["reachable"])
        self.assertEqual(record["gpu_count"], 0)
        self.assertEqual(record["cpu"]["cpus"], "64")
        self.assertEqual(len(record["nics"]), 1)

    def test_empty_output_is_unreachable(self):
        record = tp.parse_node_topology("", host="dead")
        self.assertFalse(record["reachable"])
        self.assertEqual(record["gpu_count"], 0)
        self.assertEqual(record["host"], "dead")

    def test_banner_only_output_is_unreachable(self):
        # An SSH failure captured as banner text has no markers.
        record = tp.parse_node_topology("Permission denied (publickey)\n", host="x")
        self.assertFalse(record["reachable"])

    def test_none_output_does_not_raise(self):
        record = tp.parse_node_topology(None, host="x")
        self.assertFalse(record["reachable"])


_DMIDECODE_SAMPLE = """Memory Device
\tArray Handle: 0x001E
\tSize: 128 GB
\tForm Factor: DIMM
\tLocator: DIMM 0
\tBank Locator: P0 CHANNEL A
\tType: DDR5
\tSpeed: 8000 MT/s
\tManufacturer: Samsung
\tPart Number: M321RAJA0MB2-CQT5X
\tRank: 2

Memory Device
\tSize: No Module Installed
\tForm Factor: DIMM
\tLocator: DIMM 0
\tBank Locator: P0 CHANNEL C

Memory Device
\tSize: 432 GB
\tForm Factor: Die
\tLocator: HBM
\tBank Locator: n/a
\tType: HBM3
"""


class TestParseDimms(unittest.TestCase):
    def test_dmidecode_keeps_only_populated_system_dimms(self):
        dimms, summary = tp._parse_dimms(_DMIDECODE_SAMPLE)
        # The empty slot and the HBM "Die" are both excluded; one real DIMM stays.
        self.assertEqual(len(dimms), 1)
        self.assertEqual(summary, {})
        dimm = dimms[0]
        self.assertEqual(dimm["size"], "128 GB")
        self.assertEqual(dimm["type"], "DDR5")
        self.assertEqual(dimm["speed"], "8000 MT/s")
        self.assertEqual(dimm["bank"], "P0 CHANNEL A")
        self.assertEqual(dimm["manufacturer"], "Samsung")

    def test_hbm_die_is_never_a_dimm(self):
        dimms, _ = tp._parse_dimms(_DMIDECODE_SAMPLE)
        self.assertTrue(all(d["type"] != "HBM3" for d in dimms))

    def test_lsmem_fallback_when_no_dmidecode(self):
        text = "LSMEM: RANGE SIZE STATE\nLSMEM: Total online memory: 512G\n"
        dimms, summary = tp._parse_dimms(text)
        self.assertEqual(dimms, [])
        self.assertEqual(summary, {"total_online": "512G"})

    def test_empty_section(self):
        dimms, summary = tp._parse_dimms("")
        self.assertEqual(dimms, [])
        self.assertEqual(summary, {})


class TestParseNvme(unittest.TestCase):
    def test_parses_name_capacity_numa_model(self):
        text = (
            "nvme0n1 sectors=7501476528 numa=1 model=SDS6BA138OKP8X7\n"
            "nvme1n1 sectors=7501476528 numa=0 model=SDS6BA138OKP8X7\n"
        )
        nvme = tp._parse_nvme(text)
        self.assertEqual(len(nvme), 2)
        self.assertEqual(nvme[0]["name"], "nvme0n1")
        self.assertEqual(nvme[0]["capacity"], "3.8 TB")
        self.assertEqual(nvme[0]["numa_node"], "1")
        self.assertEqual(nvme[0]["model"], "SDS6BA138OKP8X7")

    def test_capacity_helper(self):
        self.assertEqual(tp._nvme_capacity("7501476528"), "3.8 TB")
        self.assertEqual(tp._nvme_capacity("1953525168"), "1.0 TB")
        self.assertIsNone(tp._nvme_capacity("notanumber"))

    def test_empty_section(self):
        self.assertEqual(tp._parse_nvme(""), [])


class TestParseFullProbeFixture(unittest.TestCase):
    """End-to-end parse of a real captured probe with DIMM + NVMe sections."""

    def test_real_node_dimms_and_nvme(self):
        raw = _fixture("b072_full_probe.txt")
        rec = tp.parse_node_topology(raw, host="b07-2")
        self.assertTrue(rec["reachable"])
        self.assertEqual(rec["gpu_count"], 4)
        # 4x 128 GB DDR5 system DIMMs (HBM excluded).
        self.assertEqual(len(rec["dimms"]), 4)
        self.assertTrue(all(d["type"] == "DDR5" for d in rec["dimms"]))
        self.assertEqual(sum(int(d["size"].split()[0]) for d in rec["dimms"]), 512)
        # 2x NVMe.
        self.assertEqual(len(rec["nvme"]), 2)
        self.assertTrue(all(n["capacity"] == "3.8 TB" for n in rec["nvme"]))


class TestParsePartition(unittest.TestCase):
    def test_uniform_mode(self):
        text = _fixture("b072_amdsmi_partition.json")
        per_gpu, mode = tp._parse_partition(text)
        self.assertEqual(len(per_gpu), 4)
        self.assertEqual(mode, "SPX/NPS1")
        self.assertEqual(per_gpu[0]["accelerator_type"], "SPX")
        self.assertEqual(per_gpu[0]["memory"], "NPS1")

    def test_mixed_mode(self):
        text = (
            '{"current_partition": ['
            '{"gpu_id": 0, "accelerator_type": "SPX", "memory": "NPS1"},'
            '{"gpu_id": 1, "accelerator_type": "CPX", "memory": "NPS4"}]}'
        )
        _, mode = tp._parse_partition(text)
        self.assertEqual(mode, "mixed")

    def test_empty(self):
        self.assertEqual(tp._parse_partition(""), ([], None))


class TestParseRdma(unittest.TestCase):
    def setUp(self):
        self.rdma = tp._parse_rdma(_fixture("b072_rdma.txt"))

    def test_counts_and_backend_fabric(self):
        self.assertEqual(len(self.rdma), 9)  # 1 front-end + 8 backend
        backend = [r for r in self.rdma if r["gbps"] == 400]
        self.assertEqual(len(backend), 8)
        self.assertTrue(all(r["fabric"] == "RoCE" for r in backend))
        self.assertTrue(all(r["active"] for r in backend))
        self.assertEqual(backend[0]["gen"], "NDR")

    def test_front_end_separated_by_domain(self):
        fe = [r for r in self.rdma if (r["bdf"] or "").split(":")[0] == "0000"]
        self.assertEqual(len(fe), 1)
        self.assertEqual(fe[0]["gbps"], 100)

    def test_infiniband_maps_to_ib(self):
        text = (
            "rdma=mlx5_0 netdev=ib0 bdf=0000:aa:00.0 link_layer=InfiniBand state=4: ACTIVE rate=[400 Gb/sec (4X NDR)]"
        )
        rec = tp._parse_rdma(text)[0]
        self.assertEqual(rec["fabric"], "IB")

    def test_empty(self):
        self.assertEqual(tp._parse_rdma(""), [])


class TestParseLldp(unittest.TestCase):
    def test_keyvalue_neighbor(self):
        text = "lldp.enP1p68s0.chassis.name=tor-g02-1\nlldp.enP1p68s0.port.descr=Eth1/5\n"
        result = tp._parse_lldp(text)
        self.assertEqual(result, {"enP1p68s0": {"switch_name": "tor-g02-1", "switch_port": "Eth1/5"}})

    def test_absent_daemon_is_empty(self):
        self.assertEqual(tp._parse_lldp(""), {})

    def test_entry_without_switch_name_dropped(self):
        self.assertEqual(tp._parse_lldp("lldp.eth0.port.descr=Eth1/5\n"), {})


class TestParseFullProbeScaleFields(unittest.TestCase):
    def test_full_fixture_has_partition_and_rdma(self):
        raw = _fixture("b072_full_probe.txt")
        rec = tp.parse_node_topology(raw, host="b07-2")
        # The committed full-probe fixture predates the scale sections, so those
        # parse empty; this asserts the keys exist and degrade cleanly.
        self.assertIn("scale_up_mode", rec)
        self.assertIn("rdma", rec)
        self.assertIn("lldp", rec)
        self.assertEqual(rec["lldp"], {})


if __name__ == "__main__":
    unittest.main()
