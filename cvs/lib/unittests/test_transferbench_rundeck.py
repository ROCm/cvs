'''Unit tests for cvs.lib.transferbench_rundeck.'''

import unittest

from cvs.lib import transferbench_rundeck

A2A_LINE = "RTotal 90.0 110.0 120.0 130.0 140.0 150.0 160.0 170.0\n"
A2A_EXPECT = {"gpu_to_gpu_a2a_rtotal": "100"}

P2P_OK = "Averages (During UniDir): 1.1 2.2 3.3 40.5\nAverages (During BiDir): 1.1 2.2 3.3 50.5\n"
P2P_EXPECT = {
    "avg_gpu_to_gpu_p2p_unidir_bw": "33.9",
    "avg_gpu_to_gpu_p2p_bidir_bw": "43.9",
}
P2P_ABORT = "cannot allocate memory due to process memory policy/cpuset\n"

SCALING_OK = "Best 10.0(  8) 20.0(  4) 55.5\n"
SCHMOO_OK = "  32  10.0  20.0  30.0  40.0  50.0  60.0\n"
SCHMOO_EXPECT = {
    "32_cu_local_read": "9",
    "32_cu_local_write": "19",
    "32_cu_local_copy": "29",
    "32_cu_rem_read": "39",
    "32_cu_rem_write": "49",
    "32_cu_rem_copy": "70",
}


def _node(results, group, node="n1"):
    return results["groups"][group]["nodes"][node]


class TestMakeMeta(unittest.TestCase):
    def test_labels_transferbench(self):
        meta = transferbench_rundeck.make_meta({"cluster_name": "helios"}, "transferbench_cvs")
        self.assertEqual(meta["cluster"], "helios")
        self.assertEqual(meta["version_label"], "TransferBench")
        self.assertEqual(meta["suite"], "transferbench_cvs")


class TestRecordA2a(unittest.TestCase):
    def test_gpu_under_threshold_fails(self):
        results = {}
        transferbench_rundeck.record_a2a(results, {"n1": A2A_LINE}, A2A_EXPECT)
        node = _node(results, "a2a")
        self.assertEqual(node["status"], "fail")
        self.assertEqual(node["items"][0]["name"], "GPU0")
        self.assertEqual(node["items"][0]["status"], "fail")
        self.assertEqual(node["items"][1]["status"], "pass")
        self.assertIn("90.0", node["items_summary"])

    def test_missing_rtotal_fails(self):
        results = {}
        transferbench_rundeck.record_a2a(results, {"n1": "no table"}, A2A_EXPECT)
        self.assertEqual(_node(results, "a2a")["status"], "fail")
        self.assertIn("not found", _node(results, "a2a")["items"][0]["message"])

    def test_traceback_fails_passing_rtotal(self):
        results = {}
        text = "RTotal 110.0 120.0 130.0 140.0 150.0 160.0 170.0 180.0\nTraceback (most recent call last):\n"
        transferbench_rundeck.record_a2a(results, {"n1": text}, A2A_EXPECT)
        node = _node(results, "a2a")
        self.assertEqual(node["status"], "fail")
        self.assertEqual(node["items"][0]["status"], "pass")
        self.assertIn("Traceback", node["items"][-1]["message"])

    def test_per_link_avg_gate_passes(self):
        results = {}
        transferbench_rundeck.record_a2a(results, {"n1": A2A_BOX}, {"gpu_to_gpu_a2a_avg": "32.9"})
        node = _node(results, "a2a")
        self.assertEqual(node["status"], "pass")
        self.assertEqual(node["items"][0]["name"], "per-link avg")
        self.assertIn("per-link avg", node["items_summary"])

    def test_benign_vmware_warning_does_not_fail_passing_metrics(self):
        results = {}
        text = A2A_BOX + "Failed to open VMware provider: No such file or directory\n"
        transferbench_rundeck.record_a2a(results, {"n1": text}, {"gpu_to_gpu_a2a_avg": "32.9"})
        self.assertEqual(_node(results, "a2a")["status"], "pass")


class TestRecordP2p(unittest.TestCase):
    def test_averages_above_threshold_pass(self):
        results = {}
        transferbench_rundeck.record_p2p(results, {"n1": P2P_OK}, P2P_EXPECT)
        node = _node(results, "p2p")
        self.assertEqual(node["status"], "pass")
        self.assertEqual(node["items_summary"], "UniDir 40.5 / BiDir 50.5 GB/s")

    def test_abort_without_averages_fails(self):
        results = {}
        transferbench_rundeck.record_p2p(results, {"n1": P2P_ABORT}, P2P_EXPECT)
        node = _node(results, "p2p")
        self.assertEqual(node["status"], "fail")
        self.assertEqual(node["items"][0]["message"], "UniDir averages not found")

    def test_abort_fails_passing_averages(self):
        results = {}
        transferbench_rundeck.record_p2p(results, {"n1": P2P_OK + "ABORT\n"}, P2P_EXPECT)
        node = _node(results, "p2p")
        self.assertEqual(node["status"], "fail")
        self.assertEqual(node["items"][0]["status"], "pass")
        self.assertEqual(node["items"][-1]["message"], "ABORT")


class TestRecordScalingAndSchmoo(unittest.TestCase):
    def test_scaling_best_gpu00(self):
        results = {}
        transferbench_rundeck.record_scaling(results, {"n1": SCALING_OK}, {"best_gpu0_bw": "50"})
        node = _node(results, "scaling")
        self.assertEqual(node["status"], "pass")
        self.assertEqual(node["items"][0]["message"], "55.5 GB/s (threshold 50)")

    def test_scaling_below_threshold(self):
        results = {}
        transferbench_rundeck.record_scaling(results, {"n1": SCALING_OK}, {"best_gpu0_bw": "80"})
        self.assertEqual(_node(results, "scaling")["status"], "fail")

    def test_abort_fails_passing_scaling_and_schmoo(self):
        results = {}
        transferbench_rundeck.record_scaling(results, {"n1": SCALING_OK + "ABORT\n"}, {"best_gpu0_bw": "50"})
        self.assertEqual(_node(results, "scaling")["status"], "fail")
        self.assertEqual(_node(results, "scaling")["items"][-1]["message"], "ABORT")
        results = {}
        passing = "  32  10.0  20.0  30.0  40.0  50.0  80.0\nABORT\n"
        transferbench_rundeck.record_schmoo(results, {"n1": passing}, SCHMOO_EXPECT)
        self.assertEqual(_node(results, "schmoo")["status"], "fail")
        self.assertEqual(_node(results, "schmoo")["items"][-1]["message"], "ABORT")

    def test_schmoo_remote_copy_below_threshold(self):
        results = {}
        transferbench_rundeck.record_schmoo(results, {"n1": SCHMOO_OK}, SCHMOO_EXPECT)
        node = _node(results, "schmoo")
        self.assertEqual(node["status"], "fail")
        remote_copy = node["items"][-1]
        self.assertEqual(remote_copy["name"], "remote copy")
        self.assertEqual(remote_copy["status"], "fail")
        self.assertEqual(node["items"][0]["status"], "pass")


A2A_BOX = (
    "TransferBench v1.67.00 (HEAD:2bc42cd) (Single-node mode)\n"
    "│  RTotal │ 326.11   325.01   326.48   324.77   324.87   326.37   324.68   326.75 │    2605.03 │\n"
)
P2P_REAL = (
    "TransferBench v1.67.00 (HEAD:2bc42cd) (Single-node mode)\n"
    "                           CPU->CPU  CPU->GPU  GPU->CPU  GPU->GPU\n"
    "Averages (During UniDir):    100.21     44.83     54.70     48.76\n"
    "Averages (During  BiDir):     89.50     44.73     44.81     45.82\n"
)


def _metric(node, name):
    for item in node.get("metrics") or []:
        if item["name"] == name:
            return item
    raise AssertionError(name)


def _series_named(node, name):
    for item in node.get("series") or []:
        if item["name"] == name:
            return item
    raise AssertionError(name)


def _heatmap_named(node, name):
    for item in node.get("heatmaps") or []:
        if item["name"] == name:
            return item
    raise AssertionError(name)


HEALTHCHECK_REAL = """\
TransferBench v1.67.00 (HEAD:2bc42cd) (Single-node mode)
Testing HBM performance [READ]             ........PASS
Testing unidirectional host to device copy ........PASS
Testing bidirectional host<->device copies ........FAIL (8 test(s))
 GPU 00: Measured:  86.17 GB/s      Criteria:  87.30 GB/s
 GPU 01: Measured:  86.09 GB/s      Criteria:  87.30 GB/s
Testing all-to-all XGMI copies             ........FAIL (56 test(s))
 GPU 00 to GPU 01:  40.30 GB/s      Criteria:  43.65 GB/s
 GPU 00 to GPU 02:  40.73 GB/s      Criteria:  43.65 GB/s
 GPU 01 to GPU 00:  41.22 GB/s      Criteria:  43.65 GB/s
 GPU 01 to GPU 02:  40.71 GB/s      Criteria:  43.65
"""


class TestA2aCharts(unittest.TestCase):
    def test_box_rtotal_emits_per_gpu_metrics(self):
        results = {}
        meta = transferbench_rundeck.make_meta({"cluster_name": "helios"}, "transferbench_cvs")
        transferbench_rundeck.record_a2a(results, {"node-a": A2A_BOX}, {"gpu_to_gpu_a2a_rtotal": "320"}, meta=meta)
        node = _node(results, "a2a", "node-a")
        self.assertEqual(node["status"], "pass")
        self.assertEqual(node["items"][0]["name"], "GPU0")
        gpu00 = _metric(node, "GPU00")
        self.assertEqual(gpu00["value"], 326.11)
        self.assertEqual(gpu00["unit"], "GB/s")
        self.assertEqual(gpu00["threshold"], 320.0)
        self.assertEqual(gpu00["direction"], "higher")
        self.assertEqual(gpu00["status"], "pass")
        self.assertEqual(_metric(node, "GPU07")["value"], 326.75)
        self.assertNotIn("series", node)
        self.assertEqual(results["_meta"]["version"], "1.67.00")

    def test_simple_line_marks_gpu_under_threshold(self):
        results = {}
        transferbench_rundeck.record_a2a(results, {"n1": A2A_LINE}, A2A_EXPECT)
        node = _node(results, "a2a")
        self.assertEqual(node["status"], "fail")
        self.assertEqual(_metric(node, "GPU00")["status"], "fail")
        self.assertEqual(_metric(node, "GPU00")["value"], 90.0)
        self.assertEqual(_metric(node, "GPU01")["status"], "pass")

    def test_missing_rtotal_omits_charts_but_keeps_version(self):
        results = {}
        meta = transferbench_rundeck.make_meta({"cluster_name": "helios"}, "transferbench_cvs")
        transferbench_rundeck.record_a2a(results, {"n1": "TransferBench v1.67.00\nno table\n"}, A2A_EXPECT, meta=meta)
        node = _node(results, "a2a")
        self.assertEqual(node["status"], "fail")
        self.assertNotIn("metrics", node)
        self.assertNotIn("series", node)
        self.assertEqual(results["_meta"]["version"], "1.67.00")


class TestP2pCharts(unittest.TestCase):
    def test_double_space_bidir_and_unidir_paths(self):
        results = {}
        meta = transferbench_rundeck.make_meta({"name": "lab"}, "transferbench_cvs")
        transferbench_rundeck.record_p2p(results, {"node-a": P2P_REAL}, P2P_EXPECT, meta=meta)
        node = _node(results, "p2p", "node-a")
        self.assertEqual(node["status"], "pass")
        self.assertEqual(node["items_summary"], "UniDir 48.76 / BiDir 45.82 GB/s")
        self.assertEqual(node["items"][0]["name"], "UniDir")
        self.assertEqual(node["items"][0]["status"], "pass")
        self.assertNotIn("metrics", node)
        paths = _series_named(node, "UniDir")["points"]
        self.assertEqual([point["x"] for point in paths], ["CPU->CPU", "CPU->GPU", "GPU->CPU", "GPU->GPU"])
        self.assertEqual(paths[0]["y"], 100.21)
        self.assertEqual(paths[-1]["y"], 48.76)
        self.assertEqual(_series_named(node, "BiDir")["points"][0]["y"], 89.50)
        self.assertEqual(results["_meta"]["version"], "1.67.00")

    def test_below_threshold_fails_without_dropping_series(self):
        results = {}
        low = dict(P2P_EXPECT)
        low["avg_gpu_to_gpu_p2p_bidir_bw"] = "90"
        transferbench_rundeck.record_p2p(results, {"n1": P2P_REAL}, low)
        node = _node(results, "p2p")
        self.assertEqual(node["status"], "fail")
        self.assertEqual(node["items"][1]["name"], "BiDir")
        self.assertEqual(node["items"][1]["status"], "fail")
        self.assertEqual(node["items"][0]["status"], "pass")

    def test_abort_omits_charts(self):
        results = {}
        transferbench_rundeck.record_p2p(results, {"n1": P2P_ABORT}, P2P_EXPECT)
        node = _node(results, "p2p")
        self.assertEqual(node["status"], "fail")
        self.assertNotIn("metrics", node)
        self.assertNotIn("series", node)


class TestRecordCompletion(unittest.TestCase):
    def test_clean_healthcheck_passes(self):
        results = {}
        transferbench_rundeck.record_completion(results, "healthcheck", {"n1": "preset finished\n"})
        self.assertEqual(_node(results, "healthcheck")["status"], "pass")
        self.assertEqual(_node(results, "healthcheck")["items_summary"], "completed")

    def test_numa_abort_fails(self):
        results = {}
        transferbench_rundeck.record_completion(results, "a2asweep", {"n1": P2P_ABORT})
        node = _node(results, "a2asweep")
        self.assertEqual(node["status"], "fail")
        self.assertIn("allocate memory", node["items"][0]["message"])


class TestHealthcheckCharts(unittest.TestCase):
    def test_subtests_metrics_and_xgmi_heatmap(self):
        results = {}
        meta = transferbench_rundeck.make_meta({"cluster_name": "helios"}, "transferbench_cvs")
        transferbench_rundeck.record_completion(results, "healthcheck", {"node-a": HEALTHCHECK_REAL}, meta=meta)
        node = _node(results, "healthcheck", "node-a")
        self.assertEqual(node["status"], "fail")
        self.assertEqual(node["items_summary"], "2 pass, 2 fail")
        names = [item["name"] for item in node["items"]]
        self.assertEqual(
            names,
            [
                "HBM performance [READ]",
                "unidirectional host to device copy",
                "bidirectional host<->device copies",
                "all-to-all XGMI copies",
            ],
        )
        self.assertEqual(node["items"][0]["status"], "pass")
        self.assertEqual(node["items"][2]["status"], "fail")
        self.assertEqual(node["items"][2]["message"], "FAIL (8 test(s))")
        gpu00 = _metric(node, "GPU00")
        self.assertEqual(gpu00["value"], 86.17)
        self.assertEqual(gpu00["threshold"], 87.30)
        self.assertEqual(gpu00["status"], "fail")
        self.assertEqual(_metric(node, "GPU01")["value"], 86.09)
        heat = _heatmap_named(node, "XGMI")
        self.assertEqual(heat["rows"], ["GPU00", "GPU01", "GPU02"])
        self.assertEqual(heat["cols"], heat["rows"])
        self.assertEqual(heat["row_label"], "Src")
        self.assertEqual(heat["col_label"], "Dst")
        self.assertEqual(heat["threshold"], 43.65)
        self.assertEqual(heat["direction"], "higher")
        self.assertEqual(heat["values"][0], [None, 40.30, 40.73])
        self.assertEqual(heat["values"][1][0], 41.22)
        self.assertIsNone(heat["values"][1][1])
        self.assertEqual(heat["values"][2], [None, None, None])
        self.assertEqual(results["_meta"]["version"], "1.67.00")

    def test_scan_failure_still_fails_the_node(self):
        results = {}
        text = HEALTHCHECK_REAL + "\n" + P2P_ABORT
        transferbench_rundeck.record_completion(results, "healthcheck", {"n1": text})
        node = _node(results, "healthcheck")
        self.assertEqual(node["status"], "fail")
        self.assertIn("allocate memory", node["items"][-1]["message"])
        self.assertIn("metrics", node)
        self.assertIn("heatmaps", node)

    def test_subtest_fail_fails_the_cell(self):
        results = {}
        text = "Testing HBM performance [READ] ........FAIL\n"
        transferbench_rundeck.record_completion(results, "healthcheck", {"n1": text})
        node = _node(results, "healthcheck")
        self.assertEqual(node["status"], "fail")
        self.assertEqual(node["items_summary"], "0 pass, 1 fail")
        self.assertEqual(node["items"][0]["status"], "fail")

    def test_garbled_pairs_omit_heatmap(self):
        results = {}
        text = "Testing HBM performance [READ] ........PASS\n GPU 00 to GPU 01: nope GB/s Criteria: 1\n"
        transferbench_rundeck.record_completion(results, "healthcheck", {"n1": text})
        node = _node(results, "healthcheck")
        self.assertEqual(node["status"], "pass")
        self.assertEqual(node["items_summary"], "1 pass, 0 fail")
        self.assertNotIn("heatmaps", node)
        self.assertNotIn("metrics", node)


A2A_SWEEP = """\
TransferBench v1.67.00 (HEAD:2bc42cd) (Single-node mode)
 BlkS   UnR    SE 004   SE 008   SE 012   SE 016   SE 024   SE 032
  256     1    113.41   215.96   296.31   319.75   324.02   322.96
  512     1    203.89   311.82   310.15   319.70   325.20   327.09
=======================================================================================
Highest GPU-event-timed (min) bandwidth found:  327.09 GB/s
          BlockSize  :     512
          Unroll     :       1
          NumSubExec :      32
"""

SCALING_REAL = (
    "TransferBench v1.67.00 (HEAD:2bc42cd) (Single-node mode)\n"
    "NumCUs CPU00 CPU01 GPU00 GPU01 GPU02 GPU03 GPU04 GPU05 GPU06 GPU07\n"
    "1 22.74 22.83 34.77 35.69 35.48 35.48 35.32 35.28 35.41 35.41\n"
    "2 41.39 41.55 69.35 49.24 49.28 49.79 49.12 49.80 48.66 49.00\n"
    "32 56.28 56.29 843.62 48.80 48.85 49.14 48.57 48.98 47.68 48.69\n"
    "Best 56.28( 32) 56.29( 32) 843.62( 32) 49.43(  6) 49.90(  6) "
    "49.79(  6) 49.12(  2) 49.80(  2) 48.66(  2) 49.78(  5)\n"
)

SCHMOO_REAL = """\
TransferBench v1.67.00 (HEAD:2bc42cd) (Single-node mode)
       | Local Read  | Local Write | Local Copy  | Remote Read | Remote Write| Remote Copy |
  #CUs |G00->G00->N00|N00->G00->G00|G00->G00->G00|G01->G00->N00|N00->G00->G01|G00->G00->G01|
    16      3000.000      1500.000       700.000        40.000        41.000        42.000
    32      3465.248      1642.561       806.209        49.072        49.911        49.312
"""
SCHMOO_REAL_EXPECT = {
    "32_cu_local_read": "1650",
    "32_cu_local_write": "1250.0",
    "32_cu_local_copy": "1250.0",
    "32_cu_rem_read": "48.0",
    "32_cu_rem_write": "48.0",
    "32_cu_rem_copy": "48.0",
}


class TestSweepScalingSchmooCharts(unittest.TestCase):
    def test_a2asweep_heatmap_and_best_config(self):
        results = {}
        meta = transferbench_rundeck.make_meta({"cluster_name": "helios"}, "transferbench_cvs")
        transferbench_rundeck.record_completion(results, "a2asweep", {"node-a": A2A_SWEEP}, meta=meta)
        node = _node(results, "a2asweep", "node-a")
        self.assertEqual(node["status"], "pass")
        self.assertEqual(node["items_summary"], "completed")
        heat = _heatmap_named(node, "a2a sweep")
        self.assertEqual(heat["rows"], ["256/1", "512/1"])
        self.assertEqual(heat["cols"], ["SE 004", "SE 008", "SE 012", "SE 016", "SE 024", "SE 032"])
        self.assertEqual(heat["row_label"], "BlkS/UnR")
        self.assertEqual(heat["col_label"], "SubExec")
        self.assertEqual(heat["values"][1][-1], 327.09)
        self.assertNotIn("threshold", heat)
        self.assertNotIn("metrics", node)
        self.assertEqual(results["_meta"]["version"], "1.67.00")

    def test_a2asweep_abort_keeps_fail_and_charts(self):
        results = {}
        transferbench_rundeck.record_completion(results, "a2asweep", {"n1": P2P_ABORT + A2A_SWEEP})
        node = _node(results, "a2asweep")
        self.assertEqual(node["status"], "fail")
        self.assertIn("allocate memory", node["items"][0]["message"])
        self.assertIn("heatmaps", node)

    def test_a2asweep_without_table_omits_heatmap(self):
        results = {}
        text = "Highest GPU-event-timed (min) bandwidth found:  10.5 GB/s\n          BlockSize  :     256\n"
        transferbench_rundeck.record_completion(results, "a2asweep", {"n1": text})
        node = _node(results, "a2asweep")
        self.assertEqual(node["status"], "pass")
        self.assertNotIn("heatmaps", node)
        self.assertNotIn("metrics", node)

    def test_scaling_series_per_endpoint(self):
        results = {}
        transferbench_rundeck.record_scaling(results, {"node-a": SCALING_REAL}, {"best_gpu0_bw": "480"})
        node = _node(results, "scaling", "node-a")
        self.assertEqual(node["status"], "pass")
        self.assertEqual(node["items"][0]["name"], "GPU00")
        self.assertEqual(node["items"][0]["status"], "pass")
        gpu00 = _series_named(node, "GPU00")
        self.assertEqual([point["x"] for point in gpu00["points"]], [1, 2, 32])
        self.assertEqual(gpu00["points"][-1]["y"], 843.62)
        self.assertEqual(_series_named(node, "CPU00")["points"][0]["y"], 22.74)
        self.assertEqual(_series_named(node, "GPU07")["points"][1]["y"], 49.00)
        self.assertEqual(len(node["series"]), 10)
        self.assertNotIn("metrics", node)

    def test_scaling_gpu00_below_threshold_fails_node(self):
        results = {}
        transferbench_rundeck.record_scaling(results, {"n1": SCALING_REAL}, {"best_gpu0_bw": "900"})
        node = _node(results, "scaling")
        self.assertEqual(node["status"], "fail")
        self.assertEqual(node["items"][0]["status"], "fail")

    def test_schmoo_local_copy_misses_threshold_on_its_own_series(self):
        results = {}
        transferbench_rundeck.record_schmoo(results, {"node-a": SCHMOO_REAL}, SCHMOO_REAL_EXPECT)
        node = _node(results, "schmoo", "node-a")
        self.assertEqual(node["status"], "fail")
        local_copy = _metric(node, "local copy")
        self.assertEqual(local_copy["value"], 806.209)
        self.assertEqual(local_copy["threshold"], 1250.0)
        self.assertEqual(local_copy["status"], "fail")
        self.assertEqual(_metric(node, "local read")["status"], "pass")
        self.assertEqual(_metric(node, "remote copy")["value"], 49.312)
        self.assertEqual(_metric(node, "remote copy")["status"], "pass")
        local_series = _series_named(node, "local copy")
        remote_series = _series_named(node, "remote copy")
        self.assertEqual([point["y"] for point in local_series["points"]], [700.0, 806.209])
        self.assertEqual([point["y"] for point in remote_series["points"]], [42.0, 49.312])
        self.assertGreater(local_series["points"][-1]["y"], remote_series["points"][-1]["y"] * 10)

    def test_schmoo_without_32_row_keeps_series_and_fails(self):
        results = {}
        text = "    16      1.0      2.0      3.0      4.0      5.0      6.0\n"
        transferbench_rundeck.record_schmoo(results, {"n1": text}, SCHMOO_REAL_EXPECT)
        node = _node(results, "schmoo")
        self.assertEqual(node["status"], "fail")
        self.assertIn("not found", node["items"][0]["message"])
        self.assertNotIn("metrics", node)
        self.assertNotIn("series", node)

    def test_unknown_preset_text_does_not_raise(self):
        results = {}
        for recorder, args in (
            (transferbench_rundeck.record_scaling, ({"n1": "not a table"}, {"best_gpu0_bw": "1"})),
            (transferbench_rundeck.record_schmoo, ({"n1": "not a table"}, SCHMOO_REAL_EXPECT)),
        ):
            recorder(results, *args)
        transferbench_rundeck.record_completion(results, "a2asweep", {"n1": "BlkS UnR\nno columns\n"})
        self.assertEqual(_node(results, "scaling")["status"], "fail")
        self.assertNotIn("series", _node(results, "scaling"))
        self.assertEqual(_node(results, "schmoo")["status"], "fail")
        self.assertEqual(_node(results, "a2asweep")["status"], "pass")
        self.assertNotIn("heatmaps", _node(results, "a2asweep"))


if __name__ == "__main__":
    unittest.main()
