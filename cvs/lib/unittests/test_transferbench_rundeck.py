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
    def test_box_rtotal_emits_per_gpu_metrics_and_series(self):
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
        series = _series_named(node, "RTotal")
        self.assertEqual([point["x"] for point in series["points"]], [f"GPU{idx:02d}" for idx in range(8)])
        self.assertEqual(series["points"][3]["y"], 324.77)
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
        unidir = _metric(node, "UniDir")
        self.assertEqual(unidir["value"], 48.76)
        self.assertEqual(unidir["threshold"], 33.9)
        self.assertEqual(unidir["status"], "pass")
        bidir = _metric(node, "BiDir")
        self.assertEqual(bidir["value"], 45.82)
        self.assertEqual(bidir["threshold"], 43.9)
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
        self.assertEqual(_metric(node, "BiDir")["status"], "fail")
        self.assertEqual(_metric(node, "UniDir")["status"], "pass")

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
        self.assertEqual(node["status"], "pass")
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

    def test_garbled_pairs_omit_heatmap(self):
        results = {}
        text = "Testing HBM performance [READ] ........PASS\n GPU 00 to GPU 01: nope GB/s Criteria: 1\n"
        transferbench_rundeck.record_completion(results, "healthcheck", {"n1": text})
        node = _node(results, "healthcheck")
        self.assertEqual(node["status"], "pass")
        self.assertEqual(node["items_summary"], "1 pass, 0 fail")
        self.assertNotIn("heatmaps", node)
        self.assertNotIn("metrics", node)


if __name__ == "__main__":
    unittest.main()
