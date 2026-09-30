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


if __name__ == "__main__":
    unittest.main()
