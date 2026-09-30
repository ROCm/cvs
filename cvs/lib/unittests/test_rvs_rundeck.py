'''Unit tests for cvs.lib.rvs_rundeck (Run Deck result capture from RVS stdout).'''

import unittest

from cvs.lib import rvs_rundeck


class TestMakeMeta(unittest.TestCase):
    def test_pulls_cluster_and_version(self):
        meta = rvs_rundeck.make_meta({"cluster_name": "helios"}, "1.3.0", "rvs_cvs")
        self.assertEqual(meta["cluster"], "helios")
        self.assertEqual(meta["version"], "1.3.0")
        self.assertEqual(meta["version_label"], "RVS version")
        self.assertEqual(meta["suite"], "rvs_cvs")

    def test_defaults_when_missing(self):
        meta = rvs_rundeck.make_meta({}, "", "")
        self.assertEqual(meta["cluster"], "—")
        self.assertEqual(meta["version"], "—")
        self.assertEqual(meta["suite"], "rvs_cvs")


class TestClassifyOutput(unittest.TestCase):
    def test_error_pattern_fails_node(self):
        status, items = rvs_rundeck.classify_output("module [ERROR ] gst failed", [r"\[ERROR\s*\]"])
        self.assertEqual(status, "fail")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["status"], "fail")

    def test_clean_output_passes(self):
        status, items = rvs_rundeck.classify_output("RVS module completed", [r"\[ERROR\s*\]"])
        self.assertEqual(status, "pass")
        self.assertEqual(items, [])

    def test_gpu_enumeration_no_gpu_line(self):
        status, _items = rvs_rundeck.classify_output(
            "No supported GPUs available on this host",
            [r"No supported GPUs available"],
        )
        self.assertEqual(status, "fail")

    def test_level_reports_each_matched_pattern(self):
        output = "gst [ERROR ] boom\nmem FAIL seen"
        status, items = rvs_rundeck.classify_output(output, [r"\[ERROR\s*\]", r"\bFAIL\b"])
        self.assertEqual(status, "fail")
        self.assertEqual(len(items), 2)

    def test_invalid_regex_is_ignored(self):
        status, items = rvs_rundeck.classify_output("anything", ["["])
        self.assertEqual(status, "pass")
        self.assertEqual(items, [])


class TestRecordOutputs(unittest.TestCase):
    def test_records_pass_and_fail_nodes(self):
        results = {}
        meta = rvs_rundeck.make_meta({"name": "lab"}, "1.2.0", "rvs_cvs")
        rvs_rundeck.record_outputs(
            results,
            "gst_single",
            {"n1": "ok", "n2": "saw [ERROR ] here"},
            [r"\[ERROR\s*\]"],
            meta=meta,
        )
        nodes = results["groups"]["gst_single"]["nodes"]
        self.assertEqual(nodes["n1"]["status"], "pass")
        self.assertEqual(nodes["n1"]["items_summary"], "passed")
        self.assertEqual(nodes["n2"]["status"], "fail")
        self.assertEqual(nodes["n2"]["items_summary"], "1 failure pattern(s)")
        self.assertEqual(results["_meta"]["cluster"], "lab")
        self.assertEqual(results["_meta"]["version_label"], "RVS version")

    def test_later_real_version_fills_placeholder(self):
        results = {}
        rvs_rundeck.record_outputs(results, "gpu_enumeration", {"n1": "ok"}, [], meta={"version": "—", "cluster": "c"})
        rvs_rundeck.record_outputs(
            results,
            "mem_test",
            {"n1": "ok"},
            [],
            meta={"version": "1.3.0", "cluster": "c", "version_label": "RVS version"},
        )
        self.assertEqual(results["_meta"]["version"], "1.3.0")

    def test_rerun_overwrites_same_node(self):
        results = {}
        rvs_rundeck.record_outputs(results, "mem_test", {"n1": "[ERROR ]"}, [r"\[ERROR\s*\]"])
        rvs_rundeck.record_outputs(results, "mem_test", {"n1": "clean"}, [r"\[ERROR\s*\]"])
        self.assertEqual(results["groups"]["mem_test"]["nodes"]["n1"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
