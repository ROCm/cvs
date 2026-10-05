'''Unit tests for cvs.lib.agfhc_rundeck (Run Deck result capture from AGFHC stdout).'''

import unittest

from cvs.lib import agfhc_rundeck

_SUCCESS = "Program exiting with return code AGFHC_SUCCESS [0]\n"
_RECIPE_INFO = """\
Log Directory: /root/agfhc/logs/agfhc_20250707-082147
Name: all_lvl5
Title: A ~2h check across system
Contents:
Test Title Mode Approximate Time
pcie_link_status PCIe Link Status 1 iteration 0:00:08
hbm_bw HBM BW 1 iteration 0:01:05
---------
Total: 02:08:47
Summary:
Tests: 0 Total, 0 Executed, 0 Skipped
Total Time: 00:00:03
Log directory: /root/agfhc/logs/agfhc_20250707-082147
Program exiting with return code AGFHC_SUCCESS [0]
"""


class TestMakeMeta(unittest.TestCase):
    def test_pulls_cluster_name(self):
        meta = agfhc_rundeck.make_meta({"cluster_name": "helios"}, "agfhc_cvs")
        self.assertEqual(meta["cluster"], "helios")
        self.assertEqual(meta["version"], "—")
        self.assertEqual(meta["version_label"], "AGFHC version")
        self.assertEqual(meta["suite"], "agfhc_cvs")

    def test_defaults_when_missing(self):
        meta = agfhc_rundeck.make_meta({}, "")
        self.assertEqual(meta["cluster"], "—")
        self.assertEqual(meta["suite"], "agfhc_cvs")


class TestClassifyOutput(unittest.TestCase):
    def test_success_code_passes(self):
        status, items, summary = agfhc_rundeck.classify_output(_SUCCESS)
        self.assertEqual(status, "pass")
        self.assertEqual(items, [])
        self.assertEqual(summary, "passed")

    def test_missing_success_code_fails(self):
        status, items, summary = agfhc_rundeck.classify_output("recipe finished\n")
        self.assertEqual(status, "fail")
        self.assertEqual(items[0]["name"], "AGFHC_SUCCESS")
        self.assertEqual(summary, "AGFHC_SUCCESS not seen")

    def test_failure_token_fails_even_with_success_word_nearby(self):
        text = "Program exiting with return code AGFHC_FAILURE [1]\n"
        status, items, _summary = agfhc_rundeck.classify_output(text)
        self.assertEqual(status, "fail")
        self.assertEqual(items[0]["name"], "AGFHC_SUCCESS")
        self.assertIn("AGFHC_FAILURE", items[1]["message"])

    def test_scan_line_fails_a_success_banner(self):
        text = "hbm check ERROR on GPU0\n" + _SUCCESS
        status, items, _summary = agfhc_rundeck.classify_output(text)
        self.assertEqual(status, "fail")
        self.assertEqual(items[0]["name"], "scan")
        self.assertIn("ERROR", items[0]["message"])

    def test_simple_rows_keep_worst_status(self):
        text = "\n".join(
            [
                "hbm_bw                                  failed",
                "hbm_bw                                  passed",
                "pcie_link_status                        passed",
                "xgmi_a2a .................... SKIPPED",
                _SUCCESS.strip(),
            ]
        )
        status, items, summary = agfhc_rundeck.classify_output(text)
        by_name = {item["name"]: item for item in items}
        self.assertEqual(status, "fail")
        self.assertEqual(by_name["hbm_bw"]["status"], "fail")
        self.assertEqual(by_name["pcie_link_status"]["status"], "pass")
        self.assertEqual(by_name["xgmi_a2a"]["status"], "na")
        self.assertEqual(summary, "1 pass, 1 fail, 1 skipped")

    def test_pipe_and_colon_rows(self):
        text = "\n".join(
            [
                "| dma_bidi_peak | Passed |",
                "gfx_dgemm: FAILED",
                _SUCCESS.strip(),
            ]
        )
        status, items, _summary = agfhc_rundeck.classify_output(text)
        by_name = {item["name"]: item for item in items}
        self.assertEqual(status, "fail")
        self.assertEqual(by_name["dma_bidi_peak"]["status"], "pass")
        self.assertEqual(by_name["gfx_dgemm"]["status"], "fail")

    def test_recipe_info_is_not_a_result_table(self):
        status, items, summary = agfhc_rundeck.classify_output(_RECIPE_INFO)
        self.assertEqual(status, "pass")
        self.assertEqual(items, [])
        self.assertEqual(summary, "0 total, 0 executed, 0 skipped")
        self.assertNotIn("pcie_link_status", summary)

    def test_skipped_row_does_not_fail_a_success_run(self):
        text = "mall                                    skipped\n" + _SUCCESS
        status, items, summary = agfhc_rundeck.classify_output(text)
        self.assertEqual(status, "pass")
        self.assertEqual(items[0]["status"], "na")
        self.assertEqual(summary, "0 pass, 0 fail, 1 skipped")


class TestRecordOutputs(unittest.TestCase):
    def test_records_pass_and_fail_nodes(self):
        results = {}
        meta = agfhc_rundeck.make_meta({"name": "lab"}, "agfhc_cvs")
        agfhc_rundeck.record_outputs(
            results,
            "hbm",
            {"n1": _SUCCESS, "n2": "saw FAIL here\n"},
            meta=meta,
        )
        nodes = results["groups"]["hbm"]["nodes"]
        self.assertEqual(nodes["n1"]["status"], "pass")
        self.assertEqual(nodes["n2"]["status"], "fail")
        self.assertEqual(results["_meta"]["cluster"], "lab")
        self.assertEqual(results["_meta"]["version_label"], "AGFHC version")

    def test_version_banner_fills_placeholder(self):
        results = {}
        text = "agfhc version:\n1.24.1\npackage: mi300x\n" + _SUCCESS
        agfhc_rundeck.record_outputs(
            results,
            "gfx_lvl1",
            {"node-a": text},
            meta=agfhc_rundeck.make_meta({"cluster_name": "helios"}, "agfhc_cvs"),
        )
        self.assertEqual(results["_meta"]["version"], "1.24.1")
        self.assertEqual(results["groups"]["gfx_lvl1"]["nodes"]["node-a"]["status"], "pass")

    def test_later_real_version_fills_placeholder(self):
        results = {}
        agfhc_rundeck.record_outputs(
            results,
            "hbm",
            {"n1": _SUCCESS},
            meta={"version": "—", "cluster": "c", "version_label": "AGFHC version"},
        )
        agfhc_rundeck.record_outputs(
            results,
            "pcie_lvl1",
            {"n1": _SUCCESS},
            meta={"version": "1.22.0", "cluster": "c", "version_label": "AGFHC version"},
        )
        self.assertEqual(results["_meta"]["version"], "1.22.0")

    def test_rerun_overwrites_same_node(self):
        results = {}
        agfhc_rundeck.record_outputs(results, "dma_lvl1", {"n1": "FAIL\n"})
        agfhc_rundeck.record_outputs(results, "dma_lvl1", {"n1": _SUCCESS})
        self.assertEqual(results["groups"]["dma_lvl1"]["nodes"]["n1"]["status"], "pass")

    def test_dict_output_uses_output_field(self):
        results = {}
        agfhc_rundeck.record_outputs(results, "all_perf", {"n1": {"output": _SUCCESS}})
        self.assertEqual(results["groups"]["all_perf"]["nodes"]["n1"]["status"], "pass")


class TestRecordVersionCheck(unittest.TestCase):
    def test_version_banner_passes_and_fills_meta(self):
        results = {}
        text = "agfhc version:\n1.24.1\npackage: mi300x\n"
        agfhc_rundeck.record_version_check(
            results,
            {"node-a": text},
            meta=agfhc_rundeck.make_meta({"cluster_name": "helios"}, "csp_qual_agfhc"),
        )
        node = results["groups"]["version_check"]["nodes"]["node-a"]
        self.assertEqual(node["status"], "pass")
        self.assertEqual(results["_meta"]["suite"], "csp_qual_agfhc")
        self.assertEqual(results["_meta"]["version"], "1.24.1")

    def test_missing_version_banner_fails(self):
        results = {}
        agfhc_rundeck.record_version_check(results, {"n1": "not installed\n"})
        self.assertEqual(results["groups"]["version_check"]["nodes"]["n1"]["status"], "fail")


if __name__ == "__main__":
    unittest.main()
