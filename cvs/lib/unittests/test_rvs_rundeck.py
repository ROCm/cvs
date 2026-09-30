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


class TestGstAndIetMetrics(unittest.TestCase):
    def test_gst_final_line_sets_threshold_and_ignores_interval(self):
        text = "\n".join(
            [
                "[compute-fp8-trig] [GPU::  2987] GFLOPS 1400000",
                "[compute-fp8-trig] [GPU::  2987] GFLOPS 1299650 Target GFLOPS: 983000 met: TRUE",
            ]
        )
        perf = rvs_rundeck.parse_node_performance(text, module_hint="gst_single")
        self.assertEqual(len(perf["metrics"]), 1)
        metric = perf["metrics"][0]
        self.assertEqual(metric["name"], "fp8")
        self.assertEqual(metric["unit"], "GFLOPS")
        self.assertNotEqual(metric["unit"], "TFLOPS")
        self.assertAlmostEqual(metric["value"], 1299650.0)
        self.assertAlmostEqual(metric["threshold"], 983000.0)
        self.assertEqual(metric["direction"], "higher")
        self.assertEqual(metric["status"], "pass")
        self.assertEqual(metric["group"], "gst")
        self.assertEqual(perf["series"][0]["points"], [{"x": "2987", "y": 1299650.0}])

    def test_gst_precisions_and_slowest_gpu(self):
        text = "\n".join(
            [
                "[GPU:: 5 - 2987 - 0000:15:00.0]",
                "[GPU:: 2 - 9091 - 0000:75:00.0]",
                "[compute-fp8-trig] [GPU::  2987] GFLOPS 1299650 Target GFLOPS: 983000 met: TRUE",
                "[compute-fp8-trig] [GPU::  9091] GFLOPS 100 Target GFLOPS: 983000 met: FALSE",
                "[compute-fp16-trig] [GPU:: 2987] GFLOPS 717247 Target GFLOPS: 524000 met: TRUE",
                "[compute-bf16-trig] [GPU:: 2987] GFLOPS 774410 Target GFLOPS: 554000 met: TRUE",
                "[compute-fp32-trig] [GPU:: 2987] GFLOPS 119547 Target GFLOPS: 100000 met: TRUE",
                "[compute-fp64-trig] [GPU:: 9091] GFLOPS 116652 Target GFLOPS: 70000 met: TRUE",
            ]
        )
        perf = rvs_rundeck.parse_node_performance(text, module_hint="level_config")
        by_name = {metric["name"]: metric for metric in perf["metrics"]}
        self.assertEqual(list(by_name), ["fp8", "fp16", "bf16", "fp32", "fp64"])
        self.assertAlmostEqual(by_name["fp8"]["value"], 100.0)
        self.assertEqual(by_name["fp8"]["status"], "fail")
        self.assertEqual(by_name["fp64"]["status"], "pass")
        fp8 = next(item for item in perf["series"] if item["name"] == "fp8")
        self.assertEqual([point["x"] for point in fp8["points"]], ["GPU2", "GPU5"])
        self.assertAlmostEqual(fp8["points"][0]["y"], 100.0)
        self.assertAlmostEqual(fp8["points"][1]["y"], 1299650.0)

    def test_gst_metric_does_not_change_node_verdict(self):
        text = "[compute-fp8-trig] [GPU:: 2987] GFLOPS 10 Target GFLOPS: 983000 met: FALSE"
        results = {}
        rvs_rundeck.record_outputs(results, "gst_single", {"node-a": text}, [])
        node = results["groups"]["gst_single"]["nodes"]["node-a"]
        self.assertEqual(node["status"], "pass")
        self.assertEqual(node["metrics"][0]["status"], "fail")

        results = {}
        rvs_rundeck.record_outputs(results, "gst_single", {"node-a": text}, [r"met:\s*FALSE"])
        node = results["groups"]["gst_single"]["nodes"]["node-a"]
        self.assertEqual(node["status"], "fail")
        self.assertEqual(node["items_summary"], "1 failure pattern(s)")

    def test_iet_peak_power_per_gpu(self):
        text = "\n".join(
            [
                "[GPU:: 2 - 9091 - 0000:75:00.0]",
                "[GPU:: 5 - 2987 - 0000:15:00.0]",
                "Module name :iet",
                "[power-stress] [GPU::  9091] Power(W) 148.000000",
                "[power-stress] [GPU::  9091] Power(W) 992.000000",
                "[power-stress] [GPU::  2987] Power(W) 991.000000",
                "[power-stress] [GPU::  9091] pass: TRUE",
                "[power-stress] [GPU::  2987] pass: FALSE",
                "Module name :gst",
                "[compute-fp64-trig] [GPU:: 9091] GFLOPS 116652 Target GFLOPS: 70000 met: TRUE",
            ]
        )
        perf = rvs_rundeck.parse_node_performance(text, module_hint="iet_stress")
        power = [metric for metric in perf["metrics"] if metric["name"].startswith("power ")]
        self.assertEqual([metric["name"] for metric in power], ["power GPU2", "power GPU5"])
        self.assertAlmostEqual(power[0]["value"], 992.0)
        self.assertEqual(power[0]["unit"], "W")
        self.assertEqual(power[0]["group"], "iet")
        self.assertEqual(power[0]["status"], "pass")
        self.assertEqual(power[1]["status"], "fail")
        self.assertTrue(all("temp" not in metric["name"].lower() for metric in perf["metrics"]))
        series = next(item for item in perf["series"] if item["name"] == "power")
        self.assertEqual(series["points"], [{"x": "GPU2", "y": 992.0}, {"x": "GPU5", "y": 991.0}])
        self.assertEqual(series["group"], "iet")
        gst = next(metric for metric in perf["metrics"] if metric["name"] == "fp64")
        self.assertEqual(gst["group"], "gst")

    def test_missing_or_unknown_output_has_no_metrics(self):
        for text in ("", "RVS module completed", None):
            perf = rvs_rundeck.parse_node_performance(text, module_hint="gst_single")
            self.assertEqual(perf, {"metrics": [], "series": [], "heatmaps": []})
        interval_only = "[compute-fp8-trig] [GPU:: 2987] GFLOPS 1299650"
        perf = rvs_rundeck.parse_node_performance(interval_only)
        self.assertEqual(perf["metrics"], [])
        results = {}
        rvs_rundeck.record_outputs(results, "gpu_enumeration", {"node-b": "No supported GPUs available"}, [])
        node = results["groups"]["gpu_enumeration"]["nodes"]["node-b"]
        self.assertEqual(node["status"], "pass")
        self.assertNotIn("metrics", node)


if __name__ == "__main__":
    unittest.main()
