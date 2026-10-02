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
        self.assertEqual(items[0]["name"], r"\[ERROR\s*\]")
        self.assertEqual(items[0]["message"], "module [ERROR ] gst failed")

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
        self.assertEqual([item["message"] for item in items], ["gst [ERROR ] boom", "mem FAIL seen"])

    def test_invalid_regex_is_ignored(self):
        status, items = rvs_rundeck.classify_output("anything", ["["])
        self.assertEqual(status, "pass")
        self.assertEqual(items, [])

    def test_scan_indicator_fails_a_passing_measurement(self):
        output = "[gst] [GPU:: 0] GFLOPS 100.0 Target GFLOPS: 50.0 met: TRUE\nTraceback (most recent call last):\n"
        status, items = rvs_rundeck.classify_output(output, [r"\[ERROR\s*\]"])
        self.assertEqual(status, "fail")
        self.assertEqual(items[0]["name"], "scan")
        self.assertIn("Traceback", items[0]["message"])


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
        self.assertEqual(perf["series"], [])

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
        self.assertEqual(perf["series"], [])
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


class TestPebbAndPbqtMetrics(unittest.TestCase):
    def test_pebb_duration_per_gpu_direction_skips_intervals(self):
        text = "\n".join(
            [
                "[pcie_d2h_bandwidth] pcie-bandwidth (*) [CPU:: 0] [GPU:: 2 - 9091 - 0000:75:00.0] "
                "h2d::false d2h::true 10.000 GBps duration: 0.1 secs",
                "[pcie_d2h_bandwidth] pcie-bandwidth [ 1/16] [CPU:: 0] [GPU:: 2 - 9091 - 0000:75:00.0] "
                "h2d::false d2h::true 53.438 GBps duration: 0.080373 secs",
                "[pcie_d2h_bandwidth] pcie-bandwidth [ 2/16] [CPU:: 1] [GPU:: 2 - 9091 - 0000:75:00.0] "
                "h2d::false d2h::true 52.971 GBps duration: 0.081081 secs",
                "[pcie_d2h_bandwidth] pcie-bandwidth [CPU:: 0] [GPU:: 2 - 9091 - 0000:75:00.0] distance:20 PCIe:20",
                "[pcie_h2d_bandwidth] pcie-bandwidth [ 1/16] [CPU:: 0] [GPU:: 2 - 9091 - 0000:75:00.0] "
                "h2d::true d2h::false 54.828 GBps duration: 0.078336 secs",
                "[pcie_d2h_bandwidth] pcie-bandwidth [ 7/16] [CPU:: 0] [GPU:: 5 - 2987 - 0000:15:00.0] "
                "h2d::false d2h::true 52.849 GBps duration: 0.081268 secs",
            ]
        )
        perf = rvs_rundeck.parse_node_performance(text, module_hint="pebb_single")
        by_name = {metric["name"]: metric for metric in perf["metrics"]}
        self.assertEqual(set(by_name), {"h2d GPU2", "d2h GPU2", "d2h GPU5"})
        self.assertAlmostEqual(by_name["d2h GPU2"]["value"], 53.438)
        self.assertAlmostEqual(by_name["h2d GPU2"]["value"], 54.828)
        self.assertEqual(by_name["d2h GPU2"]["unit"], "GB/s")
        self.assertEqual(by_name["d2h GPU2"]["direction"], "higher")
        self.assertEqual(by_name["d2h GPU2"]["group"], "pebb")
        self.assertEqual(perf["series"], [])
        self.assertEqual(perf["heatmaps"], [])

    def test_pbqt_pair_heatmap_and_slowest_link(self):
        text = "\n".join(
            [
                "[xgmi_d2d_bandwidth] p2p-bandwidth[ 1/56] [GPU:: 2 - 9091 - 0000:75:00.0] "
                "[GPU:: 3 - 51110 - 0000:05:00.0] bidirectional: true 101.231 GBps duration: 0.254565 secs",
                "[xgmi_d2d_bandwidth] p2p-bandwidth[ 8/56] [GPU:: 3 - 51110 - 0000:05:00.0] "
                "[GPU:: 2 - 9091 - 0000:75:00.0] bidirectional: true 99.100 GBps duration: 0.254476 secs",
                "[xgmi_d2d_bandwidth] p2p-bandwidth (*) [GPU:: 2 - 9091 - 0000:75:00.0] "
                "[GPU:: 3 - 51110 - 0000:05:00.0] bidirectional: true 1.000 GBps duration: 0.1 secs",
                "[xgmi_d2d_bandwidth] p2p-bandwidth[ 2/56] [GPU:: 2 - 9091 - 0000:75:00.0] "
                "[GPU:: 4 - 61326 - 0000:65:00.0] bidirectional: true 101.770 GBps",
            ]
        )
        perf = rvs_rundeck.parse_node_performance(text, module_hint="pbqt_single")
        self.assertEqual(perf["metrics"], [])
        heat = perf["heatmaps"][0]
        self.assertEqual(heat["name"], "xgmi")
        self.assertEqual(heat["rows"], ["GPU2", "GPU3"])
        self.assertEqual(heat["cols"], ["GPU2", "GPU3"])
        self.assertEqual(heat["row_label"], "Src")
        self.assertEqual(heat["col_label"], "Dst")
        self.assertEqual(heat["group"], "pbqt")
        self.assertEqual(heat["unit"], "GB/s")
        self.assertIsNone(heat["values"][0][0])
        self.assertAlmostEqual(heat["values"][0][1], 101.231)
        self.assertAlmostEqual(heat["values"][1][0], 99.1)
        self.assertIsNone(heat["values"][1][1])

    def test_bandwidth_metrics_do_not_change_node_verdict(self):
        text = (
            "[pcie_h2d_bandwidth] pcie-bandwidth [ 1/16] [CPU:: 0] [GPU:: 2 - 9091 - 0000:75:00.0] "
            "h2d::true d2h::false 54.828 GBps duration: 0.078 secs [ERROR ]"
        )
        results = {}
        rvs_rundeck.record_outputs(results, "pebb_single", {"node-a": text}, [r"\[ERROR\s*\]"])
        node = results["groups"]["pebb_single"]["nodes"]["node-a"]
        self.assertEqual(node["status"], "fail")
        self.assertEqual(node["metrics"][0]["name"], "h2d GPU2")
        self.assertAlmostEqual(node["metrics"][0]["value"], 54.828)


class TestBabelAndMemMetrics(unittest.TestCase):
    def test_babel_uses_first_column_and_gpu_kernel_heatmap(self):
        text = "\n".join(
            [
                "[GPU:: 5 - 2987 - 0000:15:00.0]",
                "[GPU:: 2 - 9091 - 0000:75:00.0]",
                "Action name :hbm_full",
                "Module name :babel",
                "GPU Id      Function    MiBytes/sec    Max MiB/s      Min MiB/s      Avg MiB/s",
                "2987        Read        100.0          5068077.546    4583316.549    4978787.260",
                "2987        Triad       50.0           4360496.063    3957406.353    4285494.042",
                "9091        Read        80.0           5054319.123    4568593.596    4985863.158",
                "Module name :gst",
                "2987        Read        999.0          1              1              1",
            ]
        )
        perf = rvs_rundeck.parse_node_performance(text, module_hint="level_config")
        self.assertEqual(perf["metrics"], [])
        heat = perf["heatmaps"][0]
        self.assertEqual(heat["unit"], "MB/s")
        self.assertEqual(heat["group"], "babel")
        self.assertEqual(heat["name"], "babel")
        self.assertEqual(heat["rows"], ["GPU2", "GPU5"])
        self.assertEqual(heat["cols"], ["Read", "Triad"])
        self.assertEqual(heat["row_label"], "GPU")
        self.assertEqual(heat["col_label"], "Kernel")
        self.assertAlmostEqual(heat["values"][0][0], 80.0)
        self.assertIsNone(heat["values"][0][1])
        self.assertAlmostEqual(heat["values"][1][0], 100.0)
        self.assertAlmostEqual(heat["values"][1][1], 50.0)

    def test_babel_action_context_and_per_module_hint(self):
        headed = "\n".join(
            [
                "Action name :hbm_full",
                "2987 Read 4011893.551 0.00020 0.00035 0.00028",
            ]
        )
        perf = rvs_rundeck.parse_node_performance(headed, module_hint="level_config")
        self.assertEqual(perf["heatmaps"][0]["cols"], ["Read"])
        self.assertAlmostEqual(perf["heatmaps"][0]["values"][0][0], 4011893.551)

        bare = "2987 Read 10.0 1 1 1"
        self.assertEqual(rvs_rundeck.parse_node_performance(bare, module_hint="gst_single")["heatmaps"], [])
        hinted = rvs_rundeck.parse_node_performance(bare, module_hint="babel_stream")
        self.assertEqual(hinted["heatmaps"][0]["group"], "babel")
        self.assertAlmostEqual(hinted["heatmaps"][0]["values"][0][0], 10.0)

    def test_babel_survives_clipped_module_token(self):
        text = "\n".join(
            [
                "Action name :hbm_full",
                "Module name :babe",
                "2987        Read        100.0    9    8    7",
            ]
        )
        perf = rvs_rundeck.parse_node_performance(text, module_hint="level_config")
        self.assertEqual(perf["heatmaps"][0]["group"], "babel")
        self.assertAlmostEqual(perf["heatmaps"][0]["values"][0][0], 100.0)

        gst = "\n".join(["Action name :hbm_full", "Module name :gst", "2987 Read 100.0 9 8 7"])
        self.assertEqual(rvs_rundeck.parse_node_performance(gst, module_hint="level_config")["heatmaps"], [])

    def test_mem_bandwidth_or_verdict_only(self):
        text = "\n".join(
            [
                "Module name :mem",
                "[memtest] mem Test 1 : PASS",
                "[memtest] mem Test 11: elapsedtime = 23682.550781 bandwidth = 2161.934570GB/s",
                "[memtest] mem Test 11: elapsedtime = 23617.277344 bandwidth = 2167.909912GB/s",
            ]
        )
        perf = rvs_rundeck.parse_node_performance(text, module_hint="mem_test")
        self.assertEqual(len(perf["metrics"]), 2)
        self.assertEqual([metric["name"] for metric in perf["metrics"]], ["bandwidth", "bandwidth"])
        self.assertAlmostEqual(perf["metrics"][0]["value"], 2161.934570)
        self.assertAlmostEqual(perf["metrics"][1]["value"], 2167.909912)
        self.assertEqual(perf["metrics"][0]["unit"], "GB/s")
        self.assertEqual(perf["metrics"][0]["group"], "mem")
        self.assertEqual(perf["series"], [])
        self.assertEqual(perf["heatmaps"], [])

        pass_only = "[memtest] mem Test 1 : PASS"
        perf = rvs_rundeck.parse_node_performance(pass_only, module_hint="mem_test")
        self.assertEqual(perf["metrics"], [])
        results = {}
        rvs_rundeck.record_outputs(results, "mem_test", {"node-b": pass_only}, [])
        node = results["groups"]["mem_test"]["nodes"]["node-b"]
        self.assertEqual(node["status"], "pass")
        self.assertNotIn("metrics", node)


class TestLevelOutputContext(unittest.TestCase):
    def test_level_run_keeps_one_verdict_and_labels_gpus_across_modules(self):
        # Compact LEVEL excerpt: device ids in Babel/GST/IET, indexes only on PEBB/PBQT.
        text = "\n".join(
            [
                "Action name :hbm_full",
                "Module name :babel",
                "2987        Read        100.0    9    8    7",
                "2987        Triad       40.0     9    8    7",
                "Action name :pcie_d2h_bandwidth",
                "Module name :pebb",
                "[pcie_d2h_bandwidth] pcie-bandwidth [ 1/16] [CPU:: 0] [GPU:: 5 - 2987 - 0000:15:00.0] "
                "h2d::false d2h::true 52.849 GBps duration: 0.081268 secs",
                "Action name :xgmi_d2d_bandwidth",
                "Module name :pbqt",
                "[xgmi_d2d_bandwidth] p2p-bandwidth[ 1/56] [GPU:: 5 - 2987 - 0000:15:00.0] "
                "[GPU:: 2 - 9091 - 0000:75:00.0] bidirectional: true 101.231 GBps duration: 0.254565 secs",
                "Action name :memtest",
                "Module name :mem",
                "[memtest] mem Test 1 : PASS",
                "[memtest] mem Test 11: elapsedtime = 23617.277344 bandwidth = 2167.909912GB/s",
                "Action name :compute-fp8-trig",
                "Module name :gst",
                "[compute-fp8-trig] [GPU:: 2987] GFLOPS 1400000",
                "[compute-fp8-trig] [GPU:: 2987] GFLOPS 1299650 Target GFLOPS: 983000 met: TRUE",
                "[compute-fp16-trig] [GPU:: 2987] GFLOPS 10 Target GFLOPS: 524000 met: FALSE",
                "Action name :power-stress",
                "Module name :iet",
                "[power-stress] [GPU:: 2987] Power(W) 148.000000",
                "[power-stress] [GPU:: 2987] Power(W) 992.000000",
                "[power-stress] [GPU:: 2987] pass: TRUE",
            ]
        )
        perf = rvs_rundeck.parse_node_performance(text, module_hint="level_config")
        by_name = {metric["name"]: metric for metric in perf["metrics"]}
        babel = next(item for item in perf["heatmaps"] if item["name"] == "babel")
        self.assertEqual(babel["group"], "babel")
        self.assertAlmostEqual(babel["values"][0][babel["cols"].index("Read")], 100.0)
        self.assertEqual(by_name["d2h GPU5"]["group"], "pebb")
        xgmi = next(item for item in perf["heatmaps"] if item["name"] == "xgmi")
        self.assertEqual(xgmi["group"], "pbqt")
        self.assertAlmostEqual(by_name["bandwidth"]["value"], 2167.909912)
        self.assertEqual(by_name["bandwidth"]["group"], "mem")
        self.assertEqual(by_name["fp8"]["group"], "gst")
        self.assertAlmostEqual(by_name["fp8"]["value"], 1299650.0)
        self.assertEqual(by_name["fp8"]["status"], "pass")
        self.assertEqual(by_name["fp16"]["status"], "fail")
        self.assertEqual(by_name["power GPU5"]["group"], "iet")
        self.assertAlmostEqual(by_name["power GPU5"]["value"], 992.0)
        self.assertEqual(perf["series"], [])
        babel = next(item for item in perf["heatmaps"] if item["name"] == "babel")
        self.assertEqual(babel["rows"], ["GPU5"])
        xgmi = next(item for item in perf["heatmaps"] if item["name"] == "xgmi")
        self.assertEqual(xgmi["rows"], ["GPU2", "GPU5"])
        self.assertAlmostEqual(xgmi["values"][1][0], 101.231)

        results = {}
        rvs_rundeck.record_outputs(
            results,
            "level_config",
            {"node-a": text, "node-b": "module completed"},
            [r"met:\s*FALSE"],
            meta=rvs_rundeck.make_meta({"cluster_name": "lab"}, "1.7.0", "rvs_cvs"),
        )
        self.assertEqual(list(results["groups"]), ["level_config"])
        self.assertEqual(results["_meta"]["version"], "1.7.0")
        failed = results["groups"]["level_config"]["nodes"]["node-a"]
        self.assertEqual(failed["status"], "fail")
        self.assertIn("fp8", {metric["name"] for metric in failed["metrics"]})
        quiet = results["groups"]["level_config"]["nodes"]["node-b"]
        self.assertEqual(quiet["status"], "pass")
        self.assertNotIn("metrics", quiet)

        results = {}
        rvs_rundeck.record_outputs(results, "level_config", {"node-a": text}, [])
        self.assertEqual(results["groups"]["level_config"]["nodes"]["node-a"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
