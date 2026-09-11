'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for RVS stdout parsing in cvs/lib/health/rvs_parsing.py.
'''

import unittest

from cvs.lib.health.rvs_parsing import append_rvs_records, parse_rvs_output


class TestParseRvsOutput(unittest.TestCase):
    def test_gst_final_line(self):
        line = "[gst-Tflops-8K-trig-fp64] [GPU:: 42583] GFLOPS 6478.066983 Target GFLOPS: 5000.000000 met: TRUE"
        records = parse_rvs_output(line, "node1", module="gst_single")
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["node"], "node1")
        self.assertEqual(rec["gpu"], "42583")
        self.assertEqual(rec["module"], "gst")
        self.assertEqual(rec["action"], "gst-Tflops-8K-trig-fp64")
        self.assertEqual(rec["metric"], "gflops")
        self.assertAlmostEqual(rec["value"], 6478.066983)
        self.assertEqual(rec["unit"], "GFLOPS")
        self.assertAlmostEqual(rec["target"], 5000.0)
        self.assertTrue(rec["passed"])

    def test_gst_interval_line_without_target_ignored(self):
        interval = "[gst-Tflops-8K-trig-fp64] [GPU:: 42583] GFLOPS 6100.123456"
        final = "[gst-Tflops-8K-trig-fp64] [GPU:: 42583] GFLOPS 6478.066983 Target GFLOPS: 5000.000000 met: TRUE"
        records = parse_rvs_output("\n".join([interval, final]), "node1", module="gst")
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0]["value"], 6478.066983)

    def test_pebb_duration_line(self):
        line = (
            "[pcie_h2d_bandwidth] pcie-bandwidth [ 1/16] [CPU:: 0] "
            "[GPU:: 2 - 42583 - 0000:05:00.0] h2d::true d2h::false "
            "57.678 GBps duration: 0.223392 secs"
        )
        records = parse_rvs_output(line, "node1", module="pebb")
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["gpu"], "42583")
        self.assertEqual(rec["metric"], "pcie_gbps")
        self.assertAlmostEqual(rec["value"], 57.678)
        self.assertEqual(rec["unit"], "GB/s")
        self.assertEqual(rec["h2d"], "true")
        self.assertEqual(rec["d2h"], "false")

    def test_pbqt_duration_line(self):
        line = (
            "[xgmi_d2d_unidir_bandwidth] p2p-bandwidth[48/56] "
            "[GPU:: 8 - 11806 - 0000:e5:00.0] [GPU:: 7 - 51771 - 0000:95:00.0] "
            "bidirectional: false 61.370 GBps duration: 0.227452 secs"
        )
        records = parse_rvs_output(line, "node1", module="pbqt")
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["gpu"], "11806")
        self.assertEqual(rec["metric"], "p2p_gbps")
        self.assertAlmostEqual(rec["value"], 61.370)
        self.assertEqual(rec["dst"], "51771")
        self.assertEqual(rec["bidirectional"], "false")

    def test_iet_power_peak_and_pass(self):
        text = "\n".join(
            [
                "[iet_stress] [GPU:: 42583] Power(W) 210.5",
                "[iet_stress] [GPU:: 42583] Power(W) 245.8",
                "[iet_stress] [GPU:: 42583] Power(W) 230.1",
                "[iet_stress] [GPU:: 42583] pass: TRUE",
            ]
        )
        records = parse_rvs_output(text, "node1", module="iet")
        power = [r for r in records if r["metric"] == "power_w"]
        status = [r for r in records if r["metric"] == "status"]
        self.assertEqual(len(power), 1)
        self.assertAlmostEqual(power[0]["value"], 245.8)
        self.assertEqual(len(status), 1)
        self.assertTrue(status[0]["passed"])

    def test_babel_triad_row_uses_avg_column(self):
        line = "42583 Triad 6031416.066 6031416.066 5879289.313 5950804.403"
        records = parse_rvs_output(line, "node1", module="babel")
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["gpu"], "42583")
        self.assertEqual(rec["action"], "Triad")
        self.assertEqual(rec["metric"], "babel_mbytes_s")
        self.assertAlmostEqual(rec["value"], 5950804.403)
        self.assertEqual(rec["unit"], "MB/s")

    def test_gpu_enumeration_no_supported_gpus(self):
        records = parse_rvs_output("No supported GPUs available", "node1", module="gpu_enumeration")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["module"], "gpup")
        self.assertEqual(records[0]["metric"], "status")
        self.assertFalse(records[0]["passed"])

    def test_iet_power_keeps_iet_module_after_later_module_line(self):
        text = "\n".join(
            [
                "Module name :iet",
                "[iet-stress-1400W-true] [GPU:: 42583] Power(W) 241.0",
                "Module name :gst",
                "[gst-Tflops-8K-trig-fp64] [GPU:: 42583] GFLOPS 100.0 Target GFLOPS: 50.0 met: TRUE",
            ]
        )
        records = parse_rvs_output(text, "node1", module="level_config")
        power = [r for r in records if r["metric"] == "power_w"]
        self.assertEqual(len(power), 1)
        self.assertEqual(power[0]["module"], "iet")
        self.assertAlmostEqual(power[0]["value"], 241.0)


class TestAppendRvsRecords(unittest.TestCase):
    def test_append_empty_output_adds_status_row(self):
        store = {}
        recs = append_rvs_records(store, "", "node1", module="gst_single")
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["metric"], "status")
        self.assertIsNone(recs[0]["passed"])
        self.assertEqual(store["records"], recs)

    def test_append_failed_sets_passed_false_on_unset_records(self):
        line = (
            "[pcie_h2d_bandwidth] pcie-bandwidth [ 1/16] [CPU:: 0] "
            "[GPU:: 2 - 42583 - 0000:05:00.0] h2d::true d2h::false "
            "57.678 GBps duration: 0.223392 secs"
        )
        store = {}
        recs = append_rvs_records(store, line, "node1", module="pebb", failed=True)
        self.assertEqual(len(recs), 1)
        self.assertFalse(recs[0]["passed"])

    def test_append_success_sets_passed_true_on_unset_records(self):
        line = (
            "[pcie_h2d_bandwidth] pcie-bandwidth [ 1/16] [CPU:: 0] "
            "[GPU:: 2 - 42583 - 0000:05:00.0] h2d::true d2h::false "
            "57.678 GBps duration: 0.223392 secs"
        )
        store = {}
        recs = append_rvs_records(store, line, "node1", module="pebb", failed=False)
        self.assertEqual(len(recs), 1)
        self.assertTrue(recs[0]["passed"])


if __name__ == "__main__":
    unittest.main()
