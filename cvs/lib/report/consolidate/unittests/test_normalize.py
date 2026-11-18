'''Tests for summarizing one Run Deck by dataset builder.'''

import unittest

from cvs.lib.report.consolidate.normalize import (
    MAX_ITEMS_IN_MESSAGE,
    UNIT_CHECKS,
    UNIT_CONFIGURATIONS,
    UNIT_ROWS,
    combined,
    metric,
    normalize,
    run_card_metadata,
    status_of,
    text,
    unavailable,
)
from cvs.lib.report.consolidate.unittests.deck_fixtures import deck, status_matrix


def _sweep_cell(label, tiers, host="10.0.0.1", metrics=None):
    return {"cell_id": label, "label": label, "host": host, "tiers": tiers, "metrics": metrics or []}


class TestStatusMatrix(unittest.TestCase):
    def test_counts_one_result_per_node_and_check_group(self):
        dataset = status_matrix({"n1": {"mem": "pass", "gst": "fail"}, "n2": {"mem": "na", "gst": "pass"}})
        result, detail = normalize(deck("rvs_cvs", "fail", dataset=dataset))
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["unit"], UNIT_CHECKS)
        self.assertEqual(result["total"], 4)
        self.assertEqual({k: v for k, v in result["counts"].items() if v}, {"pass": 2, "fail": 1, "na": 1})
        self.assertEqual(result["nodes"], ["n1", "n2"])
        self.assertEqual(result["failed_nodes"], ["n1"])
        self.assertTrue(result["host_rows"])
        self.assertEqual(result["scope"], "2 nodes")
        self.assertFalse(result["partial"])
        self.assertEqual(detail["matrix"]["grid"]["n1"], {"gst": "fail", "mem": "pass"})

    def test_failed_check_evidence_names_failed_items_and_metrics(self):
        dataset = status_matrix({"n1": {"gst": "fail"}})
        cell = dataset["grid"]["n1"]["gst"]
        cell["items"] = [
            {"name": "fp8", "status": "fail", "message": "fp8 below target"},
            {"name": "ok", "status": "pass"},
        ]
        cell["metrics"] = [{"name": "fp8", "value": 410.5, "threshold": 500, "direction": "higher", "status": "fail"}]
        _, detail = normalize(deck("rvs_cvs", "fail", dataset=dataset))
        self.assertEqual(
            detail["evidence"],
            [
                {
                    "subject": "n1",
                    "check": "gst",
                    "status": "fail",
                    "message": "fp8 below target",
                    "metrics": [
                        {"name": "fp8", "actual": "410.5", "target": "\u2265 500", "unit": "", "status": "fail"}
                    ],
                }
            ],
        )

    def test_long_failure_lists_are_shortened(self):
        dataset = status_matrix({"n1": {"agfhc": "fail"}})
        dataset["grid"]["n1"]["agfhc"]["items"] = [
            {"name": "t%d" % i, "status": "fail"} for i in range(MAX_ITEMS_IN_MESSAGE + 3)
        ]
        _, detail = normalize(deck("agfhc_cvs", "fail", dataset=dataset))
        self.assertEqual(detail["evidence"][0]["message"], "t0; t1; t2; t3; t4; and 3 more")

    def test_items_summary_when_no_item_failed(self):
        dataset = status_matrix({"n1": {"g": "fail"}})
        dataset["grid"]["n1"]["g"]["items_summary"] = "Items: 4 Total | 3 PASSED, 1 FAILED"
        _, detail = normalize(deck("anc_test_cpu", "fail", dataset=dataset))
        self.assertEqual(detail["evidence"][0]["message"], "Items: 4 Total | 3 PASSED, 1 FAILED")

    def test_test_case_rows_are_not_hosts(self):
        for suite_id in ("ib_perf_bw_test", "mori_benchmark_test"):
            with self.subTest(suite_id=suite_id):
                result, _ = normalize(deck(suite_id, dataset=status_matrix({"case": {"bw": "pass"}})))
                self.assertFalse(result["host_rows"])
                self.assertEqual(result["scope"], "1 test case")

    def test_empty_matrix_is_partial(self):
        result, _ = normalize(deck("rvs_cvs", "na", dataset=status_matrix({})))
        self.assertTrue(result["partial"])
        self.assertIn("Node/check details are unavailable.", result["notes"])
        self.assertIsNone(result["scope"])


class TestSweep(unittest.TestCase):
    def test_counts_configurations_from_all_cells(self):
        cells = [
            _sweep_cell("c1", {"throughput": "pass", "latency": "pass"}),
            _sweep_cell("c2", {"throughput": "fail", "latency": "pass"}),
            _sweep_cell("c3", {"throughput": "record"}),
        ]
        data = deck("vllm", "fail", "sweep", {"all_cells": cells, "cells": cells[:1]}, cells=cells[:1])
        result, detail = normalize(data)
        self.assertEqual(result["unit"], UNIT_CONFIGURATIONS)
        self.assertEqual(result["total"], 3)
        self.assertEqual({k: v for k, v in result["counts"].items() if v}, {"pass": 1, "fail": 1, "record": 1})
        self.assertEqual(result["failed_nodes"], ["10.0.0.1"])
        self.assertEqual(result["scope"], "10.0.0.1")
        self.assertEqual([c["label"] for c in detail["configurations"]], ["c1", "c2", "c3"])
        self.assertEqual(detail["evidence"][0]["subject"], "c2")
        self.assertEqual(detail["evidence"][0]["message"], "Failed gates: throughput")

    def test_metric_status_decides_when_there_are_no_tiers(self):
        metrics = [{"label": "tok/s", "actual": 90, "spec": {"kind": "min", "value": 100}, "status": "fail"}]
        result, detail = normalize(
            deck("megatron", "fail", "training_sweep", {"all_cells": [_sweep_cell("m", {}, metrics=metrics)]})
        )
        self.assertEqual(result["counts"]["fail"], 1)
        self.assertEqual(detail["evidence"][0]["message"], "One or more measured metrics failed.")
        self.assertEqual(detail["evidence"][0]["metrics"][0]["target"], "\u2265 100")

    def test_scope_names_up_to_two_hosts_and_counts_more(self):
        two = [_sweep_cell("a", {"t": "pass"}, host="cluster"), _sweep_cell("b", {"t": "pass"}, host="node-01")]
        result, _ = normalize(deck("megatron", "pass", "training_sweep", {"all_cells": two}))
        self.assertEqual(result["scope"], "cluster, node-01")
        three = [_sweep_cell(str(i), {"t": "pass"}, host="node-0%d" % i) for i in range(3)]
        result, _ = normalize(deck("vllm", "pass", "sweep", {"all_cells": three}))
        self.assertEqual(result["scope"], "3 hosts")

    def test_truncated_cells_without_all_cells_are_partial(self):
        data = deck("sglang", "pass", "sweep", {}, cells=[_sweep_cell("c1", {"throughput": "pass"})])
        data["summary"] = {"mode": "truncated", "total_cells": 4}
        result, _ = normalize(data)
        self.assertEqual(result["total"], 4)
        self.assertEqual(result["counts"]["unknown"], 3)
        self.assertTrue(result["partial"])
        self.assertIn("Only 1 of 4 configurations have details.", result["notes"])


class TestSeries(unittest.TestCase):
    def test_rows_are_recorded_without_a_verdict(self):
        table = {
            "headers": ["Collective", "Size", "Bus BW"],
            "rows": [["all_reduce", 1024, 101.25], ["all_gather", 2048, 99.0]],
        }
        result, detail = normalize(deck("rccl", "record", "series", {"results_table": table}))
        self.assertEqual(result["unit"], UNIT_ROWS)
        self.assertEqual((result["total"], result["counts"]["record"]), (2, 2))
        self.assertIsNone(result["scope"])
        self.assertEqual(detail["table"]["rows"][0], ["all_reduce", "1024", "101.25"])
        self.assertIn("Measurements are recorded without a pass/fail verdict.", result["notes"])

    def test_no_record_note_once_the_deck_reports_a_verdict(self):
        result, _ = normalize(deck("rccl", "fail", "series", {"results_table": {"headers": [], "rows": []}}))
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["notes"], [])

    def test_falls_back_to_the_top_level_results_table(self):
        data = deck("rccl", "record", "series", {}, results_table={"headers": ["h"], "rows": [["v"]]})
        _, detail = normalize(data)
        self.assertEqual(detail["table"], {"headers": ["h"], "rows": [["v"]]})


class TestPartialDecks(unittest.TestCase):
    def test_unsupported_dataset_keeps_the_verdict_and_is_partial(self):
        result, _ = normalize(deck("rccl", "pass", "matrix", {"rows": []}))
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["partial"])
        self.assertIn("Detailed results use an unsupported dataset format.", result["notes"])

    def test_unrecognized_overall_status(self):
        result, _ = normalize(deck("rvs_cvs", "maybe", dataset=status_matrix({"n1": {"g": "pass"}})))
        self.assertEqual(result["status"], "unknown")
        self.assertIn("The report has an unrecognized overall status.", result["notes"])

    def test_unavailable(self):
        result, detail = unavailable("Could not read it.")
        self.assertEqual(
            (result["status"], result["partial"], result["notes"]), ("unknown", True, ["Could not read it."])
        )
        self.assertEqual(detail, {"metadata": [], "evidence": []})


class TestRunCard(unittest.TestCase):
    def test_cluster_and_metadata_skip_link_rows(self):
        rows = [["Cluster", "helios", False], ["Pytest report", "../rvs_cvs.html", True], ["Nodes", 8, False]]
        data = deck("rvs_cvs", dataset=status_matrix({"n1": {"g": "pass"}}), run_card_display=rows)
        result, detail = normalize(data)
        self.assertEqual(result["cluster"], "helios")
        self.assertEqual(detail["metadata"], [["Cluster", "helios"], ["Nodes", "8"]])

    def test_placeholder_cluster_is_ignored(self):
        data = deck(
            "rvs_cvs", dataset=status_matrix({"n1": {"g": "pass"}}), run_card_display=[["Cluster", "\u2014", False]]
        )
        self.assertIsNone(normalize(data)[0]["cluster"])
        self.assertEqual(run_card_metadata({"run_card_display": "bad"}), [])


class TestValues(unittest.TestCase):
    def test_text(self):
        self.assertEqual(text(2.0), "2")
        self.assertEqual(text(1.23456789), "1.2346")
        self.assertEqual(text(True), "true")
        self.assertEqual(text(7), "7")
        self.assertEqual(text(None), "\u2014")
        self.assertEqual(text({"a": 1}), "\u2014")

    def test_status_of(self):
        self.assertEqual(status_of("n/a"), "na")
        self.assertEqual(status_of("record"), "record")
        self.assertEqual(status_of("PASS"), "unknown")
        self.assertEqual(status_of(None), "unknown")

    def test_combined(self):
        self.assertEqual(combined(["pass", "fail", "unknown"]), "fail")
        self.assertEqual(combined(["pass", "unknown"]), "unknown")
        self.assertEqual(combined([]), "unknown")
        self.assertEqual(combined(["record", "pass"]), "pass")
        self.assertEqual(combined(["record", "na"]), "na")
        self.assertEqual(combined(["record"]), "record")

    def test_metric_targets(self):
        self.assertEqual(metric({"name": "lat", "spec": {"kind": "max_ms", "value": 50}})["target"], "\u2264 50")
        self.assertEqual(metric({"name": "bw", "threshold": 40.0, "direction": "lower"})["target"], "\u2264 40")
        self.assertEqual(metric({"name": "bw", "threshold": 40})["target"], "40")
        self.assertEqual(metric({"name": "bw", "value": 41})["target"], "Not gated")
        self.assertEqual(metric({"metric": "m", "actual": None})["actual"], "\u2014")


if __name__ == "__main__":
    unittest.main()
