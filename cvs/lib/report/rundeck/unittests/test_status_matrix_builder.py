'''Unit tests for the Run Deck ``status_matrix`` dataset builder.'''

import unittest

from cvs.lib.report.rundeck.dataset_builders.registry import build_datasets
from cvs.lib.report.rundeck.dataset_builders.status_matrix import build_status_matrix_datasets


def _sources(results):
    return {"results": results, "cvs_results_dict": results}


_RESULTS = {
    "_meta": {"cluster": "c1", "version": "1.4.9", "suite": "anc_test_cpu", "generated_at": "t0"},
    "groups": {
        "cpu_content_check": {
            "nodes": {
                "n1": {"status": "pass", "items": [{"name": "a", "status": "pass"}]},
                "n2": {"status": "pass", "items": []},
            }
        },
        "hbm_lvl3": {
            "nodes": {
                "n1": {"status": "pass", "items": []},
                "n2": {"status": "fail", "items": [{"name": "ecc", "status": "fail", "message": "bad"}]},
            }
        },
        "gpu_content_check": {
            "nodes": {
                "n1": {"status": "na", "items": []},
            }
        },
    },
}


class TestStatusMatrixBuilder(unittest.TestCase):
    def test_registered_under_status_matrix_id(self):
        out = build_datasets("status_matrix", _sources(_RESULTS), {})
        self.assertIn("grid", out)

    def test_nodes_discovered_in_first_seen_order(self):
        out = build_status_matrix_datasets(_sources(_RESULTS), {})
        self.assertEqual(out["nodes"], ["n1", "n2"])

    def test_groups_preserve_insertion_order(self):
        out = build_status_matrix_datasets(_sources(_RESULTS), {})
        self.assertEqual(out["groups"], ["cpu_content_check", "hbm_lvl3", "gpu_content_check"])

    def test_overall_status_fail_when_any_cell_fails(self):
        out = build_status_matrix_datasets(_sources(_RESULTS), {})
        self.assertEqual(out["overall_status"], "fail")

    def test_overall_status_pass_when_no_fail(self):
        results = {"groups": {"g": {"nodes": {"n1": {"status": "pass", "items": []}}}}}
        out = build_status_matrix_datasets(_sources(results), {})
        self.assertEqual(out["overall_status"], "pass")

    def test_overall_status_na_when_only_na(self):
        results = {"groups": {"g": {"nodes": {"n1": {"status": "na", "items": []}}}}}
        out = build_status_matrix_datasets(_sources(results), {})
        self.assertEqual(out["overall_status"], "na")

    def test_grid_status_per_node_group(self):
        out = build_status_matrix_datasets(_sources(_RESULTS), {})
        grid = out["grid"]
        self.assertEqual(grid["n2"]["hbm_lvl3"]["status"], "fail")
        self.assertEqual(grid["n1"]["cpu_content_check"]["status"], "pass")
        self.assertEqual(grid["n1"]["gpu_content_check"]["status"], "na")

    def test_missing_node_group_cell_defaults_na(self):
        out = build_status_matrix_datasets(_sources(_RESULTS), {})
        # n2 has no gpu_content_check entry -> na in the grid.
        self.assertEqual(out["grid"]["n2"]["gpu_content_check"]["status"], "na")

    def test_counts_and_run_card(self):
        out = build_status_matrix_datasets(_sources(_RESULTS), {})
        self.assertEqual(out["counts"], {"pass": 3, "fail": 1, "na": 2})
        labels = {r[0]: r[1] for r in out["run_card_display"]}
        self.assertEqual(labels["Cluster"], "c1")
        self.assertEqual(labels["ANC version"], "1.4.9")
        self.assertEqual(labels["Groups"], "3")

    def test_version_label_comes_from_meta(self):
        results = {
            "_meta": {
                "cluster": "c1",
                "version": "1.3.0",
                "version_label": "RVS version",
                "suite": "rvs_cvs",
            },
            "groups": {"mem": {"nodes": {"n1": {"status": "pass", "items": []}}}},
        }
        out = build_status_matrix_datasets(_sources(results), {})
        labels = {r[0]: r[1] for r in out["run_card_display"]}
        self.assertEqual(labels["RVS version"], "1.3.0")
        self.assertNotIn("ANC version", labels)

    def test_empty_results_yield_empty_grid(self):
        out = build_status_matrix_datasets(_sources({}), {})
        self.assertEqual(out["nodes"], [])
        self.assertEqual(out["grid"], {})
        self.assertEqual(out["overall_status"], "na")

    def test_unknown_status_coerced_to_na(self):
        results = {"groups": {"g": {"nodes": {"n1": {"status": "weird", "items": []}}}}}
        out = build_status_matrix_datasets(_sources(results), {})
        self.assertEqual(out["grid"]["n1"]["g"]["status"], "na")

    def test_verdict_only_cells_have_empty_performance_fields(self):
        out = build_status_matrix_datasets(_sources(_RESULTS), {})
        cell = out["grid"]["n1"]["cpu_content_check"]
        self.assertEqual(cell["metrics"], [])
        self.assertEqual(cell["series"], [])
        self.assertEqual(cell["heatmaps"], [])
        self.assertEqual(out["metric_charts"], {"metrics": [], "series": [], "heatmaps": []})

    def test_overview_pass_rate_excludes_na(self):
        out = build_status_matrix_datasets(_sources(_RESULTS), {})
        overview = out["overview"]
        self.assertEqual(overview["counts"], {"pass": 3, "fail": 1, "na": 2})
        self.assertEqual(overview["evaluated"], 4)
        self.assertEqual(overview["total"], 6)
        self.assertAlmostEqual(overview["pass_rate"], 0.75)
        self.assertEqual(overview["failures_by_node"][0]["node"], "n2")
        self.assertEqual(overview["failures_by_node"][0]["fail"], 1)
        self.assertEqual(overview["failures_by_group"][0]["group"], "hbm_lvl3")

    def test_empty_results_overview_and_charts(self):
        out = build_status_matrix_datasets(_sources({}), {})
        overview = out["overview"]
        self.assertIsNone(overview["pass_rate"])
        self.assertEqual(overview["evaluated"], 0)
        self.assertEqual(overview["total"], 0)
        self.assertEqual(overview["failures_by_node"], [])
        self.assertEqual(overview["failures_by_group"], [])
        self.assertEqual(out["metric_charts"]["metrics"], [])

    def test_metrics_series_and_heatmaps_roll_up(self):
        results = {
            "groups": {
                "a2a": {
                    "nodes": {
                        "n1": {
                            "status": "pass",
                            "metrics": [
                                {
                                    "name": "rtotal",
                                    "value": 412.5,
                                    "unit": "GB/s",
                                    "threshold": 400,
                                    "direction": "higher",
                                    "status": "pass",
                                },
                                {"name": "skipped", "value": "nope"},
                            ],
                            "series": [
                                {"name": "power", "unit": "W", "points": [["GPU0", 10], {"x": "GPU1", "y": 12}]}
                            ],
                            "heatmaps": [
                                {
                                    "name": "xgmi",
                                    "rows": ["GPU0", "GPU1"],
                                    "cols": ["GPU0", "GPU1"],
                                    "values": [[None, "48.2"], [47.1, None]],
                                    "threshold": 40,
                                    "direction": "higher",
                                    "unit": "GB/s",
                                }
                            ],
                        },
                        "n2": {
                            "status": "fail",
                            "metrics": [{"name": "rtotal", "value": "390", "status": "fail", "group": "a2a"}],
                        },
                    }
                }
            }
        }
        out = build_status_matrix_datasets(_sources(results), {})
        cell = out["grid"]["n1"]["a2a"]
        self.assertEqual(len(cell["metrics"]), 1)
        self.assertEqual(cell["metrics"][0]["group"], "a2a")
        self.assertEqual(cell["series"][0]["points"], [{"x": "GPU0", "y": 10.0}, {"x": "GPU1", "y": 12.0}])
        self.assertIsNone(cell["heatmaps"][0]["values"][0][0])
        self.assertEqual(cell["heatmaps"][0]["values"][0][1], 48.2)
        charts = out["metric_charts"]
        self.assertEqual(len(charts["metrics"]), 1)
        metric = charts["metrics"][0]
        self.assertEqual(metric["name"], "rtotal")
        self.assertEqual(metric["threshold"], 400.0)
        self.assertEqual(metric["direction"], "higher")
        self.assertEqual([point["node"] for point in metric["points"]], ["n1", "n2"])
        self.assertEqual(metric["points"][1]["value"], 390.0)
        self.assertEqual(metric["points"][1]["status"], "fail")
        self.assertEqual(charts["series"][0]["node"], "n1")
        self.assertNotIn("node", cell["series"][0])
        self.assertEqual(charts["heatmaps"][0]["node"], "n1")

    def test_malformed_performance_entries_are_dropped(self):
        results = {
            "groups": {
                "g": {
                    "nodes": {
                        "n1": {
                            "status": "pass",
                            "metrics": "nope",
                            "series": [{"name": "s", "points": []}, {"points": [[1, 2]]}],
                            "heatmaps": [{"name": "h", "rows": ["a"], "cols": ["b", "c"], "values": [[1]]}],
                        }
                    }
                }
            }
        }
        out = build_status_matrix_datasets(_sources(results), {})
        cell = out["grid"]["n1"]["g"]
        self.assertEqual(cell["metrics"], [])
        self.assertEqual(cell["series"], [])
        self.assertEqual(cell["heatmaps"], [])
        self.assertEqual(out["overall_status"], "pass")


class TestResultsTable(unittest.TestCase):
    def test_metric_item_and_summary_rows(self):
        results = {
            "groups": {
                "gst_single": {
                    "nodes": {
                        "node-a": {
                            "status": "fail",
                            "items_summary": "1 failure pattern(s)",
                            "items": [{"name": "scan", "status": "fail", "message": "gst action failed"}],
                            "metrics": [
                                {
                                    "name": "fp8",
                                    "value": 1299650,
                                    "unit": "GFLOPS",
                                    "threshold": 983000,
                                    "status": "pass",
                                }
                            ],
                            "errors_json_href": "errors.json",
                            "log_tarball_href": "logs.tar.gz",
                        }
                    }
                },
                "mem_test": {"nodes": {"node-a": {"status": "pass", "items_summary": "passed", "items": []}}},
            }
        }
        table = build_status_matrix_datasets(_sources(results), {})["results_table"]
        self.assertEqual(
            table["headers"],
            ["Node", "Check", "Result", "Value", "Unit", "Threshold", "Status", "Message"],
        )
        by_result = {row[2]: row for row in table["rows"]}
        self.assertEqual(by_result["fp8"][3:8], ["1299650", "GFLOPS", "983000", "pass", "1 failure pattern(s)"])
        self.assertEqual(by_result["scan"][6:8], ["fail", "gst action failed"])
        self.assertEqual(by_result["mem_test"][6:8], ["pass", "passed"])
        self.assertEqual([row[2] for row in table["rows"] if row[6] == "fail"], ["scan"])
        self.assertLess(table["rows"].index(by_result["scan"]), table["rows"].index(by_result["fp8"]))

    def test_missing_cell_is_one_summary_row(self):
        results = {"groups": {"hbm": {"nodes": {"n1": {"status": "pass", "items": []}}}}}
        table = build_status_matrix_datasets(_sources(results), {})["results_table"]
        missing = [row for row in table["rows"] if row[0] == "n1" and row[1] == "hbm" and row[6] == "na"]
        self.assertEqual(missing, [])
        only = table["rows"][0]
        self.assertEqual(only[2], "hbm")
        self.assertEqual(only[6], "pass")

    def test_requested_count_columns_come_from_the_items_summary(self):
        results = {
            "groups": {
                "hbm_lvl1": {
                    "nodes": {
                        "node-11": {
                            "status": "fail",
                            "items_summary": "Items: 6 Total | 5 PASSED, 1 FAILED",
                            "items": [{"name": "oblex_remix2", "status": "fail", "message": "PDC Failed"}],
                        }
                    }
                }
            }
        }
        profile = {"cards": [{"id": "results", "columns": ["Node", "Passed", "Failed", "Status"]}]}
        table = build_status_matrix_datasets(_sources(results), profile)["results_table"]
        self.assertEqual(table["headers"][-2:], ["Passed", "Failed"])
        row = table["rows"][0]
        self.assertEqual(row[2], "oblex_remix2")
        self.assertEqual(row[6:], ["fail", "PDC Failed", "5", "1"])

    def test_logs_column_carries_artifact_hrefs_only_when_requested(self):
        results = {
            "groups": {
                "hbm_lvl1": {
                    "nodes": {
                        "node-11": {
                            "status": "fail",
                            "items_summary": "Items: 6 Total | 5 PASSED, 1 FAILED",
                            "items": [{"name": "oblex_remix2", "status": "fail", "message": "PDC Failed"}],
                            "errors_json_href": "n1_errors.json",
                            "log_tarball_href": "n1_logs.tar.gz",
                        },
                        "node-12": {"status": "pass", "items_summary": "Items: 6 Total | 6 PASSED, 0 FAILED"},
                    }
                }
            }
        }
        without = build_status_matrix_datasets(_sources(results), {})["results_table"]
        self.assertNotIn("Logs", without["headers"])
        self.assertTrue(all(len(row) == 8 for row in without["rows"]))
        profile = {"cards": [{"id": "results", "columns": ["Node", "Message", "Logs"]}]}
        table = build_status_matrix_datasets(_sources(results), profile)["results_table"]
        self.assertEqual(table["headers"][-1], "Logs")
        failed = next(row for row in table["rows"] if row[2] == "oblex_remix2")
        self.assertEqual(
            failed[-1],
            [
                {"label": "errors.json", "href": "n1_errors.json"},
                {"label": "logs.tar.gz", "href": "n1_logs.tar.gz"},
            ],
        )
        passed = next(row for row in table["rows"] if row[0] == "node-12")
        self.assertEqual(passed[-1], "\u2014")

    def test_absent_group_on_a_node_is_not_evaluated(self):
        results = {
            "groups": {
                "a": {"nodes": {"n1": {"status": "pass", "items": []}, "n2": {"status": "pass", "items": []}}},
                "b": {"nodes": {"n1": {"status": "pass", "items": []}}},
            }
        }
        table = build_status_matrix_datasets(_sources(results), {})["results_table"]
        absent = next(row for row in table["rows"] if row[0] == "n2" and row[1] == "b")
        self.assertEqual(absent[6], "na")
        self.assertEqual(absent[7], "group not on node")


if __name__ == "__main__":
    unittest.main()
