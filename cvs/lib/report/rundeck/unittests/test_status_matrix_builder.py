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


if __name__ == "__main__":
    unittest.main()
