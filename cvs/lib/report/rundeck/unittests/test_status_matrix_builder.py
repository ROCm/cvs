'''Unit tests for the Run Deck ``status_matrix`` dataset builder.'''

import unittest

from cvs.lib.report.rundeck.dataset_builders.registry import build_datasets
from cvs.lib.report.rundeck.dataset_builders.status_matrix import build_status_matrix_datasets


def _sources(results):
    return {"results": results, "cvs_results_dict": results}


_RESULTS = {
    "_meta": {"cluster": "c1", "version": "1.4.9", "suite": "anc_test_cpu", "generated_at": "t0"},
    "groups": {
        "cpu_sanity": {
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
        self.assertEqual(out["groups"], ["cpu_sanity", "hbm_lvl3", "gpu_content_check"])

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
        self.assertEqual(grid["n1"]["cpu_sanity"]["status"], "pass")
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
        self.assertEqual(labels["Groups"], "3")

    def test_empty_results_yield_empty_grid(self):
        out = build_status_matrix_datasets(_sources({}), {})
        self.assertEqual(out["nodes"], [])
        self.assertEqual(out["grid"], {})
        self.assertEqual(out["overall_status"], "na")

    def test_unknown_status_coerced_to_na(self):
        results = {"groups": {"g": {"nodes": {"n1": {"status": "weird", "items": []}}}}}
        out = build_status_matrix_datasets(_sources(results), {})
        self.assertEqual(out["grid"]["n1"]["g"]["status"], "na")


if __name__ == "__main__":
    unittest.main()
