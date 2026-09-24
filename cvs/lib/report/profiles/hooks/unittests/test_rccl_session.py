'''Unit tests for RCCL report configuration and result publication.'''

import copy
import unittest
from unittest.mock import patch

from cvs.lib.report.profiles.hooks.rccl_session import publish_graph, variant_from_config
from cvs.lib.report.registry import ReportSessionStore


class TestRcclSession(unittest.TestCase):
    def test_collective_precedence_matches_suite_shapes(self):
        nested = {"rccl_test_params": {"rccl_collective": ["all_gather_perf"]}}
        cases = [
            ("rccl_perf", nested, ["all_gather_perf"]),
            ("rccl_regression", nested, ["all_reduce_perf"]),
            ("rccl_pairwise", nested, ["all_reduce_perf"]),
            (
                "rccl_regression",
                {**nested, "rccl_collective": ["broadcast_perf"]},
                ["broadcast_perf"],
            ),
            (None, {**nested, "rccl_collective": []}, []),
        ]
        for suite, config, expected in cases:
            with self.subTest(suite=suite, config=config):
                before = copy.deepcopy(config)
                variant = variant_from_config(config, {}, suite_name=suite)
                self.assertEqual(variant.rccl_test_params["rccl_collective"], expected)
                self.assertEqual(config, before)

    def test_threshold_flags_and_cluster_count(self):
        for flag in ("verify_bus_bw", "verify_bw_dip", "verify_lat_dip"):
            for value in (True, "True", "1", "yes", False, "False"):
                with self.subTest(flag=flag, value=value):
                    variant = variant_from_config({"cvs_params": {flag: value}}, {"node_dict": {"n0": {}, "n1": {}}})
                    self.assertEqual(variant.nnodes, 2)
                    self.assertEqual(variant.enforce_thresholds, value not in (False, "False"))

    def test_publish_preserves_bound_store_identity_and_legacy_graph_values(self):
        store = {}
        registry = ReportSessionStore()
        registry.bind_results(cvs_results_dict=store)
        store["stale"] = 1
        raw = {
            "all_reduce_perf": [
                {"name": "all_reduce_perf", "size": 1024, "inPlace": 0, "busBw": 1.5, "algBw": 1.2, "time": 10.0},
                {"name": "all_reduce_perf", "size": 1024, "inPlace": 1, "busBw": 2.5, "algBw": 2.2, "time": 9.0},
            ]
        }
        before = copy.deepcopy(raw)
        graph = publish_graph(raw, store)
        self.assertIs(registry.get_results()["cvs_results_dict"], store)
        self.assertEqual(store, graph)
        self.assertNotIn("stale", store)
        self.assertEqual(graph["all_reduce_perf"][1024], {"bus_bw": 2.5, "alg_bw": 2.2, "time": 9.0})
        self.assertEqual(raw, before)

    def test_optional_store_failure_does_not_break_legacy_reporting(self):
        with patch("cvs.lib.report.profiles.hooks.rccl_session.globals.log.warning") as warning:
            self.assertEqual(publish_graph({}, None), {})
        warning.assert_called_once()

    def test_empty_results_clear_stale_graph(self):
        store = {"stale": 1}
        self.assertEqual(publish_graph(None, store), {})
        self.assertEqual(store, {})


if __name__ == "__main__":
    unittest.main()
