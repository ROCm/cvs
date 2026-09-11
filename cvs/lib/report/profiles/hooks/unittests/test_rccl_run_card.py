'''Unit tests for the RCCL Run Deck run-card hook.'''

import unittest
from types import SimpleNamespace

from cvs.lib.report.profiles.hooks.rccl_run_card import rccl_run_card_display
from cvs.lib.report.profiles.hooks.rccl_session import publish_graph, variant_from_config


class TestRcclRunCard(unittest.TestCase):
    def test_rows_include_mpi_and_collectives(self):
        variant = SimpleNamespace(
            framework="rccl",
            nnodes=2,
            mpi_params={"no_of_nodes": "2", "no_of_local_ranks": "8"},
            rccl_test_params={
                "rccl_collective": ["all_reduce_perf", "all_gather_perf"],
                "start_msg_size": "1024",
                "end_msg_size": "16g",
            },
            cvs_params={"nic_model": "thor", "verify_bus_bw": "False"},
            enforce_thresholds=False,
        )
        rows = rccl_run_card_display(variant, {"pytest_html_path": "/tmp/report.html"})
        labels = [label for label, _value, _link in rows]
        self.assertIn("MPI nodes", labels)
        self.assertIn("Collectives", labels)
        self.assertIn("NIC model", labels)
        self.assertIn("Pytest report", labels)
        collectives = next(value for label, value, _link in rows if label == "Collectives")
        self.assertIn("all_reduce_perf", collectives)

    def test_variant_and_publish_graph(self):
        variant = variant_from_config(
            {
                "mpi_params": {"no_of_nodes": "2"},
                "rccl_test_params": {"start_msg_size": "1024"},
                "cvs_params": {"verify_bus_bw": "False", "verify_bw_dip": "True", "nic_model": "thor"},
            },
            {"node_dict": {"n0": {}, "n1": {}}},
        )
        self.assertEqual(variant.nnodes, 2)
        self.assertTrue(variant.enforce_thresholds)

        store = {"stale": 1}
        raw = {
            "all_reduce_perf": [
                {
                    "name": "all_reduce_perf",
                    "size": 8,
                    "inPlace": 0,
                    "busBw": 1.5,
                    "algBw": 1.2,
                    "time": 10.0,
                }
            ]
        }
        graph = publish_graph(raw, store)
        self.assertNotIn("stale", store)
        self.assertEqual(graph["all_reduce_perf"][8]["bus_bw"], 1.5)
        self.assertEqual(store["all_reduce_perf"][8]["bus_bw"], 1.5)


if __name__ == "__main__":
    unittest.main()
