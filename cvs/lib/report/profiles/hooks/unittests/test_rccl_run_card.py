'''Unit tests for the RCCL Run Deck run-card hook.'''

import unittest
from types import SimpleNamespace

from cvs.lib.report.profiles.hooks.rccl_run_card import rccl_run_card_display
from cvs.lib.report.profiles.hooks.rccl_session import variant_from_config


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

    def test_missing_metadata(self):
        rows = {label: value for label, value, _ in rccl_run_card_display(None, {})}
        for label in ("Nodes", "MPI nodes", "Local ranks", "MPI ranks", "Collectives"):
            self.assertEqual(rows[label], "\u2014")

    def test_run_card_uses_results_collected_after_fixture_resolution(self):
        raw = {}
        variant = variant_from_config(
            {"rccl_test_params": {"rccl_collective": ["all_gather_perf"]}},
            {},
            raw_results=raw,
        )
        raw["all_reduce_perf-NCCL_ALGO=Ring"] = [
            {"name": "all_reduce_perf", "size": 1024},
            {"name": "all_reduce_perf", "size": 2048},
        ]
        rows = {label: value for label, value, _ in rccl_run_card_display(variant, {})}
        self.assertEqual(rows["Collectives"], "all_reduce_perf")
        self.assertEqual(rows["Msg size (bytes)"], "1024 .. 2048")

    def test_empty_results_do_not_claim_configured_collectives_ran(self):
        variant = variant_from_config({"rccl_collective": ["all_reduce_perf"]}, {}, raw_results={"failed": []})
        rows = {label: value for label, value, _ in rccl_run_card_display(variant, {})}
        self.assertEqual(rows["Collectives"], "\u2014")

    def test_pairwise_card_tracks_actual_node_and_rank_counts(self):
        variant = variant_from_config(
            {"mpi_params": {"no_of_nodes": "99", "no_of_local_ranks": "8"}},
            {"node_dict": {"n0": {}, "n1": {}, "unused": {}}},
            run_nodes={"Phase0": ["n0"], "Phase1": ["n0", "n1"], "Phase2": ["n0", "n1"]},
        )
        rows = {label: value for label, value, _ in rccl_run_card_display(variant, {})}
        self.assertEqual(rows["Nodes"], "2")
        self.assertEqual(rows["MPI nodes"], "1, 2")
        self.assertEqual(rows["MPI ranks"], "8, 16")


if __name__ == "__main__":
    unittest.main()
