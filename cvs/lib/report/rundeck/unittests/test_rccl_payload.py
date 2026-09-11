'''Unit tests for RCCL series Run Deck payload and HTML.'''

import unittest
from types import SimpleNamespace

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.dataset_builders import series  # noqa: F401
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html


def _graph():
    return {
        "all_reduce_perf": {
            1024: {"bus_bw": 12.5, "alg_bw": 11.0, "time": 100.0},
            2048: {"bus_bw": 40.0, "alg_bw": 36.0, "time": 180.0},
        },
        "all_gather_perf": {
            1024: {"bus_bw": 10.0, "alg_bw": 9.0, "time": 120.0},
        },
    }


class TestRcclRundeckPayload(unittest.TestCase):
    def test_payload_and_html_include_series_panels(self):
        profile = load_json_profile("rccl")
        self.assertIsNotNone(profile)
        store = {
            "cvs_results_dict": _graph(),
            "variant_config": SimpleNamespace(
                framework="rccl",
                nnodes=2,
                mpi_params={"no_of_nodes": "2", "no_of_local_ranks": "8"},
                rccl_test_params={
                    "rccl_collective": ["all_reduce_perf"],
                    "start_msg_size": "1024",
                    "end_msg_size": "16g",
                },
                cvs_params={"nic_model": "thor"},
                enforce_thresholds=False,
            ),
        }
        payload = build_rundeck_payload(profile=profile, store=store, cvs_version="1.0.0")
        series_ds = payload["datasets"]["series"]
        self.assertIn("bus_bw", series_ds["charts"])
        self.assertTrue(series_ds["results_table"]["rows"])
        self.assertEqual(payload["results_table"]["headers"][0], "Collective")
        labels = [row[0] for row in payload["run_card_display"]]
        self.assertIn("MPI nodes", labels)
        self.assertIn("Collectives", labels)

        doc = render_rundeck_html(payload)
        self.assertIn("RCCL Run Deck", doc)
        self.assertIn("Bus bandwidth vs message size", doc)
        self.assertIn("Algorithm bandwidth vs message size", doc)
        self.assertIn("Time vs message size", doc)
        self.assertIn("all_reduce_perf", doc)
        self.assertIn("Full results", doc)
        self.assertIn("1K", doc)
        self.assertNotIn("C=1024", doc)
        self.assertIn("<title>RCCL Run Deck &mdash; rccl</title>", doc)


if __name__ == "__main__":
    unittest.main()
