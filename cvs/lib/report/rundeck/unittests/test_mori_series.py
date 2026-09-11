'''Unit tests for the Mori series Run Deck.'''

import unittest

from cvs.lib.mori_reporting import record_ibgda_results, record_io_results
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.dataset_builders.registry import build_datasets
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html


def _mori_results():
    results = {}
    record_io_results(
        results,
        {
            "operation": "io_read",
            "buffer_size": 16384,
            "transfer_batch_size": 128,
            "qp_count": 1,
            "ranks": {
                0: {
                    "rows": [
                        {
                            "MsgSize_B": 524288,
                            "BatchSize": 128,
                            "TotalSize_MB": 64,
                            "Max_BW_GBps": 50.0,
                            "Avg_BW_GBps": 47.5,
                            "Min_Lat_us": 1000.0,
                            "Avg_Lat_us": 1200.0,
                        },
                        {
                            "MsgSize_B": 1048576,
                            "BatchSize": 128,
                            "TotalSize_MB": 128,
                            "Max_BW_GBps": 52.0,
                            "Avg_BW_GBps": 49.5,
                            "Min_Lat_us": 1800.0,
                            "Avg_Lat_us": 2100.0,
                        },
                    ]
                }
            },
        },
        case_qp_count=1,
    )
    record_ibgda_results(
        results,
        {
            "operation": "ibgda_write",
            "processes": 2,
            "ctas": 2,
            "threads": 256,
            "qp_count": 4,
            "iterations": 1,
            "nodes": {
                "node-a": {
                    "rows": [
                        {
                            "size_bytes": 33554432,
                            "bandwidth_gb": 46.5,
                            "time_ms": 390.0,
                            "rate_mpps": 1.25,
                        },
                        {
                            "size_bytes": 67108864,
                            "bandwidth_gb": 47.0,
                            "time_ms": 780.0,
                            "rate_mpps": 1.5,
                        },
                    ]
                }
            },
        },
    )
    return results


class TestMoriSeries(unittest.TestCase):
    def test_builds_all_metric_series_and_full_table(self):
        profile = load_json_profile("mori_benchmark_test")

        dataset = build_datasets("series", {"results": _mori_results()}, profile)

        self.assertEqual(
            set(dataset["charts"]),
            {"Avg_BW_GBps", "Avg_Lat_us", "bandwidth_gb", "time_ms", "rate_mpps"},
        )
        self.assertEqual(len(dataset["charts"]["Avg_BW_GBps"]), 1)
        self.assertEqual(len(dataset["charts"]["bandwidth_gb"]), 1)
        self.assertEqual(len(dataset["results_table"]["rows"]), 4)
        self.assertIn("Avg latency (us)", dataset["results_table"]["headers"])
        self.assertIn("IBGDA rate (Mpps)", dataset["results_table"]["headers"])

    def test_renders_run_card_graphs_and_table_without_viewer(self):
        profile = load_json_profile("mori_benchmark_test")
        payload = build_rundeck_payload(
            profile=profile,
            store={
                "cvs_results_dict": _mori_results(),
                "variant_config": {
                    "gpu_type": "MI355X",
                    "node_count": 2,
                    "nic_type": "thor2",
                    "mori_device_list": "rdma0,rdma1",
                    "container_image": "example/mori:latest",
                },
            },
            cvs_version="1.0.0",
        )

        document = render_rundeck_html(payload)
        self.assertIn("Mori Benchmark Run Deck", document)
        self.assertIn("Torch IO average bandwidth by message size", document)
        self.assertIn("IBGDA packet rate by message size", document)
        self.assertIn("Full Mori benchmark results", document)
        self.assertIn("MI355X", document)
        self.assertNotIn("viewer_config", payload)
        self.assertNotIn("Viewer</a>", document)


if __name__ == "__main__":
    unittest.main()
