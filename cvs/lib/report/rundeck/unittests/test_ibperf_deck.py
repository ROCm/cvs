'''Unit tests for the ibperf Run Deck profile on the series builder.'''

import unittest

from cvs.lib.ibperf_reporting import record_bw_results, record_lat_results
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.dataset_builders.registry import build_datasets
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html


def _ibperf_results():
    ib_bw_dict = {
        "ib_write_bw": {
            8192: {
                "8": {"node-a": {0: {"bw": "180.0", "pps": "2.7"}}, "node-b": {0: {"bw": "170.0", "pps": "2.6"}}},
                "16": {"node-a": {0: {"bw": "200.0", "pps": "3.0"}}, "node-b": {0: {"bw": "196.0", "pps": "2.9"}}},
            },
            65536: {
                "8": {"node-a": {0: {"bw": "380.0", "pps": "0.7"}}, "node-b": {0: {"bw": "360.0", "pps": "0.7"}}},
            },
        }
    }
    ib_lat_dict = {
        "ib_write_lat": {
            2: {"node-a": {0: {"t_avg": "2.1", "t_99_pct": "2.6", "t_max": "8.0"}}},
            65536: {"node-a": {0: {"t_avg": "9.4", "t_99_pct": "10.2", "t_max": "15.0"}}},
        }
    }
    results = {}
    for bw_test, sizes in ib_bw_dict.items():
        for msg_size, qps in sizes.items():
            for qp_count, per_node in qps.items():
                record_bw_results(results, bw_test, msg_size, qp_count, per_node)
    for lat_test, sizes in ib_lat_dict.items():
        for msg_size, per_node in sizes.items():
            record_lat_results(results, lat_test, msg_size, per_node)
    return results


class TestIbperfDeck(unittest.TestCase):
    def test_bw_and_latency_charts_share_one_results_store(self):
        dataset = build_datasets("series", {"results": _ibperf_results()}, load_json_profile("ib_perf_bw_test"))

        charts = dataset["charts"]
        self.assertEqual(
            charts["bw_min"]["ib_write_bw · QP 8"][0]["points"],
            [("8K", 170.0), ("64K", 360.0)],
        )
        self.assertEqual(set(charts["bw_mean"]), {"ib_write_bw · QP 8", "ib_write_bw · QP 16"})
        self.assertEqual(set(charts["t_99_pct"]), {"ib_write_lat"})
        self.assertEqual(charts["t_avg"]["ib_write_lat"][0]["points"], [("2", 2.1), ("64K", 9.4)])

        table = dataset["results_table"]
        self.assertEqual(len(table["rows"]), 5)
        row = dict(zip(table["headers"], table["rows"][0]))
        self.assertEqual(row["Series"], "ib_write_bw · QP 16")
        self.assertEqual(row["Min BW (Gb/s)"], 196.0)
        self.assertEqual(row["Avg latency (us)"], "—")

    def test_renders_run_card_charts_and_table(self):
        payload = build_rundeck_payload(
            profile=load_json_profile("ib_perf_bw_test"),
            store={
                "cvs_results_dict": _ibperf_results(),
                "variant_config": {"node_count": 2, "orchestrator": "baremetal"},
            },
            cvs_version="1.0.0",
        )

        document = render_rundeck_html(payload)
        for text in (
            "ibperf Run Deck",
            "Average bandwidth by message size",
            "Slowest NIC instance bandwidth by message size",
            "Worst p99 latency by message size",
            "Full ibperf results",
            "baremetal",
            "does not affect gates",
        ):
            self.assertIn(text, document)


if __name__ == "__main__":
    unittest.main()
