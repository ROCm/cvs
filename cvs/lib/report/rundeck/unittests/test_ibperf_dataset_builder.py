'''Unit tests for the ibperf Run Deck dataset builder.'''

import unittest

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.dataset_builders.registry import build_datasets
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html


def _ibperf_results():
    return {
        "ib_write_bw": {
            2: {
                "8": {
                    "node-a": {
                        0: {"bw": "100.0", "pps": "1.0"},
                        1: {"bw": "110.0", "pps": "1.1"},
                    },
                    "node-b": {
                        0: {"bw": "90.0", "pps": "0.9"},
                        1: {"bw": "95.0", "pps": "0.95"},
                    },
                },
                "16": {
                    "node-a": {
                        0: {"bw": "120.0", "pps": "1.2"},
                        1: {"bw": "125.0", "pps": "1.25"},
                    }
                },
            },
            4: {
                "8": {
                    "node-a": {
                        0: {"bw": "140.0", "pps": "1.4"},
                        1: {"bw": "150.0", "pps": "1.5"},
                    }
                }
            },
        }
    }


class TestIbperfDatasetBuilder(unittest.TestCase):
    def test_builds_qp_series_and_per_node_spread(self):
        datasets = build_datasets("ibperf", {"results": _ibperf_results()}, {})

        series = datasets["charts"]["bus_bw"]["ib_write_bw"]
        self.assertEqual([entry["label"] for entry in series], ["ib_write_bw · QP 8", "ib_write_bw · QP 16"])
        self.assertEqual(series[0]["points"], [(2, 98.75), (4, 145.0)])

        rows = datasets["results_table"]["rows"]
        node_a = next(row for row in rows if row[:4] == ["ib_write_bw", "8", 2, "node-a"])
        self.assertEqual(node_a[4:], [2, 100.0, 105.0, 110.0, 0])

    def test_profile_renders_run_card_charts_and_table(self):
        profile = load_json_profile("ib_perf_bw_test")
        payload = build_rundeck_payload(
            profile=profile,
            store={
                "cvs_results_dict": _ibperf_results(),
                "variant_config": {
                    "gid_index": "3",
                    "duration": "30",
                    "msg_size_list": [2, 4],
                    "qp_count_list": ["8", "16"],
                },
            },
            cvs_version="1.0.0",
        )

        document = render_rundeck_html(payload)
        self.assertIn("IB Performance Bandwidth Run Deck", document)
        self.assertIn("ib_write_bw · QP 8", document)
        self.assertIn("Per-node bandwidth spread", document)
        self.assertIn("Message sizes", document)
        self.assertIn("2, 4", document)


if __name__ == "__main__":
    unittest.main()
