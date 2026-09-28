'''Unit tests for RCCL series datasets.'''

import unittest

from cvs.lib.report.rundeck.dataset_builders.series import _format_msg_size, build_series_datasets


class TestSeries(unittest.TestCase):
    def test_message_size_labels(self):
        for size, expected in (
            (8, "8"),
            (1024, "1K"),
            (1048576, "1M"),
            (1 << 34, "16G"),
            (1025, "1025"),
            ("custom", "custom"),
        ):
            with self.subTest(size=size):
                self.assertEqual(_format_msg_size(size), expected)

    def test_points_stay_in_numeric_size_order(self):
        graph = {
            "all_reduce_perf": {
                "1048576": {"bus_bw": 3},
                "2048": {"bus_bw": 2},
                "1024": {"bus_bw": 1},
            }
        }
        dataset = build_series_datasets({"results": graph}, {"series": {"y_fields": "bus_bw"}})
        self.assertEqual(dataset["charts"]["bus_bw"]["all_reduce_perf"][0]["points"], [("1K", 1), ("2K", 2), ("1M", 3)])
        self.assertEqual([row[1] for row in dataset["results_table"]["rows"]], ["1024", "2048", "1048576"])

    def test_chart_labels_humanized_table_stays_numeric(self):
        # convert_to_graph_dict keys results by int size, not str — cover that shape
        # explicitly so the chart/table split isn't only exercised via string keys.
        graph = {"all_reduce_perf": {1024: {"bus_bw": 1.0}, 1536: {"bus_bw": 1.5}, 1048576: {"bus_bw": 3.0}}}
        dataset = build_series_datasets({"results": graph}, {"series": {"y_fields": "bus_bw"}})
        points = dataset["charts"]["bus_bw"]["all_reduce_perf"][0]["points"]
        self.assertEqual([p[0] for p in points], ["1K", "1536", "1M"])
        self.assertTrue(all(isinstance(p[0], str) for p in points))
        rows = dataset["results_table"]["rows"]
        self.assertEqual([row[1] for row in rows], [1024, 1536, 1048576])
        self.assertTrue(all(isinstance(row[1], int) for row in rows))


if __name__ == "__main__":
    unittest.main()
