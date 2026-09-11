"""Unit tests for the Run Deck series dataset builder."""

import unittest

from cvs.lib.report.rundeck.dataset_builders.series import build_series_datasets


class TestSeriesDatasetBuilder(unittest.TestCase):
    def test_build_record_layout_charts_and_table(self):
        records = [
            {
                "workload": "FLUX",
                "label": "node-a",
                "resolution_mp": 1.0,
                "average_time_s": 2.0,
                "sample_times_s": [2.1, 1.9],
                "status": "PASS",
            },
            {
                "workload": "FLUX",
                "label": "node-b",
                "resolution_mp": 2.0,
                "average_time_s": 3.0,
                "sample_times_s": [3.0],
                "status": "FAIL",
            },
        ]
        profile = {
            "series": {
                "record_layout": True,
                "charts": [
                    {
                        "id": "latency",
                        "x_field": "resolution_mp",
                        "y_field": "average_time_s",
                        "label_field": "workload",
                    },
                    {
                        "id": "samples",
                        "points_field": "sample_times_s",
                        "label_field": "label",
                        "x_start": 1,
                    },
                ],
                "table_columns": [["Run", "label"], ["Status", "status"]],
            }
        }

        dataset = build_series_datasets({"results": records}, profile)

        self.assertEqual(
            dataset["charts"]["latency"]["FLUX"][0]["points"],
            [(1.0, 2.0), (2.0, 3.0)],
        )
        self.assertEqual(
            dataset["charts"]["samples"]["node-a"][0]["points"],
            [(1, 2.1), (2, 1.9)],
        )
        self.assertEqual(dataset["results_table"]["headers"], ["Run", "Status"])
        self.assertEqual(dataset["results_table"]["rows"], [["node-a", "PASS"], ["node-b", "FAIL"]])

    def test_legacy_collective_layout_remains_supported(self):
        dataset = build_series_datasets(
            {"results": {"all_reduce": {"1024": {"bus_bw": 10, "alg_bw": 12, "time": 3}}}},
            {"series": {"y_fields": ["bus_bw"]}},
        )

        self.assertEqual(dataset["charts"]["bus_bw"]["all_reduce"][0]["points"], [(1024, 10.0)])
        self.assertEqual(dataset["results_table"]["rows"], [["all_reduce", "1024", 10, 12, 3]])


if __name__ == "__main__":
    unittest.main()
