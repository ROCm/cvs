'''Unit tests for the Run Deck series dataset builder.'''

import unittest

from cvs.lib.report.rundeck.dataset_builders.series import build_series_datasets


class TestSeriesDatasetBuilder(unittest.TestCase):
    def test_configured_metrics_and_table_columns(self):
        results = {
            "TraceLens per-rank trace": {
                "0": {"total_time_ms": 12.5, "compute_ratio_pct": 60.0, "parser": "TraceLensParser"},
                "1": {"total_time_ms": 13.0, "compute_ratio_pct": 58.0, "parser": "TraceLensParser"},
            }
        }
        profile = {
            "series": {
                "y_fields": ["total_time_ms", "compute_ratio_pct"],
                "table_columns": [
                    {"label": "Series", "field": "$series"},
                    {"label": "Rank", "field": "$x"},
                    {"label": "Parser", "field": "parser"},
                    {"label": "Time", "field": "total_time_ms"},
                ],
            }
        }

        datasets = build_series_datasets({"results": results}, profile)

        self.assertEqual(
            datasets["charts"]["total_time_ms"]["TraceLens per-rank trace"][0]["points"],
            [(0, 12.5), (1, 13.0)],
        )
        self.assertEqual(datasets["results_table"]["headers"], ["Series", "Rank", "Parser", "Time"])
        self.assertEqual(
            datasets["results_table"]["rows"][0],
            ["TraceLens per-rank trace", "0", "TraceLensParser", 12.5],
        )

    def test_default_rccl_table_contract_is_preserved(self):
        results = {"all_reduce": {"8": {"bus_bw": 12.5, "alg_bw": 11.0, "time": 100}}}

        datasets = build_series_datasets({"results": results}, {"series": {}})

        self.assertEqual(
            datasets["results_table"]["headers"],
            ["Collective", "Message size", "Bus BW (GB/s)", "Alg BW (GB/s)", "Time (us)"],
        )
        self.assertEqual(datasets["results_table"]["rows"], [["all_reduce", "8", 12.5, 11.0, 100]])


if __name__ == "__main__":
    unittest.main()
