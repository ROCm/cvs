'''Unit tests for ibperf Run Deck result normalization.'''

import unittest

from cvs.lib.ibperf_reporting import record_bw_results, record_lat_results


class TestIbperfReporting(unittest.TestCase):
    def test_bw_point_aggregates_every_node_and_nic(self):
        per_node = {
            "node-a": {0: {"bw": "390.5", "pps": "0.745"}, 1: {"bw": "370.5", "pps": "0.707"}},
            "node-b": {0: {"bw": "380.0", "pps": "0.725"}},
        }
        results = {}

        record_bw_results(results, "ib_write_bw", 65536, "8", per_node)

        self.assertEqual(
            results["ib_write_bw · QP 8"]["65536"],
            {
                "test": "ib_write_bw",
                "qp_count": "8",
                "bw_mean": 380.333,
                "bw_min": 370.5,
                "pps_mean": 0.726,
                "nics": 3,
                "nodes": 2,
            },
        )

    def test_bw_skips_instances_that_failed_collection(self):
        results = {}
        record_bw_results(results, "ib_write_bw", 2, "8", {"node-a": {0: {"bw": "10.0", "pps": "1.0"}, 1: {}}})
        self.assertEqual(results["ib_write_bw · QP 8"]["2"]["nics"], 1)

        untouched = {}
        record_bw_results(untouched, "ib_write_bw", 2, "8", {"node-a": {}, "node-b": {0: {}}})
        self.assertEqual(untouched, {})

    def test_lat_point_reports_mean_average_and_worst_tail(self):
        per_node = {
            "node-a": {0: {"t_avg": "2.00", "t_99_pct": "2.50", "t_max": "9.00"}},
            "node-b": {0: {"t_avg": "3.00", "t_99_pct": "4.00", "t_max": "7.00"}},
        }
        results = {}

        record_lat_results(results, "ib_write_lat", 2, per_node)

        self.assertEqual(
            results["ib_write_lat"]["2"],
            {"test": "ib_write_lat", "t_avg": 2.5, "t_99_pct": 4.0, "t_max": 9.0, "nics": 2, "nodes": 2},
        )


if __name__ == "__main__":
    unittest.main()
