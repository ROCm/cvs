'''Unit tests for Mori Run Deck result normalization.'''

import unittest

from cvs.lib.mori_reporting import record_ibgda_results, record_io_results


def _io_row(size, avg_bw, avg_lat, max_bw=None, min_lat=None):
    return {
        "MsgSize_B": size,
        "BatchSize": 128,
        "TotalSize_MB": size * 128 / 1e6,
        "Max_BW_GBps": avg_bw + 1.0 if max_bw is None else max_bw,
        "Avg_BW_GBps": avg_bw,
        "Min_Lat_us": avg_lat - 1.0 if min_lat is None else min_lat,
        "Avg_Lat_us": avg_lat,
    }


def _io_parsed(ranks):
    return {
        "operation": "io_read",
        "buffer_size": 16384,
        "transfer_batch_size": 128,
        "qp_count": 1,
        "ranks": {rank: {"rows": rows} for rank, rows in ranks.items()},
    }


CASE_LABEL = "io_read · buffer 16384 · batch 128 · QP 1"


class TestMoriReporting(unittest.TestCase):
    def test_io_ranks_collapse_into_one_series_per_case(self):
        parsed = _io_parsed(
            {
                0: [_io_row(524288, 47.0, 1200.0, max_bw=49.0, min_lat=1100.0), _io_row(1048576, 46.0, 2900.0)],
                1: [_io_row(524288, 45.0, 1300.0, max_bw=50.0, min_lat=1150.0), _io_row(1048576, 47.0, 2850.0)],
                2: [_io_row(524288, 46.3, 1250.0, max_bw=48.0, min_lat=1050.0), _io_row(1048576, 46.5, 2870.0)],
            }
        )
        results = {}

        record_io_results(results, parsed, case_qp_count=1, status="pass")

        self.assertEqual(list(results), [CASE_LABEL])
        self.assertEqual(sorted(results[CASE_LABEL], key=int), ["524288", "1048576"])
        row = results[CASE_LABEL]["524288"]
        self.assertEqual(row["ranks"], 3)
        self.assertEqual(row["Avg_BW_GBps"], 46.1)
        self.assertEqual(row["Slowest_rank"], 1)
        self.assertEqual(row["Slowest_rank_BW_GBps"], 45.0)
        self.assertEqual(row["Avg_Lat_us"], 1250.0)
        self.assertEqual(row["Worst_rank_Lat_us"], 1300.0)
        self.assertEqual(row["Max_BW_GBps"], 50.0)
        self.assertEqual(row["Min_Lat_us"], 1050.0)
        self.assertEqual(row["MsgSize_B"], 524288)
        self.assertEqual(row["BatchSize"], 128)
        self.assertEqual(row["status"], "pass")
        self.assertNotIn("rank", row)
        self.assertEqual(results[CASE_LABEL]["1048576"]["Slowest_rank"], 0)

    def test_case_qp_that_differs_from_the_run_stays_in_the_label(self):
        results = {}

        record_io_results(results, _io_parsed({0: [_io_row(8, 0.02, 40.0)]}), case_qp_count=8, status="fail")

        label = f"{CASE_LABEL} · case QP 8"
        self.assertEqual(list(results), [label])
        self.assertEqual(results[label]["8"]["case_qp_count"], 8)
        self.assertEqual(results[label]["8"]["qp_count"], 1)
        self.assertEqual(results[label]["8"]["status"], "fail")

    def test_slowest_rank_tie_resolves_to_lowest_rank_regardless_of_order(self):
        parsed = _io_parsed({7: [_io_row(64, 0.5, 40.0)], 3: [_io_row(64, 0.5, 41.0)], 5: [_io_row(64, 0.9, 42.0)]})
        results = {}

        record_io_results(results, parsed)

        row = results[CASE_LABEL]["64"]
        self.assertEqual(row["Slowest_rank"], 3)
        self.assertEqual(row["Worst_rank_Lat_us"], 42.0)

    def test_size_reported_by_only_some_ranks_aggregates_those_ranks(self):
        parsed = _io_parsed({0: [_io_row(8, 0.02, 40.0), _io_row(16, 0.04, 41.0)], 1: [_io_row(16, 0.06, 43.0)]})
        results = {}

        record_io_results(results, parsed)

        self.assertEqual(results[CASE_LABEL]["8"]["ranks"], 1)
        self.assertEqual(results[CASE_LABEL]["8"]["Avg_BW_GBps"], 0.02)
        self.assertEqual(results[CASE_LABEL]["16"]["ranks"], 2)
        self.assertEqual(results[CASE_LABEL]["16"]["Avg_BW_GBps"], 0.05)
        self.assertEqual(results[CASE_LABEL]["16"]["Slowest_rank"], 0)

    def test_non_numeric_rank_value_is_left_out_of_that_metric_only(self):
        bad = _io_row(1024, 0.0, 50.0)
        bad["Avg_BW_GBps"] = "n/a"
        parsed = _io_parsed({0: [bad], 1: [_io_row(1024, 3.0, 60.0)], 2: [_io_row(1024, 5.0, 70.0)]})
        results = {}

        record_io_results(results, parsed)

        row = results[CASE_LABEL]["1024"]
        self.assertEqual(row["ranks"], 3)
        self.assertEqual(row["Avg_BW_GBps"], 4.0)
        self.assertEqual(row["Slowest_rank"], 1)
        self.assertEqual(row["Avg_Lat_us"], 60.0)
        self.assertEqual(row["Worst_rank_Lat_us"], 70.0)

    def test_rank_bandwidth_missing_everywhere_leaves_bandwidth_fields_out(self):
        row = _io_row(2048, 1.0, 45.0)
        del row["Avg_BW_GBps"]
        results = {}

        record_io_results(results, _io_parsed({0: [row]}))

        entry = results[CASE_LABEL]["2048"]
        self.assertNotIn("Avg_BW_GBps", entry)
        self.assertNotIn("Slowest_rank", entry)
        self.assertEqual(entry["Avg_Lat_us"], 45.0)

    def test_rows_without_message_size_never_create_an_empty_series(self):
        row = _io_row(8, 0.02, 40.0)
        del row["MsgSize_B"]
        results = {}

        record_io_results(results, _io_parsed({0: [row], 1: []}))

        self.assertEqual(results, {})

    def test_separate_cases_get_separate_series(self):
        results = {}
        other = _io_parsed({0: [_io_row(8, 0.03, 60.0)]})
        other["transfer_batch_size"] = 256

        record_io_results(results, _io_parsed({0: [_io_row(8, 0.02, 40.0)]}))
        record_io_results(results, other)

        self.assertEqual(set(results), {CASE_LABEL, "io_read · buffer 16384 · batch 256 · QP 1"})

    def test_io_run_without_results_leaves_store_untouched(self):
        results = {"existing": {"1": {"Avg_BW_GBps": 1.0}}}
        for parsed in (None, {}):
            with self.subTest(parsed=parsed):
                self.assertIs(record_io_results(results, parsed, case_qp_count=1, status="fail"), results)
                self.assertEqual(results, {"existing": {"1": {"Avg_BW_GBps": 1.0}}})

    def test_records_real_ibgda_node_rows_as_series(self):
        parsed = {
            "operation": "ibgda_write",
            "processes": 2,
            "ctas": 2,
            "threads": 256,
            "qp_count": 4,
            "iterations": 1,
            "nodes": {
                "node-a": {
                    "metadata": {"blocks": 2, "threads": 256, "iterations": 1, "qps": 4},
                    "rows": [
                        {
                            "size_bytes": 33554432,
                            "bandwidth_gb": 46.5,
                            "time_ms": 390.0,
                            "rate_mpps": 1.25,
                        }
                    ],
                }
            },
        }
        results = {}

        record_ibgda_results(results, parsed, status="fail")

        label = "ibgda_write · procs 2 · CTAs 2 · QP 4 · node-a"
        row = results[label]["33554432"]
        self.assertEqual(row["bandwidth_gb"], 46.5)
        self.assertEqual(row["threads"], 256)
        self.assertEqual(row["status"], "fail")


if __name__ == "__main__":
    unittest.main()
