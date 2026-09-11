'''Unit tests for Mori Run Deck result normalization.'''

import unittest

from cvs.lib.mori_reporting import record_ibgda_results, record_io_results


class TestMoriReporting(unittest.TestCase):
    def test_records_real_io_rank_rows_as_series(self):
        parsed = {
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
                        }
                    ]
                }
            },
        }
        results = {}

        record_io_results(results, parsed, case_qp_count=8, status="pass")

        label = "io_read · buffer 16384 · batch 128 · QP 1 · rank 0 · case QP 8"
        row = results[label]["524288"]
        self.assertEqual(row["Avg_BW_GBps"], 47.5)
        self.assertEqual(row["case_qp_count"], 8)
        self.assertEqual(row["status"], "pass")

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
