'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import unittest

from cvs.lib.inference.utils.inference_suite_results_table import log_sweep_knee_summary


class TestInferenceSuiteResultsTable(unittest.TestCase):
    def test_log_sweep_knee_summary_picks_max_throughput(self):
        inf_res_dict = {
            ("m", "mi300x", "1024", "1024", "w1", 128): {
                "n1": {"client.output_throughput": 1000.0, "client.mean_ttft_ms": 50.0}
            },
            ("m", "mi300x", "1024", "1024", "w1", 256): {
                "n1": {"client.output_throughput": 2000.0, "client.mean_ttft_ms": 80.0}
            },
        }
        log_sweep_knee_summary(inf_res_dict)


if __name__ == "__main__":
    unittest.main()
