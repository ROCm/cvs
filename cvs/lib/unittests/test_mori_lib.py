'''Unit tests for Mori benchmark result capture.'''

import unittest
from unittest.mock import Mock, patch

from cvs.lib.mori_lib import MoriBenchmark


class TestMoriBenchmarkResultCapture(unittest.TestCase):
    def test_ibgda_run_returns_parsed_node_metrics(self):
        output = """
        Blocks: 2, Threads: 256, Iterations: 1, QPs: 4
        Index Size(B) bw(GB) Time(ms) Rate(Mpps)
        0 33554432 46.5 390.0 1.25
        """
        benchmark = MoriBenchmark.__new__(MoriBenchmark)
        benchmark.container_name = "mori"
        benchmark.mori_dir = "/mori"
        benchmark.expected_results_dict = {"ibgda_write": {}}
        benchmark.phdl = Mock()
        benchmark.phdl.exec.return_value = {"node-a": output}

        result = benchmark.run_ibgda_dist_write()

        self.assertEqual(result["operation"], "ibgda_write")
        self.assertEqual(result["nodes"]["node-a"]["rows"][0]["bandwidth_gb"], 46.5)
        self.assertEqual(result["nodes"]["node-a"]["metadata"]["qps"], 4)

    @patch("cvs.lib.mori_lib.time.sleep")
    def test_torch_io_run_returns_rank_metrics_with_effective_parameters(self, _sleep):
        output = """
        Initiator Rank 0
        +-------------+-----------+----------------+---------------+---------------+--------------+--------------+
        | MsgSize (B) | BatchSize | TotalSize (MB) | Max BW (GB/s) | Avg Bw (GB/s) | Min Lat (us) | Avg Lat (us) |
        +-------------+-----------+----------------+---------------+---------------+--------------+--------------+
        | 524288      | 128       | 64             | 50.0          | 47.5          | 1000.0       | 1200.0       |
        +-------------+-----------+----------------+---------------+---------------+--------------+--------------+
        """
        benchmark = MoriBenchmark.__new__(MoriBenchmark)
        benchmark.container_name = "mori"
        benchmark.mori_dir = "/mori"
        benchmark.nnodes = 1
        benchmark.host_list = ["node-a"]
        benchmark.master_addr = "node-a"
        benchmark.master_port = "1234"
        benchmark.expected_results_dict = {"io_read": {}}
        benchmark.phdl = Mock()
        benchmark.phdl.exec_cmd_list.return_value = {}
        benchmark.phdl.exec.return_value = {"node-a": output}

        result = benchmark.run_mori_torch_io_test(
            op_type="read",
            buffer_size=16384,
            transfer_batch_size=128,
            no_of_qp_per_transfer=4,
        )

        self.assertEqual(result["operation"], "io_read")
        self.assertEqual(result["qp_count"], 4)
        self.assertEqual(result["ranks"][0]["rows"][0]["Avg_BW_GBps"], 47.5)


if __name__ == "__main__":
    unittest.main()
