import unittest
from types import SimpleNamespace

from cvs.lib.report.profiles.hooks.sglang_run_card import sglang_run_card_display


class TestSglangRunCard(unittest.TestCase):
    def test_single_lists_every_execution_host_as_benchmark_node(self):
        variant = SimpleNamespace(
            model=SimpleNamespace(id="meta-llama/Llama-3.1-70B-Instruct"),
            gpu_arch="mi325",
            topology="single",
            inference={
                "nnodes": "1",
                "_execution_hosts": ["node-a", "node-b"],
                "benchmark_serv_node": ["node-a", "node-b"],
            },
            benchmark_params={"tensor_parallelism": "8", "pipeline_parallelism": "1"},
            enforce_thresholds=False,
        )
        rows = {label: value for label, value, _ in sglang_run_card_display(variant, {})}
        self.assertEqual(rows["Benchmark node"], "node-a, node-b")
        self.assertNotIn("Server nodes", rows)

    def test_distributed_uses_first_host_as_benchmark_node(self):
        variant = SimpleNamespace(
            model=SimpleNamespace(id="meta-llama/Llama-3.1-70B-Instruct"),
            gpu_arch="mi325",
            topology="distributed",
            inference={
                "nnodes": "2",
                "_execution_hosts": ["node-a", "node-b"],
                "benchmark_serv_node": "node-a",
            },
            benchmark_params={"tensor_parallelism": "8", "pipeline_parallelism": "2"},
            enforce_thresholds=False,
        )
        rows = {label: value for label, value, _ in sglang_run_card_display(variant, {})}
        self.assertEqual(rows["Server nodes"], "node-a, node-b")
        self.assertEqual(rows["nnodes"], "2")
        self.assertEqual(rows["Benchmark node"], "node-a")

    def test_disagg_lists_prefill_decode_and_first_host_roles(self):
        variant = SimpleNamespace(
            model=SimpleNamespace(id="meta-llama/Llama-3.1-70B-Instruct"),
            gpu_arch="mi325",
            topology="disaggregated",
            inference={
                "nnodes": "4",
                "prefill_node_list": ["n0", "n2"],
                "decode_node_list": ["n1", "n3"],
                "proxy_router_node": "n0",
                "benchmark_serv_node": "n0",
            },
            benchmark_params={"tensor_parallelism": "8", "pipeline_parallelism": "1"},
            enforce_thresholds=False,
        )
        rows = {label: value for label, value, _ in sglang_run_card_display(variant, {})}
        self.assertEqual(rows["Prefill nodes"], "n0, n2")
        self.assertEqual(rows["Decode nodes"], "n1, n3")
        self.assertEqual(rows["Proxy router"], "n0")
        self.assertEqual(rows["Benchmark node"], "n0")


if __name__ == "__main__":
    unittest.main()
