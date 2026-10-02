'''Unit tests for the TorchTitan Run Deck run-card hook.'''

import unittest
from types import SimpleNamespace

from cvs.lib.report.profiles.hooks.torchtitan_run_card import torchtitan_run_card_display


class TestTorchTitanRunCard(unittest.TestCase):
    def test_model_gpu_and_parallel_degrees(self):
        variant = SimpleNamespace(
            gpu_arch="MI355X",
            enforce_thresholds=True,
            train_params={
                "model_name": "llama3_1_8b",
                "tokenizer_model": "meta-llama/Llama-3.1-8B",
                "sequence_length": "8192",
                "tensor_parallel_degree": "2",
                "pipeline_parallel_degree": "1",
            },
            container=SimpleNamespace(image="rocm/torchtitan:latest", env={}),
        )
        rows = torchtitan_run_card_display(variant, {"pytest_html_href": "report.html"})
        by_label = {label: (value, is_link) for label, value, is_link in rows}
        self.assertEqual(by_label["Model"][0], "meta-llama/Llama-3.1-8B")
        self.assertEqual(by_label["GPU"][0], "MI355X")
        self.assertEqual(by_label["Framework"][0], "TorchTitan")
        self.assertEqual(by_label["Seq"][0], "8192")
        self.assertEqual(by_label["TP"][0], "2")
        self.assertEqual(by_label["PP"][0], "1")
        self.assertEqual(by_label["Thresholds"][0], "enforced")
        self.assertTrue(by_label["Pytest report"][1])

    def test_primus_image(self):
        variant = SimpleNamespace(
            gpu_name="MI300X",
            enforce_thresholds=False,
            train_params={"tokenizer_model": "meta-llama/Llama-3.1-8B"},
            container={"image": "rocm/primus:latest", "env": {}},
        )
        rows = torchtitan_run_card_display(variant, {})
        by_label = {label: value for label, value, _is_link in rows}
        self.assertEqual(by_label["Model"], "meta-llama/Llama-3.1-8B")
        self.assertEqual(by_label["Framework"], "Primus")
        self.assertEqual(by_label["Thresholds"], "record-only")

    def test_model_name_when_tokenizer_missing(self):
        variant = SimpleNamespace(
            gpu_arch="MI355X",
            enforce_thresholds=False,
            train_params={"model_name": "llama3_1_8b"},
            container=SimpleNamespace(image="rocm/torchtitan:latest", env={}),
        )
        rows = torchtitan_run_card_display(variant, {})
        by_label = {label: value for label, value, _is_link in rows}
        self.assertEqual(by_label["Model"], "llama3_1_8b")

    def test_parallelism_keys_match_results_table_order(self):
        variant = SimpleNamespace(
            gpu_arch="MI355X",
            enforce_thresholds=False,
            train_params={
                "tensor_parallelism": "4",
                "tensor_parallel_degree": "2",
                "pipeline_parallelism": "2",
                "pipeline_parallel_degree": "1",
            },
            container=SimpleNamespace(image="rocm/torchtitan:latest", env={}),
        )
        rows = torchtitan_run_card_display(variant, {})
        by_label = {label: value for label, value, _is_link in rows}
        self.assertEqual(by_label["TP"], "4")
        self.assertEqual(by_label["PP"], "2")


if __name__ == "__main__":
    unittest.main()
