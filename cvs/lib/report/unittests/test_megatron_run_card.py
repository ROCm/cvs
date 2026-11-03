'''Unit tests for the Megatron Run Deck run-card hook.'''

import unittest
from types import SimpleNamespace

from cvs.lib.report.profiles.hooks.megatron_run_card import megatron_run_card_display


class TestMegatronRunCard(unittest.TestCase):
    def test_primus_image_and_rows(self):
        variant = SimpleNamespace(
            gpu_arch="MI300X",
            enforce_thresholds=True,
            train_params={
                "tokenizer_model": "meta-llama/Llama-3.1-8B",
                "tensor_parallelism": "1",
                "pipeline_parallelism": "1",
            },
            container=SimpleNamespace(image="rocm/primus:latest", env={"NNODES": "2"}),
        )
        rows = megatron_run_card_display(variant, {"pytest_html_href": "report.html"})
        by_label = {label: (value, is_link) for label, value, is_link in rows}
        self.assertEqual(by_label["Model"][0], "meta-llama/Llama-3.1-8B")
        self.assertEqual(by_label["Framework"][0], "Primus")
        self.assertNotIn("nnodes", by_label)
        self.assertEqual(by_label["Thresholds"][0], "enforced")
        self.assertTrue(by_label["Pytest report"][1])

    def test_megatron_lm_when_image_is_not_primus(self):
        variant = SimpleNamespace(
            gpu_name="MI325X",
            enforce_thresholds=False,
            train_params={"model_name": "llama-70b", "tensor_parallelism": "8"},
            container={"image": "rocm/megatron-lm:latest", "env": {}},
        )
        rows = megatron_run_card_display(variant, {})
        by_label = {label: value for label, value, _is_link in rows}
        self.assertEqual(by_label["Model"], "llama-70b")
        self.assertEqual(by_label["GPU"], "MI325X")
        self.assertEqual(by_label["Framework"], "Megatron-LM")
        self.assertEqual(by_label["TP"], "8")
        self.assertEqual(by_label["PP"], "1")
        self.assertEqual(by_label["Thresholds"], "record-only")


if __name__ == "__main__":
    unittest.main()
