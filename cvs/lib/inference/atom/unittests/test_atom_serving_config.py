import json
import unittest
from pathlib import Path

from cvs.lib.inference.atom.atom_config_loader import AtomVariantConfig
from cvs.lib.inference.atom.atom_serving_config import (
    atom_sweep_to_serving,
    is_serving_config,
    parse_perf_cell_key,
    serving_runs_to_atom_sweep,
    serving_to_atom_variant_raw,
)


class TestAtomServingConfig(unittest.TestCase):
    def test_is_serving_config(self):
        self.assertTrue(
            is_serving_config(
                {
                    "server_params": {"backend": "sglang"},
                    "benchmark_params": {"data_set_name": "random"},
                    "sweeps": {},
                    "runs": [],
                }
            )
        )
        self.assertFalse(is_serving_config({"params": {}}))

    def test_atom_sweep_to_serving(self):
        sweeps, runs = atom_sweep_to_serving(
            {
                "sequence_combinations": [{"name": "a", "isl": "1024", "osl": "1024"}],
                "runs": [{"combo": "a", "concurrency": 16}],
            },
            "8",
            "1",
        )
        self.assertEqual(list(sweeps), ["ISL=1024,OSL=1024,TP=8,PP=1,CONC=16"])
        self.assertEqual(runs, ["ISL=1024,OSL=1024,TP=8,PP=1,CONC=16"])

    def test_serving_runs_to_atom_sweep(self):
        sweep = serving_runs_to_atom_sweep(
            ["ISL=1024,OSL=1024,TP=8,PP=1,CONC=16", "ISL=1024,OSL=1024,TP=8,PP=1,CONC=32"]
        )
        self.assertEqual(len(sweep["sequence_combinations"]), 1)
        self.assertEqual(len(sweep["runs"]), 2)

    def test_load_atom_vllm_serving_config(self):
        root = Path(__file__).resolve().parents[4]
        cfg = root / "input/config_file/inference/atom/mi3xx_atom_vllm_deepseek-r1_fp8_single.json"
        raw = json.loads(cfg.read_text(encoding="utf-8"))
        th_path = cfg.parent / raw["threshold_json"]
        thresholds = json.loads(th_path.read_text(encoding="utf-8"))
        variant_raw = serving_to_atom_variant_raw(raw, thresholds)
        variant = AtomVariantConfig(**variant_raw)
        self.assertEqual(variant.params.driver, "vllm_atom")
        self.assertEqual(len(variant.expected_cells()), 3)

    def test_load_atom_vllm_gpt_oss_serving_config(self):
        root = Path(__file__).resolve().parents[4]
        cfg = root / "input/config_file/inference/atom/mi3xx_atom_vllm_gpt-oss-120b_mxfp4_single.json"
        raw = json.loads(cfg.read_text(encoding="utf-8"))
        th_path = cfg.parent / raw["threshold_json"]
        thresholds = json.loads(th_path.read_text(encoding="utf-8"))
        variant_raw = serving_to_atom_variant_raw(raw, thresholds)
        variant = AtomVariantConfig(**variant_raw)
        self.assertEqual(variant.params.driver, "vllm_atom")
        self.assertEqual(variant.params.max_model_length, "12288")
        self.assertTrue(variant.platform.gpu_metrics_poll)

    def test_load_atom_sglang_serving_config(self):
        root = Path(__file__).resolve().parents[4]
        cfg = root / "input/config_file/inference/atom/mi3xx_atom_sglang_deepseek-r1_fp8_single.json"
        raw = json.loads(cfg.read_text(encoding="utf-8"))
        th_path = cfg.parent / raw["threshold_json"]
        thresholds = json.loads(th_path.read_text(encoding="utf-8"))
        variant_raw = serving_to_atom_variant_raw(raw, thresholds)
        variant = AtomVariantConfig(**variant_raw)
        self.assertEqual(variant.params.driver, "sglang")
        self.assertIn("--kv-cache-dtype", variant.roles.server.sglang_args)

    def test_parse_perf_cell_key(self):
        parts = parse_perf_cell_key("ISL=1024,OSL=2048,TP=8,PP=2,CONC=32")
        self.assertEqual(parts["pp"], "2")


if __name__ == "__main__":
    unittest.main()
