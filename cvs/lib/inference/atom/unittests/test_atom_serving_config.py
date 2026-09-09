import json
import unittest
from pathlib import Path

from cvs.lib.inference.atom.atom_config_loader import AtomVariantConfig
from cvs.lib.inference.atom.atom_serving_config import (
    atom_sweep_to_serving,
    is_serving_config,
    materialize_atom_sweep,
    parse_perf_cell_key,
    serving_runs_to_atom_sweep,
    serving_selected_cell_keys,
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
                    "sweep": {"runs": []},
                }
            )
        )
        self.assertTrue(
            is_serving_config(
                {
                    "server_params": {"backend": "vllm"},
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

    def test_serving_selected_cell_keys_from_sweep_runs(self):
        keys = serving_selected_cell_keys(
            {
                "sweeps": {
                    "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16": {},
                    "ISL=1024,OSL=1024,TP=8,PP=1,CONC=32": {},
                },
                "sweep": {
                    "runs": [{"combo": "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16"}],
                },
            }
        )
        self.assertEqual(keys, ["ISL=1024,OSL=1024,TP=8,PP=1,CONC=16"])

    def test_serving_selected_cell_keys_empty_runs_uses_sweeps(self):
        keys = serving_selected_cell_keys(
            {
                "sweeps": {
                    "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16": {},
                    "ISL=1024,OSL=1024,TP=8,PP=1,CONC=32": {},
                },
                "runs": [],
            }
        )
        self.assertEqual(
            keys,
            [
                "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16",
                "ISL=1024,OSL=1024,TP=8,PP=1,CONC=32",
            ],
        )

    def test_materialize_atom_sweep_empty_runs_uses_catalog(self):
        raw = materialize_atom_sweep(
            {
                "sweeps": {
                    "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16": {},
                    "ISL=1024,OSL=1024,TP=8,PP=1,CONC=32": {},
                },
                "runs": [],
            }
        )
        self.assertNotIn("sweeps", raw)
        self.assertNotIn("runs", raw)
        self.assertEqual(len(raw["sweep"]["runs"]), 2)

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

    def test_load_atom_sglang_distributed_drops_container_runtime_env(self):
        root = Path(__file__).resolve().parents[4]
        cfg = root / "input/config_file/inference/atom/mi3xx_atom_sglang_deepseek-r1_fp8_distributed.json"
        raw = json.loads(cfg.read_text(encoding="utf-8"))
        th_path = cfg.parent / raw["threshold_json"]
        thresholds = json.loads(th_path.read_text(encoding="utf-8"))
        variant_raw = serving_to_atom_variant_raw(raw, thresholds)
        variant = AtomVariantConfig(**variant_raw)
        self.assertNotIn("env", variant.container.runtime.args)
        self.assertEqual(variant.roles.server.env.get("SGLANG_USE_AITER"), "1")
        self.assertEqual(variant.roles.server.env.get("NCCL_IB_GID_INDEX"), "3")
        self.assertNotIn("NCCL_IB_HCA", variant.roles.server.env)
        self.assertNotIn("NCCL_SOCKET_IFNAME", variant.roles.server.env)
        self.assertEqual(len(variant.expected_cells()), 16)

    def test_parse_perf_cell_key(self):
        parts = parse_perf_cell_key("ISL=1024,OSL=2048,TP=8,PP=2,CONC=32")
        self.assertEqual(parts["pp"], "2")


if __name__ == "__main__":
    unittest.main()
