'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.inference.utils.inferencex_atom_vllm_config_loader.
'''

import unittest
from pathlib import Path

from cvs.lib.inference.utils.inferencex_atom_vllm_config_loader import load_variant


def _cluster_dict():
    return {"username": "testuser"}


class TestInferenceXAtomVllmConfigLoader(unittest.TestCase):
    def test_load_w1_vllm_parity_variant(self):
        root = Path(__file__).resolve().parents[3]
        config = root / (
            "input/config_file/inference/inferencex_atom_vllm_single/"
            "deepseek_r1_fp8_mi300x_atom_vllm_perf/deepseek_r1_fp8_mi300x_atom_vllm_perf_config.json"
        )
        variant = load_variant(config, _cluster_dict())
        self.assertEqual(variant.framework, "inferencex_atom_vllm_single")
        self.assertEqual(variant.params.driver, "vllm")
        self.assertEqual(
            variant.expected_cells(),
            ["ISL=1024,OSL=1024,TP=8,CONC=128", "ISL=1024,OSL=1024,TP=8,CONC=256"],
        )
        self.assertFalse(variant.enforce_thresholds)

    def test_load_gpt_oss_uplift_variant(self):
        root = Path(__file__).resolve().parents[3]
        config = root / (
            "input/config_file/inference/inferencex_atom_vllm_single/"
            "mi300x_gpt_oss_120b_single/mi300x_gpt_oss_120b_single_config.json"
        )
        variant = load_variant(config, _cluster_dict())
        self.assertEqual(variant.framework, "inferencex_atom_vllm_single")
        self.assertEqual(variant.model.id, "openai/gpt-oss-120b")
        self.assertEqual(variant.expected_cells(), ["ISL=7168,OSL=1024,TP=8,CONC=64"])

    def test_rejects_atom_driver(self):
        from cvs.lib.inference.utils.inferencex_atom_config_loader import InferenceXAtomParams
        from cvs.lib.inference.utils.inferencing_config_loader import Run, SeqCombo, Sweep
        from cvs.lib.inference.utils.inferencex_atom_vllm_config_loader import (
            InferenceXAtomVllmVariantConfig,
        )
        from cvs.lib.inference.utils.inferencex_atom_config_loader import placeholder_gated_threshold_cell

        sweep = Sweep(
            sequence_combinations=[SeqCombo(name="w1_1k_1k", isl="1024", osl="1024")],
            runs=[Run(combo="w1_1k_1k", concurrency=128)],
        )
        with self.assertRaises(ValueError):
            InferenceXAtomVllmVariantConfig(
                schema_version=1,
                framework="inferencex_atom_vllm_single",
                gpu_arch="mi300x",
                enforce_thresholds=False,
                paths={
                    "shared_fs": "/home/x",
                    "models_dir": "/home/x/models",
                    "log_dir": "/home/x/LOGS",
                    "hf_token_file": "/home/x/.hf",
                },
                model={"id": "m", "remote": 0, "precision": "fp8"},
                container={
                    "name": "c",
                    "image": "img",
                    "runtime": {"name": "docker", "args": {"volumes": ["/home/x:/home/x"]}},
                },
                params=InferenceXAtomParams(driver="atom", tensor_parallelism="8"),
                sweep=sweep,
                thresholds={
                    "ISL=1024,OSL=1024,TP=8,CONC=128": placeholder_gated_threshold_cell(),
                },
            )


if __name__ == "__main__":
    unittest.main()
