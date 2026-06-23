'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.inference.utils.inferencex_atom_config_loader.
'''

import unittest
from pathlib import Path

from cvs.lib.inference.utils.inferencex_atom_config_loader import (
    InferenceXAtomVariantConfig,
    load_variant,
    orchestrator_container_from_variant,
    placeholder_gated_threshold_cell,
)
from cvs.lib.inference.utils.inferencing_config_loader import Run, SeqCombo, Sweep


def _cluster_dict():
    return {"username": "testuser"}


class TestInferenceXAtomConfigLoader(unittest.TestCase):
    def test_load_mi300x_sample_config(self):
        root = Path(__file__).resolve().parents[3]
        config = root / (
            "input/config_file/inference/inferencex_atom_single/"
            "mi300x_gpt_oss_120b_single/mi300x_gpt_oss_120b_single_config.json"
        )
        variant = load_variant(config, _cluster_dict())
        self.assertEqual(variant.framework, "inferencex_atom_single")
        self.assertEqual(variant.params.driver, "vllm")
        self.assertEqual(variant.expected_cells(), ["ISL=7168,OSL=1024,TP=8,CONC=64"])
        self.assertIn("enforce-eager", variant.roles.server.serve_args)

    def test_load_w1_mi300x_atom_variant(self):
        root = Path(__file__).resolve().parents[3]
        config = root / (
            "input/dtni/inferencex_atom_single/"
            "deepseek_r1_fp8_mi300x_atom_perf/config.json"
        )
        variant = load_variant(config, _cluster_dict())
        self.assertEqual(variant.gpu_arch, "mi300x")
        self.assertEqual(variant.ix_recipe_id, "dsr1-fp8-mi300x-atom")
        self.assertEqual(variant.params.driver, "atom")
        self.assertEqual(
            variant.roles.server.atom_args[:4],
            ["-tp", "8", "--kv_cache_dtype", "fp8"],
        )
        self.assertEqual(
            variant.expected_cells(),
            ["ISL=1024,OSL=1024,TP=8,CONC=128", "ISL=1024,OSL=1024,TP=8,CONC=256"],
        )

    def test_load_w1_mi300x_smoke_variant(self):
        root = Path(__file__).resolve().parents[3]
        config = root / (
            "input/dtni/inferencex_atom_single/"
            "deepseek_r1_fp8_mi300x_atom_smoke/config.json"
        )
        variant = load_variant(config, _cluster_dict())
        self.assertEqual(variant.params.num_prompts, "128")
        self.assertEqual(variant.expected_cells(), ["ISL=1024,OSL=1024,TP=8,CONC=128"])

    def test_orchestrator_container_includes_server_env(self):
        sweep = Sweep(
            sequence_combinations=[SeqCombo(name="legacy_profile", isl="7168", osl="1024")],
            runs=[Run(combo="legacy_profile", concurrency=64)],
        )
        thresholds = {
            "ISL=7168,OSL=1024,TP=8,CONC=64": placeholder_gated_threshold_cell(),
        }
        variant = InferenceXAtomVariantConfig(
            schema_version=1,
            framework="inferencex_atom_single",
            gpu_arch="mi300x",
            enforce_thresholds=False,
            paths={
                "shared_fs": "/home/x",
                "models_dir": "/home/x/models",
                "log_dir": "/home/x/LOGS",
                "hf_token_file": "/home/x/.hf",
            },
            model={"id": "openai/gpt-oss-120b", "remote": 0, "precision": "bf16"},
            container={
                "name": "c",
                "image": "img",
                "runtime": {"name": "docker", "args": {"volumes": ["/home/x:/home/x"]}},
            },
            roles={"server": {"env": {"VLLM_ROCM_USE_AITER": "1"}}},
            params={"tensor_parallelism": "8"},
            sweep=sweep,
            thresholds=thresholds,
        )
        block = orchestrator_container_from_variant(variant)
        self.assertEqual(block["env"]["VLLM_ROCM_USE_AITER"], "1")


if __name__ == "__main__":
    unittest.main()
