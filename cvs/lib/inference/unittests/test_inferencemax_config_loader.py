'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.inference.utils.inferencemax_config_loader.
'''

import unittest
from pathlib import Path

from cvs.lib.inference.utils.inferencemax_config_loader import (
    InferenceMaxVariantConfig,
    benchmark_model_key,
    legacy_benchmark_params_from_variant,
    legacy_inference_dict_from_variant,
    load_variant,
    placeholder_gated_threshold_cell,
)
from cvs.lib.inference.utils.inferencing_config_loader import Run, SeqCombo, Sweep


def _cluster_dict():
    return {"username": "testuser"}


def _minimal_variant(sweep, thresholds=None, enforce=False):
    return InferenceMaxVariantConfig(
        schema_version=1,
        framework="inferencemax_single",
        gpu_arch="mi300x",
        enforce_thresholds=enforce,
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
            "runtime": {
                "name": "docker",
                "args": {
                    "volumes": ["/home/x:/home/x"],
                    "devices": ["/dev/dri", "/dev/kfd"],
                    "shm_size": "128G",
                },
            },
        },
        roles={"server": {"env": {"CVS_GPU_MEMORY_UTIL": "0.95"}}},
        params={"tensor_parallelism": "8"},
        sweep=sweep,
        thresholds=thresholds or {},
    )


class TestInferenceMaxConfigLoader(unittest.TestCase):
    def test_load_mi300x_sample_config(self):
        root = Path(__file__).resolve().parents[3]
        config = root / (
            "input/config_file/inference/inferencemax_single/"
            "mi300x_gpt_oss_120b_single/mi300x_gpt_oss_120b_single_config.json"
        )
        variant = load_variant(config, _cluster_dict())
        self.assertEqual(variant.framework, "inferencemax_single")
        self.assertEqual(benchmark_model_key(variant), "gpt-oss-120b")
        self.assertEqual(variant.expected_cells(), ["ISL=7168,OSL=1024,TP=8,CONC=64"])

    def test_legacy_adapters_map_thresholds(self):
        cell_key = "ISL=7168,OSL=1024,TP=8,CONC=64"
        thresholds = {
            cell_key: placeholder_gated_threshold_cell(
                output_throughput_min=4200,
                mean_ttft_max_ms=500,
                mean_tpot_max_ms=15,
            )
        }
        sweep = Sweep(
            sequence_combinations=[SeqCombo(name="legacy_profile", isl="7168", osl="1024")],
            runs=[Run(combo="legacy_profile", concurrency=64)],
        )
        variant = _minimal_variant(sweep, thresholds=thresholds)
        legacy_inf = legacy_inference_dict_from_variant(variant)
        self.assertEqual(legacy_inf["container_config"]["env_dict"]["CVS_GPU_MEMORY_UTIL"], "0.95")
        legacy_bp = legacy_benchmark_params_from_variant(variant)
        result = legacy_bp["gpt-oss-120b"]["result_dict"][cell_key]
        self.assertEqual(result["output_throughput_per_sec"], "4200")
        self.assertEqual(result["mean_ttft_ms"], "500")
        self.assertEqual(result["mean_tpot_ms"], "15")


if __name__ == "__main__":
    unittest.main()
