"""Unit tests for SGLang inference variant schema (inference/sglang/variant.py)."""

import unittest

from pydantic import ValidationError

from cvs.schema.config_file.inference.sglang.variant import (
    SglangSingleVariantConfig,
    perf_cell_key,
)


def _minimal_variant(**overrides):
    payload = {
        "schema_version": 1,
        "framework": "sglang_single",
        "gpu_arch": "mi30x",
        "enforce_thresholds": False,
        "threshold_json": "test_threshold.json",
        "paths": {
            "shared_fs": "/home/test",
            "models_dir": "/home/test/models",
            "log_dir": "/home/test/LOGS",
            "hf_token_file": "/home/test/.hf_token",
        },
        "model": {"id": "test/model", "remote": 0},
        "container": {
            "name": "sglang_test",
            "image": "rocm/sglang:latest",
            "runtime": {"name": "docker", "args": {}},
        },
        "benchmark_params": {
            "tensor_parallelism": "8",
            "pipeline_parallelism": "1",
            "max_concurrency": "32",
            "inference_tests": {
                "bench_serv_random": {"input_length": 128, "output_length": 2048},
            },
        },
    }
    payload.update(overrides)
    return payload


class TestPerfCellKey(unittest.TestCase):
    def test_builds_from_benchmark_params(self):
        key = perf_cell_key(
            {
                "tensor_parallelism": "8",
                "pipeline_parallelism": "2",
                "max_concurrency": "16",
                "inference_tests": {"bench_serv_random": {"input_length": 128, "output_length": 2048}},
            }
        )
        self.assertEqual(key, "ISL=128,OSL=2048,TP=8,PP=2,CONC=16")


class TestSglangSingleVariantConfig(unittest.TestCase):
    def test_minimal_payload_validates(self):
        config = SglangSingleVariantConfig.model_validate(_minimal_variant())
        self.assertEqual(config.framework, "sglang_single")
        self.assertEqual(config.perf_cell_key(), "ISL=128,OSL=2048,TP=8,PP=1,CONC=32")

    def test_syncs_legacy_inference_container_name(self):
        config = SglangSingleVariantConfig.model_validate(
            _minimal_variant(
                inference={"container_name": "old"},
            )
        )
        self.assertEqual(config.inference["container_name"], config.container.name)

    def test_unknown_top_level_field_rejected(self):
        payload = _minimal_variant()
        payload["bogus"] = True
        with self.assertRaises(ValidationError):
            SglangSingleVariantConfig.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
