"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

import os
import unittest

from cvs.lib.config.loader import parse_config
from cvs.lib.config.migrate import migrate_vllm_megaconfig

MEGA = {
    "config": {"nnodes": "1", "container_image": "img:tag"},
    "benchmark_params": {
        "gpt-oss-120b": {
            "backend": "vllm",
            "model": "openai/gpt-oss-120b",
            "concurrency_levels": [16, 32],
            "sequence_combinations": [{"isl": "1024", "osl": "1024", "name": "balanced"}],
            "tensor_parallelism": "1",
            "num_prompts": "3200",
            "max_model_length": "9216",
            "percentile_metrics": "ttft,tpot,itl,e2el",
            "server_script": "gpt.sh",
            "result_dict": {
                "ISL=1024,OSL=1024,TP=1,CONC=16": {
                    "total_throughput_per_sec": "4651",
                    "mean_ttft_ms": "70",
                    "mean_tpot_ms": "8",
                }
            },
        },
        "qwen3-80b": {
            "backend": "vllm",
            "model": "Qwen/Qwen3-Next-80B",
            "concurrency_levels": [16],
            "sequence_combinations": [{"isl": "8192", "osl": "1024", "name": "long_context"}],
            "tensor_parallelism": "8",
            "server_script": "qwen.sh",
            "result_dict": {},
        },
    },
}


class TestMigrate(unittest.TestCase):
    def setUp(self):
        # Migrated configs reference ${env:HF_TOKEN}; B3 fails closed if unset,
        # so provide it for the parse round-trip and restore afterwards.
        self._prev_token = os.environ.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = "hf_test_token"

    def tearDown(self):
        if self._prev_token is None:
            os.environ.pop("HF_TOKEN", None)
        else:
            os.environ["HF_TOKEN"] = self._prev_token

    def test_splits_per_model_and_validates(self):
        migrated = migrate_vllm_megaconfig(MEGA, target_gpu="mi355x")
        self.assertEqual(set(migrated), {"gpt_oss_120b", "qwen3_80b"})
        for v2 in migrated.values():
            cfg = parse_config(v2)  # round-trips through the typed schema
            self.assertEqual(cfg.framework, "vllm")
            self.assertEqual(cfg.schema_version, "2")

    def test_result_dict_becomes_honest_thresholds(self):
        migrated = migrate_vllm_megaconfig(MEGA, target_gpu="mi355x")
        thresholds = migrated["gpt_oss_120b"]["thresholds"]
        by_metric = {t["metric"]: t for t in thresholds}
        # Throughput floors at the observed min; latencies ceil at the max.
        self.assertEqual((by_metric["total_throughput"]["op"], by_metric["total_throughput"]["value"]), (">=", 4651.0))
        self.assertEqual((by_metric["mean_ttft_ms"]["op"], by_metric["mean_ttft_ms"]["value"]), ("<=", 70.0))
        self.assertEqual(by_metric["mean_tpot_ms"]["op"], "<=")
        # C5: a v1 mean is never relabeled as a P99 percentile.
        self.assertFalse(any(t["type"] == "percentile" for t in thresholds))
        self.assertFalse(any("percentile" in t for t in thresholds))

    def test_empty_result_dict_yields_no_thresholds(self):
        migrated = migrate_vllm_megaconfig(MEGA, target_gpu="mi355x")
        self.assertEqual(migrated["qwen3_80b"]["thresholds"], [])

    def test_emits_token_into_container_env_not_secrets(self):
        migrated = migrate_vllm_megaconfig(MEGA, target_gpu="mi355x")
        gpt = migrated["gpt_oss_120b"]
        self.assertEqual(gpt["container"]["env"]["HF_TOKEN"], "${env:HF_TOKEN}")
        self.assertNotIn("secrets", gpt)

    def test_rejects_changeme_sentinel(self):
        bad = {"config": {"hf_token_file": "<changeme>"}, "benchmark_params": {}}
        with self.assertRaises(ValueError):
            migrate_vllm_megaconfig(bad, target_gpu="mi300")

    def test_topology_gpus_from_tp(self):
        migrated = migrate_vllm_megaconfig(MEGA, target_gpu="mi355x")
        self.assertEqual(migrated["qwen3_80b"]["topology"]["roles"]["server"]["gpus_per_node"], 8)

    def test_tp_is_not_a_sweep_axis(self):
        # TP lives once, in topology.gpus_per_node; it is not also a sweep axis
        # (which would both diverge from the fixed topology and add a redundant
        # cell-ID token).
        migrated = migrate_vllm_megaconfig(MEGA, target_gpu="mi355x")
        self.assertNotIn("tensor_parallelism", migrated["gpt_oss_120b"]["sweep"])


class TestMigrateFailsClosed(unittest.TestCase):
    """Migration must fail at conversion time on inputs that would otherwise
    silently lose a workload or emit a config that only breaks when loaded.

    These need no HF_TOKEN: migration self-validates schema-only, so the
    deferred ``${env:HF_TOKEN}`` is never resolved here."""

    def test_slug_collision_raises_not_drops(self):
        # 'gpt-oss' and 'gpt_oss' both slugify to 'gpt_oss'; pre-fix the second
        # silently overwrote the first.
        mega = {
            "config": {"nnodes": "1"},
            "benchmark_params": {
                "gpt-oss": {
                    "model": "x",
                    "concurrency_levels": [1],
                    "sequence_combinations": [{"isl": "1", "osl": "1"}],
                    "server_script": "s.sh",
                    "result_dict": {},
                },
                "gpt_oss": {
                    "model": "y",
                    "concurrency_levels": [2],
                    "sequence_combinations": [{"isl": "2", "osl": "2"}],
                    "server_script": "s.sh",
                    "result_dict": {},
                },
            },
        }
        with self.assertRaises(ValueError):
            migrate_vllm_megaconfig(mega, target_gpu="mi300")

    def test_empty_axis_rejected_at_migration_not_at_load(self):
        mega = {
            "config": {"nnodes": "1"},
            "benchmark_params": {
                "bad": {
                    "model": "z",
                    "concurrency_levels": [1],
                    "sequence_combinations": [],
                    "server_script": "s.sh",
                    "result_dict": {},
                },
            },
        }
        with self.assertRaises(ValueError):
            migrate_vllm_megaconfig(mega, target_gpu="mi300")


if __name__ == "__main__":
    unittest.main()
