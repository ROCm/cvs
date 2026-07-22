'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.inference.inferencex_atom.inferencex_atom_config_loader.
'''

import unittest
from pathlib import Path

from cvs.lib.inference.inferencex_atom.inferencex_atom_config_loader import (
    InferenceXAtomVariantConfig,
    expand_sweep,
    expand_sweep_parametrize,
    load_variant,
    orchestrator_container_from_variant,
    placeholder_gated_threshold_cell,
    reuse_server_flag,
    server_session_key,
)
from cvs.lib.inference.utils.inferencing_config_loader import Run, SeqCombo, Sweep


def _cluster_dict():
    return {"username": "testuser"}


class TestInferenceXAtomConfigLoader(unittest.TestCase):
    def test_load_mi300x_sample_config(self):
        root = Path(__file__).resolve().parents[3]
        config = root / (
            "input/config_file/inference/inferencex_atom/mi300x_inferencex-atom_gpt-oss-120b_bf16_config.json"
        )
        variant = load_variant(config, _cluster_dict())
        self.assertEqual(variant.framework, "inferencex_atom")
        self.assertEqual(variant.params.driver, "vllm")
        self.assertEqual(variant.expected_cells(), ["ISL=7168,OSL=1024,TP=8,CONC=64"])
        self.assertIn("enforce-eager", variant.roles.server.serve_args)

    def test_deprecated_framework_alias_normalizes(self):
        sweep = Sweep(
            sequence_combinations=[SeqCombo(name="w1", isl="1024", osl="1024")],
            runs=[Run(combo="w1", concurrency=128)],
        )
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
            model={"id": "deepseek-ai/DeepSeek-R1-0528", "remote": 0, "precision": "fp8"},
            container={
                "name": "c",
                "image": "img",
                "runtime": {"name": "docker", "args": {"volumes": ["/home/x:/home/x"]}},
            },
            params={
                "driver": "atom",
                "tensor_parallelism": "8",
                "num_prompts": "100",
            },
            sweep=sweep,
            thresholds={},
        )
        self.assertEqual(variant.framework, "inferencex_atom")

    def test_load_w1_mi300x_atom_variant(self):
        root = Path(__file__).resolve().parents[3]
        config = root / (
            "input/config_file/inference/inferencex_atom/mi300x_inferencex-atom_deepseek-r1_fp8_perf_config.json"
        )
        variant = load_variant(config, _cluster_dict())
        self.assertEqual(variant.threshold_json, "mi300x_inferencex-atom_deepseek-r1_fp8_perf_threshold.json")
        self.assertEqual(variant.gpu_arch, "mi300x")
        self.assertEqual(variant.params.driver, "atom")
        self.assertEqual(variant.params.metric_percentiles, "95,99")
        self.assertEqual(
            variant.roles.server.atom_args[:4],
            ["-tp", "8", "--kv_cache_dtype", "fp8"],
        )
        self.assertEqual(
            variant.expected_cells(),
            ["ISL=1024,OSL=1024,TP=8,CONC=128", "ISL=1024,OSL=1024,TP=8,CONC=256"],
        )
        cell = "ISL=1024,OSL=1024,TP=8,CONC=128"
        for key in (
            "client.per_gpu_throughput",
            "client.output_tput_per_gpu",
            "client.p99_ttft_ms",
            "client.p99_tpot_ms",
            "client.p95_tpot_ms",
        ):
            self.assertIn(key, variant.thresholds[cell])

    def test_load_w1_mi300x_multinode_variant(self):
        root = Path(__file__).resolve().parents[3]
        config = root / (
            "input/config_file/inference/inferencex_atom/"
            "mi300x_inferencex-atom_deepseek-r1_fp8_perf_multi_config.json"
        )
        variant = load_variant(config, _cluster_dict())
        self.assertEqual(variant.params.nnodes, "2")
        self.assertEqual(variant.params.pipeline_parallel_size, "2")
        self.assertEqual(variant.params.scaling_baseline_output_throughput, "1500")
        self.assertTrue(variant.enforce_thresholds)
        self.assertEqual(len(variant.expected_cells()), 15)
        cell = "ISL=512,OSL=512,TP=8,PP=2,NNODES=2,CONC=16"
        self.assertIn(cell, variant.expected_cells())
        self.assertEqual(
            variant.thresholds[cell]["scaling.efficiency_pct"],
            {"kind": "min", "value": 22},
        )

    def test_load_w1_mi355x_multinode_variant(self):
        root = Path(__file__).resolve().parents[3]
        config = root / (
            "input/config_file/inference/inferencex_atom/"
            "mi355x_inferencex-atom_deepseek-r1_fp8_perf_multi_config.json"
        )
        variant = load_variant(config, _cluster_dict())
        self.assertEqual(variant.gpu_arch, "mi355x")
        self.assertEqual(variant.params.nnodes, "2")
        self.assertEqual(variant.params.pipeline_parallel_size, "2")
        self.assertEqual(variant.params.scaling_baseline_output_throughput, "4000")
        self.assertFalse(variant.enforce_thresholds)
        self.assertEqual(len(variant.expected_cells()), 15)
        cell = "ISL=512,OSL=512,TP=8,PP=2,NNODES=2,CONC=16"
        self.assertIn(cell, variant.expected_cells())
        self.assertEqual(
            variant.thresholds[cell]["scaling.efficiency_pct"],
            {"kind": "min", "value": 50},
        )

    def test_load_baseline_sweep_mi300x_variant(self):
        root = Path(__file__).resolve().parents[3]
        config = root / (
            "input/config_file/inference/inferencex_atom/"
            "mi300x_inferencex-atom_deepseek-r1_fp8_perf_baseline_sweep_config.json"
        )
        variant = load_variant(config, _cluster_dict())
        self.assertEqual(variant.params.max_model_length, "10240")
        self.assertTrue(variant.enforce_thresholds)
        self.assertEqual(len(variant.expected_cells()), 14)
        self.assertIn("ISL=1024,OSL=1024,TP=8,CONC=4", variant.expected_cells())
        self.assertIn("ISL=8192,OSL=1024,TP=8,CONC=256", variant.expected_cells())
        cell = "ISL=8192,OSL=1024,TP=8,CONC=128"
        self.assertIn("client.output_throughput", variant.thresholds[cell])
        self.assertEqual(variant.thresholds[cell]["client.success_rate"]["value"], 1)

    def test_load_baseline_sweep_multinode_mi300x_variant(self):
        root = Path(__file__).resolve().parents[3]
        config = root / (
            "input/config_file/inference/inferencex_atom/"
            "mi300x_inferencex-atom_deepseek-r1_fp8_perf_baseline_sweep_multinode_config.json"
        )
        variant = load_variant(config, _cluster_dict())
        self.assertEqual(variant.params.nnodes, "2")
        self.assertEqual(variant.params.pipeline_parallel_size, "2")
        self.assertEqual(variant.params.max_model_length, "10240")
        self.assertEqual(variant.params.scaling_baseline_output_throughput, "1500")
        self.assertTrue(variant.enforce_thresholds)
        self.assertEqual(len(variant.expected_cells()), 14)
        cell = "ISL=1024,OSL=1024,TP=8,PP=2,NNODES=2,CONC=4"
        self.assertIn(cell, variant.expected_cells())
        self.assertEqual(
            variant.thresholds[cell]["scaling.efficiency_pct"],
            {"kind": "min", "value": 50},
        )

    def test_load_baseline_sweep_mi355x_variant(self):
        root = Path(__file__).resolve().parents[3]
        config = root / (
            "input/config_file/inference/inferencex_atom/"
            "mi355x_inferencex-atom_deepseek-r1_fp8_perf_baseline_sweep_config.json"
        )
        variant = load_variant(config, _cluster_dict())
        self.assertEqual(variant.gpu_arch, "mi355x")
        self.assertFalse(variant.enforce_thresholds)
        self.assertEqual(len(variant.expected_cells()), 14)

    def test_load_w1_mi300x_smoke_variant(self):
        root = Path(__file__).resolve().parents[3]
        config = root / (
            "input/config_file/inference/inferencex_atom/"
            "mi300x_inferencex-atom_deepseek-r1_fp8_smoke_config.json"
        )
        variant = load_variant(config, _cluster_dict())
        self.assertEqual(variant.params.num_prompts, "128")
        self.assertEqual(variant.paths.models_dir, "/home/models")
        self.assertIn("/home/models:/home/models", variant.container.runtime.args["volumes"])
        self.assertEqual(variant.expected_cells(), ["ISL=1024,OSL=1024,TP=8,CONC=128"])

    def test_load_w1_mi355x_atom_perf_variant_and_thresholds(self):
        root = Path(__file__).resolve().parents[3]
        config = root / (
            "input/config_file/inference/inferencex_atom/mi355x_inferencex-atom_deepseek-r1_fp8_perf_config.json"
        )
        variant = load_variant(config, _cluster_dict())
        self.assertEqual(variant.gpu_arch, "mi355x")
        self.assertIn("--trust-remote-code", variant.roles.server.atom_args)
        self.assertEqual(
            variant.expected_cells(),
            ["ISL=1024,OSL=1024,TP=8,CONC=128", "ISL=1024,OSL=1024,TP=8,CONC=256"],
        )
        cell = "ISL=1024,OSL=1024,TP=8,CONC=128"
        self.assertEqual(
            variant.thresholds[cell]["client.output_throughput"]["value"],
            4004.66,
        )
        self.assertEqual(
            variant.thresholds[cell]["client.mean_ttft_ms"]["value"],
            362.18,
        )

    def test_load_w1_mi355x_atom_mtp3_inline_bench_args(self):
        root = Path(__file__).resolve().parents[3]
        config = root / (
            "input/config_file/inference/inferencex_atom/mi355x_inferencex-atom_deepseek-r1_fp8_mtp3_config.json"
        )
        variant = load_variant(config, _cluster_dict())
        self.assertIn("--method", variant.roles.server.atom_args)
        self.assertEqual(variant.params.bench_extra_args, "--use-chat-template")

    def test_load_w1_mi355x_atom_mtp3_thresholds(self):
        root = Path(__file__).resolve().parents[3]
        config = root / (
            "input/config_file/inference/inferencex_atom/mi355x_inferencex-atom_deepseek-r1_fp8_mtp3_config.json"
        )
        variant = load_variant(config, _cluster_dict())
        cell = "ISL=1024,OSL=1024,TP=8,CONC=256"
        self.assertEqual(
            variant.thresholds[cell]["client.output_throughput"]["value"],
            6451.59,
        )

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
            framework="inferencex_atom",
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

    def test_expand_sweep_matches_w1_perf(self):
        root = Path(__file__).resolve().parents[3]
        config = root / (
            "input/config_file/inference/inferencex_atom/mi300x_inferencex-atom_deepseek-r1_fp8_perf_config.json"
        )
        import json

        raw = json.loads(config.read_text())
        cases, ids = expand_sweep(raw["sweep"])
        self.assertEqual(len(cases), 2)
        self.assertEqual(ids[0], "w1_1k_1k-conc128")
        self.assertEqual(ids[1], "w1_1k_1k-conc256")
        self.assertEqual(cases[0][1], 128)

    def test_w1_perf_threshold_health_gates_tight_when_enforcing(self):
        root = Path(__file__).resolve().parents[3]
        config = root / (
            "input/config_file/inference/inferencex_atom/mi300x_inferencex-atom_deepseek-r1_fp8_perf_config.json"
        )
        variant = load_variant(config, _cluster_dict())
        self.assertTrue(variant.enforce_thresholds)
        cell = "ISL=1024,OSL=1024,TP=8,CONC=128"
        self.assertEqual(variant.thresholds[cell]["client.success_rate"]["value"], 1)
        self.assertEqual(variant.thresholds[cell]["client.failed"]["value"], 0)

    def test_placeholder_threshold_cell_covers_gated_metrics(self):
        cell = placeholder_gated_threshold_cell()
        from cvs.lib.inference.inferencex_atom.inferencex_atom_parsing import GATED_METRICS

        for short in GATED_METRICS:
            self.assertIn(f"client.{short}", cell, short)

    def test_atom_driver_requires_inline_atom_args(self):
        sweep = Sweep(
            sequence_combinations=[SeqCombo(name="w1", isl="1024", osl="1024")],
            runs=[Run(combo="w1", concurrency=128)],
        )
        thresholds = {"ISL=1024,OSL=1024,TP=8,CONC=128": placeholder_gated_threshold_cell()}
        with self.assertRaises(ValueError):
            InferenceXAtomVariantConfig(
                schema_version=1,
                framework="inferencex_atom",
                gpu_arch="mi300x",
                enforce_thresholds=False,
                paths={
                    "shared_fs": "/home/x",
                    "models_dir": "/home/x/models",
                    "log_dir": "/home/x/LOGS",
                    "hf_token_file": "/home/x/.hf",
                },
                model={"id": "deepseek-ai/DeepSeek-R1-0528", "remote": 0, "precision": "fp8"},
                container={
                    "name": "c",
                    "image": "img",
                    "runtime": {"name": "docker", "args": {"volumes": ["/home/x:/home/x"]}},
                },
                roles={"server": {"env": {}}},
                params={"driver": "atom", "tensor_parallelism": "8"},
                sweep=sweep,
                thresholds=thresholds,
            )

    def test_reuse_server_flag_and_session_key_helpers(self):
        from types import SimpleNamespace

        self.assertFalse(reuse_server_flag(SimpleNamespace()))
        variant = SimpleNamespace(
            model=SimpleNamespace(id="m"),
            params=SimpleNamespace(driver="atom", tensor_parallelism="8"),
            roles=SimpleNamespace(server=SimpleNamespace(atom_args=("-tp", "8"))),
        )
        self.assertNotEqual(server_session_key(variant, "1", "2"), server_session_key(variant, "3", "4"))

    def test_expand_sweep_parametrize_tier_ids(self):
        sweep = {
            "sequence_combinations": [{"name": "w1", "isl": "1024", "osl": "1024"}],
            "runs": [{"combo": "w1", "concurrency": 128}],
        }
        _, _, ids = expand_sweep_parametrize(sweep, ("metric_tier",))
        self.assertIn("w1-conc128-throughput", ids)


if __name__ == "__main__":
    unittest.main()
