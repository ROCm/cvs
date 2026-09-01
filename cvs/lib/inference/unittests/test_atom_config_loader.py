'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.inference.atom.atom_config_loader.
'''

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from cvs.lib.inference.atom.atom_config_loader import (
    AtomVariantConfig,
    expand_sweep,
    expand_sweep_parametrize,
    load_variant,
    orchestrator_container_from_variant,
    placeholder_gated_threshold_cell,
    resolve_atom_profile,
    reuse_server_flag,
    server_session_key,
)
from cvs.lib.inference.utils.inferencing_config_loader import Run, SeqCombo, Sweep


def _cluster_dict():
    return {"username": "testuser"}


def _atom_config(root: Path, name: str, profile: str | None = None):
    return load_variant(root / f"input/config_file/inference/atom/{name}", _cluster_dict(), profile=profile)


class TestATOMAtomConfigLoader(unittest.TestCase):
    def test_load_mi3xx_sample_config(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_gpt-oss-120b_mxfp4_single.json", profile="vllm")
        self.assertEqual(variant.framework, "atom")
        self.assertEqual(variant.params.driver, "vllm_atom")
        self.assertEqual(
            variant.expected_cells(), ["ISL=8192,OSL=1024,TP=4,PP=1,CONC=32", "ISL=8192,OSL=1024,TP=4,PP=1,CONC=64"]
        )
        self.assertIn("enforce-eager", variant.roles.server.serve_args)

    def test_load_w1_mi3xx_atom_variant(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_deepseek-r1_fp8_single.json", profile="perf")
        self.assertEqual(variant.threshold_json, "mi325x_atom_deepseek-r1_fp8_threshold.json")
        self.assertEqual(variant.gpu_arch, "mi3xx")
        self.assertEqual(variant.params.driver, "atom")
        self.assertEqual(variant.params.metric_percentiles, "95,99")
        self.assertEqual(
            variant.roles.server.atom_args[:4],
            ["-tp", "8", "--kv_cache_dtype", "fp8"],
        )
        self.assertEqual(
            variant.expected_cells(),
            ["ISL=1024,OSL=1024,TP=8,PP=1,CONC=128", "ISL=1024,OSL=1024,TP=8,PP=1,CONC=256"],
        )
        cell = "ISL=1024,OSL=1024,TP=8,PP=1,CONC=128"
        for key in (
            "per_gpu_throughput",
            "output_tput_per_gpu",
            "p99_ttft_ms",
            "p99_tpot_ms",
            "p95_tpot_ms",
        ):
            self.assertIn(key, variant.thresholds[cell])

    def test_load_w1_mi3xx_multinode_variant(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_deepseek-r1_fp8_distributed.json", profile="perf")
        self.assertEqual(variant.params.nnodes, "2")
        self.assertEqual(variant.params.driver, "vllm_atom")
        self.assertEqual(variant.params.pipeline_parallel_size, "2")
        self.assertEqual(variant.roles.server.ib_netdev, "auto")
        self.assertEqual(variant.roles.server.ib_hca_devices, "auto")
        self.assertEqual(variant.params.scaling_baseline_output_throughput, "1500")
        self.assertTrue(variant.enforce_thresholds)
        self.assertEqual(len(variant.expected_cells()), 15)
        cell = "ISL=512,OSL=512,TP=8,PP=2,CONC=16"
        self.assertIn(cell, variant.expected_cells())
        self.assertEqual(
            variant.thresholds[cell]["scaling.efficiency_pct"],
            {"kind": "min", "value": 11},
        )

    def test_load_w1_mi3xx_multinode_sglang_variant(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_deepseek-r1_fp8_distributed.json", profile="sglang")
        self.assertEqual(variant.params.driver, "sglang")
        self.assertEqual(variant.params.pipeline_parallel_size, "2")
        self.assertFalse(variant.enforce_thresholds)
        cell = "ISL=512,OSL=512,TP=8,PP=2,CONC=16"
        self.assertIn(cell, variant.expected_cells())

    def test_load_baseline_sweep_mi3xx_variant(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_deepseek-r1_fp8_single.json", profile="baseline_sweep")
        self.assertEqual(variant.params.max_model_length, "10240")
        self.assertTrue(variant.enforce_thresholds)
        self.assertEqual(len(variant.expected_cells()), 14)
        self.assertIn("ISL=1024,OSL=1024,TP=8,PP=1,CONC=4", variant.expected_cells())
        self.assertIn("ISL=8192,OSL=1024,TP=8,PP=1,CONC=256", variant.expected_cells())
        cell = "ISL=8192,OSL=1024,TP=8,PP=1,CONC=128"
        self.assertIn("output_throughput", variant.thresholds[cell])
        self.assertEqual(variant.thresholds[cell]["success_rate"]["value"], 1)

    def test_load_w1_mi3xx_atom_mtp3_inline_bench_args(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_deepseek-r1_fp8_single.json", profile="mtp3")
        self.assertIn("--method", variant.roles.server.atom_args)
        self.assertEqual(variant.params.bench_extra_args, "--use-chat-template")

    def test_orchestrator_container_includes_server_env(self):
        sweep = Sweep(
            sequence_combinations=[SeqCombo(name="legacy_profile", isl="7168", osl="1024")],
            runs=[Run(combo="legacy_profile", concurrency=64)],
        )
        thresholds = {
            "ISL=7168,OSL=1024,TP=8,PP=1,CONC=64": placeholder_gated_threshold_cell(),
        }
        variant = AtomVariantConfig(
            schema_version=1,
            framework="atom",
            gpu_arch="mi3xx",
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

    def test_expand_sweep_matches_w1_single(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_deepseek-r1_fp8_single.json", profile="perf")
        cases, ids = expand_sweep(variant.sweep)
        self.assertEqual(len(cases), 2)
        self.assertEqual(ids[0], "w1_1k_1k-conc128")
        self.assertEqual(ids[1], "w1_1k_1k-conc256")
        self.assertEqual(cases[0][1], 128)

    def test_w1_single_threshold_health_gates_tight_when_enforcing(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_deepseek-r1_fp8_single.json", profile="perf")
        self.assertTrue(variant.enforce_thresholds)
        cell = "ISL=1024,OSL=1024,TP=8,PP=1,CONC=128"
        self.assertEqual(variant.thresholds[cell]["success_rate"]["value"], 1)
        self.assertEqual(variant.thresholds[cell]["failed"]["value"], 0)

    def test_placeholder_threshold_cell_covers_gated_metrics(self):
        cell = placeholder_gated_threshold_cell()
        from cvs.lib.inference.atom.atom_parsing import GATED_METRICS

        for short in GATED_METRICS:
            self.assertIn(short, cell, short)

    def test_atom_driver_requires_inline_atom_args(self):
        sweep = Sweep(
            sequence_combinations=[SeqCombo(name="w1", isl="1024", osl="1024")],
            runs=[Run(combo="w1", concurrency=128)],
        )
        thresholds = {"ISL=1024,OSL=1024,TP=8,PP=1,CONC=128": placeholder_gated_threshold_cell()}
        with self.assertRaises(ValueError):
            AtomVariantConfig(
                schema_version=1,
                framework="atom",
                gpu_arch="mi3xx",
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
            params=SimpleNamespace(
                driver="atom",
                tensor_parallelism="8",
                nnodes="1",
                pipeline_parallel_size="1",
                master_addr="",
                master_port="29501",
            ),
            roles=SimpleNamespace(server=SimpleNamespace(atom_args=("-tp", "8"), serve_args={}, sglang_args=[])),
        )
        self.assertNotEqual(server_session_key(variant, "1", "2"), server_session_key(variant, "3", "4"))

    def test_expand_sweep_parametrize_tier_ids(self):
        sweep = {
            "sequence_combinations": [{"name": "w1", "isl": "1024", "osl": "1024"}],
            "runs": [{"combo": "w1", "concurrency": 128}],
        }
        _, _, ids = expand_sweep_parametrize(sweep, ("metric_tier",))
        self.assertIn("w1-conc128-throughput", ids)

    def test_ib_netdev_coerces_mlx5_hca_name_to_auto(self):
        sweep = Sweep(
            sequence_combinations=[SeqCombo(name="w1", isl="512", osl="512")],
            runs=[Run(combo="w1", concurrency=16)],
        )
        variant = AtomVariantConfig(
            schema_version=1,
            framework="atom",
            gpu_arch="mi3xx",
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
            roles={"server": {"ib_netdev": "mlx5_0"}},
            params={
                "driver": "vllm_atom",
                "tensor_parallelism": "8",
                "pipeline_parallel_size": "2",
                "nnodes": "2",
                "master_addr": "10.0.0.1",
            },
            sweep=sweep,
            thresholds={},
        )
        self.assertEqual(variant.roles.server.ib_netdev, "auto")

    def test_server_env_strips_orchestrator_network_keys(self):
        sweep = Sweep(
            sequence_combinations=[SeqCombo(name="w1", isl="512", osl="512")],
            runs=[Run(combo="w1", concurrency=16)],
        )
        variant = AtomVariantConfig(
            schema_version=1,
            framework="atom",
            gpu_arch="mi3xx",
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
            roles={
                "server": {
                    "env": {
                        "GLOO_SOCKET_IFNAME": "mlx5_0",
                        "NCCL_IB_HCA": "mlx5_0",
                        "NCCL_IB_GID_INDEX": "1",
                    }
                }
            },
            params={
                "driver": "vllm_atom",
                "tensor_parallelism": "8",
                "pipeline_parallel_size": "2",
                "nnodes": "2",
                "master_addr": "10.0.0.1",
            },
            sweep=sweep,
            thresholds={},
        )
        self.assertEqual(variant.roles.server.env, {"NCCL_IB_GID_INDEX": "1"})

    def test_load_w1_accuracy_variant(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_deepseek-r1_fp8_single.json", profile="accuracy")
        self.assertEqual(len(variant.accuracy.tasks), 9)
        self.assertEqual(variant.accuracy.tasks[0].id, "gsm8k_flex")
        task_ids = {t.id for t in variant.accuracy.tasks}
        self.assertIn("hellaswag", task_ids)
        self.assertIn("mmlu_pro", task_ids)
        self.assertIn("bbh", task_ids)
        self.assertIn("arc_challenge", task_ids)
        self.assertTrue(variant.quant_parity.enabled)
        self.assertTrue(variant.enforce_thresholds)
        self.assertIn(
            "gsm8k.exact_match__flexible-extract",
            variant.thresholds["accuracy"]["gsm8k_flex"],
        )
        self.assertIn(
            "hellaswag.acc_norm__none",
            variant.thresholds["accuracy"]["hellaswag"],
        )
        self.assertIn(
            "mmlu_pro.exact_match__custom-extract",
            variant.thresholds["accuracy"]["mmlu_pro"],
        )
        self.assertIn("quant_parity", variant.thresholds)

    def test_load_mtp3_variant_mtp_quality_enabled(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_deepseek-r1_fp8_single.json", profile="mtp3")
        self.assertTrue(variant.mtp_quality.enabled)
        self.assertIn("mtp.acceptance_rate", variant.thresholds["mtp_quality"])

    def test_mtp_quality_threshold_key_not_sweep_cell(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_deepseek-r1_fp8_single.json", profile="mtp3")
        self.assertIn("mtp_quality", variant.thresholds)
        self.assertEqual(len(variant.expected_cells()), 2)

    def test_load_w2_accuracy_long_context_cells(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_gpt-oss-120b_mxfp4_single.json", profile="accuracy")
        self.assertTrue(variant.functional.api_smoke)
        self.assertEqual(len(variant.long_context_accuracy.cells), 1)
        self.assertEqual(variant.long_context_accuracy.cells[0].id, "niah_8k")
        self.assertIn("long_context_accuracy", variant.thresholds)

    def test_load_phase_c_w2_mxfp4_perf(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_gpt-oss-120b_mxfp4_single.json", profile="perf")
        self.assertEqual(variant.params.driver, "vllm_atom")
        self.assertEqual(variant.params.tensor_parallelism, "4")
        self.assertEqual(variant.model.precision, "mxfp4")
        self.assertEqual(variant.roles.server.env.get("ATOM_USE_TRITON_MOE"), "1")
        self.assertEqual(variant.roles.server.env.get("ATOM_USE_TRITON_GEMM"), "1")
        self.assertEqual(
            variant.expected_cells(),
            ["ISL=8192,OSL=1024,TP=4,PP=1,CONC=32", "ISL=8192,OSL=1024,TP=4,PP=1,CONC=64"],
        )

    def test_load_w2_native_profile(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_gpt-oss-120b_mxfp4_single.json", profile="native")
        self.assertEqual(variant.params.driver, "atom")

    def test_w2_perf_atom_fallback_to_vllm(self):
        raw = {
            "schema_version": 2,
            "framework": "atom",
            "gpu_arch": "mi3xx",
            "default_profile": "perf",
            "model": {"id": "openai/gpt-oss-120b", "remote": 0, "precision": "mxfp4"},
            "profiles": {
                "perf": {
                    "params": {"driver": "atom", "tensor_parallelism": "4"},
                    "sweep": {"sequence_combinations": [], "runs": []},
                },
                "vllm": {
                    "params": {"driver": "vllm_atom", "tensor_parallelism": "4"},
                    "roles": {"server": {"serve_args": {"enforce-eager": True}}},
                    "sweep": {"sequence_combinations": [], "runs": []},
                },
            },
        }
        merged, _, selected = resolve_atom_profile(raw, {"profiles": {"perf": {}}}, "perf")
        self.assertEqual(selected, "perf")
        self.assertEqual(merged["params"]["driver"], "vllm_atom")

    def test_load_phase_c_w3_glm_perf(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_glm-5.1_single.json")
        self.assertEqual(variant.model.id, "zai-org/GLM-5.1")
        self.assertEqual(
            variant.expected_cells(),
            ["ISL=1024,OSL=8192,TP=8,PP=1,CONC=32", "ISL=1024,OSL=8192,TP=8,PP=1,CONC=64"],
        )

    def test_load_phase_c_w17_mxfp4_perf(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_deepseek-r1_mxfp4_single.json")
        self.assertEqual(variant.model.id, "amd/DeepSeek-R1-0528-MXFP4")
        self.assertEqual(variant.params.tensor_parallelism, "8")

    def test_load_m4_vllm_single_parity(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_deepseek-r1_fp8_single.json", profile="vllm")
        self.assertEqual(variant.params.driver, "vllm_atom")
        self.assertEqual(variant.params.nnodes, "1")
        self.assertIn("kv-cache-dtype", variant.roles.server.serve_args)

    def test_load_m4_sglang_single_parity(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_deepseek-r1_fp8_single.json", profile="sglang")
        self.assertEqual(variant.params.driver, "sglang")
        self.assertIn("--kv-cache-dtype", variant.roles.server.sglang_args)

    def test_load_qwen397b_fp8_single(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_qwen3.5-397b-a17b_fp8_single.json")
        self.assertEqual(variant.model.id, "amd/Qwen3.5-397B-A17B-FP8")
        self.assertEqual(variant.expected_cells()[0], "ISL=1024,OSL=8192,TP=8,PP=1,CONC=32")

    def test_load_w1_single_gpu_metrics_poll(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_deepseek-r1_fp8_single.json", profile="perf")
        self.assertTrue(variant.platform.gpu_metrics_poll)

    def test_w1_gsm8k_threshold_fails_below_floor(self):
        from cvs.lib.utils.verdict import ThresholdViolation, evaluate_all

        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_deepseek-r1_fp8_single.json", profile="accuracy")
        specs = variant.thresholds["accuracy"]["gsm8k_flex"]
        with self.assertRaises(ThresholdViolation):
            evaluate_all({"gsm8k.exact_match__flexible-extract": 0.90}, specs)

    def test_load_distributed_accuracy_scaffold(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_deepseek-r1_fp8_distributed.json", profile="accuracy")
        self.assertEqual(variant.params.driver, "vllm_atom")
        self.assertEqual(variant.params.nnodes, "2")
        self.assertIn("PP=2", variant.expected_cells()[0])
        self.assertIn("accuracy", variant.thresholds)

    def test_load_w2_m4_vllm_parity(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_gpt-oss-120b_mxfp4_single.json", profile="vllm")
        self.assertEqual(variant.params.driver, "vllm_atom")
        self.assertEqual(variant.model.id, "openai/gpt-oss-120b")

    def test_resolve_profile_merges_shared_fields(self):
        raw = {
            "schema_version": 2,
            "framework": "atom",
            "gpu_arch": "mi3xx",
            "default_profile": "perf",
            "threshold_json": "t.json",
            "paths": {"shared_fs": "/home/x", "models_dir": "/m", "log_dir": "/l", "hf_token_file": "/h"},
            "model": {"id": "m", "remote": 0, "precision": "fp8"},
            "profiles": {
                "perf": {
                    "params": {"driver": "atom", "tensor_parallelism": "8"},
                    "roles": {"server": {"atom_args": ["-tp", "8"]}},
                },
            },
        }
        thresholds = {"profiles": {"perf": {"ISL=1,OSL=1,TP=8,PP=1,CONC=1": {}}}}
        merged, th, name = resolve_atom_profile(raw, thresholds, "perf")
        self.assertEqual(name, "perf")
        self.assertEqual(merged["schema_version"], 1)
        self.assertEqual(merged["model"]["id"], "m")
        self.assertEqual(merged["params"]["driver"], "atom")
        self.assertIn("ISL=1,OSL=1,TP=8,PP=1,CONC=1", th)

    def test_unknown_profile_raises(self):
        raw = {
            "schema_version": 2,
            "framework": "atom",
            "gpu_arch": "mi3xx",
            "default_profile": "perf",
            "paths": {"shared_fs": "/home/x", "models_dir": "/m", "log_dir": "/l", "hf_token_file": "/h"},
            "model": {"id": "m", "remote": 0},
            "profiles": {
                "perf": {
                    "params": {"driver": "atom", "tensor_parallelism": "8"},
                    "roles": {"server": {"atom_args": ["-tp", "8"]}},
                }
            },
        }
        with self.assertRaises(ValueError):
            resolve_atom_profile(raw, {}, "missing")

    def test_legacy_flat_config_unchanged(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_glm-5.1_single.json")
        self.assertEqual(variant.schema_version, 1)
        self.assertEqual(variant.threshold_json, "mi325x_atom_glm-5.1_threshold.json")

    def test_flat_config_slices_profiled_threshold(self):
        root = Path(__file__).resolve().parents[3]
        variant = _atom_config(root, "mi3xx_atom_deepseek-r1_fp8_distributed.json", profile="perf")
        cell = "ISL=512,OSL=512,TP=8,PP=2,CONC=16"
        self.assertIn(cell, variant.expected_cells())
        self.assertIn("scaling.efficiency_pct", variant.thresholds[cell])

    def test_atom_threshold_files_use_aligned_keys_and_bare_metrics(self):
        root = Path(__file__).resolve().parents[3]
        atom_dir = root / "input/config_file/inference/atom"
        cell_no_pp = re.compile(r"^ISL=.*,TP=\d+,CONC=")
        config_platform_stem = re.compile(r"^mi325x_|^mi35x_|^mi300x_|^mi355x_")
        threshold_family_stem = re.compile(r"^mi3xx_|^mi35x_|^mi300x_|^mi355x_")
        for path in sorted(atom_dir.glob("*.json")):
            if "threshold" in path.name:
                self.assertFalse(
                    threshold_family_stem.match(path.name),
                    f"threshold must use platform stem, not family: {path.name}",
                )
                self.assertTrue(
                    path.name.startswith("mi325x_"),
                    f"shipped thresholds are mi325x-only: {path.name}",
                )
            else:
                self.assertFalse(
                    config_platform_stem.match(path.name),
                    f"config must use family stem mi3xx, not platform: {path.name}",
                )
                self.assertTrue(
                    path.name.startswith("mi3xx_"),
                    f"shipped configs use mi3xx family stem: {path.name}",
                )
        for path in sorted(atom_dir.glob("*threshold*.json")):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("client.", text, path.name)
            self.assertNotIn("NNODES=", text, path.name)
            data = json.loads(text)

            def walk(obj, prefix=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(key, str) and key.startswith("ISL="):
                            self.assertIn(",PP=", key, f"{path.name}: {key}")
                            self.assertNotRegex(key, r",NNODES=", f"{path.name}: {key}")
                            self.assertFalse(cell_no_pp.match(key), f"{path.name}: {key}")
                        if isinstance(value, dict) and key.startswith("ISL="):
                            for metric in value:
                                self.assertFalse(
                                    metric.startswith("client."),
                                    f"{path.name} {key}: {metric}",
                                )
                        walk(value, prefix)
                elif isinstance(obj, list):
                    for item in obj:
                        walk(item, prefix)

            walk(data)

    def test_all_atom_configs_load_with_aligned_thresholds(self):
        root = Path(__file__).resolve().parents[3]
        atom_dir = root / "input/config_file/inference/atom"
        cluster = _cluster_dict()
        for cfg in sorted(atom_dir.glob("*.json")):
            if "threshold" in cfg.name:
                continue
            variant = load_variant(cfg, cluster)
            if not variant.threshold_json:
                continue
            for cell in variant.expected_cells():
                self.assertIn(
                    cell,
                    variant.thresholds,
                    f"{cfg.name}: missing threshold cell {cell!r}",
                )


if __name__ == "__main__":
    unittest.main()
