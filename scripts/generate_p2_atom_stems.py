#!/usr/bin/env python3
"""One-shot generator for P2 atom perf stems (tracker #10-23)."""

from __future__ import annotations

import json
from pathlib import Path

ATOM = Path(__file__).resolve().parents[1] / "cvs/input/config_file/inference/atom"

# Kimi MXFP4 Triton MoE requires aiter shuffle_scale_moe on gfx950; no MI300X stems.
MI355X_ONLY_STEMS = frozenset({"kimi-k2.5-mxfp4"})

P2 = [
    {
        "tracker": 10,
        "stem": "deepseek-v4-pro_longctx",
        "model": "deepseek-ai/DeepSeek-V4-Pro",
        "tp": 8,
        "precision": "fp4+fp8",
        "isl": 5000,
        "osl": 1024,
        "mml": 16384,
        "combo": "v4pro_5k_1k",
        "conc": [16, 32],
    },
    {
        "tracker": 11,
        "stem": "deepseek-v4-flash",
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "tp": 4,
        "precision": "fp4+fp8",
        "isl": 1024,
        "osl": 1024,
        "mml": 4096,
        "combo": "v4flash_1k_1k",
        "conc": [128, 256],
    },
    {
        "tracker": 13,
        "stem": "glm-5.2-mxfp4",
        "model": "amd/GLM-5.2-MXFP4",
        "tp": 8,
        "precision": "mxfp4",
        "isl": 1024,
        "osl": 1024,
        "mml": 4096,
        "combo": "glm52mx_1k_1k",
        "conc": [128, 256],
    },
    {
        "tracker": 14,
        "stem": "kimi-k2.5-mxfp4",
        "model": "amd/Kimi-K2.5-MXFP4",
        "tp": 4,
        "precision": "mxfp4",
        "isl": 1024,
        "osl": 1024,
        "mml": 4096,
        "combo": "k25mx_1k_1k",
        "conc": [128, 256],
    },
    {
        "tracker": 15,
        "stem": "qwen3.5-397b-a17b",
        "model": "Qwen/Qwen3.5-397B-A17B",
        "tp": 8,
        "precision": "bf16",
        "isl": 1024,
        "osl": 1024,
        "mml": 4096,
        "combo": "qwenbf16_1k_1k",
        "conc": [128, 256],
    },
    {
        "tracker": 16,
        "stem": "glm-5.2-fp8",
        "model": "zai-org/GLM-5.2-FP8",
        "tp": 8,
        "precision": "fp8",
        "isl": 1024,
        "osl": 1024,
        "mml": 4096,
        "combo": "glm52fp8_1k_1k",
        "conc": [128, 256],
    },
    {
        "tracker": 17,
        "stem": "glm-5.2",
        "model": "zai-org/GLM-5.2",
        "tp": 8,
        "precision": "mxfp4",
        "isl": 1024,
        "osl": 1024,
        "mml": 4096,
        "combo": "glm52_1k_1k",
        "conc": [128, 256],
    },
    {
        "tracker": 19,
        "stem": "minimax-m3",
        "model": "MiniMaxAI/MiniMax-M3",
        "tp": 8,
        "precision": "bf16",
        "isl": 1024,
        "osl": 1024,
        "mml": 4096,
        "combo": "minimax_1k_1k",
        "conc": [128, 256],
    },
    {
        "tracker": 20,
        "stem": "qwen3.5-397b-a17b-mxfp4",
        "model": "amd/Qwen3.5-397B-A17B-MXFP4",
        "tp": 8,
        "precision": "mxfp4",
        "isl": 1024,
        "osl": 1024,
        "mml": 4096,
        "combo": "qwenmx_1k_1k",
        "conc": [128, 256],
    },
    {
        "tracker": 21,
        "stem": "mistral-large-3",
        "model": "mistralai/Mistral-Large-3-675B-Instruct-2512",
        "tp": 8,
        "precision": "fp8",
        "isl": 1024,
        "osl": 1024,
        "mml": 4096,
        "combo": "mistral_1k_1k",
        "conc": [128, 256],
    },
    {
        "tracker": 23,
        "stem": "mimo-v2.5-pro",
        "model": "XiaomiMiMo/MiMo-V2.5-Pro",
        "tp": 8,
        "precision": "bf16",
        "isl": 1024,
        "osl": 1024,
        "mml": 4096,
        "combo": "mimo_1k_1k",
        "conc": [128, 256],
    },
]


def make_config(gpu: str, w: dict) -> dict:
    name = f"{gpu}_atom_{w['stem']}_single"
    return {
        "_comment": f"Tracker #{w['tracker']} {w['model']} perf stem ({gpu}).",
        "schema_version": 1,
        "framework": "atom",
        "gpu_arch": gpu,
        "enforce_thresholds": False,
        "threshold_json": f"{name}_threshold.json",
        "run_card": {
            "atom_image_pin": "rocm/atom-dev:latest",
            "notes": f"P2 workload tracker #{w['tracker']}; lab pending",
        },
        "paths": {
            "shared_fs": "/home/{user-id}",
            "models_dir": "/home/models",
            "log_dir": "{shared_fs}/LOGS",
            "hf_token_file": "{shared_fs}/.hf_token",
        },
        "model": {"id": w["model"], "remote": 0, "precision": w["precision"]},
        "container": {
            "lifetime": "per_run",
            "name": f"atom_{gpu}",
            "image": "rocm/atom-dev:latest",
            "runtime": {
                "name": "docker",
                "args": {
                    "network": "host",
                    "ipc": "host",
                    "privileged": True,
                    "shm_size": "128G",
                    "volumes": [
                        "/home/{user-id}:/home/{user-id}",
                        "/home/models:/home/models",
                    ],
                    "devices": ["/dev/dri", "/dev/kfd"],
                },
            },
        },
        "roles": {
            "server": {
                "atom_args": ["-tp", str(w["tp"]), "--trust-remote-code"],
                "env": {"ATOM_DISABLE_MMAP": "true"},
            }
        },
        "params": {
            "driver": "atom",
            "port_no": "8000",
            "tensor_parallelism": str(w["tp"]),
            "random_range_ratio": "0.8",
            "num_prompts": "1000",
            "max_model_length": str(w["mml"]),
            "metric_percentiles": "95,99",
            "reuse_server_across_sweep": "true",
            "client_poll_count": "80",
            "client_poll_wait_time": "60",
        },
        "platform": {"gpu_metrics_poll": True},
        "sweep": {
            "sequence_combinations": [{"name": w["combo"], "isl": str(w["isl"]), "osl": str(w["osl"])}],
            "runs": [{"combo": w["combo"], "concurrency": c} for c in w["conc"]],
        },
    }


def make_threshold(gpu: str, w: dict) -> dict:
    cells = {
        "_comment": f"CI seed for {gpu}_atom_{w['stem']}_single",
    }
    for c in w["conc"]:
        key = f"ISL={w['isl']},OSL={w['osl']},TP={w['tp']},CONC={c}"
        cells[key] = {
            "client.output_throughput": {"kind": "min_tok_s", "value": 0},
            "client.mean_ttft_ms": {"kind": "max_ms", "value": 1000000},
            "client.mean_tpot_ms": {"kind": "max_ms", "value": 1000000},
            "client.success_rate": {"kind": "min", "value": 0},
            "client.failed": {"kind": "max", "value": 1000000},
        }
    return cells


def main() -> None:
    created = 0
    for w in P2:
        gpus = ("mi355x",) if w["stem"] in MI355X_ONLY_STEMS else ("mi300x", "mi355x")
        for gpu in gpus:
            stem = f"{gpu}_atom_{w['stem']}_single"
            (ATOM / f"{stem}.json").write_text(
                json.dumps(make_config(gpu, w), indent=2) + "\n",
                encoding="utf-8",
            )
            (ATOM / f"{stem}_threshold.json").write_text(
                json.dumps(make_threshold(gpu, w), indent=2) + "\n",
                encoding="utf-8",
            )
            created += 2
    print(f"P2 stems: {created} files")

    skip = ("vllm_single", "sglang_single")
    for path in sorted(ATOM.glob("*.json")):
        if path.name.endswith("_threshold.json"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("params", {}).get("driver", "atom") != "atom":
            continue
        if any(s in path.name for s in skip):
            continue
        if "accuracy" in path.name and "distributed" not in path.name:
            continue
        plat = data.setdefault("platform", {})
        if not plat.get("gpu_metrics_poll"):
            plat["gpu_metrics_poll"] = True
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            print("gpu poll:", path.name)


if __name__ == "__main__":
    main()
