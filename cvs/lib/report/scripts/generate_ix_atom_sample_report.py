#!/usr/bin/env python3
'''Generate sample IX Run Deck HTML from synthetic data (no lab run required).'''

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Mapping, Tuple

from cvs.lib.report.inference import write_report
from cvs.lib.report.presets.inferencex_atom import INFERENCEX_ATOM_REPORT_CONFIG

_CONFIG_ROOT = (
    Path(__file__).resolve().parents[3]
    / "input/config_file/inference/inferencex_atom_single"
)

SWEEP_CONCS = (64, 128, 256, 512)

VariantBuild = Tuple[
    Callable[[], Any],
    Callable[[], Dict[tuple, Any]],
    Callable[[], Mapping[str, list]],
    Dict[str, str],
]


def _load_thresholds(filename: str) -> dict:
    raw = json.loads((_CONFIG_ROOT / filename).read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not str(k).startswith("_")}


def _base_metrics(*, tput_scale: float, ttft_scale: float = 1.0) -> dict:
    return {
        "client.total_token_throughput": round(2400 * tput_scale, 2),
        "client.output_throughput": round(1200 * tput_scale, 2),
        "client.per_gpu_throughput": round(300 * tput_scale, 2),
        "client.output_tput_per_gpu": round(150 * tput_scale, 2),
        "client.mean_ttft_ms": round(680 * ttft_scale, 2),
        "client.median_ttft_ms": round(610, 2),
        "client.p90_ttft_ms": round(1240, 2),
        "client.p95_ttft_ms": round(1580, 2),
        "client.p99_ttft_ms": round(2100, 2),
        "client.mean_tpot_ms": round(42.5 * ttft_scale, 2),
        "client.median_tpot_ms": round(40.8, 2),
        "client.p90_tpot_ms": round(51.2, 2),
        "client.p95_tpot_ms": round(58.6, 2),
        "client.p99_tpot_ms": round(72.1, 2),
        "client.success_rate": 1.0,
        "client.failed": 0,
    }


def _w1_variant():
    return SimpleNamespace(
        model=SimpleNamespace(id="deepseek-ai/DeepSeek-R1-0528"),
        gpu_arch="mi300x",
        enforce_thresholds=True,
        thresholds=_load_thresholds(
            "mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_threshold.json"
        ),
        cell_key=lambda isl, osl, conc: f"ISL={isl},OSL={osl},TP=8,CONC={conc}",
        params=SimpleNamespace(driver="atom", tensor_parallelism="8", nnodes=1),
        run_card=SimpleNamespace(
            atom_image_pin="rocm/atom-dev:latest",
            upstream_run_url="",
            notes="Sample W1 perf data (not a lab run)",
        ),
    )


def _w1_inf_res():
    model = "deepseek-ai/DeepSeek-R1-0528"
    gpu, host = "mi300x", "10.245.135.75"
    scales = {64: 0.85, 128: 1.0, 256: 1.18, 512: 1.35}
    out = {}
    for conc in SWEEP_CONCS:
        scale = scales[conc]
        key = (model, gpu, "1024", "1024", "w1_1k_1k", conc)
        out[key] = {
            host: _base_metrics(
                tput_scale=scale * 1.3,
                ttft_scale=1.0 if conc <= 128 else 1.05,
            )
        }
    return out


def _inference_lifecycle(
    base: str,
    combo: str,
    *,
    first_server_ready: float = 312.4,
    later_server_ready: float = 4.2,
    base_client_complete: float = 1842.7,
    client_step: float = 120.0,
) -> dict:
    lifecycle = {}
    for i, conc in enumerate(SWEEP_CONCS):
        lifecycle[f"{base}::test_inferencex_atom_inference[{combo}-conc{conc}]"] = [
            ("server_ready", first_server_ready if i == 0 else later_server_ready, "s"),
            ("client_complete", base_client_complete + i * client_step, "s"),
        ]
    return lifecycle


def _w1_lifecycle():
    base = "cvs/tests/inference/inferencex_atom/inferencex_atom_single.py"
    return {
        f"{base}::test_launch_container": [("container_launch", 42.3, "s")],
        f"{base}::test_setup_sshd": [("sshd_setup", 8.1, "s")],
        f"{base}::test_model_fetch": [("model_fetch", 15.6, "s"), ("model_size", 685.2, "GB")],
        **_inference_lifecycle(base, "w1_1k_1k"),
        f"{base}::test_teardown": [("teardown", 6.8, "s")],
    }


VARIANTS: Dict[str, VariantBuild] = {
    "w1": (
        _w1_variant,
        _w1_inf_res,
        _w1_lifecycle,
        {
            "cluster_file": "mi300x_atom_single.json",
            "config_file": "mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json",
            "generated": "synthetic W1 DeepSeek R1 FP8 perf demo",
        },
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="w1",
        help="Which sample workload to simulate (default: w1)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: sample_reports/<variant>)",
    )
    args = parser.parse_args()
    out_dir = (args.output_dir or Path("sample_reports") / args.variant).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    build_variant, build_inf_res, build_lifecycle, provenance = VARIANTS[args.variant]
    config = INFERENCEX_ATOM_REPORT_CONFIG

    artifacts = write_report(
        out_dir / f"{config.report_basename}.html",
        config=config,
        variant_config=build_variant(),
        inf_res_dict=build_inf_res(),
        lifecycle_report=build_lifecycle(),
        cvs_version="sample",
        pytest_html_path=str(out_dir / f"sample_{args.variant}_pytest_results.html"),
        log_file_path=str(out_dir / f"sample_{args.variant}_run.log"),
        provenance=provenance,
    )
    print(f"Variant: {args.variant}")
    print(f"HTML:    {artifacts['html']}")
    print(f"JSON:    {artifacts['json']}")
    if artifacts.get("viewer"):
        print(f"Viewer:  {artifacts['viewer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
