'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Shared helpers and ordering for the SGLang disaggregated inference suite.
'''

from __future__ import annotations

import os
from tabulate import tabulate
from typing import Any, Mapping

from cvs.lib import globals


log = globals.log

__all__ = [
    "resolve_benchmark_variant_key",
    "SGLANG_DISAGG_TEST_ORDER",
    "test_print_results_table",
]


def resolve_benchmark_variant_key(root: Mapping[str, Any], config_path: str) -> str:
    """Pick which ``benchmark_params`` entry to run.

    Resolution order:
    1. Environment ``SGLANG_BENCHMARK_KEY`` (override for CI matrices).
    2. Top-level JSON ``active_benchmark`` (string key into ``benchmark_params``).
    3. If ``benchmark_params`` has exactly one key, use it.

    ``root`` is the full JSON object loaded from ``--config_file`` (not only ``config``).
    """
    env_key = (os.environ.get("SGLANG_BENCHMARK_KEY") or "").strip()
    bp = root.get("benchmark_params") or {}
    if not isinstance(bp, dict) or not bp:
        raise ValueError(f"benchmark_params missing or empty in {config_path!r}")

    if env_key:
        if env_key not in bp:
            raise ValueError(
                f"SGLANG_BENCHMARK_KEY={env_key!r} not found in benchmark_params "
                f"({config_path}); valid: {sorted(bp)!r}"
            )
        log.info("Using benchmark variant from env SGLANG_BENCHMARK_KEY=%r", env_key)
        return env_key

    explicit = root.get("active_benchmark")
    if explicit is not None:
        if explicit not in bp:
            raise ValueError(
                f"active_benchmark={explicit!r} not found in benchmark_params "
                f"({config_path}); valid: {sorted(bp)!r}"
            )
        log.info("Using benchmark variant from active_benchmark=%r", explicit)
        return str(explicit)

    if len(bp) == 1:
        only = next(iter(bp))
        log.info("Single benchmark_params entry; using %r", only)
        return str(only)

    raise ValueError(
        f"Multiple benchmark_params keys in {config_path!r}: {sorted(bp)!r}. "
        "Set top-level \"active_benchmark\" to one of them, or export SGLANG_BENCHMARK_KEY."
    )


# Stable test order (definition order already matches; this guards against drift / imports).
SGLANG_DISAGG_TEST_ORDER = {
    "test_cleanup_stale_containers": 0,
    "test_launch_inference_containers": 1,
    "test_setup_ibv_devices": 2,
    "test_rms_norm": 3,
    "test_launch_prefill_servers": 4,
    "test_launch_decode_servers": 5,
    "test_poll_for_server_ready": 6,
    "test_launch_proxy_router": 7,
    "test_openai_compatible_http_endpoints": 8,
    "test_run_lm_eval_hellaswag_benchmark_test": 9,
    "test_run_lm_eval_gsm8k_benchmark_test": 10,
    "test_run_lm_eval_mmlu_benchmark_test": 11,
    "test_run_performance_benchmark_test": 12,
    "test_disagg_gpu_topology": 13,
    "test_print_results_table": 14,
}


def test_print_results_table(inf_res_dict):
    phase_labels = inf_res_dict.pop("__phase_labels__", None) or {}

    smoke_results = inf_res_dict.pop("__smoke_probe_results__", None)
    if smoke_results:
        if isinstance(smoke_results, list):
            log.info(
                "\n======== OpenAI-compatible smoke results ========\n%s",
                "\n".join(str(line) for line in smoke_results),
            )
        
    acc_rows = []
    for label, key in (("HellaSwag", "accuracy_hellaswag"), ("GSM8K", "accuracy_gsm8k"), ("MMLU", "accuracy_mmlu")):
        e = phase_labels.get(key)
        if isinstance(e, dict) and "task" in e:
            acc_rows.append(
                [
                    label,
                    e.get("task", "-"),
                    e.get("metric_key", "-"),
                    f"{float(e['actual']):.4f}",
                    f"{float(e['expected']):.4f}",
                ]
            )
    if acc_rows:
        log.info(
            "\n\n\n\n======== LM-eval accuracy results ========\n%s",
            tabulate(
                acc_rows,
                headers=["Suite", "Task", "Metric", "Actual", "Expected"],
                tablefmt="github",
            ),
        )

    performance = phase_labels.get("performance_test", "-")

    if not inf_res_dict:
        log.info("inf_res_dict empty, nothing to print")
        return

    headers = [
        "Model",
        "GPU",
        "ISL",
        "OSL",
        "Policy",
        "Conc",
        "Host",
        "Performance test",
        "Req/s",
        "Total tok/s",
        "Mean TTFT (ms)",
        "Mean TPOT (ms)",
        "P99 ITL (ms)",
        "Goodput",
        "Output tok/s/GPU",
        "MFU (estimated)"
    ]
    rows = []
    for key, host_dict in inf_res_dict.items():
        model, gpu, isl, osl, policy, conc = key
        for host, m in host_dict.items():
            rows.append(
                [
                    model,
                    gpu,
                    isl,
                    osl,
                    policy,
                    conc,
                    host,
                    performance,
                    m.get("request_throughput_per_sec", "-"),
                    m.get(
                        "total_throughput_per_sec",
                        m.get("output_throughput_per_sec", "-"),
                    ),
                    m.get("mean_ttft_ms", "-"),
                    m.get("mean_tpot_ms", "-"),
                    m.get("p99_itl_ms", "-"),
                    m.get("goodput") or "-",
                    m.get("output_throughput_per_gpu_per_sec", "-"),
                    m.get("mfu", "-"),
                ]
            )
    log.info("\n\n\n\n======== Performance results ========\n")
    log.info("\n" + tabulate(rows, headers=headers, tablefmt="github"))
