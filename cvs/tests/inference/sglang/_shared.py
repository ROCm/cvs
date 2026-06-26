'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Shared helpers and ordering for the SGLang disaggregated inference suite.
'''

from __future__ import annotations

import os
import re
from tabulate import tabulate
from typing import Any, Mapping

from cvs.lib import globals


log = globals.log

__all__ = [
    "resolve_benchmark_variant_key",
    "SGLANG_DISAGG_TEST_ORDER",
    "test_print_results_table",
]

_SMOKE_LINE_RE = re.compile(r"^(.+) -> (Pass|Fail) \((\d+)\)$")

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
    "test_gpu_metric": 14,
    "test_print_results_table": 15,
}


def test_print_results_table(inf_res_dict):
    phase_labels = inf_res_dict.pop("__phase_labels__", None) or {}

    smoke_results = inf_res_dict.pop("__smoke_probe_results__", None)
    if smoke_results:
        smoke_rows = []
        for line in smoke_results:
            m = _SMOKE_LINE_RE.match(str(line).strip())
            if m:
                smoke_rows.append([m.group(1), m.group(2).upper(), m.group(3)])
            else:
                smoke_rows.append([str(line), "-", "-"])
        if smoke_rows:
            log.info(
                "\n\n\n\n======== OpenAI-compatible smoke results ========\n%s",
                tabulate(
                    smoke_rows,
                    headers=["Check", "Result", "HTTP status"],
                    tablefmt="github",
                ),
            )
        
    acc_rows = []
    for label, key in (("HellaSwag", "accuracy_hellaswag"), ("GSM8K", "accuracy_gsm8k"), ("MMLU", "accuracy_mmlu")):
        e = phase_labels.get(key)
        if isinstance(e, dict) and "task" in e:
            passed = e.get("passed")
            acc_rows.append(
                [
                    label,
                    e.get("task", "-"),
                    e.get("metric_key", "-"),
                    f"{float(e['actual']):.4f}" if e.get("actual") is not None else "-",
                    f"{float(e['expected']):.4f}",
                    "PASS" if passed is True else "FAIL" if passed is False else "-",
                ]
            )
    if acc_rows:
        log.info(
            "\n\n\n\n======== LM-eval accuracy results ========\n%s",
            tabulate(
                acc_rows,
                headers=["Suite", "Task", "Metric", "Actual", "Expected", "Result"],
                tablefmt="github",
            ),
        )

    perf_expected = phase_labels.get("performance_expected") or {}

    PERF_METRICS = [
        ("Mean TTFT (ms)", "mean_ttft_ms"),
        ("Mean TPOT (ms)", "mean_tpot_ms"),
        ("P99 ITL (ms)", "p99_itl_ms"),
        ("Mean E2E latency (ms)", "mean_e2e_latency_ms"),
        ("Req/s", "request_throughput_per_sec"),
        ("Output tok/s", "output_throughput_per_sec"),
        ("Output tok/s/GPU", "output_throughput_per_gpu_per_sec"),
        ("Goodput", "goodput"),
        ("MFU (estimated)", "mfu"),
    ]

    def _perf_result(actual, expected, metric_key: str) -> str:
        if actual is None or expected is None:
            return "-"
        a, e = float(actual), float(expected)
        if "ms" in metric_key.lower():
            return "PASS" if a <= e else "FAIL"
        return "PASS" if a >= e else "FAIL"

    perf_rows = []
    for key, host_dict in inf_res_dict.items():
        model, gpu, isl, osl, policy, conc = key
        for host, m in host_dict.items():
            for label, metric_key in PERF_METRICS:
                actual = m.get(metric_key)
                if actual is None:
                    continue

                expected = perf_expected.get(metric_key)
                perf_rows.append(
                    [
                        model,
                        gpu,
                        host,
                        label,
                        f"{float(actual):.4f}",
                        f"{float(expected):.4f}" if expected is not None else "-",
                        _perf_result(actual, expected, metric_key),
                    ]
                )

    if perf_rows:
        log.info(
            "\n\n\n\n======== Performance results ========\n%s",
            tabulate(
                perf_rows,
                headers=["Model", "GPU", "Host", "Metric", "Actual", "Expected", "Result"],
                tablefmt="github",
            ),
        )
    elif not smoke_results and not acc_rows:
        log.info("inf_res_dict empty, nothing to print")
