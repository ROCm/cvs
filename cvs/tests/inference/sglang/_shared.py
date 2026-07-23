'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Shared helpers and ordering for the SGLang inference suites (single-node and disaggregated).

Used by both ``sglang_single.py`` and ``sglang_disagg_distributed.py``. Each suite
has its own stage order (unified server vs PD disaggregation).
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
    "SGLANG_TEST_ORDER",
    "SGLANG_SINGLE_TEST_ORDER",
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


# Stable test order for sglang_disagg_distributed (PD prefill/decode/router).
SGLANG_TEST_ORDER = {
    "test_cleanup_stale_containers": 0,
    "test_launch_inference_containers": 1,
    "test_rms_norm": 2,
    "test_launch_prefill_servers": 3,
    "test_launch_decode_servers": 4,
    "test_poll_for_server_ready": 5,
    "test_launch_proxy_router": 6,
    "test_openai_compatible_http_endpoints": 7,
    "test_run_long_context_accuracy": 8,
    "test_run_lm_eval_hellaswag_benchmark_test": 9,
    "test_run_lm_eval_gsm8k_benchmark_test": 10,
    "test_run_lm_eval_mmlu_benchmark_test": 11,
    "test_run_performance_benchmark_test": 12,
    "test_disagg_gpu_topology": 13,
    "test_print_results_table": 14,
    "test_teardown": 15,
}

# Stable test order for sglang_single (one unified server, no PD).
SGLANG_SINGLE_TEST_ORDER = {
    "test_launch_container": 0,
    "test_rms_norm": 1,
    "test_launch_server": 2,
    "test_poll_for_server_ready": 3,
    "test_openai_compatible_http_endpoints": 4,
    "test_run_long_context_accuracy": 5,
    "test_run_lm_eval_hellaswag_benchmark_test": 6,
    "test_run_lm_eval_gsm8k_benchmark_test": 7,
    "test_run_lm_eval_mmlu_benchmark_test": 8,
    "test_run_performance_benchmark_test": 9,
    "test_server_gpu_topology": 10,
    "test_print_results_table": 11,
    "test_teardown": 12,
}


def _flat_threshold_specs(specs: dict) -> dict[str, float]:
    """Threshold cell specs → {metric: numeric_gate}."""
    out: dict[str, float] = {}
    for metric, spec in (specs or {}).items():
        if isinstance(spec, dict) and "value" in spec:
            out[metric] = float(spec["value"])
        elif spec is not None:
            out[metric] = float(spec)
    return out


def _perf_result(actual, expected, metric_key: str) -> str:
    if actual is None or expected is None:
        return "-"
    a, e = float(actual), float(expected)
    if "ms" in metric_key.lower():
        return "PASS" if a <= e else "FAIL"
    return "PASS" if a >= e else "FAIL"


def _thresholds_for_cell(variant_config, isl, osl, conc) -> dict[str, float]:
    if variant_config is None:
        return {}
    tp = (getattr(variant_config, "benchmark_params", None) or {}).get("tensor_parallelism", "-")
    pp = (getattr(variant_config, "benchmark_params", None) or {}).get("pipeline_parallelism", "-")
    cell_id = f"ISL={isl},OSL={osl},TP={tp},PP={pp},CONC={conc}"
    raw = (getattr(variant_config, "thresholds", None) or {}).get(cell_id) or {}
    return _flat_threshold_specs(raw)


def test_print_results_table(inf_res_dict, lifecycle, variant_config=None):
    """Log smoke, lm-eval accuracy, and perf tables (one row per metric per host per ISL/OSL cell)."""
    phase_labels = getattr(lifecycle, "phase_labels", None) or {}
    smoke_results = getattr(lifecycle, "smoke_results", None)

    if variant_config is None:
        try:
            from cvs.lib.report.registry import get_session_results

            variant_config = get_session_results().get("variant_config")
        except Exception:
            variant_config = None

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
    for label, key in (
        ("HellaSwag", "accuracy_hellaswag"),
        ("GSM8K", "accuracy_gsm8k"),
        ("MMLU", "accuracy_mmlu"),
    ):
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

    bp = (getattr(variant_config, "benchmark_params", None) or {}) if variant_config else {}
    tp = bp.get("tensor_parallelism", "8")
    pp = bp.get("pipeline_parallelism", "1")

    _ACC_CELL_RE = re.compile(r"^ACC_ISL=(?P<isl>\d+),OSL=(?P<osl>\d+)$")

    acc_by_cell = phase_labels.get("accuracy_by_cell") or {}
    if acc_by_cell:
        long_ctx_rows = []
        for cell_id, result in sorted(acc_by_cell.items()):
            m = _ACC_CELL_RE.match(str(cell_id))
            if not m:
                continue
            key = f"accuracy_long_ctx_{m.group('isl')}"
            e = phase_labels.get(key) or {}
            long_ctx_rows.append([
                f"ISL={m.group('isl')}",
                m.group("osl"),
                f"{float(e['actual']):.4f}" if e.get("actual") is not None else "-",
                f"{float(e['expected']):.4f}" if e.get("expected") is not None else "-",
                result,
            ])
        if long_ctx_rows:
            log.info(
                "\n\n\n\n======== Long-context accuracy (NIAH) ========\n%s",
                tabulate(
                    long_ctx_rows,
                    headers=["Cell", "OSL", "Pass rate", "Expected", "Result"],
                    tablefmt="github",
                ),
            )

    _CELL_RE = re.compile(
        r"^ISL=(?P<isl>\d+),OSL=(?P<osl>\d+),TP=(?P<tp>\d+),PP=(?P<pp>\d+),CONC=(?P<conc>\d+)$"
    )
    bp = (getattr(variant_config, "benchmark_params", None) or {}) if variant_config else {}
    performance_by_cell = phase_labels.get("performance_by_cell") or {}
    if performance_by_cell:
        summary_rows = []
        for cell_id, result in sorted(
            performance_by_cell.items(),
            key=lambda kv: (
                int(m.group("isl")), int(m.group("osl")), int(m.group("conc"))
            ) if (m := _CELL_RE.match(str(kv[0]))) else (0, 0, 0),
        ):
            m = _CELL_RE.match(str(cell_id))
            if m:
                summary_rows.append([
                    m.group("isl"),
                    m.group("osl"),
                    m.group("tp"),
                    m.group("pp"),
                    m.group("conc"),
                    result,
                ])
            else:
                summary_rows.append(["-", "-", tp, pp, str(cell_id), result])

        log.info(
            "\n\n\n\n======== Performance summary (by ISL/OSL cell) ========\n%s",
            tabulate(
                summary_rows,
                headers=["ISL", "OSL", "TP", "PP", "Conc", "Result"],
                tablefmt="github",
            ),
        )

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

    perf_items = [
        (k, v)
        for k, v in inf_res_dict.items()
        if isinstance(k, tuple) and len(k) == 6 and isinstance(v, dict)
    ]

    perf_rows = []
    for key, host_dict in sorted(
        perf_items,
        key=lambda kv: (int(kv[0][2]), int(kv[0][3]), int(kv[0][5])),
    ):
        model, gpu, isl, osl, policy, conc = key
        expected_map = _thresholds_for_cell(variant_config, isl, osl, conc)
        for host, m in host_dict.items():
            for label, metric_key in PERF_METRICS:
                actual = m.get(metric_key)
                if actual is None:
                    continue
                expected = expected_map.get(metric_key)
                perf_rows.append(
                    [
                        model,
                        gpu,
                        isl,
                        osl,
                        policy,
                        conc,
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
                headers=[
                    "Model",
                    "GPU",
                    "ISL",
                    "OSL",
                    "Policy",
                    "Conc",
                    "Host",
                    "Metric",
                    "Actual",
                    "Expected",
                    "Result",
                ],
                tablefmt="github",
            ),
        )
    elif not smoke_results and not acc_rows and not performance_by_cell:
        log.info("inf_res_dict empty, nothing to print")