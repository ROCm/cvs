'''Shared helpers for SGLang single-node and disaggregated inference libs.'''

from __future__ import annotations

import base64
import json
import re
import shlex
import time
from typing import Any, Callable, Mapping, Optional

from cvs.lib import globals
from cvs.lib.utils.model_query_lib import LmEvalBenchmark, OpenAIProbe
from cvs.lib.utils_lib import fail_test
from cvs.lib.utils.verdict import ThresholdViolation, _check_one, evaluate_all

log = globals.log

DEFAULT_GPU_MEM_THRESHOLD_MB = 5000
AMD_SMI_METRIC_CMD = "sudo amd-smi metric --json"

_SERVER_READY_RE = re.compile(
    r"server is fired up and ready to roll",
    re.I,
)


def textwrap_for_yml(msg_string: str) -> str:
    return '\n'.join([m.lstrip() for m in msg_string.split('\n')])


def as_node_list(value) -> list:
    """Normalize cluster JSON node field to a list of host strings."""
    if isinstance(value, str):
        return [value]
    return list(value)


def resolve_server_node_list(inf_dict: Mapping[str, Any]) -> list[str]:
    """Hosts for unified multi-node SGLang (not PD disagg).

    Resolution order:
    1. ``server_node_list`` when set.
    2. Union of ``prefill_node_list`` and ``decode_node_list`` (stable order).
    """
    explicit = inf_dict.get('server_node_list')
    if explicit:
        return as_node_list(explicit)
    seen: list[str] = []
    for key in ('prefill_node_list', 'decode_node_list'):
        for host in as_node_list(inf_dict.get(key) or []):
            if host not in seen:
                seen.append(host)
    if not seen:
        raise ValueError(
            'sglang_distributed requires server_node_list or at least one of '
            'prefill_node_list / decode_node_list in the inference config'
        )
    return seen


def resolve_distributed_client_host(
    inf_dict: Mapping[str, Any],
    *,
    rank0_node: str,
    benchmark_serv_node: str,
) -> str:
    """HTTP target for bench/smoke/lm-eval when the unified server spans multiple nodes."""
    explicit = inf_dict.get('client_host')
    if explicit:
        return str(explicit)
    if benchmark_serv_node == rank0_node:
        return '127.0.0.1'
    return rank0_node


def resolve_client_host(inf_dict: Mapping[str, Any], *, unified_server: bool = False) -> str:
    """HTTP target for smoke/bench/lm-eval clients running inside a container."""
    explicit = inf_dict.get('client_host')
    if explicit:
        return str(explicit)
    if unified_server:
        return '127.0.0.1'
    proxy = as_node_list(inf_dict['proxy_router_node'])[0]
    bench = as_node_list(inf_dict['benchmark_serv_node'])[0]
    if proxy == bench:
        return '127.0.0.1'
    return proxy


def _normalize_key_value_list(raw: Any, field_name: str) -> list[str]:
    """Normalize ``add_export_env`` entries to ``KEY=VALUE`` strings."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [f'{k}={v}' for k, v in raw.items()]
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            line = str(item).strip()
            if not line:
                continue
            if line.startswith('export '):
                line = line[7:].strip()
            out.append(line)
        return out
    raise ValueError(f'{field_name} must be a list or dict, got {type(raw).__name__}')


def _normalize_cli_flags(raw: Any) -> list[str]:
    """Normalize ``add_flags`` entries to extra ``launch_server`` CLI tokens."""
    if raw is None:
        return []
    if isinstance(raw, str):
        line = raw.strip()
        return [line] if line else []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    raise ValueError(f'add_flags must be a list or str, got {type(raw).__name__}')


def add_export_env_block(bp_dict: Mapping[str, Any], indent: str = '                      ') -> str:
    """Shell ``export`` lines from ``bp_dict['add_export_env']``."""
    env = _normalize_key_value_list(bp_dict.get('add_export_env'), 'add_export_env')
    return '\n'.join(f'{indent}export {entry}' for entry in env)


def add_cli_flags_block(bp_dict: Mapping[str, Any], indent: str = '                              ') -> str:
    """Extra ``launch_server`` CLI flag lines from ``bp_dict['add_flags']``."""
    flags = _normalize_cli_flags(bp_dict.get('add_flags'))
    if not flags:
        return ''
    return '\n'.join(f'{indent}{flag} \\' for flag in flags)


def first_float(pattern: str, text: str):
    m = re.search(pattern, text, re.I)
    return m.group(1) if m else None


def _is_sglang_latency_metric(metric_name: str) -> bool:
    name = metric_name.lower()
    return 'ms' in name or 'latency' in name


def _is_sglang_higher_is_better_metric(metric_name: str) -> bool:
    if _is_sglang_latency_metric(metric_name):
        return False
    name = metric_name.lower()
    return any(
        token in name
        for token in (
            'throughput',
            'goodput',
            'mfu',
            'request_throughput',
        )
    )


def normalize_sglang_threshold_spec(metric_name: str, spec: Any) -> dict[str, Any]:
    """Map threshold JSON specs (or legacy flat floats) to evaluate_all kinds."""
    if isinstance(spec, dict) and spec.get('kind'):
        return spec
    value = float(spec['value'] if isinstance(spec, dict) and 'value' in spec else spec)
    if _is_sglang_latency_metric(metric_name):
        return {'kind': 'max_ms', 'value': value}
    if _is_sglang_higher_is_better_metric(metric_name):
        kind = 'min_tok_s' if 'throughput' in metric_name.lower() else 'min'
        return {'kind': kind, 'value': value}
    return {'kind': 'min', 'value': value}


def coerce_sglang_actual(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_output(out_dict: dict) -> str:
    if not out_dict:
        return ""
    return next(iter(out_dict.values())) or ""


def normalize_hosts(hosts) -> list[str]:
    """Normalize cluster JSON node field to a list of host strings."""
    if hosts is None:
        return []
    return as_node_list(hosts)


def thresholds_from_expected(expected_result_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {metric: normalize_sglang_threshold_spec(metric, spec) for metric, spec in expected_result_dict.items()}


def perf_enforce_thresholds(bp_dict: Mapping[str, Any] | None) -> bool:
    """Read performance gating flag from ``inference_tests.bench_serv_random``."""
    if not bp_dict:
        return True
    bench = (bp_dict.get("inference_tests") or {}).get("bench_serv_random") or {}
    raw = bench.get("enforce_thresholds", True)
    if isinstance(raw, str):
        return raw.strip().lower() not in ("0", "false", "no")
    return bool(raw)


def node_threshold_actuals(
    inference_results_dict: dict[str, dict[str, Any]],
    node: str,
    thresholds: dict[str, dict[str, Any]],
) -> dict[str, float | None]:
    return {
        metric: coerce_sglang_actual(value)
        for metric, value in inference_results_dict[node].items()
        if metric in thresholds
    }


def metric_threshold_violation(
    metric: str,
    actuals: dict[str, float | None],
    spec: dict[str, Any],
) -> str | None:
    if metric not in actuals:
        return f"{metric}: missing from actuals"
    if actuals[metric] is None:
        return f"{metric}: value is None (metric unavailable for this run)"
    spec_with_actuals = dict(spec)
    if spec.get("kind") == "min_ratio":
        spec_with_actuals["_actuals"] = actuals
    return _check_one(metric, actuals[metric], spec_with_actuals)


def finalize_inference_verification(host_exec: Callable[..., dict]) -> dict:
    end_time = host_exec('date +"%a %b %e %H:%M"')
    time.sleep(2)
    return end_time


def verify_inference_results(
    inference_results_dict: dict[str, dict[str, Any]],
    expected_result_dict: dict[str, Any],
    host_exec: Callable[..., dict],
    *,
    test_name: str = "",
    enforce_thresholds: bool = True,
) -> dict:
    thresholds = thresholds_from_expected(expected_result_dict)
    if enforce_thresholds:
        for node in inference_results_dict:
            actuals = node_threshold_actuals(inference_results_dict, node, thresholds)
            try:
                evaluate_all(actuals, thresholds)
            except ThresholdViolation as exc:
                for msg in exc.violations:
                    fail_test(f"FAIL - {msg}")
    return finalize_inference_verification(host_exec)


def verify_inference_results_subtests(
    inference_results_dict: dict[str, dict[str, Any]],
    expected_result_dict: dict[str, Any],
    host_exec: Callable[..., dict],
    subtests,
    test_name: str,
    *,
    lifecycle=None,
    report_nodeid=None,
    enforce_thresholds: bool = True,
) -> tuple[bool, dict]:
    """Verify each metric on each node as its own pytest subtest."""
    thresholds = thresholds_from_expected(expected_result_dict)
    all_passed = bool(inference_results_dict)
    if lifecycle is not None and report_nodeid:
        metric_rows: list[dict[str, Any]] = []
        lifecycle.perf_metric_rows[report_nodeid] = metric_rows
    else:
        metric_rows = None

    for node in inference_results_dict:
        actuals = node_threshold_actuals(inference_results_dict, node, thresholds)
        if enforce_thresholds and not actuals:
            all_passed = False
        for metric, spec in thresholds.items():
            if enforce_thresholds:
                violation = metric_threshold_violation(metric, actuals, spec)
                status = 'pass' if violation is None else 'fail'
            else:
                violation = None
                status = 'pass' if metric in actuals else 'fail'
                if metric not in actuals:
                    all_passed = False
            if metric_rows is not None:
                metric_rows.append(
                    {
                        'node': node,
                        'metric': metric,
                        'status': status,
                    }
                )
            if violation is not None:
                all_passed = False
            with subtests.test(test_name=test_name, node=node, metric=metric):
                if enforce_thresholds:
                    assert violation is None, violation

    return all_passed, finalize_inference_verification(host_exec)


_BENCH_METRIC_PATTERNS: tuple[tuple[str, str], ...] = (
    (r'Successful requests:\s+([0-9]+)', 'successful_requests'),
    (r'Benchmark duration\s+\(s\):\s+([0-9\.]+)', 'benchmark_duration'),
    (r'Total input tokens:\s+([0-9\.]+)', 'total_input_tokens'),
    (r'Total generated tokens:\s+([0-9\.]+)', 'total_generated_tokens'),
    (r'Request throughput \(req/s\):\s+([0-9\.]+)', 'request_throughput_per_sec'),
    (r'Output token throughput \(tok/s\):\s+([0-9\.]+)', 'output_throughput_per_sec'),
    (r'Mean TTFT \(ms\):\s+([0-9\.]+)', 'mean_ttft_ms'),
    (r'Median TTFT \(ms\):\s+([0-9\.]+)', 'median_ttft_ms'),
    (r'P99 TTFT \(ms\):\s+([0-9\.]+)', 'p99_ttft_ms'),
    (r'Mean TPOT \(ms\):\s+([0-9\.]+)', 'mean_tpot_ms'),
    (r'Median TPOT \(ms\):\s+([0-9\.]+)', 'median_tpot_ms'),
    (r'P99 TPOT \(ms\):\s+([0-9\.]+)', 'p99_tpot_ms'),
)

_ITL_METRIC_PATTERNS: tuple[tuple[str, str], ...] = (
    (r'Mean ITL \(ms\):\s+([0-9\.]+)', 'mean_itl_ms'),
    (r'Median ITL \(ms\):\s+([0-9\.]+)', 'median_itl_ms'),
    (r'P99 ITL \(ms\):\s+([0-9\.]+)', 'p99_itl_ms'),
)

_E2E_METRIC_PATTERNS: tuple[tuple[str, str], ...] = (
    (r'Mean E2E Latency \(ms\):\s+([0-9\.]+)', 'mean_e2e_latency_ms'),
    (r'Median E2E Latency \(ms\):\s+([0-9\.]+)', 'median_e2e_latency_ms'),
    (r'P99 E2E Latency \(ms\):\s+([0-9\.]+)', 'p99_e2e_latency_ms'),
)


def parse_inference_bench_results(
    out_dict: Mapping[str, str],
    *,
    bench_num_prompts: int | None = None,
    num_gpus_for_per_gpu_throughput: int | None = None,
    include_itl: bool = False,
    include_extended_e2e_percentiles: bool = False,
) -> dict[str, dict[str, Any]]:
    """Parse sglang.bench_serving log tail output into per-node metric dicts."""
    inference_results_dict: dict[str, dict[str, Any]] = {}
    for node, text in out_dict.items():
        inference_results_dict[node] = {}
        for pattern, key in _BENCH_METRIC_PATTERNS:
            match = re.search(pattern, text, re.I)
            if match:
                inference_results_dict[node][key] = match.group(1)
        if include_itl:
            for pattern, key in _ITL_METRIC_PATTERNS:
                match = re.search(pattern, text, re.I)
                if match:
                    inference_results_dict[node][key] = match.group(1)
        for pattern, key in _E2E_METRIC_PATTERNS:
            val = first_float(pattern, text)
            if val:
                inference_results_dict[node][key] = val
        if include_extended_e2e_percentiles:
            for percentile in (90, 95, 99):
                val = first_float(
                    rf'P{percentile} E2E Latency \(ms\):\s+([0-9\.]+)',
                    text,
                )
                if val:
                    inference_results_dict[node][f'p{percentile}_e2e_latency_ms'] = val

        total_req = first_float(r'Total requests:\s+([0-9]+)', text)
        failed_req = first_float(r'Failed requests:\s+([0-9]+)', text)
        succ = inference_results_dict[node].get('successful_requests')
        if total_req:
            inference_results_dict[node]['total_requests'] = total_req
        elif succ is not None and failed_req is not None:
            inference_results_dict[node]['total_requests'] = str(int(succ) + int(failed_req))
        elif succ is not None and bench_num_prompts is not None:
            inference_results_dict[node]['total_requests'] = str(int(bench_num_prompts))
        if succ and inference_results_dict[node].get('total_requests'):
            s, t = int(succ), int(inference_results_dict[node]['total_requests'])
            inference_results_dict[node]['goodput'] = f'{(s / t):.6f}' if t else None

        out_tps = inference_results_dict[node].get('output_throughput_per_sec')
        if out_tps and num_gpus_for_per_gpu_throughput and num_gpus_for_per_gpu_throughput > 0:
            inference_results_dict[node]['output_throughput_per_gpu_per_sec'] = (
                f'{float(out_tps) / num_gpus_for_per_gpu_throughput:.6f}'
            )

    return inference_results_dict


def poll_for_inference_completion(
    fetch_log_tail: Callable[[], dict[str, str]],
    parse_results: Callable[[dict[str, str]], dict[str, dict[str, Any]]],
    *,
    iterations: int = 10,
    waittime_between_iters: int = 60,
    total_timeout: int | None = 3600,
    require_all_nodes: bool = True,
    inference_poll_iterations: int | None = None,
    log_progress: bool = False,
) -> dict[str, Any]:
    """Poll benchmark logs until completion or timeout."""
    time.sleep(60)
    start_time = time.time()
    poll_cap = inference_poll_iterations if inference_poll_iterations is not None else iterations

    def timed_out() -> bool:
        return total_timeout is not None and (time.time() - start_time) >= float(total_timeout)

    completed_pattern = re.compile('Serving Benchmark Result', re.I)

    for itr in range(1, iterations + 1):
        if log_progress:
            log.info('Starting iteration %d', itr)

        out_dict = fetch_log_tail()
        node_completion = {node: bool(completed_pattern.search(output or '')) for node, output in out_dict.items()}
        if require_all_nodes:
            all_complete = all(node_completion.values()) if node_completion else False
        else:
            all_complete = any(node_completion.values()) if node_completion else False

        if not all_complete:
            if timed_out():
                msg = f"Timeout while waiting for inference completion after ~{int(time.time() - start_time)}s"
                log.warning("%s", msg)
                return {"status": "timeout", "reason": msg}
            if log_progress:
                log.info('Inference still in progress')
            time.sleep(30 + int(waittime_between_iters))
            continue

        results = parse_results(out_dict)
        if log_progress:
            log.info('Completed Inference, returning !!!')
        return {"status": "success", "results": results}

    if timed_out():
        msg = f"Timeout after maximum iterations ({poll_cap}) and ~{int(time.time() - start_time)}s"
        log.warning("%s", msg)
        return {"status": "timeout", "reason": msg}
    msg = f"Reached iteration cap ({poll_cap}) without completion; still in progress"
    log.warning("%s", msg)
    return {"status": "stuck_in_progress", "reason": msg}


def verify_openai_compatible_endpoints(
    *,
    port: int,
    model_name: str,
    client_host: str,
    log_dir: str,
    exec_probe: Callable[[str, int | None], dict[str, str]],
    probe_host_key: str,
    log_label: str | None = None,
) -> list[str]:
    """Smoke-test OpenAI-compatible HTTP API via an in-container probe script."""
    probe_src = OpenAIProbe.probe_script(port, model_name, host=client_host)
    b64 = base64.b64encode(probe_src.encode('utf-8')).decode('ascii')
    inner = (
        f"mkdir -p {log_dir}/benchmark_node && "
        f"echo {shlex.quote(b64)} | base64 -d > /tmp/openai_mq_probe.py && "
        f"python3 /tmp/openai_mq_probe.py && rm -f /tmp/openai_mq_probe.py"
    )
    if log_label:
        log.info('%s (%s:%r)', log_label, client_host, port)
    else:
        log.info('OpenAI endpoint probe inside container (%s:%r)', client_host, port)
    out_dict = exec_probe("bash -c " + shlex.quote(inner), min(900, 480 + 180))
    raw_out = out_dict.get(probe_host_key) or first_output(out_dict)

    probe_err: Optional[str] = None
    results: dict[str, tuple[int, Any]] = {}
    if not raw_out or not str(raw_out).strip():
        probe_err = f"OpenAI-compatible probe produced no output on {probe_host_key!r}: {out_dict!r}"
    else:
        lines_out = str(raw_out).strip().splitlines()
        if not lines_out:
            probe_err = f"OpenAI-compatible probe empty lines after strip on node {probe_host_key!r}: {raw_out!r}"
        else:
            last_line = lines_out[-1]
            try:
                parsed = json.loads(last_line)
            except json.JSONDecodeError as exc:
                probe_err = f"OpenAI-compatible probe invalid JSON: {exc!r} raw={raw_out!r}"
            else:
                if not isinstance(parsed, dict):
                    probe_err = f"OpenAI-compatible probe expected JSON object, got {type(parsed).__name__!r}"
                else:
                    for step, val in parsed.items():
                        if isinstance(val, (list, tuple)) and len(val) == 2:
                            results[step] = (int(val[0]), val[1])
                        else:
                            probe_err = f"OpenAI-compatible probe bad shape at {step!r}: {val!r}"
                            break

    if probe_err is not None:
        fail_test(probe_err)
        return []

    OpenAIProbe.log_results(results, log)
    ok, err = OpenAIProbe.check_results(results, port=port, logger=log)
    if not ok:
        fail_test(f"{err}")
        return OpenAIProbe.summarize_results(results, ok, err)
    return OpenAIProbe.summarize_results(results, ok, err)


def run_lm_eval_benchmark_test(
    bench_key: str,
    *,
    bp_dict: Mapping[str, Any],
    router_serv_port: str,
    client_host: str,
    log_dir: str,
    env_script: str,
    exec_bench: Callable[[str, int | None], dict[str, str]],
) -> Any:
    spec = LM_EVAL_SPECS[bench_key]
    log.info("#================ * * * =========================#")
    log.info("lm-eval %s benchmark", spec["display"])
    log.info("#================ * * * =========================#")
    task_name = bench_key.removeprefix("lm_eval_")
    i_dict = bp_dict["inference_tests"][bench_key]
    inner_cmd, scoring = LmEvalBenchmark.prepare(
        i_dict,
        port=int(router_serv_port),
        host=client_host,
        model_id=bp_dict["model"],
        task_name=task_name,
        default_tasks=task_name,
        default_metric=spec["default_metric"],
        default_metric_key=spec["default_metric_key"],
        log_dir=log_dir,
        log_basename=f"{bench_key}.log",
        default_num_concurrent=spec["default_num_concurrent"],
    )
    inner = f"mkdir -p {log_dir}/benchmark_node && source {env_script} && {inner_cmd}"
    out_dict = exec_bench("bash -c " + shlex.quote(inner), scoring["exec_timeout_sec"])
    time.sleep(5)

    check_kwargs = LmEvalBenchmark.check_kwargs_from_scoring(scoring)
    summary = None
    errors: list[str] = []

    for node, text in out_dict.items():
        ok, node_summary, err = LmEvalBenchmark.check_results(text, **check_kwargs)
        if node_summary is not None:
            summary = node_summary
        if not ok:
            errors.append(f"lm-eval {spec['display']} on node {node!r}: {err}")

    if summary is None:
        summary = LmEvalBenchmark.fallback_summary(
            scoring,
            error=errors[-1] if errors else "no benchmark nodes produced output to score",
        )
        errors.append(f"lm-eval {spec['display']}: no benchmark nodes produced output to score")

    for msg in errors:
        fail_test(msg)

    return summary


def build_log_dir_cleanup_cmd(log_dir: str, user: str) -> str:
    """Shell command: rm -rf, recreate, chown (host namespace, not in-container)."""
    if not log_dir or not str(log_dir).strip():
        raise ValueError("log_dir must be a non-empty path")
    log_dir = str(log_dir).strip()
    quser = shlex.quote(str(user))
    qdir = shlex.quote(log_dir)
    return f"sudo rm -rf {qdir} && sudo mkdir -p {qdir} && sudo chown -R {quser}:{quser} {qdir}"


def cleanup_sglang_log_dir(
    orch: Any,
    log_dir: str,
    *,
    all_nodes: bool | None = None,
    timeout: int = 60,
) -> None:
    """Reset log root on cluster hosts via baremetal SSH (``orch.head`` / ``orch.all``)."""
    if all_nodes is None:
        all_nodes = len(orch.hosts) > 1
    cmd = build_log_dir_cleanup_cmd(log_dir, orch.user)
    if all_nodes:
        orch.all.exec(cmd, timeout=timeout)
    else:
        orch.head.exec(cmd, timeout=timeout)


LM_EVAL_SPECS = {
    'lm_eval_hellaswag': {
        'display': 'HellaSwag',
        'default_metric': 'acc_norm',
        'default_metric_key': 'acc_norm,none',
        'default_num_concurrent': '1',
    },
    'lm_eval_gsm8k': {
        'display': 'GSM8K',
        'default_metric': 'exact_match',
        'default_metric_key': 'exact_match,flexible-extract',
        'default_num_concurrent': '4',
    },
}


def _parse_amd_smi_gpu_entries(payload: str | None) -> list[dict]:
    """Unwrap amd-smi --json (list or {"gpu_data": [...]}) -> GPU entry list."""
    try:
        entries = json.loads((payload or "").strip())
    except (json.JSONDecodeError, AttributeError, TypeError):
        return []
    if isinstance(entries, dict):
        entries = entries.get("gpu_data", [])
    return entries if isinstance(entries, list) else []


def count_occupied_gpus_on_node(
    payload: str | None,
    *,
    mem_threshold_mb: int = DEFAULT_GPU_MEM_THRESHOLD_MB,
) -> int:
    count = 0
    for g in _parse_amd_smi_gpu_entries(payload):
        used_mb = g.get("mem_usage", {}).get("used_vram", {}).get("value", 0)
        if used_mb > mem_threshold_mb:
            count += 1
    return count


def count_occupied_gpus_per_node(
    out_dict: Mapping[str, str | None],
    *,
    mem_threshold_mb: int = DEFAULT_GPU_MEM_THRESHOLD_MB,
) -> dict[str, int]:
    per_node: dict[str, int] = {}
    for node, payload in out_dict.items():
        if payload is None:
            log.warning("No amd-smi output on node %s", node)
            per_node[node] = 0
            continue
        try:
            per_node[node] = count_occupied_gpus_on_node(payload, mem_threshold_mb=mem_threshold_mb)
        except (TypeError, ValueError, AttributeError):
            log.warning("Failed to parse amd-smi JSON on node %s", node)
            per_node[node] = 0
    return per_node


def collect_sglang_gpu_topology(
    host_exec: Callable[..., dict[str, str | None]],
    groups: Mapping[str, list[str]],
    *,
    mem_threshold_mb: int = DEFAULT_GPU_MEM_THRESHOLD_MB,
    amd_smi_cmd: str = AMD_SMI_METRIC_CMD,
    timeout: int | None = None,
) -> dict[str, Any]:
    """
    groups: e.g. {"server": [...]} or {"prefill": [...], "decode": [...]}
    host_exec: suite _host_exec(cmd, hosts=..., timeout=...)
    """
    group_stats: dict[str, dict[str, Any]] = {}
    total = 0

    for name, hosts in groups.items():
        if not hosts:
            group_stats[name] = {"per_node": {}, "total": 0}
            continue
        per_node = count_occupied_gpus_per_node(
            host_exec(amd_smi_cmd, hosts=hosts, timeout=timeout),
            mem_threshold_mb=mem_threshold_mb,
        )
        group_total = sum(per_node.values())
        group_stats[name] = {"per_node": per_node, "total": group_total}
        total += group_total

    return {"groups": group_stats, "total_occupied_gpus": total}


def format_sglang_gpu_topology_lines(
    *,
    configured_tp: int,
    configured_pp: int,
    groups: Mapping[str, dict[str, Any]],
    configured_nnodes: int | None = None,
) -> list[str]:
    lines = ["", f"Configured TP: {configured_tp}", f"Configured PP: {configured_pp}"]
    if configured_nnodes is not None:
        lines.append(f"Configured nnodes: {configured_nnodes}")
    for label, stats in groups.items():
        lines.extend(["", f"{label.title()}:"])
        for node, count in stats["per_node"].items():
            lines.append(f"  {node}: {count} occupied GPUs")
        lines.append(f"  Total: {stats['total']} occupied GPUs")
    lines.extend(["", "Total hardware GPUs consumed:", f"  {sum(s['total'] for s in groups.values())}"])
    return lines
