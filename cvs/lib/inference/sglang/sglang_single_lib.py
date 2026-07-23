'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Single-node SGLang inference controller (no PD disaggregation).

One container on the cluster head (via ``ContainerOrchestrator``) runs a unified
``sglang.launch_server`` on ``proxy_router_serv_port``. Benchmark/smoke/lm-eval
traffic hits that port via ``client_host`` (default ``127.0.0.1`` inside the
container).
'''

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import time
from typing import Any, Optional

from cvs.lib import globals
from cvs.lib.inference.sglang.sglang_common import (
    LM_EVAL_SPECS,
    add_cli_flags_block,
    add_export_env_block,
    coerce_sglang_actual,
    first_float,
    normalize_sglang_threshold_spec,
    resolve_client_host,
)
from cvs.lib.utils.model_query_lib import LmEvalBenchmark, OpenAIProbe
from cvs.lib.utils_lib import fail_test
from cvs.lib.utils.verdict import ThresholdViolation, evaluate_all
from cvs.lib.verify_lib import verify_dmesg_for_errors

log = globals.log

_SERVER_READY_RE = re.compile(
    r"fired up and ready to roll|Uvicorn running|Application startup complete|200 OK",
    re.I,
)


class SglangSingle:
    """Unified single-node SGLang serve + benchmark via ``ContainerOrchestrator``."""

    def __init__(
        self,
        model_name,
        inference_config_dict,
        benchmark_params_dict,
        hf_token,
        orch=None,
        gpu_type='mi300',
        user_name=None,
        priv_key_file=None,
    ):
        if orch is None:
            raise ValueError("SglangSingle requires orch= (ContainerOrchestrator)")

        self.orch = orch
        self.user_name = user_name
        self.priv_key_file = priv_key_file
        self.model_name = model_name
        self.hf_token = hf_token
        self.gpu_type = gpu_type

        self.inf_dict = inference_config_dict
        self.bp_dict = benchmark_params_dict

        self.mount_vol = self.inf_dict.get(
            'mount_vol',
            '/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so',
        )

        self.inference_results_dict = {}
        log.info("%s", self.gpu_type)

        self.home_dir = os.path.expanduser("~")
        self._apply_inf_defaults()
        self._apply_bp_defaults()

        self.container_name = self.inf_dict['container_name']
        self.nic_type = self.inf_dict['nic_type']
        self.hca_id_prefix = str(self.inf_dict['hca_id_prefix']).strip()
        self.log_dir = self.inf_dict['log_dir']
        self.inference_poll_iterations = self.bp_dict['inference_poll_iterations']

        self.inference_start_time = self._host_exec('date +"%a %b %e %H:%M"')
        self.inference_end_time = None

        log.info('single-node inference_dict = %s', self.inf_dict)
        log.info('single-node benchmark_params_dict = %s', self.bp_dict)
        log.info(
            'single-node client_host=%s router_serv_port=%s head=%s',
            self.client_host,
            self.router_serv_port,
            self._head_host,
        )

    @property
    def _head_host(self) -> str:
        return self.orch.head_node

    @property
    def server_log_path(self) -> str:
        return f"{self.log_dir}/server_node/server.log"

    @property
    def router_serv_port(self) -> str:
        """Unified server listen/client port (``proxy_router_serv_port``)."""
        return str(self.inf_dict['proxy_router_serv_port'])

    @property
    def client_host(self) -> str:
        """HTTP client target when smoke/bench/lm-eval run inside the same container."""
        return resolve_client_host(self.inf_dict, unified_server=True)

    @staticmethod
    def _first_output(out_dict: dict) -> str:
        if not out_dict:
            return ""
        return next(iter(out_dict.values())) or ""

    def _container_exec(self, cmd: str, *, timeout: int | None = None) -> dict:
        """Run ``cmd`` inside the container."""
        return self.orch.exec(cmd, timeout=timeout)

    def _container_exec_text(self, cmd: str, *, timeout: int | None = None) -> str:
        return self._first_output(self._container_exec(cmd, timeout=timeout))

    def _host_exec(self, cmd: str, *, timeout: int | None = None) -> dict:
        """Run ``cmd`` on the head host (baremetal SSH), e.g. amd-smi / dmesg."""
        return self.orch.head.exec(cmd, timeout=timeout)

    def _host_exec_text(self, cmd: str, *, timeout: int | None = None) -> str:
        return self._first_output(self._host_exec(cmd, timeout=timeout))

    def _apply_inf_defaults(self) -> None:
        self.inf_dict.setdefault('container_image', 'lmsysorg/sglang:dev')
        self.inf_dict.setdefault('container_name', 'sglang_container')
        self.inf_dict.setdefault('nic_type', 'ainic')
        self.inf_dict.setdefault('nccl_ib_hca', 'rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7')
        self.inf_dict.setdefault('hca_id_prefix', 'bnxt_')
        self.inf_dict.setdefault('nccl_socket_ifname', 'eno0')
        self.inf_dict.setdefault('gloo_socket_ifname', 'eno0')
        self.inf_dict.setdefault('nccl_ib_gid_index', '1')
        self.inf_dict.setdefault('nccl_debug', 'ERROR')
        self.inf_dict.setdefault('data_cache_dir', f'{self.home_dir}/cache')
        self.inf_dict.setdefault('log_dir', f'{self.home_dir}/LOG_DIR')
        self.inf_dict.setdefault('log_level', 'info')
        self.inf_dict.setdefault('proxy_router_serv_port', '8000')

    def _apply_bp_defaults(self) -> None:
        self.bp_dict.setdefault('backend', 'sglang')
        self.bp_dict.setdefault('max_concurrency', '64')
        self.bp_dict.setdefault('model', 'openai/gpt-oss-120b')
        self.bp_dict.setdefault('tensor_parallelism', '8')
        self.bp_dict.setdefault('memory_fraction', '0.85')
        self.bp_dict.setdefault('inference_poll_iterations', '16')

    def setup_server_container_env(self) -> None:
        """Write and source ``/tmp/server_env_script.sh`` inside the container."""
        env_body = (
            "export LD_LIBRARY_PATH=/usr/local/lib:/sgl-workspace/Mooncake/build/mooncake-common/etcd:/opt/rocm/lib:$LD_LIBRARY_PATH\n"
            f"export NCCL_DEBUG={self.inf_dict['nccl_debug']}\n"
            f"export NCCL_IB_HCA={self.inf_dict['nccl_ib_hca']}\n"
            f"export NCCL_IB_GID_INDEX={self.inf_dict['nccl_ib_gid_index']}\n"
            f"export NCCL_SOCKET_IFNAME={self.inf_dict['nccl_socket_ifname']}\n"
            f"export GLOO_SOCKET_IFNAME={self.inf_dict['gloo_socket_ifname']}\n"
            f"export GLOO_TCP_IFNAME={self.inf_dict['gloo_socket_ifname']}\n"
            f"export HSA_FORCE_FINE_GRAIN_PCIE=1\n"
            f"export MODEL={self.bp_dict['model']}\n"
            f"export TP={self.bp_dict['tensor_parallelism']}\n"
            f"export HF_TOKEN={self.hf_token}\n"
            f"{add_export_env_block(self.bp_dict, indent='')}\n"
        )
        write_cmd = "bash -c " + shlex.quote(
            f"cat > /tmp/server_env_script.sh <<'EOF'\n{env_body}EOF\n"
            "chmod 755 /tmp/server_env_script.sh && /tmp/server_env_script.sh"
        )
        time.sleep(3)
        self._container_exec(write_cmd)
        time.sleep(5)

    def launch_server(self, dtype='auto', kv_cache_dtype='auto') -> None:
        """Launch one unified SGLang server (no PD disaggregation)."""
        log.info('Launch unified SGLang server on 0.0.0.0:%s', self.router_serv_port)
        flags_block = add_cli_flags_block(self.bp_dict, indent='    ')
        launch_body = (
            f"python3 -m sglang.launch_server --model {self.bp_dict['model']} \\\n"
            f"    --host 0.0.0.0 \\\n"
            f"    --port {self.router_serv_port} \\\n"
            f"    --dtype {dtype} \\\n"
            f"    --kv-cache-dtype {kv_cache_dtype} \\\n"
            f"    --trust-remote-code \\\n"
            f"    --tp-size {self.bp_dict['tensor_parallelism']} \\\n"
            f"    --disable-radix-cache --disable-cuda-graph \\\n"
            f"    --mem-fraction-static {self.bp_dict['memory_fraction']} \\\n"
            f"{flags_block}\n"
            f"    --log-level {self.inf_dict['log_level']}\n"
        )
        start_cmd = "bash -c " + shlex.quote(
            f"cat > /tmp/server_launch_script.sh <<'EOF'\n{launch_body}EOF\n"
            f"chmod 755 /tmp/server_launch_script.sh\n"
            f"mkdir -p {self.log_dir}/server_node\n"
            f"source /tmp/server_env_script.sh\n"
            f"nohup /tmp/server_launch_script.sh > {self.server_log_path} 2>&1 &"
        )
        self._container_exec(start_cmd)
        time.sleep(5)

    def poll_for_server_ready(self, no_of_iterations=16) -> None:
        for iteration in range(1, no_of_iterations):
            log.info('Starting server readiness poll iteration %d', iteration)
            grep_cmd = (
                f"grep -B 20 -A 20 -E {_SERVER_READY_RE.pattern!r} "
                f"{shlex.quote(self.server_log_path)} || true"
            )
            text = self._container_exec_text(grep_cmd)
            if _SERVER_READY_RE.search(text):
                log.info('Wait 60 secs before serving traffic')
                time.sleep(60)
                return
            log.info('Wait 120 secs and continue polling')
            time.sleep(120)
        fail_test(
            f'Single-node server on {self._head_host} did not reach ready state '
            f'in {no_of_iterations} iterations'
        )

    def poll_and_check_server_ready(self) -> None:
        log.info('Waiting 120 secs after launching server')
        time.sleep(120)
        self.poll_for_server_ready()

    def setup_benchmark_serv_container_env(self) -> None:
        self.setup_server_container_env()

    def install_container_packages(self) -> None:
        self._container_exec(
            "bash -c " + shlex.quote(
                "sudo apt -y update && sudo apt install -y iputils-ping iproute2 net-tools"
            )
        )

    def exec_nic_setup_scripts(self) -> None:
        if re.search('broadcom|thor', self.nic_type, re.I):
            self.inf_dict['nccl_ib_gid_index'] = 3
            cmd = "bash -c " + shlex.quote(
                f"cp {self.mount_vol}.host {self.mount_vol}; sleep 2; ibv_devinfo; sleep 2;"
            )
            out_dict = self._container_exec(cmd)
            hca_id_regex = rf'hca_id:\s+{re.escape(self.hca_id_prefix)}'
            for node, out in out_dict.items():
                if not re.search(hca_id_regex, out or '', re.I):
                    fail_test(f'Broadcom libbnxt rdma driver is not properly copied on node {node}')

    def check_ibv_devices(self) -> None:
        out_dict = self._container_exec("ibv_devinfo")
        for node, out in out_dict.items():
            if re.search('No IB devices found', out or '', re.I):
                fail_test(f'IB devices not seen inside the container for node {node}')

    def run_test_rmsnorm(self, max_jobs=192) -> None:
        self._container_exec(
            "bash -c " + shlex.quote(
                f"MAX_JOBS={max_jobs} python /sgl-workspace/aiter/op_tests/test_rmsnorm2d.py "
                f"> /tmp/rsmnorm_test.log 2>&1 &"
            )
        )
        time.sleep(180)
        out_dict = self._container_exec("bash -c " + shlex.quote("cat /tmp/rsmnorm_test.log"))
        for node, out in out_dict.items():
            if re.search('fail', out or '', re.I):
                fail_test(f'Some failures observed in test rmsnorm on node {node}')

    def verify_openai_compatible_endpoints(self) -> list[str]:
        port = int(self.router_serv_port)
        probe_src = OpenAIProbe.probe_script(
            port, self.bp_dict['model'], host=self.client_host
        )
        b64 = base64.b64encode(probe_src.encode('utf-8')).decode('ascii')
        inner = (
            f"mkdir -p {self.log_dir}/benchmark_node && "
            f"echo {shlex.quote(b64)} | base64 -d > /tmp/openai_mq_probe.py && "
            f"python3 /tmp/openai_mq_probe.py && rm -f /tmp/openai_mq_probe.py"
        )
        log.info(
            'OpenAI endpoint probe inside container (%s:%r)',
            self.client_host,
            port,
        )
        out_dict = self._container_exec("bash -c " + shlex.quote(inner), timeout=min(900, 480 + 180))
        raw_out = out_dict.get(self._head_host) or self._first_output(out_dict)

        probe_err: Optional[str] = None
        results: dict[str, tuple[int, Any]] = {}
        if not raw_out or not str(raw_out).strip():
            probe_err = f"OpenAI-compatible probe produced no output on {self._head_host!r}: {out_dict!r}"
        else:
            last_line = str(raw_out).strip().splitlines()[-1]
            try:
                parsed = json.loads(last_line)
            except json.JSONDecodeError as e:
                probe_err = f"OpenAI-compatible probe invalid JSON: {e!r} raw={raw_out!r}"
            else:
                if not isinstance(parsed, dict):
                    probe_err = (
                        f"OpenAI-compatible probe expected JSON object, got "
                        f"{type(parsed).__name__!r}"
                    )
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

    def benchserv_test_random(self, d_type='auto') -> None:
        i_dict = self.bp_dict['inference_tests']['bench_serv_random']
        self._bench_num_prompts = int(i_dict['num_prompts'])
        inner = (
            f"mkdir -p {self.log_dir}/benchmark_node\n"
            f"source /tmp/server_env_script.sh\n"
            f"export PYTHONPATH=/sgl-workspace/sglang/python:${{PYTHONPATH:-}}\n"
            f"python3 -m sglang.bench_serving \\\n"
            f"  --backend {i_dict['backend']} \\\n"
            f"  --dataset-name random \\\n"
            f"  --num-prompts {i_dict['num_prompts']} \\\n"
            f"  --max-concurrency {self.bp_dict['max_concurrency']} \\\n"
            f"  --random-input {i_dict['input_length']} \\\n"
            f"  --random-output {i_dict['output_length']} \\\n"
            f"  --random-range-ratio {i_dict['random_range_ratio']} \\\n"
            f"  --host {self.client_host} --port {self.router_serv_port} \\\n"
            f"  > {self.log_dir}/benchmark_node/benchmark_results.log 2>&1"
        )
        self._container_exec("bash -c " + shlex.quote(inner), timeout=1000)
        time.sleep(5)
        self.poll_for_inference_completion(iterations=10, waittime_between_iters=60)

        tp = int(self.bp_dict.get('tensor_parallelism', 1))
        num_gpus = tp
        peak_tflops = float(i_dict.get('peak_gpu_tflops', 1300))
        num_params = float(i_dict.get('model_num_params', 70e9))
        for node, m in (self.inference_results_dict or {}).items():
            duration = float(m.get('benchmark_duration') or 0)
            in_tok = float(m.get('total_input_tokens') or 0)
            out_tok = float(m.get('total_generated_tokens') or 0)
            if duration > 0 and num_gpus > 0:
                achieved = 6.0 * num_params * (in_tok + out_tok)
                peak = peak_tflops * 1e12 * num_gpus * duration
                m['mfu'] = f'{achieved / peak:.6f}'

        self.verify_inference_results('bench_serv', i_dict['expected_results'][d_type])

    def get_inference_results_dict(self, out_dict):
        self.inference_results_dict = {}
        for node, text in out_dict.items():
            self.inference_results_dict[node] = {}
            patterns = [
                (r'Successful requests:\s+([0-9]+)', 'successful_requests'),
                (r'Benchmark duration\s+\(s\):\s+([0-9]+)', 'benchmark_duration'),
                (r'Total input tokens:\s+([0-9\.]+)', 'total_input_tokens'),
                (r'Total generated tokens:\s+([0-9\.]+)', 'total_generated_tokens'),
                (r'Request throughput \(req/s\):\s+([0-9\.]+)', 'request_throughput_per_sec'),
                (r'Output token throughput \(tok/s\):\s+([0-9\.]+)', 'output_throughput_per_sec'),
                (r'Mean TTFT \(ms\):\s+([0-9\.]+)', 'mean_ttft_ms'),
                (r'Median TTFT \(ms\):\s+([0-9\.]+)', 'median_ttft_ms'),
                (r'P99 TTFT \(ms\):\s+([0-9\.]+)', 'p99_ttft_ms'),
                (r'Mean TPOT \(ms\):\s+([0-9\.]+)', 'mean_tpot_ms'),
                (r'Median TPOT \(ms\):\s+([0-9]+)', 'median_tpot_ms'),
                (r'P99 TPOT \(ms\):\s+([0-9\.]+)', 'p99_tpot_ms'),
            ]
            for pattern, key in patterns:
                match = re.search(pattern, text, re.I)
                if match:
                    self.inference_results_dict[node][key] = match.group(1)
            for pattern, key in (
                (r'Mean E2E Latency \(ms\):\s+([0-9\.]+)', 'mean_e2e_latency_ms'),
                (r'Median E2E Latency \(ms\):\s+([0-9\.]+)', 'median_e2e_latency_ms'),
                (r'P99 E2E Latency \(ms\):\s+([0-9\.]+)', 'p99_e2e_latency_ms'),
            ):
                val = first_float(pattern, text)
                if val:
                    self.inference_results_dict[node][key] = val

            total_req = first_float(r'Total requests:\s+([0-9]+)', text)
            failed_req = first_float(r'Failed requests:\s+([0-9]+)', text)
            succ = self.inference_results_dict[node].get('successful_requests')
            if total_req:
                self.inference_results_dict[node]['total_requests'] = total_req
            elif succ is not None and failed_req is not None:
                self.inference_results_dict[node]['total_requests'] = str(int(succ) + int(failed_req))
            elif succ is not None and getattr(self, '_bench_num_prompts', None) is not None:
                self.inference_results_dict[node]['total_requests'] = str(int(self._bench_num_prompts))
            if succ and self.inference_results_dict[node].get('total_requests'):
                s, t = int(succ), int(self.inference_results_dict[node]['total_requests'])
                self.inference_results_dict[node]['goodput'] = f'{(s / t):.6f}' if t else None

        return self.inference_results_dict

    def poll_for_inference_completion(
        self, iterations=10, waittime_between_iters=60, total_timeout=3600, require_all_nodes=True
    ):
        time.sleep(60)
        start_time = time.time()
        completed_pattern = re.compile('Serving Benchmark Result', re.I)
        log_path = f"{self.log_dir}/benchmark_node/benchmark_results.log"

        for _itr in range(1, iterations + 1):
            out_dict = self._container_exec(f"tail -1000 {shlex.quote(log_path)}")
            done = all(completed_pattern.search(o or '') for o in out_dict.values()) if out_dict else False
            if done:
                self.get_inference_results_dict(out_dict)
                return {"status": "success", "results": self.inference_results_dict}
            if total_timeout and (time.time() - start_time) >= total_timeout:
                return {"status": "timeout", "reason": "benchmark timed out"}
            time.sleep(30 + int(waittime_between_iters))
        return {"status": "stuck_in_progress", "reason": "benchmark did not complete"}

    def verify_inference_results(self, test_name, expected_result_dict):
        thresholds = {
            metric: normalize_sglang_threshold_spec(metric, spec)
            for metric, spec in expected_result_dict.items()
        }
        for node in self.inference_results_dict:
            actuals = {
                metric: coerce_sglang_actual(value)
                for metric, value in self.inference_results_dict[node].items()
                if metric in thresholds
            }
            try:
                evaluate_all(actuals, thresholds)
            except ThresholdViolation as exc:
                for msg in exc.violations:
                    fail_test(f"FAIL - {msg}")

        self.inference_end_time = self._host_exec('date +"%a %b %e %H:%M"')
        time.sleep(2)
        verify_dmesg_for_errors(self.orch.head, self.inference_start_time, self.inference_end_time)

    def run_lm_eval_hellaswag_benchmark_test(self, _d_type='auto'):
        return self.run_lm_eval_benchmark_test('lm_eval_hellaswag', _d_type=_d_type)

    def run_lm_eval_gsm8k_benchmark_test(self, _d_type='auto'):
        return self.run_lm_eval_benchmark_test('lm_eval_gsm8k', _d_type=_d_type)

    def run_lm_eval_benchmark_test(self, bench_key: str, _d_type='auto'):
        spec = LM_EVAL_SPECS[bench_key]
        task_name = bench_key.removeprefix('lm_eval_')
        i_dict = self.bp_dict['inference_tests'][bench_key]
        inner_cmd, scoring = LmEvalBenchmark.prepare(
            i_dict,
            port=int(self.router_serv_port),
            host=self.client_host,
            model_id=self.bp_dict['model'],
            task_name=task_name,
            default_tasks=task_name,
            default_metric=spec['default_metric'],
            default_metric_key=spec['default_metric_key'],
            log_dir=self.log_dir,
            log_basename=f'{bench_key}.log',
            default_num_concurrent=spec['default_num_concurrent'],
        )
        inner = (
            f"mkdir -p {self.log_dir}/benchmark_node && "
            f"source /tmp/server_env_script.sh && {inner_cmd}"
        )
        out_dict = self._container_exec(
            "bash -c " + shlex.quote(inner),
            timeout=scoring['exec_timeout_sec'],
        )
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
                error=errors[-1] if errors else 'no benchmark nodes produced output to score',
            )
            errors.append(f"lm-eval {spec['display']}: no benchmark nodes produced output to score")

        for msg in errors:
            fail_test(msg)
        return summary

    def sglang_disagg_gpu_counts(self, mem_threshold_mb=5000):
        tp = int(self.bp_dict['tensor_parallelism'])
        per_node = {}
        for node, payload in self._host_exec('sudo amd-smi metric --json').items():
            count = 0
            try:
                entries = json.loads(payload.strip())
            except (json.JSONDecodeError, AttributeError):
                per_node[node] = 0
                continue
            if isinstance(entries, dict) and 'gpu_data' in entries:
                entries = entries['gpu_data']
            if not isinstance(entries, list):
                per_node[node] = 0
                continue
            for g in entries:
                used_mb = g.get('mem_usage', {}).get('used_vram', {}).get('value', 0)
                if used_mb > mem_threshold_mb:
                    count += 1
            per_node[node] = count

        occupied = sum(per_node.values())
        return {
            'configured_tp': tp,
            'server_per_node': per_node,
            'server_occupied_gpus': occupied,
        }