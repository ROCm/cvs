'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Single-node SGLang inference controller (no PD disaggregation).

One container on ``benchmark_serv_node`` (via ``ContainerOrchestrator``) runs a unified
``sglang.launch_server`` on ``proxy_router_serv_port``. Benchmark/smoke/lm-eval
traffic hits that port via ``client_host`` (default ``127.0.0.1`` inside the
container).
'''

from __future__ import annotations

import os
import re
import shlex
import time

from cvs.lib import globals
from cvs.core.orchestrators.baremetal import BaremetalOrchestrator
from cvs.lib.inference.sglang.sglang_common import (
    add_cli_flags_block,
    add_export_env_block,
    as_node_list,
    first_output,
    parse_inference_bench_results,
    poll_for_inference_completion as poll_for_inference_completion_common,
    resolve_client_host,
    run_lm_eval_benchmark_test as run_lm_eval_benchmark_test_common,
    verify_inference_results as verify_inference_results_common,
    verify_inference_results_subtests as verify_inference_results_subtests_common,
    verify_openai_compatible_endpoints as verify_openai_compatible_endpoints_common,
    _SERVER_READY_RE,
)
from cvs.lib.utils_lib import fail_test

log = globals.log


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

        self.inference_results_dict = {}
        log.info("%s", self.gpu_type)

        self.home_dir = os.path.expanduser("~")
        self._apply_inf_defaults()
        self._apply_bp_defaults()

        self.container_name = self.inf_dict['container_name']
        self.log_dir = self.inf_dict['log_dir']
        self.inference_poll_iterations = self.bp_dict['inference_poll_iterations']
        self.benchmark_serv_node = self._resolve_benchmark_serv_node()

        self.inference_start_time = self._host_exec('date +"%a %b %e %H:%M"')
        self.inference_end_time = None

        log.info('single-node inference_dict = %s', self.inf_dict)
        log.info('single-node benchmark_params_dict = %s', self.bp_dict)
        log.info(
            'single-node client_host=%s router_serv_port=%s benchmark_serv_node=%s',
            self.client_host,
            self.router_serv_port,
            self.benchmark_serv_node,
        )

    def _resolve_benchmark_serv_node(self) -> str:
        raw = self.inf_dict.get('benchmark_serv_node')
        if not raw:
            raise ValueError("SglangSingle requires benchmark_serv_node in the inference config")
        hosts = as_node_list(raw)
        if len(hosts) != 1:
            raise ValueError(f"SglangSingle requires exactly one benchmark_serv_node, got {hosts!r}")
        return hosts[0]

    @property
    def _head_host(self) -> str:
        return self.benchmark_serv_node

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

    def _container_exec(self, cmd: str, *, timeout: int | None = None) -> dict:
        """Run ``cmd`` inside the container."""
        return self.orch.exec(cmd, timeout=timeout)

    def _container_exec_text(self, cmd: str, *, timeout: int | None = None) -> str:
        return first_output(self._container_exec(cmd, timeout=timeout))

    def _host_exec(self, cmd: str, *, timeout: int | None = None) -> dict:
        """Run ``cmd`` on ``benchmark_serv_node`` (baremetal), e.g. amd-smi / dmesg."""
        host = self.benchmark_serv_node
        if host == self.orch.head_node and len(self.orch.hosts) == 1:
            return self.orch.head.exec(cmd, timeout=timeout)
        return BaremetalOrchestrator.exec(self.orch, cmd, hosts=[host], timeout=timeout)

    def _host_exec_text(self, cmd: str, *, timeout: int | None = None) -> str:
        return first_output(self._host_exec(cmd, timeout=timeout))

    def _apply_inf_defaults(self) -> None:
        self.inf_dict.setdefault('container_image', 'lmsysorg/sglang:dev')
        self.inf_dict.setdefault('container_name', 'sglang_container')
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
            grep_cmd = f"grep -B 20 -A 20 -E {_SERVER_READY_RE.pattern!r} {shlex.quote(self.server_log_path)} || true"
            text = self._container_exec_text(grep_cmd)
            if _SERVER_READY_RE.search(text):
                log.info('Wait 60 secs before serving traffic')
                time.sleep(60)
                return
            log.info('Wait 120 secs and continue polling')
            time.sleep(120)
        fail_test(f'Single-node server on {self._head_host} did not reach ready state in {no_of_iterations} iterations')

    def poll_and_check_server_ready(self) -> None:
        log.info('Waiting 120 secs after launching server')
        time.sleep(120)
        self.poll_for_server_ready()

    def setup_benchmark_serv_container_env(self) -> None:
        self.setup_server_container_env()

    def install_container_packages(self) -> None:
        self._container_exec(
            "bash -c " + shlex.quote("sudo apt -y update && sudo apt install -y iputils-ping iproute2 net-tools")
        )

    def run_test_rmsnorm(self, max_jobs=192) -> None:
        self._container_exec(
            "bash -c "
            + shlex.quote(
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
        probe_src = OpenAIProbe.probe_script(port, self.bp_dict['model'], host=self.client_host)
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

    def benchserv_test_random(self, d_type='auto') -> None:
        return verify_openai_compatible_endpoints_common(
            port=int(self.router_serv_port),
            model_name=self.bp_dict['model'],
            client_host=self.client_host,
            log_dir=self.log_dir,
            exec_probe=lambda cmd, timeout: self._container_exec(cmd, timeout=timeout),
            probe_host_key=self._head_host,
        )

    def benchserv_test_random(self, d_type='auto', *, verify=True) -> None:
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

        if verify:
            self.verify_inference_results('bench_serv', i_dict['expected_results'][d_type])

    def get_inference_results_dict(self, out_dict):
        self.inference_results_dict = parse_inference_bench_results(
            out_dict,
            bench_num_prompts=getattr(self, '_bench_num_prompts', None),
        )
        return self.inference_results_dict

    def poll_for_inference_completion(
        self, iterations=10, waittime_between_iters=60, total_timeout=3600, require_all_nodes=True
    ):
        log_path = f"{self.log_dir}/benchmark_node/benchmark_results.log"

        def fetch_log_tail():
            return self._container_exec(f"tail -1000 {shlex.quote(log_path)}")

        result = poll_for_inference_completion_common(
            fetch_log_tail,
            self.get_inference_results_dict,
            iterations=iterations,
            waittime_between_iters=waittime_between_iters,
            total_timeout=total_timeout,
            require_all_nodes=require_all_nodes,
            inference_poll_iterations=self.inference_poll_iterations,
        )
        if result.get('status') == 'success':
            self.inference_results_dict = result['results']
        return result

    def verify_inference_results(self, test_name, expected_result_dict):
        thresholds = {
            metric: normalize_sglang_threshold_spec(metric, spec) for metric, spec in expected_result_dict.items()
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

        self.inference_end_time = verify_inference_results_common(
            self.inference_results_dict,
            expected_result_dict,
            self._host_exec,
            test_name=test_name,
        )

    def verify_inference_results_subtests(
        self,
        subtests,
        test_name,
        expected_result_dict,
        *,
        lifecycle=None,
        report_nodeid=None,
    ) -> bool:
        all_passed, self.inference_end_time = verify_inference_results_subtests_common(
            self.inference_results_dict,
            expected_result_dict,
            self._host_exec,
            subtests,
            test_name,
            lifecycle=lifecycle,
            report_nodeid=report_nodeid,
        )
        return all_passed

    def run_lm_eval_hellaswag_benchmark_test(self, _d_type='auto'):
        return self.run_lm_eval_benchmark_test('lm_eval_hellaswag', _d_type=_d_type)

    def run_lm_eval_gsm8k_benchmark_test(self, _d_type='auto'):
        return self.run_lm_eval_benchmark_test('lm_eval_gsm8k', _d_type=_d_type)

    def run_lm_eval_benchmark_test(self, bench_key: str, _d_type='auto'):
        return run_lm_eval_benchmark_test_common(
            bench_key,
            bp_dict=self.bp_dict,
            router_serv_port=self.router_serv_port,
            client_host=self.client_host,
            log_dir=self.log_dir,
            env_script='/tmp/server_env_script.sh',
            exec_bench=lambda cmd, timeout: self._container_exec(cmd, timeout=timeout),
        )
        inner = f"mkdir -p {self.log_dir}/benchmark_node && source /tmp/server_env_script.sh && {inner_cmd}"
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
