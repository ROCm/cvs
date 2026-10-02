'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Single-node SGLang inference controller (no PD disaggregation).

The first host in ``cluster.json`` gets a container and a full-model unified
``sglang.launch_server`` (local TP). Extra hosts are ignored. HTTP defaults to
port 8000. Benchmark, smoke, and lm-eval traffic hits ``127.0.0.1`` in that container.
'''

from __future__ import annotations

import os
import re
import shlex
import time

from cvs.lib import globals
from cvs.lib.inference.sglang.sglang_common import (
    DEFAULT_SGLANG_SERVE_PORT,
    add_cli_flags_block,
    add_export_env_block,
    as_node_list,
    log_sglang_log_matches,
    parse_inference_bench_results,
    perf_enforce_thresholds,
    poll_for_inference_completion as poll_for_inference_completion_common,
    resolve_client_host,
    run_lm_eval_benchmark_test as run_lm_eval_benchmark_test_common,
    scan_sglang_error_logs,
    SGLANG_ERROR_PATTERNS,
    stage_and_start_launch_script,
    verify_inference_results as verify_inference_results_common,
    verify_inference_results_subtests as verify_inference_results_subtests_common,
    verify_openai_compatible_endpoints as verify_openai_compatible_endpoints_common,
    _SERVER_READY_RE,
)
from cvs.lib.utils_lib import fail_test

log = globals.log


class SglangSingle:
    """Full-model SGLang serve + benchmark on the first cluster host only."""

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
        self.inference_poll_iterations = int(self.bp_dict['inference_poll_iterations'])
        self.inference_poll_total_timeout_sec = int(self.bp_dict['inference_poll_total_timeout_sec'])
        self.execution_hosts = self._resolve_execution_hosts()
        self.inf_dict['benchmark_serv_node'] = self.execution_hosts[0]

        self.inference_start_time = self._host_exec('date +"%a %b %e %H:%M"')
        self.inference_end_time = None

        log.info('single-node inference_dict = %s', self.inf_dict)
        log.info('single-node benchmark_params_dict = %s', self.bp_dict)
        log.info(
            'single-node client_host=%s router_serv_port=%s execution_hosts=%s',
            self.client_host,
            self.router_serv_port,
            self.execution_hosts,
        )

    def _resolve_execution_hosts(self):
        hosts = list(self.inf_dict.get('_execution_hosts') or self.orch.hosts or [])
        if not hosts:
            raise ValueError("SglangSingle requires at least one orchestrator host from cluster.json")
        return hosts[:1]

    def _node_log_dir(self, host):
        return f"{self.log_dir}/{host}"

    def _server_log_path(self, host):
        return f"{self._node_log_dir(host)}/server.log"

    def _bench_log_path(self, host):
        return f"{self._node_log_dir(host)}/benchmark_results.log"

    @property
    def router_serv_port(self):
        """Unified server listen/client port (defaults to 8000)."""
        return str(self.inf_dict.get('proxy_router_serv_port') or DEFAULT_SGLANG_SERVE_PORT)

    @property
    def client_host(self) -> str:
        """HTTP client target when smoke/bench/lm-eval run inside the same container."""
        return resolve_client_host(self.inf_dict, unified_server=True)

    def _container_exec(self, cmd, *, timeout=None, hosts=None):
        """Run ``cmd`` inside the container (default: every execution host)."""
        kwargs = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        if hosts is not None:
            kwargs["hosts"] = as_node_list(hosts)
        return self.orch.exec(cmd, **kwargs)

    def _container_exec_per_host(self, cmd_for_host, *, timeout=None):
        cmds = [cmd_for_host(host) for host in self.execution_hosts]
        kwargs = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return self.orch.exec_cmd_list(cmds, **kwargs)

    def _host_exec(self, cmd, *, timeout=None):
        """Run ``cmd`` on the first cluster host OS (not inside the container)."""
        return self.orch.exec_on_host(cmd, hosts=self.execution_hosts, timeout=timeout)

    def _apply_inf_defaults(self):
        self.inf_dict.setdefault('container_image', 'lmsysorg/sglang:dev')
        self.inf_dict.setdefault('container_name', 'sglang_container')
        self.inf_dict.setdefault('nccl_debug', 'ERROR')
        self.inf_dict.setdefault('data_cache_dir', f'{self.home_dir}/cache')
        self.inf_dict.setdefault('log_dir', f'{self.home_dir}/LOG_DIR')
        self.inf_dict.setdefault('log_level', 'info')
        self.inf_dict.setdefault('proxy_router_serv_port', DEFAULT_SGLANG_SERVE_PORT)

    def _apply_bp_defaults(self) -> None:
        self.bp_dict.setdefault('backend', 'sglang')
        self.bp_dict.setdefault('max_concurrency', '64')
        self.bp_dict.setdefault('model', 'openai/gpt-oss-120b')
        self.bp_dict.setdefault('tensor_parallelism', '8')
        self.bp_dict.setdefault('memory_fraction', '0.85')
        self.bp_dict.setdefault('inference_poll_iterations', '16')
        self.bp_dict.setdefault('inference_poll_total_timeout_sec', '3600')

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

    def launch_server(self, dtype='auto', kv_cache_dtype='auto'):
        """Launch one unified SGLang server on the first cluster host (full local TP)."""
        log.info(
            'Launch unified SGLang server on 0.0.0.0:%s hosts=%s',
            self.router_serv_port,
            self.execution_hosts,
        )
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

        for host in self.execution_hosts:
            stage_and_start_launch_script(
                self._container_exec,
                host,
                '/tmp/server_launch_script.sh',
                launch_body,
                '/tmp/server_env_script.sh',
                self._server_log_path(host),
            )
        time.sleep(5)

    def poll_for_server_ready(self, no_of_iterations=16):
        pending = set(self.execution_hosts)
        for iteration in range(1, no_of_iterations):
            log.info('Starting server readiness poll iteration %d remaining=%s', iteration, sorted(pending))

            def grep_cmd(host):
                return (
                    f"grep -B 20 -A 20 -E {_SERVER_READY_RE.pattern!r} "
                    f"{shlex.quote(self._server_log_path(host))} || true"
                )

            out_dict = self._container_exec_per_host(grep_cmd)
            ready = [host for host in list(pending) if _SERVER_READY_RE.search(out_dict.get(host) or '')]
            pending.difference_update(ready)
            if not pending:
                log.info('Wait 60 secs before serving traffic')
                time.sleep(60)
                return
            log.info('Wait 120 secs and continue polling; not ready: %s', sorted(pending))
            time.sleep(120)
        fail_test(
            f'Single-node servers did not reach ready state in {no_of_iterations} iterations on hosts {sorted(pending)}'
        )

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

    def verify_openai_compatible_endpoints(self):
        summaries = []
        self.openai_completions_5xx_or_hang = False
        for host in self.execution_hosts:
            host_summaries, completions_5xx_or_hang = verify_openai_compatible_endpoints_common(
                port=int(self.router_serv_port),
                model_name=self.bp_dict['model'],
                client_host=self.client_host,
                log_dir=self._node_log_dir(host),
                exec_probe=lambda cmd, timeout, h=host: self._container_exec(cmd, hosts=[h], timeout=timeout),
                probe_host_key=host,
            )
            summaries.extend(host_summaries)
            self.openai_completions_5xx_or_hang = self.openai_completions_5xx_or_hang or completions_5xx_or_hang
        self.log_server_error_logs()
        return summaries

    def benchserv_test_random(self, d_type='auto', *, verify=True):
        i_dict = self.bp_dict['inference_tests']['bench_serv_random']
        self._bench_num_prompts = int(i_dict['num_prompts'])

        def inner(host):
            log_dir = self._node_log_dir(host)
            log_path = self._bench_log_path(host)
            return "bash -c " + shlex.quote(
                f"mkdir -p {shlex.quote(log_dir)}\n"
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
                f"  > {shlex.quote(log_path)} 2>&1"
            )

        self._container_exec_per_host(inner, timeout=1000)
        time.sleep(5)
        self.poll_for_inference_completion()

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
        self, iterations=None, waittime_between_iters=60, total_timeout=None, require_all_nodes=True
    ):
        if iterations is None:
            iterations = int(self.inference_poll_iterations)
        if total_timeout is None:
            total_timeout = int(self.inference_poll_total_timeout_sec)

        def fetch_log_tail():
            return self._container_exec_per_host(lambda host: f"tail -1000 {shlex.quote(self._bench_log_path(host))}")

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

    def _server_log_targets(self):
        return [(host, self._server_log_path(host), 'server') for host in self.execution_hosts]

    def scan_for_inference_errors(self):
        """Scan this run's server logs once after server-ready polling fails."""
        return scan_sglang_error_logs(
            self._server_log_targets(),
            lambda host, cmd: self._container_exec(cmd, hosts=[host], timeout=60),
        )

    def log_server_error_logs(self):
        """Copy error-signature lines from server logs into the current test."""
        log_sglang_log_matches(
            self._server_log_targets(),
            lambda host, cmd: self._container_exec(cmd, hosts=[host], timeout=60),
            SGLANG_ERROR_PATTERNS,
            heading='Server log matches',
        )

    def verify_inference_results(self, test_name, expected_result_dict):
        self.inference_end_time = verify_inference_results_common(
            self.inference_results_dict,
            expected_result_dict,
            self._host_exec,
            test_name=test_name,
            enforce_thresholds=perf_enforce_thresholds(self.bp_dict),
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
            enforce_thresholds=perf_enforce_thresholds(self.bp_dict),
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
