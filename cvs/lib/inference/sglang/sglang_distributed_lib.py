'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Multi-node unified SGLang inference controller (TP/PP across server nodes, no PD disagg).

Each host in ``server_node_list`` (or the union of ``prefill_node_list`` +
``decode_node_list``) runs ``sglang.launch_server`` with ``--nnodes`` /
``--node-rank`` / ``--dist-init-addr``. Benchmark/smoke/lm-eval run on
``benchmark_serv_node`` and target rank-0 HTTP (``127.0.0.1`` when bench is rank 0).
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
    collect_sglang_gpu_topology,
    first_output,
    format_sglang_gpu_topology_lines,
    parse_inference_bench_results,
    poll_for_inference_completion as poll_for_inference_completion_common,
    resolve_distributed_client_host,
    resolve_server_node_list,
    run_lm_eval_benchmark_test as run_lm_eval_benchmark_test_common,
    verify_inference_results as verify_inference_results_common,
    verify_inference_results_subtests as verify_inference_results_subtests_common,
    verify_openai_compatible_endpoints as verify_openai_compatible_endpoints_common,
    _SERVER_READY_RE,
)
from cvs.lib.utils_lib import fail_test

log = globals.log


class SglangDistributed:
    """Unified multi-node SGLang serve + benchmark via ``ContainerOrchestrator``."""

    def __init__(
        self,
        model_name,
        inference_config_dict,
        benchmark_params_dict,
        hf_token,
        orch=None,
        gpu_type='mi325',
        user_name=None,
        priv_key_file=None,
    ):
        if orch is None:
            raise ValueError("SglangDistributed requires orch= (ContainerOrchestrator)")

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

        self.server_node_list = resolve_server_node_list(self.inf_dict)
        self.nnodes = int(self.inf_dict.get('nnodes') or len(self.server_node_list))
        if self.nnodes != len(self.server_node_list):
            raise ValueError(
                f"sglang_distributed nnodes={self.nnodes} must match "
                f"server node count {len(self.server_node_list)} ({self.server_node_list!r})"
            )
        self.rank0_node = self.server_node_list[0]
        self.dist_init_addr = self._resolve_dist_init_addr()
        self.benchmark_serv_node = self._resolve_benchmark_serv_node()

        self.container_name = self.inf_dict['container_name']
        self.nic_type = self.inf_dict['nic_type']
        self.hca_id_prefix = str(self.inf_dict['hca_id_prefix']).strip()
        self.log_dir = self.inf_dict['log_dir']
        self.inference_poll_iterations = self.bp_dict['inference_poll_iterations']

        self.inference_start_time = self._host_exec('date +"%a %b %e %H:%M"')
        self.inference_end_time = None

        log.info('distributed inference_dict = %s', self.inf_dict)
        log.info('distributed benchmark_params_dict = %s', self.bp_dict)
        log.info(
            'distributed server_node_list=%s nnodes=%s rank0=%s client_host=%s '
            'router_serv_port=%s benchmark_serv_node=%s dist_init=%s',
            self.server_node_list,
            self.nnodes,
            self.rank0_node,
            self.client_host,
            self.router_serv_port,
            self.benchmark_serv_node,
            self.dist_init_addr,
        )

    def _resolve_dist_init_addr(self) -> str:
        addr = self.inf_dict.get('dist_init_addr') or self.rank0_node
        port = self.inf_dict.get('dist_init_port') or '40001'
        return f"{addr}:{port}"

    def _resolve_benchmark_serv_node(self) -> str:
        raw = self.inf_dict.get('benchmark_serv_node')
        if not raw:
            return self.rank0_node
        hosts = as_node_list(raw)
        if len(hosts) != 1:
            raise ValueError(f"SglangDistributed requires exactly one benchmark_serv_node, got {hosts!r}")
        return hosts[0]

    @property
    def _head_host(self) -> str:
        return self.rank0_node

    def server_log_path(self, rank: int = 0) -> str:
        return f"{self.log_dir}/server_node{rank}/server.log"

    @property
    def router_serv_port(self) -> str:
        return str(self.inf_dict['proxy_router_serv_port'])

    @property
    def client_host(self) -> str:
        return resolve_distributed_client_host(
            self.inf_dict,
            rank0_node=self.rank0_node,
            benchmark_serv_node=self.benchmark_serv_node,
        )

    def _container_exec(
        self,
        cmd: str,
        *,
        hosts=None,
        timeout: int | None = None,
    ) -> dict:
        normalized = as_node_list(hosts) if hosts is not None else self.server_node_list
        return self.orch.exec(cmd, hosts=normalized, timeout=timeout)

    def _bench_exec(self, cmd: str, *, timeout: int | None = None) -> dict:
        return self.orch.exec(cmd, hosts=[self.benchmark_serv_node], timeout=timeout)

    def _container_exec_text(
        self,
        cmd: str,
        *,
        hosts=None,
        timeout: int | None = None,
    ) -> str:
        return first_output(self._container_exec(cmd, hosts=hosts, timeout=timeout))

    def _host_exec(
        self,
        cmd: str,
        *,
        hosts=None,
        timeout: int | None = None,
    ) -> dict:
        """Run ``cmd`` on baremetal (``orch.head`` / ``orch.all``), e.g. amd-smi / dmesg."""
        if hosts is None:
            host = self.benchmark_serv_node
            if host == self.orch.head_node and len(self.orch.hosts) == 1:
                return self.orch.head.exec(cmd, timeout=timeout)
            return BaremetalOrchestrator.exec(self.orch, cmd, hosts=[host], timeout=timeout)
        normalized = as_node_list(hosts)
        if not normalized:
            return {}
        if len(normalized) == 1 and normalized[0] == self._head_host:
            return self.orch.head.exec(cmd, timeout=timeout)
        if set(normalized) == set(self.orch.hosts):
            return self.orch.all.exec(cmd, timeout=timeout)
        return BaremetalOrchestrator.exec(self.orch, cmd, hosts=normalized, timeout=timeout)

    def _host_exec_text(
        self,
        cmd: str,
        *,
        hosts=None,
        timeout: int | None = None,
    ) -> str:
        return first_output(self._host_exec(cmd, hosts=hosts, timeout=timeout))

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
        self.bp_dict.setdefault('pipeline_parallelism', '1')
        self.bp_dict.setdefault('memory_fraction', '0.85')
        self.bp_dict.setdefault('inference_poll_iterations', '16')

    def _server_env_body(self) -> str:
        return (
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
            f"export PP={self.bp_dict.get('pipeline_parallelism', '1')}\n"
            f"export HF_TOKEN={self.hf_token}\n"
            f"{add_export_env_block(self.bp_dict, indent='')}\n"
        )

    def _write_server_env_on_hosts(self, hosts: list[str]) -> None:
        env_body = self._server_env_body()
        write_cmd = "bash -c " + shlex.quote(
            f"cat > /tmp/server_env_script.sh <<'EOF'\n{env_body}EOF\n"
            "chmod 755 /tmp/server_env_script.sh && /tmp/server_env_script.sh"
        )
        self._container_exec(write_cmd, hosts=hosts)

    def setup_server_container_env(self) -> None:
        """Write and source ``/tmp/server_env_script.sh`` on all server nodes."""
        time.sleep(3)
        self._write_server_env_on_hosts(self.server_node_list)
        time.sleep(5)

    def setup_benchmark_serv_container_env(self) -> None:
        self.setup_server_container_env()
        if self.benchmark_serv_node not in self.server_node_list:
            self._write_server_env_on_hosts([self.benchmark_serv_node])

    def launch_server(self, dtype='auto', kv_cache_dtype='auto') -> None:
        """Launch unified multi-node ``sglang.launch_server`` (no PD disagg)."""
        log.info(
            'Launch unified multi-node SGLang on %d nodes (rank0=%s:%s)',
            self.nnodes,
            self.rank0_node,
            self.router_serv_port,
        )
        flags_block = add_cli_flags_block(self.bp_dict, indent='    ')
        pp = self.bp_dict.get('pipeline_parallelism', '1')

        for i, node in enumerate(self.server_node_list):
            host_flag = '0.0.0.0' if i == 0 else node
            launch_body = (
                f"export NNODES={self.nnodes}\n"
                f"export NODE_RANK={i}\n"
                f"python3 -m sglang.launch_server --model {self.bp_dict['model']} \\\n"
                f"    --host {host_flag} \\\n"
                f"    --port {self.router_serv_port} \\\n"
                f"    --dtype {dtype} \\\n"
                f"    --kv-cache-dtype {kv_cache_dtype} \\\n"
                f"    --trust-remote-code \\\n"
                f"    --tp-size {self.bp_dict['tensor_parallelism']} \\\n"
                f"    --pp-size {pp} \\\n"
                f"    --nnodes {self.nnodes} \\\n"
                f"    --node-rank {i} \\\n"
                f"    --dist-init-addr {self.dist_init_addr} \\\n"
                f"    --disable-radix-cache --disable-cuda-graph \\\n"
                f"    --mem-fraction-static {self.bp_dict['memory_fraction']} \\\n"
                f"{flags_block}\n"
                f"    --log-level {self.inf_dict['log_level']}\n"
            )
            write_cmd = "bash -c " + shlex.quote(f"cat > /tmp/server_launch_script.sh <<'EOF'\n{launch_body}EOF")
            self._container_exec(write_cmd, hosts=[node])

        for i, node in enumerate(self.server_node_list):
            start_cmd = "bash -c " + shlex.quote(
                f"chmod 755 /tmp/server_launch_script.sh\n"
                f"mkdir -p {self.log_dir}/server_node{i}\n"
                f"source /tmp/server_env_script.sh\n"
                f"nohup /tmp/server_launch_script.sh > {self.server_log_path(i)} 2>&1 &"
            )
            self._container_exec(start_cmd, hosts=[node])
        time.sleep(5)

    def poll_for_server_ready(self, no_of_iterations=16) -> None:
        log_path = self.server_log_path(0)
        for iteration in range(1, no_of_iterations):
            log.info('Starting rank-0 server readiness poll iteration %d', iteration)
            grep_cmd = f"grep -B 20 -A 20 -E {_SERVER_READY_RE.pattern!r} {shlex.quote(log_path)} || true"
            text = self._container_exec_text(grep_cmd, hosts=[self.rank0_node])
            if _SERVER_READY_RE.search(text):
                log.info('Wait 60 secs before serving traffic')
                time.sleep(60)
                return
            log.info('Wait 120 secs and continue polling')
            time.sleep(120)
        fail_test(
            f'Distributed rank-0 server on {self.rank0_node} did not reach ready state in {no_of_iterations} iterations'
        )

    def poll_and_check_server_ready(self) -> None:
        log.info('Waiting 120 secs after launching distributed server')
        time.sleep(120)
        self.poll_for_server_ready()

    def install_container_packages(self) -> None:
        self._container_exec(
            "bash -c " + shlex.quote("sudo apt -y update && sudo apt install -y iputils-ping iproute2 net-tools")
        )

    def exec_nic_setup_scripts(self) -> None:
        if re.search('broadcom|thor', self.nic_type, re.I):
            self.inf_dict['nccl_ib_gid_index'] = 3
            cmd = "bash -c " + shlex.quote(f"cp {self.mount_vol}.host {self.mount_vol}; sleep 2; ibv_devinfo; sleep 2;")
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
            'OpenAI endpoint probe inside bench container (%s:%r)',
            self.client_host,
            port,
        )
        out_dict = self._bench_exec("bash -c " + shlex.quote(inner), timeout=min(900, 480 + 180))
        raw_out = out_dict.get(self.benchmark_serv_node) or self._first_output(out_dict)

        probe_err: Optional[str] = None
        results: dict[str, tuple[int, Any]] = {}
        if not raw_out or not str(raw_out).strip():
            probe_err = f"OpenAI-compatible probe produced no output on {self.benchmark_serv_node!r}: {out_dict!r}"
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
            exec_probe=lambda cmd, timeout: self._bench_exec(cmd, timeout=timeout),
            probe_host_key=self.benchmark_serv_node,
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
        self._bench_exec("bash -c " + shlex.quote(inner), timeout=1000)
        time.sleep(5)
        self.poll_for_inference_completion(iterations=10, waittime_between_iters=60)

        tp = int(self.bp_dict.get('tensor_parallelism', 1))
        int(self.bp_dict.get('pipeline_parallelism', 1))
        num_gpus = self.nnodes * tp
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
            return self._bench_exec(f"tail -1000 {shlex.quote(log_path)}")

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

    def sglang_distributed_gpu_counts(self, mem_threshold_mb=5000):
        tp = int(self.bp_dict["tensor_parallelism"])
        pp = int(self.bp_dict.get("pipeline_parallelism", 1))

        topo = collect_sglang_gpu_topology(
            self._host_exec,
            {"server": self.server_node_list},
            mem_threshold_mb=mem_threshold_mb,
        )
        server = topo["groups"]["server"]

        result = {
            "configured_tp": tp,
            "configured_pp": pp,
            "configured_nnodes": self.nnodes,
            "server_per_node": server["per_node"],
            "total_occupied_gpus": topo["total_occupied_gpus"],
        }
        log.info(
            "\n".join(
                format_sglang_gpu_topology_lines(
                    configured_tp=tp,
                    configured_pp=pp,
                    configured_nnodes=self.nnodes,
                    groups={"Server nodes": server},
                )
            )
        )
        return result

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
            exec_bench=lambda cmd, timeout: self._bench_exec(cmd, timeout=timeout),
        )

        inner = f"mkdir -p {self.log_dir}/benchmark_node && source /tmp/server_env_script.sh && {inner_cmd}"
        out_dict = self._bench_exec(
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
