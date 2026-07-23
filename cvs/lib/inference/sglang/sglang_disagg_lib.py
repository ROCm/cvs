'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Disaggregated Prefill/Decode (PD) SGLang inference controller.

Prefill, decode, proxy-router, and benchmark workloads run inside containers
on their respective cluster nodes via ``ContainerOrchestrator`` (``orch=``).
Bare-metal SSH (``orch.head`` / ``orch.all``) is used for ``amd-smi`` and
``dmesg`` verification.
'''

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import time
from typing import Any, Mapping, Optional

from cvs.lib import globals
from cvs.core.orchestrators.baremetal import BaremetalOrchestrator
from cvs.lib.inference.sglang.sglang_common import (
    LM_EVAL_SPECS,
    add_cli_flags_block,
    add_export_env_block,
    as_node_list,
    coerce_sglang_actual,
    first_float,
    normalize_sglang_threshold_spec,
    resolve_client_host,
)
from cvs.lib.utils.model_query_lib import LmEvalBenchmark, LongContextNiahBenchmark, OpenAIProbe
from cvs.lib.utils_lib import fail_test
from cvs.lib.utils.verdict import ThresholdViolation, evaluate_all
from cvs.lib.verify_lib import verify_dmesg_for_errors

log = globals.log

_SERVER_READY_RE = re.compile(
    r"fired up and ready to roll|Uvicorn running|Application startup complete|200 OK",
    re.I,
)

inference_err_dict = {
    'NCCL ERROR': 'NCCL ERROR|NCCL timeout|local work queue catastrophic error',
    'GPU HW ERROR': 'HW Exception by GPU|GPU Hang|Uncorrectable error|GPU Reset',
    'AssertionError': 'AssertionError|ValueError:|During handling of the above exception|triggered the following exception|RuntimeError|Python error: Aborted',
    'rocm Err': 'FAILED_PRECONDITION: No visible GPU devices|failed call to hipInit: HIP_ERROR_NoDevice|librocm reported version is: NOT_FOUND',
    'python err': 'ModuleNotFoundError: No module named|Fatal Python error:',
    'resource': 'RESOURCE_EXHAUSTED: Out of memory|failed: RESOURCE_EXHAUSTED|urllib.error.URLError|ConnectionRefusedError,HSA_STATUS_ERROR_OUT_OF_RESOURCES',
    'app_err': 'Service Unavailable|No decode workers available|No prefill workers available|Please check if decode servers are configured and healthy|Please check if prefill servers are configured and healthy|Cannot access gated repo|You must have access to it and be authenticated',
}

err_counters_pattern = 'err|retransmit|drop|discard|naks|invalid|oflow|out_of_buffer|reset|fail'


class SglangDisaggPD:
    """Disaggregated Prefill/Decode SGLang controller via ``ContainerOrchestrator``."""

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
        """
        Initialize a Disaggregated Prefill/Decode (PD) inference controller
        for SGLang.

        This class encapsulates:
          - Cluster topology (prefill, decode, proxy, benchmark nodes)
          - Container execution via ``ContainerOrchestrator`` (``orch=``)
          - Inference configuration (networking, containers, env vars)
          - Benchmark configuration (load, concurrency, prompt sizes)

        Args:
            model_name (str): HuggingFace or local model identifier
            inference_config_dict (dict): Cluster and runtime configuration
            benchmark_params_dict (dict): Benchmark workload parameters
            hf_token (str): HuggingFace access token
            orch: Required ``ContainerOrchestrator`` instance
            gpu_type (str): GPU type (e.g., mi300, mi325)
            user_name (str): SSH username for remote nodes (optional override)
            priv_key_file (str): SSH private key file (optional override)
        """
        if orch is None:
            raise ValueError("SglangDisaggPD requires orch= (ContainerOrchestrator)")

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

        self.prefill_node_list = self._normalize_hosts(self.inf_dict['prefill_node_list'])
        self.decode_node_list = self._normalize_hosts(self.inf_dict['decode_node_list'])
        self.prefill_nnodes = len(self.prefill_node_list)
        self.decode_nnodes = len(self.decode_node_list)

        self.proxy_node = self._normalize_hosts(self.inf_dict['proxy_router_node'])
        self.benchmark_serv_node = self._normalize_hosts(self.inf_dict['benchmark_serv_node'])

        self.job_cmd = ''
        self.job_cmd_list = []
        self.inference_results_dict = {}
        log.info("%s", self.gpu_type)

        self.rdma_stats_dict_before = {}
        self.ethtool_stats_dict_before = {}
        self.rdma_stats_dict_after = {}
        self.home_dir = os.path.expanduser("~")
        self._apply_inf_defaults()
        self._apply_bp_defaults()

        self.container_name = self.inf_dict['container_name']
        self.nic_type = self.inf_dict['nic_type']
        self.nccl_ib_hca_list = self.inf_dict['nccl_ib_hca_list']
        self.nccl_ib_hca = self.inf_dict['nccl_ib_hca']
        self.nccl_socket_ifname = self.inf_dict['nccl_socket_ifname']
        self.gloo_socket_ifname = self.inf_dict['gloo_socket_ifname']
        self.nccl_ib_gid_index = self.inf_dict['nccl_ib_gid_index']
        self.nccl_debug = self.inf_dict['nccl_debug']
        self.data_cache_dir = self.inf_dict['data_cache_dir']
        self.log_dir = self.inf_dict['log_dir']
        self.hca_id_prefix = str(self.inf_dict['hca_id_prefix']).strip()
        self.inference_poll_iterations = self.bp_dict['inference_poll_iterations']

        self.inference_start_time = self._host_exec('date +"%a %b %e %H:%M"')
        self.inference_end_time = None

        log.info('disagg inference_dict = %s', self.inf_dict)
        log.info('disagg benchmark_params_dict = %s', self.bp_dict)
        log.info(
            'disagg client_host=%s router_serv_port=%s head=%s',
            self.client_host,
            self.router_serv_port,
            self._head_host,
        )

    @property
    def _head_host(self) -> str:
        return self.orch.head_node

    @property
    def router_serv_port(self) -> str:
        """Client-facing proxy router port (bench/smoke/lm-eval)."""
        return str(self.inf_dict['proxy_router_serv_port'])

    @property
    def client_host(self) -> str:
        return resolve_client_host(self.inf_dict, unified_server=False)

    @staticmethod
    def _first_output(out_dict: dict) -> str:
        if not out_dict:
            return ""
        return next(iter(out_dict.values())) or ""

    @staticmethod
    def _normalize_hosts(hosts) -> list[str]:
        """Normalize cluster JSON node field to a list of host strings."""
        if hosts is None:
            return []
        return as_node_list(hosts)

    def _container_exec(
        self,
        cmd: str,
        *,
        hosts=None,
        timeout: int | None = None,
    ) -> dict:
        """Run ``cmd`` inside the container on ``hosts`` (default: all orch hosts)."""
        normalized = self._normalize_hosts(hosts) if hosts is not None else None
        return self.orch.exec(cmd, hosts=normalized, timeout=timeout)

    def _container_exec_text(
        self,
        cmd: str,
        *,
        hosts=None,
        timeout: int | None = None,
    ) -> str:
        return self._first_output(self._container_exec(cmd, hosts=hosts, timeout=timeout))

    def _host_exec(
        self,
        cmd: str,
        *,
        hosts=None,
        timeout: int | None = None,
    ) -> dict:
        """Run ``cmd`` on baremetal (``orch.head`` / ``orch.all``), e.g. amd-smi / dmesg."""
        if hosts is None:
            return self.orch.head.exec(cmd, timeout=timeout)
        normalized = self._normalize_hosts(hosts)
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
        return self._first_output(self._host_exec(cmd, hosts=hosts, timeout=timeout))

    def _apply_inf_defaults(self) -> None:
        self.inf_dict.setdefault('container_image', 'lmsysorg/sglang:dev')
        self.inf_dict.setdefault('container_name', 'sglang_container')
        self.inf_dict.setdefault('nic_type', 'ainic')
        self.inf_dict.setdefault('nccl_ib_hca_list', 'rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7')
        self.inf_dict.setdefault('nccl_ib_hca', 'rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7')
        self.inf_dict.setdefault('hca_id_prefix', 'bnxt_')
        self.inf_dict.setdefault('nccl_socket_ifname', 'eno0')
        self.inf_dict.setdefault('gloo_socket_ifname', 'eno0')
        self.inf_dict.setdefault('nccl_ib_gid_index', '1')
        self.inf_dict.setdefault('nccl_debug', 'ERROR')
        self.inf_dict.setdefault('data_cache_dir', f'{self.home_dir}/cache')
        self.inf_dict.setdefault('log_dir', f'{self.home_dir}/LOG_DIR')
        self.inf_dict.setdefault('log_level', 'info')
        self.inf_dict.setdefault('prefill_serv_port', '30001')
        self.inf_dict.setdefault('decode_serv_port', '30002')
        self.inf_dict.setdefault('proxy_router_port', '8000')
        self.inf_dict.setdefault('proxy_router_serv_port', '8000')
        self.inf_dict.setdefault('max_concurrent_requests', '-1')
        self.inf_dict.setdefault('queue_size', '100')
        self.inf_dict.setdefault('queue_timeout_secs', '60')
        self.inf_dict.setdefault('max_retries', '5')

    def _apply_bp_defaults(self) -> None:
        self.bp_dict.setdefault('backend', 'sglang')
        self.bp_dict.setdefault('dataset_name', 'sharegpt')
        self.bp_dict.setdefault('max_concurrency', '64')
        self.bp_dict.setdefault('model', 'openai/gpt-oss-120b')
        self.bp_dict.setdefault('num_prompts', '1000')
        self.bp_dict.setdefault('input_sequence_length', '8192')
        self.bp_dict.setdefault('burstiness', '1.0')
        self.bp_dict.setdefault('seed', '0')
        self.bp_dict.setdefault('request_rate', 'inf')
        self.bp_dict.setdefault('random_range_ration', '1.0')
        self.bp_dict.setdefault('random_prefix_len', '0')
        self.bp_dict.setdefault('tensor_parallelism', '8')
        self.bp_dict.setdefault('pipeline_parallelism', '1')
        self.bp_dict.setdefault('context_length', '131072')
        self.bp_dict.setdefault('port_no', '8000')
        self.bp_dict.setdefault('tokenizer_mode', 'auto')
        self.bp_dict.setdefault('percentile_metrics', 'ttft,tpot,itl,e2el')
        self.bp_dict.setdefault('metric_percentiles', '99')
        self.bp_dict.setdefault('inference_poll_iterations', '16')
        self.bp_dict.setdefault('memory_fraction', '0.85')

    def install_container_packages(
        self,
    ):
        """
        Install required system networking utilities inside inference containers.

        Purpose:
        --------
        This method prepares the container environment for distributed inference
        by installing basic networking and diagnostic tools that are commonly
        needed for:
        - Connectivity validation between nodes
        - Debugging network paths (ping, ip route, ifconfig)
        - Verifying NIC and routing configuration
        - Troubleshooting NCCL/Gloo/RDMA-related issues

        These tools are installed inside the running container on:
        - Prefill nodes
        - Decode nodes
        - Proxy/router nodes
        """
        log.info('Run pre inference tasks')
        cmd = "bash -c " + shlex.quote(
            "sudo apt -y update && "
            "sudo apt install -y iputils-ping iproute2 net-tools"
        )
        for hosts in (self.prefill_node_list, self.decode_node_list, self.proxy_node):
            self._container_exec(cmd, hosts=hosts)

    def exec_nic_setup_scripts(
        self,
    ):
        """
        Execute NIC-related setup steps inside the inference container.

        Behavior:
        - Only runs for distributed inference.
        - If NIC type appears to be Broadcom/Thor, applies a temporary workaround:
          * Copies the bnxt RDMA library from the host-named file to the container's expected path.
          * Verifies that ibv_devinfo shows a bnxt_ HCA (to confirm RDMA is wired correctly).
        - Forces NCCL GID index to 3 for Broadcom/Thor (common requirement).

        Assumptions:
        - sudo is non-interactive within the container.
        - The bnxt library file paths exist in the container base image.
        """
        if re.search('broadcom|thor', self.nic_type, re.I):
            self.nccl_ib_gid_index = 3
            cmd = "bash -c " + shlex.quote(
                f"cp {self.mount_vol}.host {self.mount_vol}; sleep 2; ibv_devinfo; sleep 2;"
            )
            hca_id_regex = rf'hca_id:\s+{re.escape(self.hca_id_prefix)}'
            for hosts in (self.prefill_node_list, self.decode_node_list):
                out_dict = self._container_exec(cmd, hosts=hosts)
                for node, out in out_dict.items():
                    if not re.search(hca_id_regex, out or '', re.I):
                        log.info("%s", out)
                        fail_test(f'Broadcom libbnxt rdma driver is not properly copied on node {node}')

    def check_ibv_devices(
        self,
    ):
        """
        Verify that InfiniBand / RDMA devices are visible inside the container
        on all relevant nodes.

        Purpose:
        --------
        This method ensures that RDMA-capable devices (e.g., InfiniBand HCAs)
        are correctly exposed inside the container environment. This is a
        critical prerequisite for:
        - NCCL / RCCL over RDMA
        - High-performance distributed inference
        - Low-latency, high-bandwidth GPU communication

        The check is performed on:
        - Prefill nodes
        - Decode nodes

        Proxy and benchmark nodes typically do not require RDMA access.
        """
        for hosts in (self.prefill_node_list, self.decode_node_list):
            out_dict = self._container_exec("ibv_devinfo", hosts=hosts)
            for node, out in out_dict.items():
                if re.search('No IB devices found', out or '', re.I):
                    fail_test(f'IB devices not seen inside the container for node {node}')

    def setup_prefill_container_env(
        self,
    ):
        """Write and source ``/tmp/prefill_env_script.sh`` on prefill nodes."""
        env_body = (
            "export LD_LIBRARY_PATH=/usr/local/lib:/sgl-workspace/Mooncake/build/mooncake-common/etcd:/opt/rocm/lib:$LD_LIBRARY_PATH\n"
            f"export NCCL_DEBUG={self.inf_dict['nccl_debug']}\n"
            f"export NCCL_IB_HCA={self.inf_dict['nccl_ib_hca']}\n"
            f"export NCCL_IB_GID_INDEX={self.inf_dict['nccl_ib_gid_index']}\n"
            f"export NCCL_SOCKET_IFNAME={self.inf_dict['nccl_socket_ifname']}\n"
            f"export GLOO_SOCKET_IFNAME={self.inf_dict['gloo_socket_ifname']}\n"
            f"export GLOO_TCP_IFNAME={self.inf_dict['gloo_socket_ifname']}\n"
            f"export HSA_FORCE_FINE_GRAIN_PCIE=1\n"
            f"export MASTER_PREFILL_ADDR={self.inf_dict['prefill_coordinator_addr']}\n"
            f"export MASTER_PREFILL_PORT={self.inf_dict['prefill_coordinator_port']}\n"
            f"export MODEL={self.bp_dict['model']}\n"
            f"export TP={self.bp_dict['tensor_parallelism']}\n"
            f"export PP={self.bp_dict['pipeline_parallelism']}\n"
            f"export HF_TOKEN={self.hf_token}\n"
            f"{add_export_env_block(self.bp_dict, indent='')}\n"
        )
        write_cmd = "bash -c " + shlex.quote(
            f"cat > /tmp/prefill_env_script.sh <<'EOF'\n{env_body}EOF\n"
            "chmod 755 /tmp/prefill_env_script.sh && /tmp/prefill_env_script.sh"
        )
        time.sleep(3)
        self._container_exec(write_cmd, hosts=self.prefill_node_list)

    def setup_decode_container_env(
        self,
    ):
        """Write and source ``/tmp/decode_env_script.sh`` on decode nodes."""
        env_body = (
            "export LD_LIBRARY_PATH=/usr/local/lib:/sgl-workspace/Mooncake/build/mooncake-common/etcd:/opt/rocm/lib:$LD_LIBRARY_PATH\n"
            f"export NCCL_DEBUG={self.inf_dict['nccl_debug']}\n"
            f"export NCCL_IB_HCA={self.inf_dict['nccl_ib_hca']}\n"
            f"export NCCL_IB_GID_INDEX={self.inf_dict['nccl_ib_gid_index']}\n"
            f"export NCCL_SOCKET_IFNAME={self.inf_dict['nccl_socket_ifname']}\n"
            f"export GLOO_SOCKET_IFNAME={self.inf_dict['gloo_socket_ifname']}\n"
            f"export GLOO_TCP_IFNAME={self.inf_dict['gloo_socket_ifname']}\n"
            f"export HSA_FORCE_FINE_GRAIN_PCIE=1\n"
            f"export MASTER_DECODE_ADDR={self.inf_dict['decode_coordinator_addr']}\n"
            f"export MASTER_DECODE_PORT={self.inf_dict['decode_coordinator_port']}\n"
            f"export MODEL={self.bp_dict['model']}\n"
            f"export TP={self.bp_dict['tensor_parallelism']}\n"
            f"export PP={self.bp_dict['pipeline_parallelism']}\n"
            f"export HF_TOKEN={self.hf_token}\n"
            f"{add_export_env_block(self.bp_dict, indent='')}\n"
        )
        write_cmd = "bash -c " + shlex.quote(
            f"cat > /tmp/decode_env_script.sh <<'EOF'\n{env_body}EOF\n"
            "chmod 755 /tmp/decode_env_script.sh && /tmp/decode_env_script.sh"
        )
        time.sleep(3)
        self._container_exec(write_cmd, hosts=self.decode_node_list)

    def setup_proxy_router_container_env(
        self,
    ):
        """Write and source ``/tmp/router_env_script.sh`` on proxy/router nodes."""
        env_body = (
            "export LD_LIBRARY_PATH=/usr/local/lib:/sgl-workspace/Mooncake/build/mooncake-common/etcd:/opt/rocm/lib:$LD_LIBRARY_PATH\n"
            f"export NCCL_DEBUG={self.inf_dict['nccl_debug']}\n"
            f"export NCCL_IB_HCA={self.inf_dict['nccl_ib_hca']}\n"
            f"export NCCL_IB_GID_INDEX={self.inf_dict['nccl_ib_gid_index']}\n"
            f"export NCCL_SOCKET_IFNAME={self.inf_dict['nccl_socket_ifname']}\n"
            f"export GLOO_SOCKET_IFNAME={self.inf_dict['gloo_socket_ifname']}\n"
            f"export GLOO_TCP_IFNAME={self.inf_dict['gloo_socket_ifname']}\n"
            f"export HSA_FORCE_FINE_GRAIN_PCIE=1\n"
            f"export HF_TOKEN={self.hf_token}\n"
        )
        write_cmd = "bash -c " + shlex.quote(
            f"cat > /tmp/router_env_script.sh <<'EOF'\n{env_body}EOF\n"
            "chmod 755 /tmp/router_env_script.sh && /tmp/router_env_script.sh"
        )
        time.sleep(3)
        self._container_exec(write_cmd, hosts=self.proxy_node)

    def setup_benchmark_serv_container_env(
        self,
    ):
        """Write and source ``/tmp/benchmark_env_script.sh`` on benchmark nodes."""
        env_body = (
            "export LD_LIBRARY_PATH=/usr/local/lib:/sgl-workspace/Mooncake/build/mooncake-common/etcd:/opt/rocm/lib:$LD_LIBRARY_PATH\n"
            f"export NCCL_DEBUG={self.inf_dict['nccl_debug']}\n"
            f"export NCCL_IB_HCA={self.inf_dict['nccl_ib_hca']}\n"
            f"export NCCL_IB_GID_INDEX={self.inf_dict['nccl_ib_gid_index']}\n"
            f"export NCCL_SOCKET_IFNAME={self.inf_dict['nccl_socket_ifname']}\n"
            f"export GLOO_SOCKET_IFNAME={self.inf_dict['gloo_socket_ifname']}\n"
            f"export GLOO_TCP_IFNAME={self.inf_dict['gloo_socket_ifname']}\n"
            f"export HSA_FORCE_FINE_GRAIN_PCIE=1\n"
            f"export HF_TOKEN={self.hf_token}\n"
        )
        write_cmd = "bash -c " + shlex.quote(
            f"cat > /tmp/benchmark_env_script.sh <<'EOF'\n{env_body}EOF\n"
            "chmod 755 /tmp/benchmark_env_script.sh && /tmp/benchmark_env_script.sh"
        )
        time.sleep(3)
        self._container_exec(write_cmd, hosts=self.benchmark_serv_node)
        time.sleep(5)

    def run_test_rmsnorm(self, max_jobs=192):
        """
        Run RMSNorm 2D operator tests inside the SGLang container across
        relevant nodes and validate correctness.

        Purpose:
        --------
        This method executes the AITER RMSNorm 2D operator test, which validates:
        - Correctness of RMSNorm kernel implementation
        - Stability under high parallel job execution
        - GPU kernel behavior under concurrent workloads

        The test is executed on:
        - Prefill nodes
        - Decode nodes
        - Proxy/router nodes

        Args:
        max_jobs (int): Maximum number of concurrent jobs to launch within
                        the RMSNorm test to stress the kernel.
        """
        log.info('#================ * * * =========================#')
        log.info('Run rmsnorm2d')
        log.info('#================ * * * =========================#')
        cmd = "bash -c " + shlex.quote(
            f"MAX_JOBS={max_jobs} python /sgl-workspace/aiter/op_tests/test_rmsnorm2d.py "
            f"> /tmp/rsmnorm_test.log 2>&1 &"
        )
        for hosts in (self.prefill_node_list, self.decode_node_list, self.proxy_node):
            self._container_exec(cmd, hosts=hosts)
        log.info('Wait 180 secs for tests to complete')
        time.sleep(180)
        for hosts in (self.prefill_node_list, self.decode_node_list, self.proxy_node):
            out_dict = self._container_exec(
                "bash -c " + shlex.quote("cat /tmp/rsmnorm_test.log"),
                hosts=hosts,
            )
            for node, out in out_dict.items():
                if re.search('fail', out or '', re.I):
                    log.warning(f'Some failures observed in test rmsnorm on node {node}')
                    fail_test(f'Some failures observed in test rmsnorm on node {node}')

    def launch_prefill_servers(self, dtype='auto', kv_cache_dtype='auto'):
        """
        Generate and stage Prefill server launch scripts on all Prefill nodes
        for SGLang disaggregated inference.

        Purpose:
        --------
        This method prepares the launch script for SGLang Prefill servers.
        In disaggregated PD (Prefill / Decode) mode:
        - Prefill servers are responsible for processing input prompts
        - They generate KV cache entries
        - KV cache is later consumed by Decode servers

        This method:
        - Creates one launch script per Prefill node
        - Sets distributed environment variables (NNODES, NODE_RANK)
        - Configures SGLang for Prefill-only execution
        - Does NOT start the servers yet; it stages the script for later execution

        Args:
        dtype (str): Model compute datatype (e.g., fp16, bf16, auto)
        kv_cache_dtype (str): KV cache datatype (e.g., fp16, bf16, auto)
        """
        log.info('#================ * * * =========================#')
        log.info('Create Prefill launch script on Prefill nodes')
        log.info('#================ * * * =========================#')

        prefill_node_list = self.prefill_node_list
        log.info('%%%% self.prefill_nnodes {}'.format(self.prefill_nnodes))
        dist_init_addr = f"{self.inf_dict['prefill_coordinator_addr']}:{self.inf_dict['prefill_coordinator_port']}"
        flags_block = add_cli_flags_block(self.bp_dict, indent='    ')

        for i in range(0, int(self.prefill_nnodes)):
            node = prefill_node_list[i]
            launch_body = (
                f"export NNODES={self.prefill_nnodes}\n"
                f"export NODE_RANK={i}\n"
                f"python3 -m sglang.launch_server --model {self.bp_dict['model']} \\\n"
                f"    --disaggregation-mode prefill \\\n"
                f"    --disaggregation-ib-device {self.inf_dict['nccl_ib_hca']} \\\n"
                f"    --host {node} \\\n"
                f"    --port {self.inf_dict['prefill_serv_port']} \\\n"
                f"    --dtype {dtype} \\\n"
                f"    --kv-cache-dtype {kv_cache_dtype} \\\n"
                f"    --trust-remote-code \\\n"
                f"    --tp-size {self.bp_dict['tensor_parallelism']} \\\n"
                f"    --pp-size {self.bp_dict['pipeline_parallelism']} \\\n"
                f"    --nnodes {self.prefill_nnodes} \\\n"
                f"    --node-rank {i} \\\n"
                f"    --dist-init-addr {dist_init_addr} \\\n"
                f"    --disable-radix-cache --disable-cuda-graph \\\n"
                f"    --mem-fraction-static {self.bp_dict['memory_fraction']} \\\n"
                f"{flags_block}\n"
                f"    --log-level {self.inf_dict['log_level']}\n"
            )
            write_cmd = "bash -c " + shlex.quote(
                f"cat > /tmp/prefill_launch_script.sh <<'EOF'\n{launch_body}EOF"
            )
            self._container_exec(write_cmd, hosts=[node])

        log.info('#================ * * * =========================#')
        log.info('Launching Prefill servers on Prefill nodes')
        log.info('#================ * * * =========================#')
        for i in range(0, int(self.prefill_nnodes)):
            node = prefill_node_list[i]
            start_cmd = "bash -c " + shlex.quote(
                f"chmod 755 /tmp/prefill_launch_script.sh\n"
                f"mkdir -p {self.log_dir}/prefill_node{i}\n"
                f"source /tmp/prefill_env_script.sh\n"
                f"nohup /tmp/prefill_launch_script.sh > "
                f"{self.log_dir}/prefill_node{i}/prefill_server.log 2>&1 &"
            )
            self._container_exec(start_cmd, hosts=[node])
        time.sleep(5)

    def launch_decode_servers(self, dtype='auto', kv_cache_dtype='auto'):
        """
        Generate and deploy Decode server launch scripts on all Decode nodes
        for SGLang disaggregated inference.

        Purpose:
        --------
        In disaggregated PD (Prefill / Decode) inference:
        - Decode servers are responsible for token generation
        - They consume KV cache generated by Prefill servers
        - They perform the latency- and throughput-critical decode loop

        This method:
        - Creates one Decode launch script per Decode node
        - Sets distributed environment variables (NNODES, NODE_RANK)
        - Configures SGLang for Decode-only execution
        - Deploys the scripts to Decode nodes for later execution

        Args:
        dtype (str): Model compute datatype (e.g., fp16, bf16, auto)
        kv_cache_dtype (str): KV cache datatype (e.g., fp16, bf16, auto)
        """
        log.info('#================ * * * =========================#')
        log.info('Create Decode launch script on Decode nodes')
        log.info('#================ * * * =========================#')

        decode_node_list = self.decode_node_list
        log.info('%%%% self.decode_nnodes {}'.format(self.decode_nnodes))
        dist_init_addr = f"{self.inf_dict['decode_coordinator_addr']}:{self.inf_dict['decode_coordinator_port']}"
        flags_block = add_cli_flags_block(self.bp_dict, indent='    ')

        for i in range(0, int(self.decode_nnodes)):
            node = decode_node_list[i]
            launch_body = (
                f"export NNODES={self.decode_nnodes}\n"
                f"export NODE_RANK={i}\n"
                f"python3 -m sglang.launch_server --model {self.bp_dict['model']} \\\n"
                f"    --disaggregation-mode decode \\\n"
                f"    --disaggregation-ib-device {self.inf_dict['nccl_ib_hca']} \\\n"
                f"    --host {node} \\\n"
                f"    --port {self.inf_dict['decode_serv_port']} \\\n"
                f"    --trust-remote-code \\\n"
                f"    --dtype {dtype} \\\n"
                f"    --kv-cache-dtype {kv_cache_dtype} \\\n"
                f"    --tp-size {self.bp_dict['tensor_parallelism']} \\\n"
                f"    --pp-size {self.bp_dict['pipeline_parallelism']} \\\n"
                f"    --nnodes {self.decode_nnodes} \\\n"
                f"    --node-rank {i} \\\n"
                f"    --dist-init-addr {dist_init_addr} \\\n"
                f"    --disable-radix-cache --disable-cuda-graph \\\n"
                f"    --mem-fraction-static {self.bp_dict['memory_fraction']} \\\n"
                f"{flags_block}\n"
                f"    --log-level {self.inf_dict['log_level']}\n"
            )
            write_cmd = "bash -c " + shlex.quote(
                f"cat > /tmp/decode_launch_script.sh <<'EOF'\n{launch_body}EOF"
            )
            self._container_exec(write_cmd, hosts=[node])

        log.info('#================ * * * =========================#')
        log.info('Launching Decode servers on Decode nodes')
        log.info('#================ * * * =========================#')
        for i in range(0, int(self.decode_nnodes)):
            node = decode_node_list[i]
            start_cmd = "bash -c " + shlex.quote(
                f"chmod 755 /tmp/decode_launch_script.sh\n"
                f"mkdir -p {self.log_dir}/decode_node{i}\n"
                f"source /tmp/decode_env_script.sh\n"
                f"nohup bash /tmp/decode_launch_script.sh > "
                f"{self.log_dir}/decode_node{i}/decode_server.log 2>&1 &"
            )
            self._container_exec(start_cmd, hosts=[node])

    def poll_and_check_server_ready(
        self,
    ):
        """
        Wait for Prefill and Decode servers to initialize and verify that they
        are fully ready to accept inference requests.

        Purpose:
        --------
        After launching Prefill and Decode server scripts, the servers require
        time to:
        - Initialize Python runtime
        - Load model weights
        - Allocate GPU memory
        - Initialize RDMA / NCCL / Gloo communication
        - Bind to network ports

        This method enforces a startup delay and then actively polls each server
        to confirm readiness before inference traffic is sent.
        """
        log.info('Waiting 120 secs after launching decode script')
        time.sleep(120)
        self.poll_for_server_ready(0, 'prefill')
        self.poll_for_server_ready(0, 'decode')

    def launch_proxy_router(
        self,
    ):
        """
        Generate and launch the SGLang Proxy Router for disaggregated
        Prefill/Decode (PD) inference.

        Purpose:
        --------
        The Proxy Router is the control-plane and data-plane entry point for
        inference traffic in a disaggregated PD deployment.

        Responsibilities:
        - Accept incoming inference requests
        - Route prefill requests to Prefill servers
        - Route decode requests to Decode servers
        - Coordinate Prefill -> Decode handoff

        This method:
        - Builds routing configuration dynamically based on cluster topology
        - Creates a launch script on the Proxy Router node
        - Launches the router as a background service
        """
        prefill_str = (
            f"--prefill http://{self.inf_dict['prefill_coordinator_addr']}:{self.inf_dict['prefill_serv_port']} "
        )
        decode_str = (
            f"--decode http://{self.inf_dict['decode_coordinator_addr']}:{self.inf_dict['decode_serv_port']} "
        )
        log.info('#================ * * * =========================#')
        log.info('Create Proxy Router launch script on Proxy Router nodes')
        log.info('#================ * * * =========================#')

        launch_body = (
            "python3 -m sglang_router.launch_router \\\n"
            f"    --pd-disaggregation \\\n"
            f"    {prefill_str.strip()} \\\n"
            f"    {decode_str.strip()} \\\n"
            f"    --host 0.0.0.0 \\\n"
            f"    --port {self.router_serv_port} \\\n"
            f"    --log-dir {self.inf_dict['log_dir']}\n"
        )
        write_cmd = "bash -c " + shlex.quote(
            f"cat > /tmp/proxy_router_launch_script.sh <<'EOF'\n{launch_body}EOF"
        )
        self._container_exec(write_cmd, hosts=self.proxy_node)

        log.info('#================ * * * =========================#')
        log.info('Launch Proxy Router script on Proxy Router nodes')
        log.info('#================ * * * =========================#')
        start_cmd = "bash -c " + shlex.quote(
            f"chmod 755 /tmp/proxy_router_launch_script.sh\n"
            f"mkdir -p {self.log_dir}/proxy_router_node\n"
            f"source /tmp/router_env_script.sh\n"
            f"nohup bash /tmp/proxy_router_launch_script.sh > "
            f"{self.log_dir}/proxy_router_node/proxy_router.log 2>&1 &"
        )
        self._container_exec(start_cmd, hosts=self.proxy_node)
        log.info('Waiting 120 secs after launching proxy router script')
        time.sleep(120)

    def benchserv_test_random(self, d_type='auto'):
        """
        Run SGLang serving benchmark using a synthetic random dataset and
        validate inference performance and correctness.

        Purpose:
        --------
        This benchmark exercises the inference serving stack using randomly
        generated input/output sequences to:
        - Stress-test request scheduling and batching
        - Evaluate sustained throughput under synthetic load
        - Validate end-to-end serving stability independent of real datasets

        The benchmark targets the Proxy Router endpoint, ensuring that
        Prefill, Decode, and routing logic work together correctly.

        Args:
        d_type (str): Data type identifier used to select expected
                      performance thresholds (e.g., fp16, bf16, auto).
        """
        log.info('#================ * * * =========================#')
        log.info('Benchmark Random Dataset')
        log.info('#================ * * * =========================#')
        i_dict = self.bp_dict['inference_tests']['bench_serv_random']
        self._bench_num_prompts = int(i_dict['num_prompts'])
        inner = (
            f"mkdir -p {self.log_dir}/benchmark_node\n"
            f"source /tmp/benchmark_env_script.sh\n"
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
        self._container_exec(
            "bash -c " + shlex.quote(inner),
            hosts=self.benchmark_serv_node,
            timeout=1000,
        )
        time.sleep(5)
        self.poll_for_inference_completion(iterations=10, waittime_between_iters=60)

        peak_tflops = float(i_dict.get("peak_gpu_tflops", 1300))
        num_params = float(i_dict.get("model_num_params", 70e9))
        tp = int(self.bp_dict.get("tensor_parallelism", 1))
        pp = int(self.bp_dict.get("pipeline_parallelism", 1))
        num_gpus = (int(self.prefill_nnodes) + int(self.decode_nnodes)) * tp * pp
        for node, m in (self.inference_results_dict or {}).items():
            duration = float(m.get("benchmark_duration") or 0)
            in_tok = float(m.get("total_input_tokens") or 0)
            out_tok = float(
                m.get("total_generated_tokens") or m.get("Total generated tokens:") or 0
            )
            if duration > 0 and num_gpus > 0:
                achieved = 6.0 * num_params * (in_tok + out_tok)
                peak = peak_tflops * 1e12 * num_gpus * duration
                m["mfu"] = f"{achieved / peak:.6f}"

        log_path = f"{self.log_dir}/benchmark_node/benchmark_results.log"
        for node, m in (self.inference_results_dict or {}).items():
            gp = m.get("goodput", "n/a")
            tpg = m.get("output_throughput_per_gpu_per_sec", "n/a")
            tr = m.get("total_requests", "n/a")
            sr = m.get("successful_requests", "n/a")
            mfu = m.get("mfu", "n/a")
            append_inner = (
                f"echo '' >> {log_path} && "
                f"echo '============ Derived Benchmark Results ============' >> {log_path} && "
                f"echo 'Goodput (successful / total): {sr} / {tr}  =>  {gp}' >> {log_path} && "
                f"echo 'Output token throughput per GPU (tok/s/GPU): {tpg}' >> {log_path} && "
                f"echo 'MFU (estimated): {mfu}' >> {log_path} && "
                f"echo '=====================================================================' >> {log_path}"
            )
            self._container_exec(
                "bash -c " + shlex.quote(append_inner),
                hosts=self.benchmark_serv_node,
            )

        self.verify_inference_results('bench_serv', i_dict['expected_results'][d_type])

    def poll_for_server_ready(self, node_no, sglang_function, no_of_iterations=16):
        """Poll Prefill or Decode server logs inside the container for readiness."""
        if re.search('prefill', sglang_function):
            self._poll_role_log_ready(
                f"{self.log_dir}/prefill_node{node_no}/prefill_server.log",
                [self.prefill_node_list[node_no]],
                f'Prefill node {node_no}',
                no_of_iterations,
            )
        elif re.search('decode', sglang_function):
            self._poll_role_log_ready(
                f"{self.log_dir}/decode_node{node_no}/decode_server.log",
                [self.decode_node_list[node_no]],
                f'Decode node {node_no}',
                no_of_iterations,
            )

    def _poll_role_log_ready(
        self,
        log_path: str,
        hosts: list[str],
        label: str,
        no_of_iterations: int = 16,
    ) -> None:
        for iteration in range(1, no_of_iterations):
            log.info('Starting %s readiness poll iteration %d', label, iteration)
            grep_cmd = (
                f"grep -B 20 -A 20 -E {_SERVER_READY_RE.pattern!r} "
                f"{shlex.quote(log_path)} || true"
            )
            text = self._container_exec_text(grep_cmd, hosts=hosts)
            if _SERVER_READY_RE.search(text):
                log.info('Wait 60 secs before serving traffic')
                time.sleep(60)
                return
            log.info('Wait 120 secs and continue polling')
            time.sleep(120)
        fail_test(
            f'{label} on {hosts[0]!r} did not reach ready state '
            f'in {no_of_iterations} iterations'
        )

    def get_inference_results_dict(self, out_dict):
        """
        Parse inference benchmark output logs and extract key performance metrics
        into a structured dictionary.

        Purpose:
        --------
        This method processes raw text output generated by inference benchmarks
        (e.g., sglang.bench_serving) and extracts important metrics such as:
        - Request counts
        - Token throughput
        - Latency statistics (TTFT, TPOT)
        - Benchmark duration

        The extracted metrics are stored per node in:
        self.inference_results_dict

        Args:
        out_dict (dict):
            Dictionary keyed by node identifier, where each value is the
            raw stdout/stderr text produced by the benchmark on that node.
        """
        self.inference_results_dict = {}
        log.info('Inside get_inference_results_dict')
        log.info("%s", out_dict)

        for node in out_dict.keys():
            self.inference_results_dict[node] = {}
            if re.search('Successful requests:', out_dict[node], re.I):
                match = re.search('Successful requests:\s+([0-9]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['successful_requests'] = match.group(1)
            if re.search('Benchmark duration\s+\(s\):\s+([0-9]+)', out_dict[node], re.I):
                match = re.search('Benchmark duration\s+\(s\):\s+([0-9]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['benchmark_duration'] = match.group(1)
            if re.search('Total input tokens:', out_dict[node], re.I):
                match = re.search('Total input tokens:\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['total_input_tokens'] = match.group(1)
            if re.search('Total generated tokens:', out_dict[node], re.I):
                match = re.search('Total generated tokens:\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['total_generated_tokens'] = match.group(1)
            if re.search('Request throughput \(req/s\):', out_dict[node], re.I):
                match = re.search('Request throughput \(req/s\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['request_throughput_per_sec'] = match.group(1)
            if re.search('Output token throughput \(tok/s\):', out_dict[node], re.I):
                match = re.search('Output token throughput \(tok/s\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['output_throughput_per_sec'] = match.group(1)
            if re.search('Mean TTFT \(ms\):', out_dict[node], re.I):
                match = re.search('Mean TTFT \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['mean_ttft_ms'] = match.group(1)
            if re.search('Median TTFT (ms):', out_dict[node], re.I):
                match = re.search('Median TTFT \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['median_ttft_ms'] = match.group(1)
            if re.search('P99 TTFT (ms):', out_dict[node], re.I):
                match = re.search('P99 TTFT \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['p99_ttft_ms'] = match.group(1)
            if re.search('Mean TPOT \(ms\)', out_dict[node], re.I):
                match = re.search('Mean TPOT \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['mean_tpot_ms'] = match.group(1)
            if re.search('Median TPOT \(ms\):', out_dict[node], re.I):
                match = re.search('Median TPOT \(ms\):\s+([0-9]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['median_tpot_ms'] = match.group(1)
            if re.search('P99 TPOT (ms):', out_dict[node], re.I):
                match = re.search('P99 TPOT \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['p99_tpot_ms'] = match.group(1)
            if re.search('Mean ITL \(ms\):', out_dict[node], re.I):
                match = re.search('Mean ITL \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['mean_itl_ms'] = match.group(1)
            if re.search('Median ITL \(ms\):', out_dict[node], re.I):
                match = re.search('Median ITL \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['median_itl_ms'] = match.group(1)
            if re.search('P99 ITL \(ms\):', out_dict[node], re.I):
                match = re.search('P99 ITL \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['p99_itl_ms'] = match.group(1)
            m = first_float(r'Mean E2E Latency \(ms\):\s+([0-9\.]+)', out_dict[node])
            if m:
                self.inference_results_dict[node]['mean_e2e_latency_ms'] = m
            m = first_float(r'Median E2E Latency \(ms\):\s+([0-9\.]+)', out_dict[node])
            if m:
                self.inference_results_dict[node]['median_e2e_latency_ms'] = m
            for p in (90, 95, 99):
                m = first_float(rf'P{p} E2E Latency \(ms\):\s+([0-9\.]+)', out_dict[node])
                if m:
                    self.inference_results_dict[node][f'p{p}_e2e_latency_ms'] = m

            total_req = first_float(r"Total requests:\s+([0-9]+)", out_dict[node])
            failed_req = first_float(r"Failed requests:\s+([0-9]+)", out_dict[node])
            succ = self.inference_results_dict[node].get("successful_requests")
            if total_req:
                self.inference_results_dict[node]["total_requests"] = total_req
            elif succ is not None and failed_req is not None:
                self.inference_results_dict[node]["total_requests"] = str(int(succ) + int(failed_req))
            elif succ is not None and getattr(self, "_bench_num_prompts", None) is not None:
                self.inference_results_dict[node]["total_requests"] = str(int(self._bench_num_prompts))
            if succ and self.inference_results_dict[node].get("total_requests"):
                s, t = int(succ), int(self.inference_results_dict[node]["total_requests"])
                self.inference_results_dict[node]["goodput"] = f"{(s / t):.6f}" if t else None

            out_tps = self.inference_results_dict[node].get("output_throughput_per_sec")
            if out_tps:
                tp = int(self.bp_dict.get("tensor_parallelism", "1"))
                pp = int(self.bp_dict.get("pipeline_parallelism", "1"))
                ng = (int(self.prefill_nnodes) + int(self.decode_nnodes)) * tp * pp
                if ng > 0:
                    self.inference_results_dict[node]["output_throughput_per_gpu_per_sec"] = (
                        f"{float(out_tps) / ng:.6f}"
                    )

        log.info("%s", self.inference_results_dict)
        return self.inference_results_dict

    def scan_for_inference_errors(
        self,
    ):
        """
        Scan Prefill and Decode server logs for known inference error patterns
        and fail the test if any are detected.

        Purpose:
        --------
        This method performs a post-inference health check by scanning
        server logs for known error signatures that indicate:
        - Runtime failures
        - Communication errors (RDMA/NCCL)
        - Out-of-memory conditions
        - Kernel or backend crashes
        - Fatal exceptions during inference

        The method ensures that even if benchmarks complete, silent or
        non-fatal errors do not go unnoticed.
        """
        log.info('Scan for inference errors')
        inference_pass = True

        for j in range(0, int(self.prefill_nnodes)):
            node = self.prefill_node_list[j]
            out_dict = self._container_exec(
                f"tail -100 {shlex.quote(f'{self.log_dir}/prefill_node{j}/prefill_server.log')}",
                hosts=[node],
            )
            out = out_dict.get(node, '')
            for err_key in inference_err_dict:
                if re.search(f'{inference_err_dict[err_key]}', out):
                    fail_test(f'ERROR {inference_err_dict[err_key]} seen in inference logs ..')
                    log.error('Aborting inference log polling')
                    inference_pass = False

        for j in range(0, int(self.decode_nnodes)):
            node = self.decode_node_list[j]
            out_dict = self._container_exec(
                f"tail -500 {shlex.quote(f'{self.log_dir}/decode_node{j}/decode_server.log')}",
                hosts=[node],
            )
            out = out_dict.get(node, '')
            for err_key in inference_err_dict:
                if re.search(f'{inference_err_dict[err_key]}', out):
                    fail_test(f'ERROR {inference_err_dict[err_key]} seen in inference logs ..')
                    log.error('Aborting inference log polling')
                    inference_pass = False

        return inference_pass

    def poll_for_inference_completion(
        self, iterations=10, waittime_between_iters=60, total_timeout=3600, require_all_nodes=True
    ):
        """
        Poll benchmark logs to detect inference completion and extract results.

        Purpose:
        --------
        This method monitors inference progress by periodically inspecting
        benchmark output logs. It determines when inference has completed,
        detects early failures, and enforces a global timeout.

        Completion criteria:
        --------------------
        Inference is considered complete when the benchmark output contains
        the pattern 'Serving Benchmark Result'.

        Failure criteria:
        -----------------
        Any known inference error detected in Prefill or Decode logs
        immediately aborts the process.

        Args:
        iterations (int):
            Maximum number of polling iterations.
        waittime_between_iters (int):
            Time (seconds) to wait between polling attempts.
        total_timeout (int or None):
            Maximum wall-clock time (seconds) allowed for inference.
        require_all_nodes (bool):
            If True, all nodes must report completion.
            If False, completion by any node is sufficient.
        """
        time.sleep(60)

        start_time = time.time()

        def timed_out() -> bool:
            return total_timeout is not None and (time.time() - start_time) >= float(total_timeout)

        completed_pattern = re.compile('Serving Benchmark Result', re.I)
        log_path = f"{self.log_dir}/benchmark_node/benchmark_results.log"

        for itr in range(1, iterations + 1):
            log.info(f'Starting iteration {itr}')

            out_dict = self._container_exec(
                f"tail -1000 {shlex.quote(log_path)}",
                hosts=self.benchmark_serv_node,
            )

            node_completion = {}
            for node, output in out_dict.items():
                node_completion[node] = bool(completed_pattern.search(output or ''))

            if require_all_nodes:
                all_complete = all(node_completion.values()) if node_completion else False
            else:
                all_complete = any(node_completion.values()) if node_completion else False

            if not all_complete:
                if timed_out():
                    msg = f"Timeout while waiting for inference completion after ~{int(time.time() - start_time)}s"
                    log.warning("%s", msg)
                    return {"status": "timeout", "reason": msg}
                log.info('Inference still in progress')
                time.sleep(30)
                time.sleep(int(waittime_between_iters))
                continue

            self.get_inference_results_dict(out_dict)
            log.info('Completed Inference, returning !!!')
            return {"status": "success", "results": self.inference_results_dict}

        if timed_out():
            msg = f"Timeout after maximum iterations ({self.inference_poll_iterations}) and ~{int(time.time() - start_time)}s"
            log.warning("%s", msg)
            return {"status": "timeout", "reason": msg}
        msg = f"Reached iteration cap ({self.inference_poll_iterations}) without completion; still in progress"
        log.warning("%s", msg)
        return {"status": "stuck_in_progress", "reason": msg}

    def verify_inference_results(self, test_name, expected_result_dict):
        """
        Validate inference benchmark results against expected performance
        thresholds and check for system-level errors.

        Comparison rules (via ``evaluate_all`` + threshold ``kind``):
        - Throughput, req/s, goodput, MFU: actual >= expected
        - Latency (*_ms, *latency*): actual <= expected

        Threshold entries may be full specs ``{"kind": ..., "value": ...}`` from
        threshold.json or legacy flat floats from ``flat_expected_from_specs``.
        """
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
        verify_dmesg_for_errors(self.orch.all, self.inference_start_time, self.inference_end_time)

    def sglang_disagg_gpu_counts(self, mem_threshold_mb=5000):
        """
        After model load, count occupied GPUs per prefill/decode node via amd-smi.
        """
        tp = int(self.bp_dict["tensor_parallelism"])
        pp = int(self.bp_dict.get("pipeline_parallelism", 1))

        def _count_per_node(out_dict):
            per_node = {}
            for node, payload in out_dict.items():
                count = 0
                try:
                    entries = json.loads((payload or '').strip())
                except (json.JSONDecodeError, AttributeError):
                    log.warning("Failed to parse amd-smi JSON on node %s", node)
                    per_node[node] = 0
                    continue
                if isinstance(entries, dict) and "gpu_data" in entries:
                    entries = entries["gpu_data"]
                if not isinstance(entries, list):
                    per_node[node] = 0
                    continue
                for g in entries:
                    used_mb = g.get("mem_usage", {}).get("used_vram", {}).get("value", 0)
                    if used_mb > mem_threshold_mb:
                        count += 1
                per_node[node] = count
            return per_node

        prefill_per_node = _count_per_node(
            self._host_exec('sudo amd-smi metric --json', hosts=self.prefill_node_list)
        )
        decode_per_node = _count_per_node(
            self._host_exec('sudo amd-smi metric --json', hosts=self.decode_node_list)
        )
        occupied_prefill = sum(prefill_per_node.values())
        occupied_decode = sum(decode_per_node.values())

        result = {
            "configured_tp": tp,
            "configured_pp": pp,
            "prefill_per_node": prefill_per_node,
            "decode_per_node": decode_per_node,
            "prefill_occupied_gpus": occupied_prefill,
            "decode_occupied_gpus": occupied_decode,
            "total_occupied_gpus": occupied_prefill + occupied_decode,
        }

        lines = [
            "",
            f"Configured TP: {tp}",
            f"Configured PP: {pp}",
            "",
            "Prefill:",
        ]
        for node, count in prefill_per_node.items():
            lines.append(f"  {node}: {count} occupied GPUs")
        lines.append(f"  Total: {occupied_prefill} occupied GPUs")
        lines.append("")
        lines.append("Decode:")
        for node, count in decode_per_node.items():
            lines.append(f"  {node}: {count} occupied GPUs")
        lines.append(f"  Total: {occupied_decode} occupied GPUs")
        lines.append("")
        lines.append("Total hardware GPUs consumed:")
        lines.append(f"  {occupied_prefill + occupied_decode}")

        log.info("\n".join(lines))
        return result

    def verify_openai_compatible_endpoints(self) -> list[str]:
        """
        Smoke-test OpenAI-compatible HTTP API on the proxy router (inside the
        benchmark container): GET /v1/models,
        POST /v1/chat/completions, POST /v1/completions, and structured JSON
        (book) via chat completions.
        """
        port = int(self.router_serv_port)
        model_name = self.bp_dict["model"]

        probe_src = OpenAIProbe.probe_script(port, model_name, host=self.client_host)
        b64 = base64.b64encode(probe_src.encode("utf-8")).decode("ascii")
        inner = (
            f"mkdir -p {self.log_dir}/benchmark_node && "
            f"echo {shlex.quote(b64)} | base64 -d > /tmp/openai_mq_probe.py && "
            f"python3 /tmp/openai_mq_probe.py && rm -f /tmp/openai_mq_probe.py"
        )
        log.info(
            "OpenAI endpoint probe inside benchmark container (%s:%r), same pattern as GSM8K/benchserv",
            self.client_host,
            port,
        )
        out_dict = self._container_exec(
            "bash -c " + shlex.quote(inner),
            hosts=self.benchmark_serv_node,
            timeout=min(900, 480 + 180),
        )
        bench_host = self.benchmark_serv_node[0]
        raw_out = out_dict.get(bench_host) or self._first_output(out_dict)

        probe_err: Optional[str] = None
        results: dict[str, tuple[int, Any]] = {}
        if not raw_out or not str(raw_out).strip():
            probe_err = f"OpenAI-compatible probe produced no output on {bench_host!r}: {out_dict!r}"
        else:
            lines_out = str(raw_out).strip().splitlines()
            if not lines_out:
                probe_err = (
                    f"OpenAI-compatible probe empty lines after strip on node "
                    f"{bench_host!r}: {raw_out!r}"
                )
            else:
                last_line = lines_out[-1]
                try:
                    parsed = json.loads(last_line)
                except json.JSONDecodeError as e:
                    probe_err = (
                        f"OpenAI-compatible probe invalid JSON: {e!r} raw={raw_out!r}"
                    )
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
                                probe_err = (
                                    f"OpenAI-compatible probe bad shape at "
                                    f"{step!r}: {val!r}"
                                )
                                break

        if probe_err is not None:
            fail_test(probe_err)
            return []

        OpenAIProbe.log_results(results, log)

        ok, err = OpenAIProbe.check_results(results, port=port, logger=log)
        if not ok:
            summary = OpenAIProbe.summarize_results(results, ok, err)
            fail_test(f"{err}")
            return summary

        summary = OpenAIProbe.summarize_results(results, ok, err)
        return summary

    def run_lm_eval_hellaswag_benchmark_test(self, _d_type="auto"):
        return self.run_lm_eval_benchmark_test("lm_eval_hellaswag", _d_type=_d_type)

    def run_lm_eval_gsm8k_benchmark_test(self, _d_type="auto"):
        return self.run_lm_eval_benchmark_test("lm_eval_gsm8k", _d_type=_d_type)

    def run_lm_eval_benchmark_test(self, bench_key: str, _d_type="auto"):
        spec = LM_EVAL_SPECS[bench_key]
        log.info("#================ * * * =========================#")
        log.info("lm-eval %s benchmark", spec["display"])
        log.info("#================ * * * =========================#")
        task_name = bench_key.removeprefix("lm_eval_")
        i_dict = self.bp_dict["inference_tests"][bench_key]
        inner_cmd, scoring = LmEvalBenchmark.prepare(
            i_dict,
            port=int(self.router_serv_port),
            host=self.client_host,
            model_id=self.bp_dict["model"],
            task_name=task_name,
            default_tasks=task_name,
            default_metric=spec["default_metric"],
            default_metric_key=spec["default_metric_key"],
            log_dir=self.log_dir,
            log_basename=f"{bench_key}.log",
            default_num_concurrent=spec["default_num_concurrent"],
        )

        inner = (
            f"mkdir -p {self.log_dir}/benchmark_node && "
            f"source /tmp/benchmark_env_script.sh && {inner_cmd}"
        )
        out_dict = self._container_exec(
            "bash -c " + shlex.quote(inner),
            hosts=self.benchmark_serv_node,
            timeout=scoring["exec_timeout_sec"],
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
                error=errors[-1] if errors else "no benchmark nodes produced output to score",
            )
            errors.append(f"lm-eval {spec['display']}: no benchmark nodes produced output to score")

        for msg in errors:
            fail_test(msg)

        return summary

    def run_long_context_niah_accuracy(self, *, isl: int, osl: int, d_type: str = "auto"):
        """NIAH long-context accuracy at fixed ISL/OSL via /v1/chat/completions."""
        bench_key = "long_ctx_niah"
        i_dict = self.bp_dict["inference_tests"][bench_key]
        port = int(self.router_serv_port)
        log_basename = f"long_ctx_niah_isl{isl}_osl{osl}.log"

        inner_cmd, scoring = LongContextNiahBenchmark.prepare(
            i_dict,
            port=port,
            host=self.client_host,
            model_id=self.bp_dict["model"],
            isl=int(isl),
            osl=int(osl),
            log_dir=self.log_dir,
            log_basename=log_basename,
        )
        probe_src = LongContextNiahBenchmark.probe_script(**scoring["probe_kwargs"])
        b64 = base64.b64encode(probe_src.encode("utf-8")).decode("ascii")

        inner = (
            f"mkdir -p {self.log_dir}/benchmark_node && "
            f"echo {shlex.quote(b64)} | base64 -d > /tmp/long_ctx_niah_probe.py && "
            f"source /tmp/benchmark_env_script.sh && {inner_cmd}"
        )
        out_dict = self._container_exec(
            "bash -c " + shlex.quote(inner),
            hosts=self.benchmark_serv_node,
            timeout=int(scoring["exec_timeout_sec"]),
        )
        time.sleep(5)

        check_kwargs = LongContextNiahBenchmark.check_kwargs_from_scoring(scoring)
        summary = None
        errors: list[str] = []

        for node, text in out_dict.items():
            ok, node_summary, err = LongContextNiahBenchmark.check_results(text, **check_kwargs)
            if node_summary is not None:
                summary = node_summary
            if not ok:
                errors.append(f"long_ctx_niah on node {node!r}: {err}")

        if summary is None:
            summary = {
                "task": "long_ctx_niah",
                "metric_key": scoring["metric_key"],
                "actual": None,
                "expected": float(scoring["expected"]),
                "passed": False,
                "error": errors[-1] if errors else "no benchmark output",
            }
            errors.append("long_ctx_niah: no benchmark nodes produced output to score")

        for msg in errors:
            fail_test(msg)

        return summary
