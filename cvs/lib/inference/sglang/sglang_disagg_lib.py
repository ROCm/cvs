'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import base64
import json
import os
import re
import shlex
import time
from typing import Any, Optional

from cvs.lib import globals
from cvs.lib.docker_lib import get_docker_cmd
from cvs.lib.utils.model_query_lib import LmEvalBenchmark, LongContextNiahBenchmark, OpenAIProbe
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *


log = globals.log

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

def textwrap_for_yml(msg_string):
    return '\n'.join([m.lstrip() for m in msg_string.split('\n')])

def _as_node_list(value):
    """
    Normalize cluster JSON node field to a list of host strings.

    Config may use a single hostname/IP string or a list. ``list(str)``
    would split into characters (e.g. ``'10.0.0.1'`` -> ``'1'``), breaking
    SSH and HTTP clients.
    """
    if isinstance(value, str):
        return [value]
    return list(value)

def _first_float(pattern, text):
    m = re.search(pattern, text, re.I)
    return m.group(1) if m else None

LM_EVAL_SPECS = {
    "lm_eval_hellaswag": {
        "display": "HellaSwag",
        "default_metric": "acc_norm",
        "default_metric_key": "acc_norm,none",
        "default_num_concurrent": "1",
    },
    "lm_eval_gsm8k": {
        "display": "GSM8K",
        "default_metric": "exact_match",
        "default_metric_key": "exact_match,flexible-extract",
        "default_num_concurrent": "4",
    },
    "lm_eval_mmlu": {
        "display": "MMLU",
        "default_metric": "acc",
        "default_metric_key": "acc,none",
        "default_num_concurrent": "1",
    },
}


class SglangDisaggPD:
    def __init__(
        self,
        model_name,
        inference_config_dict,
        benchmark_params_dict,
        hf_token,
        p_phdl=None,
        d_phdl=None,
        r_phdl=None,
        b_phdl=None,
        gpu_type='mi300',
        user_name=None,
        priv_key_file=None,
    ):
        """
        Initialize a Disaggregated Prefill/Decode (PD) inference controller
        for SGLang.

        This class encapsulates:
          - Cluster topology (prefill, decode, proxy, benchmark nodes)
          - SSH-based remote execution (via Pssh handlers)
          - Inference configuration (networking, containers, env vars)
          - Benchmark configuration (load, concurrency, prompt sizes)

        Args:
            model_name (str): HuggingFace or local model identifier
            inference_config_dict (dict): Cluster and runtime configuration
            benchmark_params_dict (dict): Benchmark workload parameters
            hf_token (str): HuggingFace access token
            p_phdl, d_phdl, r_phdl, b_phdl: Optional pre-created SSH handlers
            gpu_type (str): GPU type (e.g., mi300, mi325)
            user_name (str): SSH username for remote nodes
            priv_key_file (str): SSH private key file
        """

        # ------------------------------------------------------------------
        # Basic identity and authentication parameters
        # ------------------------------------------------------------------
        self.user_name = user_name
        self.priv_key_file = priv_key_file
        self.model_name = model_name
        self.hf_token = hf_token
        self.gpu_type = gpu_type

        # ------------------------------------------------------------------
        # Store inference and benchmark configuration dictionaries
        # These are typically loaded from a JSON/YAML configuration file
        # ------------------------------------------------------------------
        self.inf_dict = inference_config_dict
        self.bp_dict = benchmark_params_dict

        self.model_name = model_name
        self.hf_token = hf_token
        self.gpu_type = gpu_type

        # ------------------------------------------------------------------
        # Extract cluster topology for disaggregated inference
        #
        # Prefill nodes  : Handle prompt ingestion + KV cache creation
        # Decode nodes   : Handle token generation
        # Proxy node     : Routes requests between prefill/decode
        # Benchmark node : Generates inference load
        # ------------------------------------------------------------------
        self.mount_vol = self.inf_dict.get(
            'mount_vol',
            '/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so',
        )
        
        self.prefill_node_list = self.inf_dict['prefill_node_list']
        self.decode_node_list = self.inf_dict['decode_node_list']
        self.prefill_nnodes = len(self.prefill_node_list)
        self.decode_nnodes = len(self.decode_node_list)

        self.proxy_node = _as_node_list(self.inf_dict['proxy_router_node'])
        self.benchmark_serv_node = _as_node_list(self.inf_dict['benchmark_serv_node'])

        # ------------------------------------------------------------------
        # SSH handlers for each node group
        #
        # p_phdl : Prefill nodes
        # d_phdl : Decode nodes
        # r_phdl : Proxy/router node
        # b_phdl : Benchmark client node
        # ------------------------------------------------------------------
        self.p_phdl = p_phdl
        self.d_phdl = d_phdl
        self.r_phdl = r_phdl
        self.b_phdl = b_phdl

        if self.p_phdl is None:
            self.p_phdl = Pssh(log, self.prefill_node_list, user=self.user_name, pkey=self.priv_key_file)

        if self.d_phdl is None:
            self.d_phdl = Pssh(log, self.decode_node_list, user=self.user_name, pkey=self.priv_key_file)

        if self.r_phdl is None:
            self.r_phdl = Pssh(log, self.proxy_node, user=self.user_name, pkey=self.priv_key_file)

        if self.b_phdl is None:
            self.b_phdl = Pssh(log, self.benchmark_serv_node, user=self.user_name, pkey=self.priv_key_file)

        self.job_cmd = ''
        self.job_cmd_list = []
        self.inference_results_dict = {}
        log.info("%s", self.gpu_type)

        # ------------------------------------------------------------------
        # Extract commonly used inference parameters for convenience
        # ------------------------------------------------------------------
        # Needed only in the case of distributed inference - placeholder for future
        # Intialize cluster stats dicts ..
        self.rdma_stats_dict_before = {}
        self.ethtool_stats_dict_before = {}
        self.rdma_stats_dict_after = {}
        self.inference_start_time = p_phdl.exec('date +"%a %b %e %H:%M"')
        self.inference_end_time = None

        # ------------------------------------------------------------------
        # Set default benchmark parameters if not provided
        # These control request generation and performance measurement
        # ------------------------------------------------------------------
        self.home_dir = os.path.expanduser("~")
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
        self.inf_dict.setdefault('max_concurrent_requests', '-1')
        self.inf_dict.setdefault('queue_size', '100')
        self.inf_dict.setdefault('queue_timeout_secs', '60')
        self.inf_dict.setdefault('max_retries', '5')

        log.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        log.info(f'inference_dict = {self.inf_dict}')
        log.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        self.container_image = self.inf_dict['container_image']
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

        # set defaults for benchmark param dict if not passed via JSON file
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

        self.inference_poll_iterations = self.bp_dict['inference_poll_iterations']

        log.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        log.info(f'benchmark_params_dict = {self.bp_dict}')
        log.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    # Helper function for installing container packages
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
        # Install ip tools
        cmd = f'{get_docker_cmd()} exec {self.container_name} /bin/bash -c " \
            sudo apt -y update; \
            sudo apt install -y iputils-ping; \
            sudo apt install -y iproute2; \
            sudo apt install -y net-tools" '
        self.p_phdl.exec(cmd)
        self.d_phdl.exec(cmd)
        self.r_phdl.exec(cmd)

    # Helper function for executing NIC setup scripts
    def exec_nic_setup_scripts(
        self,
    ):
        """
        Execute NIC-related setup steps inside the inference container.

        Behavior:
        - Only runs for distributed inference.
        - If NIC type appears to be Broadcom/Thor, applies a temporary workaround:
          * Copies the bnxt RDMA library from the host-named file to the container?s expected path.
          * Verifies that ibv_devinfo shows a bnxt_ HCA (to confirm RDMA is wired correctly).
        - Forces NCCL GID index to 3 for Broadcom/Thor (common requirement).

        Assumptions:
        - self.s_phdl.exec runs a shell command and returns a dict: {node: stdout}.
        - sudo is non-interactive within the container.
        - The bnxt library file paths exist in the container base image.
        """

        # This is a temporary hack needed for broadcom nics to work within containers ..
        if re.search('broadcom|thor', self.nic_type, re.I):
            # override the gid_index to 3 for broadcom
            self.nccl_ib_gid_index = 3
            cmd = (
                    f'{get_docker_cmd()} exec {self.container_name} /bin/bash -c "sudo '
                    f'cp {self.mount_vol}.host {self.mount_vol}; '
                    f'sleep 2; ibv_devinfo; sleep 2;" '
                )
            pout_dict = self.p_phdl.exec(cmd)
            dout_dict = self.d_phdl.exec(cmd)
            hca_id_regex = rf'hca_id:\s+{re.escape(self.hca_id_prefix)}'
            for node in pout_dict.keys():
                if not re.search(hca_id_regex, pout_dict[node], re.I):
                    log.info("%s", pout_dict[node])
                    fail_test(f'Broadcom libbnxt rdma driver is not properly copied on node {node}')
            for node in dout_dict.keys():
                if not re.search(hca_id_regex, dout_dict[node], re.I):
                    log.info("%s", dout_dict[node])
                    fail_test(f'Broadcom libbnxt rdma driver is not properly copied on node {node}')

    # Helper function for checking IBV devices
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
        for hdl in [self.p_phdl, self.d_phdl]:
            cmd = f'''{get_docker_cmd()} exec {self.container_name} /bin/bash -c "ibv_devinfo" '''
            out_dict = hdl.exec(cmd)
            for node in out_dict.keys():
                if re.search('No IB devices found', out_dict[node], re.I):
                    fail_test(f'IB devices not seen inside the container for node {node}')

    # Helper function for setting up prefill container environment
    def setup_prefill_container_env(
        self,
    ):
        # Env setup for Prefill Nodes ..
        p_cmd = f'''{get_docker_cmd()} exec {self.container_name} /bin/bash -c "echo '

                    export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
                    export NCCL_DEBUG={self.inf_dict['nccl_debug']}
                    export NCCL_IB_HCA={self.inf_dict['nccl_ib_hca']}
                    export NCCL_IB_GID_INDEX={self.inf_dict['nccl_ib_gid_index']}
                    export HSA_FORCE_FINE_GRAIN_PCIE=1
                    export NCCL_SOCKET_IFNAME={self.inf_dict['nccl_socket_ifname']}
                    export GLOO_SOCKET_IFNAME={self.inf_dict['gloo_socket_ifname']}
                    export GLOO_TCP_IFNAME={self.inf_dict['gloo_socket_ifname']}
                    export HSA_FORCE_FINE_GRAIN_PCIE=1

                    export MASTER_PREFILL_ADDR={self.inf_dict['prefill_coordinator_addr']}
                    export MASTER_PREFILL_PORT={self.inf_dict['prefill_coordinator_port']}

                    export MODEL={self.bp_dict['model']}
                    export TP={self.bp_dict['tensor_parallelism']}
                    export HF_TOKEN={self.hf_token}
                    '  > /tmp/prefill_env_script.sh"
                    '''
        time.sleep(3)
        formatted_p_cmd = textwrap_for_yml(p_cmd)
        self.p_phdl.exec(formatted_p_cmd)
        cmd = f'''{get_docker_cmd()} exec {self.container_name} /bin/bash -c " \
                   chmod 755 /tmp/prefill_env_script.sh; /tmp/prefill_env_script.sh" '''
        self.p_phdl.exec(cmd)

    # Helper function for setting up decode container environment
    def setup_decode_container_env(
        self,
    ):
        # Env setup for Decode Nodes ..
        d_cmd = f'''{get_docker_cmd()} exec {self.container_name} /bin/bash -c "echo '

                    export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
                    export NCCL_DEBUG={self.inf_dict['nccl_debug']}
                    export NCCL_IB_HCA={self.inf_dict['nccl_ib_hca']}
                    export NCCL_IB_GID_INDEX={self.inf_dict['nccl_ib_gid_index']}
                    export HSA_FORCE_FINE_GRAIN_PCIE=1
                    export NCCL_SOCKET_IFNAME={self.inf_dict['nccl_socket_ifname']}
                    export GLOO_SOCKET_IFNAME={self.inf_dict['gloo_socket_ifname']}
                    export GLOO_TCP_IFNAME={self.inf_dict['gloo_socket_ifname']}
                    export HSA_FORCE_FINE_GRAIN_PCIE=1

                    export MASTER_DECODE_ADDR={self.inf_dict['decode_coordinator_addr']}
                    export MASTER_DECODE_PORT={self.inf_dict['decode_coordinator_port']}

                    export MODEL={self.bp_dict['model']}
                    export TP={self.bp_dict['tensor_parallelism']}
                    export HF_TOKEN={self.hf_token}
                    '  > /tmp/decode_env_script.sh"
                    '''
        time.sleep(3)
        formatted_d_cmd = textwrap_for_yml(d_cmd)
        self.d_phdl.exec(formatted_d_cmd)
        cmd = f'''{get_docker_cmd()} exec {self.container_name} /bin/bash -c " \
                   chmod 755 /tmp/decode_env_script.sh; /tmp/decode_env_script.sh" '''
        self.d_phdl.exec(cmd)

    # Helper function for setting up proxy router container environment
    def setup_proxy_router_container_env(
        self,
    ):
        # Env setup for Proxy Router Node ..
        r_cmd = f'''{get_docker_cmd()} exec {self.container_name} /bin/bash -c "echo '

                    export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
                    export NCCL_DEBUG={self.inf_dict['nccl_debug']}
                    export NCCL_IB_HCA={self.inf_dict['nccl_ib_hca']}
                    export NCCL_IB_GID_INDEX={self.inf_dict['nccl_ib_gid_index']}
                    export HSA_FORCE_FINE_GRAIN_PCIE=1
                    export NCCL_SOCKET_IFNAME={self.inf_dict['nccl_socket_ifname']}
                    export GLOO_SOCKET_IFNAME={self.inf_dict['gloo_socket_ifname']}
                    export GLOO_TCP_IFNAME={self.inf_dict['gloo_socket_ifname']}
                    export HSA_FORCE_FINE_GRAIN_PCIE=1

                    export HF_TOKEN={self.hf_token}
                    '  > /tmp/router_env_script.sh"
                    '''
        time.sleep(3)
        formatted_r_cmd = textwrap_for_yml(r_cmd)
        self.r_phdl.exec(formatted_r_cmd)
        cmd = f'''{get_docker_cmd()} exec {self.container_name} /bin/bash -c " \
                   chmod 755 /tmp/router_env_script.sh; /tmp/router_env_script.sh" '''
        self.r_phdl.exec(cmd)

    # Helper function for setting up benchmark server container environment
    def setup_benchmark_serv_container_env(
        self,
    ):
        # Env setup for Benchserv node ..
        b_cmd = f'''{get_docker_cmd()} exec {self.container_name} /bin/bash -c "echo '

                    export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
                    export NCCL_DEBUG={self.inf_dict['nccl_debug']}
                    export NCCL_IB_HCA={self.inf_dict['nccl_ib_hca']}
                    export NCCL_IB_GID_INDEX={self.inf_dict['nccl_ib_gid_index']}
                    export HSA_FORCE_FINE_GRAIN_PCIE=1
                    export NCCL_SOCKET_IFNAME={self.inf_dict['nccl_socket_ifname']}
                    export GLOO_SOCKET_IFNAME={self.inf_dict['gloo_socket_ifname']}
                    export GLOO_TCP_IFNAME={self.inf_dict['gloo_socket_ifname']}
                    export HSA_FORCE_FINE_GRAIN_PCIE=1
                    export HF_TOKEN={self.hf_token}
                    '  > /tmp/benchmark_env_script.sh"
                    '''
        time.sleep(3)
        formatted_b_cmd = textwrap_for_yml(b_cmd)
        self.b_phdl.exec(formatted_b_cmd)
        cmd = f'''{get_docker_cmd()} exec {self.container_name} /bin/bash -c " \
                   chmod 755 /tmp/benchmark_env_script.sh; /tmp/benchmark_env_script.sh" '''
        self.b_phdl.exec(cmd)
        time.sleep(5)

    # Helper function for running RMSNorm test
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
        # ------------------------------------------------------------------
        # Construct command to run RMSNorm test inside the container
        #
        # Details:
        #   - MAX_JOBS controls parallelism inside the test
        #   - Output is redirected to a per-container log file
        #   - Command is executed in the background to allow parallel execution
        # ------------------------------------------------------------------
        cmd = f'''{get_docker_cmd()} exec {self.container_name} /bin/bash -c  "MAX_JOBS={max_jobs} \
                python /sgl-workspace/aiter/op_tests/test_rmsnorm2d.py > /tmp/rsmnorm_test.log 2>&1 &" '''
        for hdl in [self.p_phdl, self.d_phdl, self.r_phdl]:
            out_dict = hdl.exec(cmd)
        log.info('Wait 180 secs for tests to complete')
        time.sleep(180)
        for hdl in [self.p_phdl, self.d_phdl, self.r_phdl]:
            cmd = f'''{get_docker_cmd()} exec {self.container_name} /bin/bash -c  "cat /tmp/rsmnorm_test.log" '''
            out_dict = hdl.exec(cmd)
            for node in out_dict.keys():
                if re.search('fail', out_dict[node], re.I):
                    log.warning(f'Some failures observed in test rmsnorm on node {node}')
                    fail_test(f'Some failures observed in test rmsnorm on node {node}')

    # supported --dtype {auto,half,float16,bfloat16,float,float32}
    # supported --kv-cache-dtype {auto,fp8_e5m2,fp8_e4m3,bf16,bfloat16,fp4_e2m1}
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

        cmd_list = []
        prefill_node_list = self.inf_dict['prefill_node_list']
        log.info('%%%% self.prefill_nnodes {}'.format(self.prefill_nnodes))
        dist_init_addr = f"{self.inf_dict['prefill_coordinator_addr']}:{self.inf_dict['prefill_coordinator_port']}"
        for i in range(0, int(self.prefill_nnodes)):
            cmd = f'''{get_docker_cmd()} exec {self.container_name} /bin/bash -c  "echo  '
                      export NNODES={self.prefill_nnodes}
                      export NODE_RANK={i}
                      export SGLANG_USE_AITER=1
                      export AMDGCN_USE_BUFFER_OPS=1
                      export ROCM_QUICK_REDUCE_QUANTIZATION=INT8
                      export LD_LIBRARY_PATH=/usr/local/lib:/sgl-workspace/Mooncake/build/mooncake-common/etcd:/opt/rocm/lib:$LD_LIBRARY_PATH
                      python3 -m sglang.launch_server --model {self.bp_dict['model']} \
                              --disaggregation-mode prefill \
                              --disaggregation-ib-device {self.inf_dict['nccl_ib_hca']} \
                              --host {prefill_node_list[i]} \
                              --port {self.inf_dict['prefill_serv_port']} \
                              --dtype {dtype} \
                              --kv-cache-dtype {kv_cache_dtype} \
                              --trust-remote-code \
                              --tp-size {self.bp_dict['tensor_parallelism']} \
                              --pp-size {self.bp_dict['pipeline_parallelism']} \
                              --nnodes {self.prefill_nnodes} \
                              --node-rank {i} \
                              --dist-init-addr {dist_init_addr} \
                              --disable-radix-cache --disable-cuda-graph \
                              --mem-fraction-static {self.bp_dict['memory_fraction']} \
                              --attention-backend aiter \
                              --context-length {self.bp_dict['context_length']} \
                              --log-level {self.inf_dict['log_level']}' > /tmp/prefill_launch_script.sh" '''
            formatted_cmd = textwrap_for_yml(cmd)
            cmd_list.append(formatted_cmd)
        log.info('%%%%%%%%%%%%%%%%%%%')
        log.info("%s", cmd_list)
        log.info('%%%%%%%%%%%%%%%%%%%')
        self.p_phdl.exec_cmd_list(cmd_list)
        log.info('#================ * * * =========================#')
        log.info('Launching Prefill servers on Prefill nodes')
        log.info('#================ * * * =========================#')
        cmd_list = []
        for i in range(0, int(self.prefill_nnodes)):
            cmd = f'''{get_docker_cmd()} exec {self.container_name} /bin/bash -c " \
                   chmod 755 /tmp/prefill_launch_script.sh; \
                   mkdir -p {self.log_dir}/prefill_node{i}; \
                   source /tmp/prefill_env_script.sh && \
                   nohup /tmp/prefill_launch_script.sh > \
                   {self.log_dir}/prefill_node{i}/prefill_server.log 2>&1 &" '''
            formatted_cmd = textwrap_for_yml(cmd)
            cmd_list.append(formatted_cmd)
        self.p_phdl.exec_cmd_list(cmd_list)
        time.sleep(5)

    # Helper function for launching Decode servers
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
        cmd_list = []
        decode_node_list = self.inf_dict['decode_node_list']
        log.info('%%%% self.decode_nnodes {}'.format(self.decode_nnodes))
        dist_init_addr = f"{self.inf_dict['decode_coordinator_addr']}:{self.inf_dict['decode_coordinator_port']}"
        for i in range(0, int(self.decode_nnodes)):
            cmd = f'''{get_docker_cmd()} exec {self.container_name} /bin/bash -c  "echo  '
                      export NNODES={self.decode_nnodes}
                      export NODE_RANK={i}
                      export SGLANG_USE_AITER=1
                      export AMDGCN_USE_BUFFER_OPS=1 
                      export ROCM_QUICK_REDUCE_QUANTIZATION=INT8
                      export LD_LIBRARY_PATH=/usr/local/lib:/sgl-workspace/Mooncake/build/mooncake-common/etcd:/opt/rocm/lib:$LD_LIBRARY_PATH
                      python3 -m sglang.launch_server --model {self.bp_dict['model']} \
                              --disaggregation-mode decode \
                              --disaggregation-ib-device {self.inf_dict['nccl_ib_hca']} \
                              --host {decode_node_list[i]} \
                              --port {self.inf_dict['decode_serv_port']} \
                              --trust-remote-code \
                              --dtype {dtype} \
                              --kv-cache-dtype {kv_cache_dtype} \
                              --tp-size {self.bp_dict['tensor_parallelism']} \
                              --pp-size {self.bp_dict['pipeline_parallelism']} \
                              --nnodes {self.decode_nnodes} \
                              --node-rank {i} \
                              --dist-init-addr {dist_init_addr} \
                              --disable-radix-cache --disable-cuda-graph \
                              --mem-fraction-static {self.bp_dict['memory_fraction']} \
                              --attention-backend aiter \
                              --context-length {self.bp_dict['context_length']} \
                              --log-level {self.inf_dict['log_level']}' > /tmp/decode_launch_script.sh" '''
            formatted_cmd = textwrap_for_yml(cmd)
            cmd_list.append(formatted_cmd)
        log.info('%%%%%%%%%%%%%%%%%%%')
        log.info("%s", cmd_list)
        log.info('%%%%%%%%%%%%%%%%%%%')
        self.d_phdl.exec_cmd_list(cmd_list)
        log.info('#================ * * * =========================#')
        log.info('Launching Decode servers on Decode nodes')
        log.info('#================ * * * =========================#')
        cmd_list = []
        for i in range(0, int(self.decode_nnodes)):
            cmd = f'''{get_docker_cmd()} exec {self.container_name} /bin/bash -c " \
                   chmod 755 /tmp/decode_launch_script.sh; \
                   mkdir -p {self.log_dir}/decode_node{i}; \
                   source /tmp/decode_env_script.sh && \
                   nohup bash /tmp/decode_launch_script.sh > \
                   {self.log_dir}/decode_node{i}/decode_server.log 2>&1 &" '''
            formatted_cmd = textwrap_for_yml(cmd)
            cmd_list.append(formatted_cmd)
        self.d_phdl.exec_cmd_list(cmd_list)

    # Helper function for polling for server readiness
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
        # for node_no in range(0, self.prefill_nnodes):
        #     self.poll_for_server_ready(node_no, 'prefill')
        # for node_no in range(0, self.decode_nnodes):
        #     self.poll_for_server_ready(node_no, 'decode')
        self.poll_for_server_ready(0, 'prefill')
        self.poll_for_server_ready(0, 'decode')

    
    # Helper function for launching Proxy Router
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
        - Coordinate Prefill ? Decode handoff

        This method:
        - Builds routing configuration dynamically based on cluster topology
        - Creates a launch script on the Proxy Router node
        - Launches the router as a background service
        """

        # ------------------------------------------------------------------
        # Build Prefill endpoint arguments for the router
        #
        # Each Prefill server is specified as:
        #   --prefill http://<host>:<port>
        # ------------------------------------------------------------------

        prefill_str = (
            f"--prefill http://{self.inf_dict['prefill_coordinator_addr']}:{self.inf_dict['prefill_serv_port']} "
        )
        # ------------------------------------------------------------------
        # Build Decode endpoint arguments for the router
        #
        # Each Decode server is specified as:
        #   --decode http://<host>:<port>
        # ------------------------------------------------------------------

        decode_str = f"--decode http://{self.inf_dict['decode_coordinator_addr']}:{self.inf_dict['decode_serv_port']} "
        log.info('#================ * * * =========================#')
        log.info('Create Proxy Router launch script on Proxy Router nodes')
        log.info('#================ * * * =========================#')

        # ------------------------------------------------------------------
        # Create the Proxy Router launch script
        #
        # Key flags:
        #   --pd-disaggregation : Enable Prefill/Decode disaggregation
        #   --prefill / --decode: Upstream Prefill and Decode endpoints
        #   --host 0.0.0.0      : Listen on all interfaces
        #   --port              : External router port
        #   --log-dir           : Directory for router logs
        #
        # NOTE:
        #   The script is written to disk but not executed here.
        # ------------------------------------------------------------------
        cmd = f'''{get_docker_cmd()} exec {self.container_name} /bin/bash -c  "echo  '
                      python3 -m sglang_router.launch_router \
                              --pd-disaggregation \
                              {prefill_str} \
                              {decode_str} \
                              --host 0.0.0.0 \
                              --port {self.inf_dict['proxy_router_port']} \
                              --log-dir {self.inf_dict['log_dir']} \
                      '  > /tmp/proxy_router_launch_script.sh"
                    '''
        formatted_cmd = textwrap_for_yml(cmd)
        self.r_phdl.exec(formatted_cmd)
        log.info('#================ * * * =========================#')
        log.info('Launch Proxy Router script on Proxy Router nodes')
        log.info('#================ * * * =========================#')
        cmd = f'''{get_docker_cmd()} exec {self.container_name} /bin/bash -c " \
                   chmod 755 /tmp/proxy_router_launch_script.sh; \
                   mkdir -p {self.log_dir}/proxy_router_node; \
                   source /tmp/router_env_script.sh && \
                   nohup bash /tmp/proxy_router_launch_script.sh > \
                   {self.log_dir}/proxy_router_node/proxy_router.log 2>&1 &" '''
        formatted_cmd = textwrap_for_yml(cmd)
        self.r_phdl.exec(formatted_cmd)
        log.info('Waiting 120 secs after launching proxy router script')
        time.sleep(120)

    
    # Helper function for running SGLang serving benchmark with random dataset
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
        # ------------------------------------------------------------------
        # Construct command to run sglang.bench_serving with random dataset
        #
        # Key parameters:
        #   --dataset-name random     : Use synthetic random prompts
        #   --num-prompts             : Total number of inference requests
        #   --random-input            : Input token length per request
        #   --random-output           : Output token length per request
        #   --random-range-ratio      : Variability in input/output lengths
        #   --host / --port           : Proxy Router endpoint
        #
        # Output is redirected to a log file for later inspection.
        # ------------------------------------------------------------------
        cmd = f'''{get_docker_cmd()} exec {self.container_name} /bin/bash -c  "
                      mkdir -p {self.log_dir}/benchmark_node; \
                      source /tmp/benchmark_env_script.sh && \
                      pip install --upgrade --no-deps sglang[all] && \
                      python3 -m sglang.bench_serving --backend {i_dict['backend']} \
                      --dataset-name random \
                      --num-prompts {i_dict['num_prompts']} \
                      --max-concurrency {self.bp_dict['max_concurrency']} \
                      --random-input {i_dict['input_length']} \
                      --random-output {i_dict['output_length']} \
                      --random-range-ratio {i_dict['random_range_ratio']} \
                      --host 0.0.0.0 --port {self.inf_dict['proxy_router_serv_port']} \
                      > {self.log_dir}/benchmark_node/benchmark_results.log 2>&1" '''
        formatted_cmd = textwrap_for_yml(cmd)
        self.b_phdl.exec(formatted_cmd, timeout=1000)
        time.sleep(5)
        self.poll_for_inference_completion(iterations=10, waittime_between_iters=60)
        
        # MFU (derived from same bench metrics as TTFT/TPOT)
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
            inner = (
                f"echo '' >> {log_path} && "
                f"echo '============ Derived Benchmark Results ============' >> {log_path} && "
                f"echo 'Goodput (successful / total): {sr} / {tr}  =>  {gp}' >> {log_path} && "
                f"echo 'Output token throughput per GPU (tok/s/GPU): {tpg}' >> {log_path} && "
                 f"echo 'MFU (estimated): {mfu}' >> {log_path} && "
                f"echo '=====================================================================' >> {log_path}"
            )
            cmd = f"{get_docker_cmd()} exec {self.container_name} /bin/bash -c {shlex.quote(inner)}"
            self.b_phdl.exec(cmd)

        self.verify_inference_results('bench_serv', i_dict['expected_results'][d_type])

    # Helper function for polling for server readiness
    def poll_for_server_ready(self, node_no, sglang_function, no_of_iterations=16):
        """
        Poll SGLang Prefill or Decode server logs to determine when the server
        is ready to accept inference traffic.

        Readiness definition:
        ---------------------
        A server is considered "ready" when its log shows successful HTTP
        requests (HTTP 200 OK), indicating that:
        - The server process has started
        - The model is loaded
        - Network endpoints are listening
        - Request handling is functional

        Assumptions:
        ------------
        - Log directory is located on a shared filesystem (e.g., NFS)
        - Logs are accessible from a designated head node
        - Each server writes logs to a predictable per-node path

        Args:
        node_no (int): Index of the Prefill or Decode node being checked
        sglang_function (str): Server role ('prefill' or 'decode')
        no_of_iterations (int): Maximum number of polling attempts before
                                declaring failure
        """
        # ------------------------------------------------------------------
        # Prefill server readiness check
        # ------------------------------------------------------------------
        if re.search('prefill', sglang_function):
            for j in range(1, no_of_iterations):
                log.info(f'Starting poll iteration {j}')
                out_dict = self.p_phdl.exec(
                    f'grep -B 20 -A 20 "200 OK" {self.log_dir}/prefill_node{node_no}/prefill_server.log'
                )
                target_pnode = self.prefill_node_list[node_no]
                if re.search('GET|POST', out_dict[target_pnode], re.I):
                    log.info('Wait 60 secs to start serving traffic')
                    time.sleep(60)
                    # if re.search('fired up and ready to roll', out_dict[target_pnode], re.I ):
                    #    print('Prefill server {node_no} ready to serve')
                    return
                else:
                    log.info('Wait for 120 secs and continue polling')
                    time.sleep(120)

            log.warning(f'Prefill node {node_no} did not get to ready to serve 200 OK state in {j} iterations')
            fail_test(f'Prefill node {node_no} did not get to ready to serve 200 OK state in {j} iterations')
        # ------------------------------------------------------------------
        # Decode server readiness check
        # ------------------------------------------------------------------
        elif re.search('decode', sglang_function):
            for j in range(1, no_of_iterations):
                log.info(f'Starting poll iteration {j}')
                out_dict = self.d_phdl.exec(
                    f'grep -B 20 -A 20 "200 OK" {self.log_dir}/decode_node{node_no}/decode_server.log'
                )
                target_dnode = self.decode_node_list[node_no]
                if re.search('GET|POST', out_dict[target_dnode]):
                    log.info('Wait 60 secs to start serving traffic')
                    time.sleep(60)
                    # if re.search('fired up and ready to roll', out_dict[target_dnode], re.I ):
                    #    print('Decode server {node_no} ready to serve')
                    return
                else:
                    log.info('Wait for 120 secs and continue polling')
                    time.sleep(120)
            log.warning(f'Decode node {node_no} did not get to ready to serve 200 OK state in {j} iterations')
            fail_test(f'Decode node {node_no} did not get to ready to serve 200 OK state in {j} iterations')

    # Helper function for getting inference results dictionary
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
            # --- SGLang "E2E Latency" wording -----
            m = _first_float(r'Mean E2E Latency \(ms\):\s+([0-9\.]+)', out_dict[node])
            if m:
                self.inference_results_dict[node]['mean_e2e_latency_ms'] = m
            m = _first_float(r'Median E2E Latency \(ms\):\s+([0-9\.]+)', out_dict[node])
            if m:
                self.inference_results_dict[node]['median_e2e_latency_ms'] = m
            for p in (90, 95, 99):
                m = _first_float(rf'P{p} E2E Latency \(ms\):\s+([0-9\.]+)', out_dict[node])
                if m:
                    self.inference_results_dict[node][f'p{p}_e2e_latency_ms'] = m

            # Goodput: log totals, or successful+failed, or benchmark num_prompts
            total_req = _first_float(r"Total requests:\s+([0-9]+)", out_dict[node])
            failed_req = _first_float(r"Failed requests:\s+([0-9]+)", out_dict[node])
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

            # Per-GPU throughput (derived): total output tok/s / GPUs in PD fleet
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

    # Helper function for scanning for inference errors
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

        # Build the list of commands to read each node's inference log file
        cmd_list = []

        # Scan all prefill nodes
        for j in range(0, int(self.prefill_nnodes)):
            cmd = f"sudo tail -100 {self.log_dir}/prefill_node{j}/prefill_server.log"
            cmd_list.append(cmd)
        out_dict = self.p_phdl.exec_cmd_list(cmd_list)

        # Check the log content against all known inference error patterns
        for node in out_dict.keys():
            for err_key in inference_err_dict:
                if re.search(f'{inference_err_dict[err_key]}', out_dict[node]):
                    fail_test(f'ERROR {inference_err_dict[err_key]} seen in inference logs ..')
                    log.error('Aborting inference log polling')
                    inference_pass = False

        # Scan all decode nodes
        cmd_list = []
        for j in range(0, int(self.decode_nnodes)):
            cmd = f"sudo tail -500 {self.log_dir}/decode_node{j}/decode_server.log"
            cmd_list.append(cmd)
        out_dict = self.d_phdl.exec_cmd_list(cmd_list)

        # Check the log content against all known inference error patterns
        for node in out_dict.keys():
            for err_key in inference_err_dict:
                if re.search(f'{inference_err_dict[err_key]}', out_dict[node]):
                    fail_test(f'ERROR {inference_err_dict[err_key]} seen in inference logs ..')
                    log.error('Aborting inference log polling')
                    inference_pass = False

        return inference_pass

    # Helper function for polling for inference completion
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
        # Initial wait to give inference time to start logging
        time.sleep(60)

        # Track wall-clock timeout if specified
        start_time = time.time()

        def timed_out() -> bool:
            return total_timeout is not None and (time.time() - start_time) >= float(total_timeout)

        completed_pattern = re.compile('Serving Benchmark Result', re.I)
        # ------------------------------------------------------------------
        # Poll loop: periodically inspect benchmark logs for completion
        # ------------------------------------------------------------------
        for itr in range(1, iterations + 1):
            log.info(f'Starting iteration {itr}')

            # --------------------------------------------------------------
            # Early exit if any inference errors are detected
            #
            # This scans Prefill and Decode logs for known failure patterns
            # (e.g., OOM, RDMA failures, backend crashes).
            # --------------------------------------------------------------
            # Early abort on inference errors
            # if not self.scan_for_inference_errors():
            #     msg = 'Failures seen in inference logs, Aborting!!!'
            #     fail_test(msg)
            #     return {"status": "error", "reason": msg}

            # --------------------------------------------------------------
            # Read the most recent benchmark output
            #
            # Tail only the last 1000 lines to reduce I/O and parsing cost.
            # --------------------------------------------------------------
            cmd = f"sudo tail -1000 {self.log_dir}/benchmark_node/benchmark_results.log"

            out_dict = self.b_phdl.exec(cmd)

            # Determine completion across nodes
            node_completion = {}
            for node, output in out_dict.items():
                node_completion[node] = bool(completed_pattern.search(output))

            # --------------------------------------------------------------
            # Determine overall completion based on policy
            #
            # - require_all_nodes=True  ? all nodes must complete
            # - require_all_nodes=False ? any node completing is sufficient
            # --------------------------------------------------------------
            if require_all_nodes:
                all_complete = all(node_completion.values()) if node_completion else False
            else:
                all_complete = any(node_completion.values()) if node_completion else False

            # --------------------------------------------------------------
            # If inference is still running, wait and retry
            # --------------------------------------------------------------
            if not all_complete:
                if timed_out():
                    msg = f"Timeout while waiting for inference completion after ~{int(time.time() - start_time)}s"
                    log.warning("%s", msg)
                    return {"status": "timeout", "reason": msg}
                log.info('Inference still in progress')
                # Short progress wait before the longer inter-iteration sleep
                time.sleep(30)
                time.sleep(int(waittime_between_iters))
                continue

            # --------------------------------------------------------------
            # Inference completed successfully
            #
            # Parse benchmark results and return structured output.
            # --------------------------------------------------------------
            self.get_inference_results_dict(out_dict)
            log.info('Completed Inference, returning !!!')
            return {"status": "success", "results": self.inference_results_dict}

            # If we reached here, it means poll for inference completion failed

        # If we exhaust the iteration cap without completing, treat as timeout (or in_progress if no wall-clock limit)
        if timed_out():
            msg = f"Timeout after maximum iterations ({self.inference_poll_iterations}) and ~{int(time.time() - start_time)}s"
            log.warning("%s", msg)
            return {"status": "timeout", "reason": msg}
        else:
            # If no wall-clock timeout was set and we hit the iteration cap, report in-progress
            msg = f"Reached iteration cap ({self.inference_poll_iterations}) without completion; still in progress"
            log.warning("%s", msg)
            return {"status": "stuck_in_progress", "reason": msg}

    # Helper function for verifying inference results
    def verify_inference_results(self, test_name, expected_result_dict):
        """
        Validate inference benchmark results against expected performance
        thresholds and check for system-level errors.

        Purpose:
        --------
        This method verifies that:
        - Inference completed successfully on all nodes
        - Performance metrics meet or exceed expected baselines
        - Latency metrics stay below defined thresholds
        - No kernel-level (dmesg) errors occurred during inference

        It acts as the final gate for inference validation.
        """
        log.info('Verify Inference Completion Msg')
        log.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        log.info("%s", self.inference_results_dict)
        log.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        # ------------------------------------------------------------------
        # Validate metrics on a per-node basis
        # ------------------------------------------------------------------
        for node in self.inference_results_dict.keys():
            log.info('%%%% node {}'.format(node))
            for metric_name in expected_result_dict.keys():
                log.info('%%% metric_name {}'.format(metric_name))
                if metric_name in self.inference_results_dict[node].keys():
                    # latency metric, so actual should be lower than expected ..
                    log.info('%% metric found in inference results ^^^')
                    # ------------------------------------------------------
                    # Latency metrics (e.g., TTFT, TPOT)
                    #
                    # For latency, lower values are better.
                    # Fail if actual latency exceeds expected threshold.
                    # ------------------------------------------------------
                    if re.search('ms', metric_name, re.I):
                        log.info("%s", self.inference_results_dict[node][metric_name])
                        log.info("%s", expected_result_dict[metric_name])
                        if float(self.inference_results_dict[node][metric_name]) > float(
                            expected_result_dict[metric_name]
                        ):
                            fail_test(
                                f"FAIL - The metric {metric_name} actual value higher than expected \
                                Actual = {self.inference_results_dict[node][metric_name]},  \
                                Expected = {expected_result_dict[metric_name]}"
                            )
                    # ------------------------------------------------------
                    # Throughput and count metrics
                    #
                    # For throughput, higher values are better.
                    # Fail if actual throughput is lower than expected.
                    # ------------------------------------------------------
                    else:
                        if float(self.inference_results_dict[node][metric_name]) < float(
                            expected_result_dict[metric_name]
                        ):
                            fail_test(
                                f"FAIL - The metric {metric_name} actual value lower than expected \
                                Actual = {self.inference_results_dict[node][metric_name]}, \
                                Expected = {expected_result_dict[metric_name]}"
                            )

        # ------------------------------------------------------------------
        # Perform kernel-level (dmesg) error checks
        #
        # This ensures no silent hardware or driver errors occurred during
        # inference (e.g., GPU resets, RDMA failures, IOMMU errors).
        # ------------------------------------------------------------------
        self.inference_end_time = self.p_phdl.exec('date +"%a %b %e %H:%M"')
        time.sleep(2)
        verify_dmesg_for_errors(self.p_phdl, self.inference_start_time, self.inference_end_time)
        verify_dmesg_for_errors(self.d_phdl, self.inference_start_time, self.inference_end_time)
        verify_dmesg_for_errors(self.r_phdl, self.inference_start_time, self.inference_end_time)
        verify_dmesg_for_errors(self.b_phdl, self.inference_start_time, self.inference_end_time)
        log.info("%s", self.inference_results_dict)

    # Helper function for counting occupied GPUs per prefill/decode node
    def sglang_disagg_gpu_counts(self, mem_threshold_mb=5000):
        """
        After model load, count occupied GPUs per prefill/decode node via amd-smi.
        """
        tp = int(self.bp_dict["tensor_parallelism"])
        pp = int(self.bp_dict.get("pipeline_parallelism", 1))

        def _count_per_node(phdl):
            per_node = {}
            for node, payload in phdl.exec("sudo amd-smi metric --json").items():
                count = 0
                try:
                    entries = json.loads(payload.strip())
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

        prefill_per_node = _count_per_node(self.p_phdl)
        decode_per_node = _count_per_node(self.d_phdl)
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

    # Helper function for verifying OpenAI-compatible endpoints
    def verify_openai_compatible_endpoints(self) -> list[str]:
        """
        Smoke-test OpenAI-compatible HTTP API on the proxy router (inside the
        benchmark container via ``{get_docker_cmd()} exec``): GET /v1/models,
        POST /v1/chat/completions, POST /v1/completions, and structured JSON
        (book) via chat completions.
        """
        port = int(self.inf_dict["proxy_router_serv_port"])
        model_name = self.bp_dict["model"]

        probe_src = OpenAIProbe.probe_script(port, model_name)
        b64 = base64.b64encode(probe_src.encode("utf-8")).decode("ascii")
        cmd = f'''{get_docker_cmd()} exec {self.container_name} /bin/bash -c  "
                  mkdir -p {self.log_dir}/benchmark_node; \
                  echo '{b64}' | base64 -d > /tmp/openai_mq_probe.py && \
                  python3 /tmp/openai_mq_probe.py && \
                  rm -f /tmp/openai_mq_probe.py" '''
        formatted_cmd = textwrap_for_yml(cmd)
        log.info(
            "OpenAI endpoint probe inside benchmark container (0.0.0.0:%r), same pattern as GSM8K/benchserv",
            port,
        )
        out_dict = self.b_phdl.exec(
            formatted_cmd,
            timeout=min(900, 480 + 180),
        )
        bench_host = self.benchmark_serv_node[0]
        raw_out = out_dict.get(bench_host)
        if raw_out is None and out_dict:
            raw_out = next(iter(out_dict.values()))

        probe_err: Optional[str] = None
        results: dict[str, tuple[int, Any]] = {}
        if not raw_out or not str(raw_out).strip():
            probe_err = (
                f"OpenAI-compatible probe produced no output node {bench_host!r}: "
                f"{out_dict!r}"
            )
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

    # Helper functions for running LM-Eval benchmarks
    def run_lm_eval_hellaswag_benchmark_test(self, _d_type="auto"):
        return self.run_lm_eval_benchmark_test("lm_eval_hellaswag", _d_type=_d_type)

    # Helper functions for running GSM8K benchmarks
    def run_lm_eval_gsm8k_benchmark_test(self, _d_type="auto"):
        return self.run_lm_eval_benchmark_test("lm_eval_gsm8k", _d_type=_d_type)

    # Helper functions for running MMLU benchmarks
    def run_lm_eval_mmlu_benchmark_test(self, _d_type="auto"):
        return self.run_lm_eval_benchmark_test("lm_eval_mmlu", _d_type=_d_type)


    # Helper function for running LM-Eval benchmarks    
    def run_lm_eval_benchmark_test(self, bench_key: str, _d_type="auto"):
        spec = LM_EVAL_SPECS[bench_key]
        log.info("#================ * * * =========================#")
        log.info("lm-eval %s benchmark", spec["display"])
        log.info("#================ * * * =========================#")
        task_name = bench_key.removeprefix("lm_eval_")
        i_dict = self.bp_dict["inference_tests"][bench_key]
        inner_cmd, scoring = LmEvalBenchmark.prepare(
            i_dict,
            port=int(self.inf_dict["proxy_router_serv_port"]),
            model_id=self.bp_dict["model"],
            task_name=task_name,
            default_tasks=task_name,
            default_metric=spec["default_metric"],
            default_metric_key=spec["default_metric_key"],
            log_dir=self.log_dir,
            log_basename=f"{bench_key}.log",
            default_num_concurrent=spec["default_num_concurrent"],
        )

        cmd = f'''{get_docker_cmd()} exec {self.container_name} /bin/bash -c  "
                mkdir -p {self.log_dir}/benchmark_node; \\
                source /tmp/benchmark_env_script.sh && \\
                {inner_cmd}" '''
        out_dict = self.b_phdl.exec(textwrap_for_yml(cmd), timeout=scoring["exec_timeout_sec"])
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
        port = int(self.inf_dict["proxy_router_serv_port"])
        log_basename = f"long_ctx_niah_isl{isl}_osl{osl}.log"

        inner_cmd, scoring = LongContextNiahBenchmark.prepare(
            i_dict,
            port=port,
            model_id=self.bp_dict["model"],
            isl=int(isl),
            osl=int(osl),
            log_dir=self.log_dir,
            log_basename=log_basename,
        )
        probe_src = LongContextNiahBenchmark.probe_script(**scoring["probe_kwargs"])
        b64 = base64.b64encode(probe_src.encode("utf-8")).decode("ascii")

        cmd = f'''{get_docker_cmd()} exec {self.container_name} /bin/bash -c  "
                mkdir -p {self.log_dir}/benchmark_node; \\
                echo '{b64}' | base64 -d > /tmp/long_ctx_niah_probe.py && \\
                source /tmp/benchmark_env_script.sh && \\
                {inner_cmd}" '''
        out_dict = self.b_phdl.exec(
            textwrap_for_yml(cmd),
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