'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import os
import re
import time

from cvs.lib import globals
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
        self.prefill_node_list = self.inf_dict['prefill_node_list']
        self.decode_node_list = self.inf_dict['decode_node_list']
        self.prefill_nnodes = len(self.prefill_node_list)
        self.decode_nnodes = len(self.decode_node_list)

        self.proxy_node = list(self.inf_dict['proxy_router_node'])
        self.benchmark_serv_node = list(self.inf_dict['benchmark_serv_node'])

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
        print(self.gpu_type)

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

        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print(f'inference_dict = {self.inf_dict}')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
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
        self.bp_dict.setdefault('max_model_length', '9216')
        self.bp_dict.setdefault('random_range_ration', '1.0')
        self.bp_dict.setdefault('random_prefix_len', '0')
        self.bp_dict.setdefault('tensor_parallelism', '8')
        self.bp_dict.setdefault('port_no', '8000')
        self.bp_dict.setdefault('tokenizer_mode', 'auto')
        self.bp_dict.setdefault('percentile_metrics', 'ttft,tpot,itl,e2el')
        self.bp_dict.setdefault('metric_percentiles', '99')
        self.bp_dict.setdefault('inference_poll_iterations', '16')

        self.inference_poll_iterations = self.bp_dict['inference_poll_iterations']

        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print(f'benchmark_params_dict = {self.bp_dict}')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

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

        print('Run pre inference tasks')
        # Install ip tools
        cmd = f'docker exec {self.container_name} /bin/bash -c " \
            sudo apt -y update; \
            sudo apt install -y iputils-ping; \
            sudo apt install -y iproute2; \
            sudo apt install -y net-tools" '
        self.p_phdl.exec(cmd)
        self.d_phdl.exec(cmd)
        self.r_phdl.exec(cmd)

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
            cmd = f'docker exec {self.container_name} /bin/bash -c "sudo \
                    cp /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.host \
                    /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so;" '
            pout_dict = self.p_phdl.exec(cmd)
            dout_dict = self.d_phdl.exec(cmd)
            for node in pout_dict.keys():
                if not re.search('hca_id:\s+bnxt_', out_dict[node], re.I):
                    print(pout_dict[node])
                    fail_test(f'Broadcom libbnxt rdma driver is not properly copied on node {node}')
            for node in dout_dict.keys():
                if not re.search('hca_id:\s+bnxt_', out_dict[node], re.I):
                    print(dout_dict[node])
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
        for hdl in [self.p_phdl, self.d_phdl]:
            cmd = f'''docker exec {self.container_name} /bin/bash -c "ibv_devinfo" '''
            out_dict = hdl.exec(cmd)
            for node in out_dict.keys():
                if re.search('No IB devices found', out_dict[node], re.I):
                    fail_test(f'IB devices not seen inside the container for node {node}')

    def setup_prefill_container_env(
        self,
    ):
        # Env setup for Prefill Nodes ..
        p_cmd = f'''docker exec {self.container_name} /bin/bash -c "echo '

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
        cmd = f'''docker exec {self.container_name} /bin/bash -c " \
                   chmod 755 /tmp/prefill_env_script.sh; /tmp/prefill_env_script.sh" '''
        self.p_phdl.exec(cmd)

    def setup_decode_container_env(
        self,
    ):
        # Env setup for Decode Nodes ..
        d_cmd = f'''docker exec {self.container_name} /bin/bash -c "echo '

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
        cmd = f'''docker exec {self.container_name} /bin/bash -c " \
                   chmod 755 /tmp/decode_env_script.sh; /tmp/decode_env_script.sh" '''
        self.d_phdl.exec(cmd)

    def setup_proxy_router_container_env(
        self,
    ):
        # Env setup for Proxy Router Node ..
        r_cmd = f'''docker exec {self.container_name} /bin/bash -c "echo '

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
        cmd = f'''docker exec {self.container_name} /bin/bash -c " \
                   chmod 755 /tmp/router_env_script.sh; /tmp/router_env_script.sh" '''
        self.r_phdl.exec(cmd)

    def setup_benchmark_serv_container_env(
        self,
    ):
        # Env setup for Benchserv node ..
        b_cmd = f'''docker exec {self.container_name} /bin/bash -c "echo '

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
        cmd = f'''docker exec {self.container_name} /bin/bash -c " \
                   chmod 755 /tmp/benchmark_env_script.sh; /tmp/benchmark_env_script.sh" '''
        self.b_phdl.exec(cmd)
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
        print('#================ * * * =========================#')
        print('Run rmsnorm2d')
        print('#================ * * * =========================#')
        # ------------------------------------------------------------------
        # Construct command to run RMSNorm test inside the container
        #
        # Details:
        #   - MAX_JOBS controls parallelism inside the test
        #   - Output is redirected to a per-container log file
        #   - Command is executed in the background to allow parallel execution
        # ------------------------------------------------------------------
        cmd = f'''docker exec {self.container_name} /bin/bash -c  "MAX_JOBS={max_jobs} \
                python /sgl-workspace/aiter/op_tests/test_rmsnorm2d.py > /tmp/rsmnorm_test.log 2>&1 &" '''
        for hdl in [self.p_phdl, self.d_phdl, self.r_phdl]:
            out_dict = hdl.exec(cmd)
        print('Wait 180 secs for tests to complete')
        time.sleep(180)
        for hdl in [self.p_phdl, self.d_phdl, self.r_phdl]:
            cmd = f'''docker exec {self.container_name} /bin/bash -c  "cat /tmp/rsmnorm_test.log" '''
            out_dict = hdl.exec(cmd)
            for node in out_dict.keys():
                if re.search('fail', out_dict[node], re.I):
                    print(f'Some failures observed in test rmsnorm on node {node}')
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
        print('#================ * * * =========================#')
        print('Create Prefill launch script on Prefill nodes')
        print('#================ * * * =========================#')

        cmd_list = []
        prefill_node_list = self.inf_dict['prefill_node_list']
        print('%%%% self.prefill_nnodes {}'.format(self.prefill_nnodes))
        for i in range(0, int(self.prefill_nnodes)):
            cmd = f'''docker exec {self.container_name} /bin/bash -c  "echo  '
                      export NNODES={self.prefill_nnodes}
                      export NODE_RANK={i}
                      export SGLANG_USE_AITER=1
                      python3 -m sglang.launch_server --model {self.bp_dict['model']} \
                              --disaggregation-mode prefill \
                              --disaggregation-ib-device {self.inf_dict['nccl_ib_hca']} \
                              --host {prefill_node_list[i]} \
                              --port {self.inf_dict['prefill_serv_port']} \
                              --dtype {dtype} \
                              --kv-cache-dtype {kv_cache_dtype} \
                              --trust-remote-code \
                              --tp {self.bp_dict['tensor_parallelism']} \
                              --disable-radix-cache --disable-cuda-graph \
                              --mem-fraction-static {self.bp_dict['memory_fraction']} \
                              --attention-backend aiter \
                              --log-level {self.inf_dict['log_level']}' > /tmp/prefill_launch_script.sh" '''
            formatted_cmd = textwrap_for_yml(cmd)
            cmd_list.append(formatted_cmd)
        print('%%%%%%%%%%%%%%%%%%%')
        print(cmd_list)
        print('%%%%%%%%%%%%%%%%%%%')
        self.p_phdl.exec_cmd_list(cmd_list)
        print('#================ * * * =========================#')
        print('Launching Prefill servers on Prefill nodes')
        print('#================ * * * =========================#')
        cmd_list = []
        for i in range(0, int(self.prefill_nnodes)):
            cmd = f'''docker exec {self.container_name} /bin/bash -c " \
                   chmod 755 /tmp/prefill_launch_script.sh; \
                   mkdir -p {self.log_dir}/prefill_node{i}; \
                   source /tmp/prefill_env_script.sh && \
                   nohup /tmp/prefill_launch_script.sh > \
                   {self.log_dir}/prefill_node{i}/prefill_server.log 2>&1 &" '''
            formatted_cmd = textwrap_for_yml(cmd)
            cmd_list.append(formatted_cmd)
        self.p_phdl.exec_cmd_list(cmd_list)
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
        print('#================ * * * =========================#')
        print('Create Decode launch script on Decode nodes')
        print('#================ * * * =========================#')
        cmd_list = []
        decode_node_list = self.inf_dict['decode_node_list']
        print('%%%% self.decode_nnodes {}'.format(self.decode_nnodes))
        for i in range(0, int(self.decode_nnodes)):
            # ------------------------------------------------------------------
            # Construct a command that writes a Decode server launch script
            # into /tmp/decode_launch_script.sh inside the container
            #
            # Key configuration details:
            #   - NNODES / NODE_RANK: Distributed topology for SGLang
            #   - disaggregation-mode decode: Run in Decode-only mode
            #   - disaggregation-ib-device: RDMA device used for KV transfers
            #   - host / port: Network endpoint for this Decode server
            #   - dtype / kv-cache-dtype: Compute and KV precision
            #   - tensor parallelism: Model sharding across GPUs
            #   - aiter backend: Optimized attention backend for AMD GPUs
            #   - memory fraction: Static GPU memory reservation
            #
            # NOTE:
            #   The script is written (echo > file), not executed here.
            #   Execution is handled by a separate orchestration step.
            # ------------------------------------------------------------------
            cmd = f'''docker exec {self.container_name} /bin/bash -c  "echo  '
                      export NNODES={self.decode_nnodes}
                      export NODE_RANK={i}
                      export SGLANG_USE_AITER=1
                      python3 -m sglang.launch_server --model {self.bp_dict['model']} \
                              --disaggregation-mode decode \
                              --disaggregation-ib-device {self.inf_dict['nccl_ib_hca']} \
                              --host {decode_node_list[i]} \
                              --port {self.inf_dict['decode_serv_port']} \
                              --trust-remote-code \
                              --dtype {dtype} \
                              --kv-cache-dtype {kv_cache_dtype} \
                              --tp {self.bp_dict['tensor_parallelism']} \
                              --disable-radix-cache --disable-cuda-graph \
                              --mem-fraction-static {self.bp_dict['memory_fraction']} \
                              --attention-backend aiter \
                              --log-level {self.inf_dict['log_level']}' > /tmp/decode_launch_script.sh" '''
            formatted_cmd = textwrap_for_yml(cmd)
            cmd_list.append(formatted_cmd)
        print('%%%%%%%%%%%%%%%%%%%')
        print(cmd_list)
        print('%%%%%%%%%%%%%%%%%%%')
        self.d_phdl.exec_cmd_list(cmd_list)
        print('#================ * * * =========================#')
        print('Launching Decode servers on Decode nodes')
        print('#================ * * * =========================#')
        cmd_list = []
        for i in range(0, int(self.decode_nnodes)):
            cmd = f'''docker exec {self.container_name} /bin/bash -c " \
                   chmod 755 /tmp/decode_launch_script.sh; \
                   mkdir -p {self.log_dir}/decode_node{i}; \
                   source /tmp/decode_env_script.sh && \
                   nohup bash /tmp/decode_launch_script.sh > \
                   {self.log_dir}/decode_node{i}/decode_server.log 2>&1 &" '''
            formatted_cmd = textwrap_for_yml(cmd)
            cmd_list.append(formatted_cmd)
        self.d_phdl.exec_cmd_list(cmd_list)

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
        print('Waiting 120 secs after launching decode script')
        time.sleep(120)
        for node_no in range(0, self.prefill_nnodes):
            self.poll_for_server_ready(node_no, 'prefill')
        for node_no in range(0, self.decode_nnodes):
            self.poll_for_server_ready(node_no, 'decode')

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
        prefill_str = ''
        for prefill_node in self.prefill_node_list:
            prefill_str = prefill_str + f"--prefill http://{prefill_node}:{self.inf_dict['prefill_serv_port']} "
        # ------------------------------------------------------------------
        # Build Decode endpoint arguments for the router
        #
        # Each Decode server is specified as:
        #   --decode http://<host>:<port>
        # ------------------------------------------------------------------
        decode_str = ''
        for decode_node in self.decode_node_list:
            decode_str = decode_str + f"--decode http://{decode_node}:{self.inf_dict['decode_serv_port']} "
        print('#================ * * * =========================#')
        print('Create Proxy Router launch script on Proxy Router nodes')
        print('#================ * * * =========================#')

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
        cmd = f'''docker exec {self.container_name} /bin/bash -c  "echo  '
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
        print('#================ * * * =========================#')
        print('Launch Proxy Router script on Proxy Router nodes')
        print('#================ * * * =========================#')
        cmd = f'''docker exec {self.container_name} /bin/bash -c " \
                   chmod 755 /tmp/proxy_router_launch_script.sh; \
                   mkdir -p {self.log_dir}/proxy_router_node; \
                   source /tmp/router_env_script.sh && \
                   nohup bash /tmp/proxy_router_launch_script.sh > \
                   {self.log_dir}/proxy_router_node/proxy_router.log 2>&1 &" '''
        formatted_cmd = textwrap_for_yml(cmd)
        self.r_phdl.exec(formatted_cmd)
        print('Waiting 120 secs after launching proxy router script')
        time.sleep(120)

    def run_gsm8k_benchmark_test(
        self,
    ):
        """
        Run the GSM8K inference benchmark against the SGLang disaggregated
        Prefill/Decode deployment and validate throughput.

        Purpose:
        --------
        This method executes a real-world inference workload (GSM8K question
        answering) to:
        - Validate end-to-end correctness of the inference pipeline
        - Measure sustained output token throughput
        - Ensure performance meets expected SLA thresholds

        The benchmark traffic is sent to the Proxy Router, which:
        - Routes requests to Prefill servers
        - Coordinates Decode servers for token generation
        """
        print('#================ * * * =========================#')
        print('Create Benchmark script')
        print('#================ * * * =========================#')

        i_dict = self.bp_dict['inference_tests']['gsm8k']
        # ------------------------------------------------------------------
        # Construct command to run GSM8K benchmark inside the container
        #
        # Key steps:
        #   - Create a directory to store benchmark logs
        #   - Navigate to the GSM8K benchmark directory
        #   - Source environment variables required for benchmark execution
        #   - Launch the benchmark using nohup to allow async execution
        #
        # Benchmark parameters:
        #   --num-questions : Total GSM8K questions to run
        #   --parallel      : Maximum concurrent inference requests
        #   --host / --port : Proxy Router endpoint for inference
        # ------------------------------------------------------------------
        cmd = f'''docker exec {self.container_name} /bin/bash -c  "
                      mkdir -p {self.log_dir}/benchmark_node; \
                      cd /sgl-workspace/sglang/benchmark/gsm8k; \
                      source /tmp/benchmark_env_script.sh && \
                      nohup python3 ./bench_sglang.py \
                      --num-questions {i_dict['num_questions']} \
                      --parallel {i_dict['max_concurrency']} \
                      --host http://0.0.0.0 --port {self.inf_dict['proxy_router_serv_port']}" '''
        formatted_cmd = textwrap_for_yml(cmd)
        out_dict = self.b_phdl.exec(formatted_cmd, timeout=800)
        time.sleep(5)
        for node in out_dict.keys():
            if not re.search('Output throughput', out_dict[node], re.I):
                fail_test(f'Benchmark test did not complete properly on node {node}, no throughput pattern seen')
            else:
                match = re.search('Output throughput:\s+([0-9\.]+)\s+token', out_dict[node], re.I)
                actual_tps = match.group(1)
                if float(actual_tps) < float(i_dict['expected_results']['tokens_per_sec']):
                    fail_test(
                        f"Test FAILED due to low performance, \
                            expected tokens per sec = {i_dict['expected_results']['tokens_per_sec']}, \
                            actual tokens per sec = {actual_tps}"
                    )

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
        print('#================ * * * =========================#')
        print('Benchmark Random Dataset')
        print('#================ * * * =========================#')
        i_dict = self.bp_dict['inference_tests']['bench_serv_random']
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
        cmd = f'''docker exec {self.container_name} /bin/bash -c  "
                      mkdir -p {self.log_dir}/benchmark_node; \
                      source /tmp/benchmark_env_script.sh && \
                      python3 -m sglang.bench_serving --backend {i_dict['backend']} \
                      --dataset-name random \
                      --num-prompts {i_dict['num_prompts']} \
                      --random-input {i_dict['input_length']} \
                      --random-output {i_dict['output_length']} \
                      --random-range-ratio {i_dict['random_range_ratio']} \
                      --host 0.0.0.0 --port {self.inf_dict['proxy_router_serv_port']} \
                      > {self.log_dir}/benchmark_node/benchmark_results.log" '''
        formatted_cmd = textwrap_for_yml(cmd)
        self.b_phdl.exec(formatted_cmd, timeout=500)
        time.sleep(5)
        self.poll_for_inference_completion(iterations=10, waittime_between_iters=60)
        self.verify_inference_results('bench_serv', i_dict['expected_results'][d_type])

    def poll_for_server_ready(self, node_no, sglang_function, no_of_iterations=11):
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
            head_node = self.prefill_node_list[0]
            for j in range(1, no_of_iterations):
                print(f'Starting poll iteration {j}')
                out_dict = self.p_phdl.exec(
                    f'grep -B 20 -A 20 "200 OK" {self.log_dir}/prefill_node{node_no}/prefill_server.log'
                )
                if re.search('GET|POST', out_dict[head_node], re.I):
                    print('Wait 60 secs to start serving traffic')
                    time.sleep(60)
                    # if re.search('fired up and ready to roll', out_dict[head_node], re.I ):
                    #    print('Prefill server {node_no} ready to serve')
                    return
                else:
                    print('Wait for 120 secs and continue polling')
                    time.sleep(120)
            head_node = self.prefill_node_list[0]
            print(f'Prefill node {node_no} did not get to ready to serve 200 OK state in {j} iterations')
            fail_test(f'Prefill node {node_no} did not get to ready to serve 200 OK state in {j} iterations')
        # ------------------------------------------------------------------
        # Decode server readiness check
        # ------------------------------------------------------------------
        elif re.search('decode', sglang_function):
            head_node = self.decode_node_list[0]
            for j in range(1, no_of_iterations):
                print(f'Starting poll iteration {j}')
                out_dict = self.d_phdl.exec(
                    f'grep -B 20 -A 20 "200 OK" {self.log_dir}/decode_node{node_no}/decode_server.log'
                )
                if re.search('GET|POST', out_dict[head_node]):
                    print('Wait 60 secs to start serving traffic')
                    time.sleep(60)
                    # if re.search('fired up and ready to roll', out_dict[head_node], re.I ):
                    #    print('Decode server {node_no} ready to serve')
                    return
                else:
                    print('Wait for 120 secs and continue polling')
                    time.sleep(120)
            print(f'Decode node {node_no} did not get to ready to serve 200 OK state in {j} iterations')
            fail_test(f'Decode node {node_no} did not get to ready to serve 200 OK state in {j} iterations')

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
        print('Inside get_inference_results_dict')
        print(out_dict)
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
                self.inference_results_dict[node]['Total generated tokens:'] = match.group(1)
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
            if re.search('Mean E2EL \(ms\):', out_dict[node], re.I):
                match = re.search('Mean E2EL \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['mean_e2el_ms'] = match.group(1)
            if re.search('Median E2EL \(ms\):', out_dict[node], re.I):
                match = re.search('Median E2EL \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['median_e2el_ms'] = match.group(1)
            if re.search('P99 E2EL \(ms\):', out_dict[node], re.I):
                match = re.search('P99 E2EL \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['p99_e2el_ms'] = match.group(1)

        print(self.inference_results_dict)
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
        print('Scan for inference errors')
        inference_pass = True

        # Build the list of commands to read each node's inference log file
        cmd_list = []

        # Scan all prefill nodes
        for j in range(0, int(self.prefill_nnodes)):
            cmd = f"sudo cat {self.log_dir}/prefill_node{j}/prefill_server.log"
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
        for j in range(0, int(self.decode_nnodes)):
            cmd = f"sudo cat {self.log_dir}/decode_node{j}/decode_server.log"
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
            print(f'Starting iteration {itr}')

            # --------------------------------------------------------------
            # Early exit if any inference errors are detected
            #
            # This scans Prefill and Decode logs for known failure patterns
            # (e.g., OOM, RDMA failures, backend crashes).
            # --------------------------------------------------------------
            # Early abort on inference errors
            if not self.scan_for_inference_errors():
                msg = 'Failures seen in inference logs, Aborting!!!'
                fail_test(msg)
                return {"status": "error", "reason": msg}

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
                    print(msg)
                    return {"status": "timeout", "reason": msg}
                print('Inference still in progress')
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
            print('Completed Inference, returning !!!')
            return {"status": "success", "results": self.inference_results_dict}

            # If we reached here, it means poll for inference completion failed

        # If we exhaust the iteration cap without completing, treat as timeout (or in_progress if no wall-clock limit)
        if timed_out():
            msg = f"Timeout after maximum iterations ({self.inference_poll_iterations}) and ~{int(time.time() - start_time)}s"
            print(msg)
            return {"status": "timeout", "reason": msg}
        else:
            # If no wall-clock timeout was set and we hit the iteration cap, report in-progress
            msg = f"Reached iteration cap ({self.inference_poll_iterations}) without completion; still in progress"
            print(msg)
            return {"status": "stuck_in_progress", "reason": msg}

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
        print('Verify Inference Completion Msg')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print(self.inference_results_dict)
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        # ------------------------------------------------------------------
        # Validate metrics on a per-node basis
        # ------------------------------------------------------------------
        for node in self.inference_results_dict.keys():
            print('%%%% node {}'.format(node))
            for metric_name in expected_result_dict.keys():
                print('%%% metric_name {}'.format(metric_name))
                if metric_name in self.inference_results_dict[node].keys():
                    # latency metric, so actual should be lower than expected ..
                    print('%% metric found in inference results ^^^')
                    # ------------------------------------------------------
                    # Latency metrics (e.g., TTFT, TPOT)
                    #
                    # For latency, lower values are better.
                    # Fail if actual latency exceeds expected threshold.
                    # ------------------------------------------------------
                    if re.search('ms', metric_name, re.I):
                        print(self.inference_results_dict[node][metric_name])
                        print(expected_result_dict[metric_name])
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
        print(self.inference_results_dict)
