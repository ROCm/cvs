'''
Copyright 2025 Advanced Micro Devices, Inc.
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
from cvs.lib import linux_utils


log = globals.log

inference_err_dict = {
    'NCCL ERROR': 'NCCL ERROR|NCCL timeout|local work queue catastrophic error',
    'GPU HW ERROR': 'HW Exception by GPU|GPU Hang|Uncorrectable error|GPU Reset',
    'AssertionError': 'AssertionError|ValueError:|During handling of the above exception|triggered the following exception',
    'rocm Err': 'FAILED_PRECONDITION: No visible GPU devices|failed call to hipInit: HIP_ERROR_NoDevice|librocm reported version is: NOT_FOUND',
    'python err': 'ModuleNotFoundError: No module named|Fatal Python error:',
    'resource': 'RESOURCE_EXHAUSTED: Out of memory|failed: RESOURCE_EXHAUSTED',
}

err_counters_pattern = 'err|retransmit|drop|discard|naks|invalid|oflow|out_of_buffer|reset|fail'


def textwrap_for_yml(msg_string):
    return '\n'.join([m.lstrip() for m in msg_string.split('\n')])


class InferenceBaseJob:
    """Base class for inference jobs supporting multiple frameworks."""

    def __init__(
        self,
        c_phdl,
        s_phdl,
        model_name,
        inference_config_dict,
        benchmark_params_dict,
        hf_token,
        gpu_type='mi300',
        distributed_inference=False,
        server_launch_poll_count=20,
    ):
        # Client instance phdl
        self.c_phdl = c_phdl
        # Server instance phdl
        self.s_phdl = s_phdl

        self.c_host_list = c_phdl.host_list
        self.s_host_list = s_phdl.host_list

        self.model_name = model_name
        self.hf_token = hf_token
        self.gpu_type = gpu_type
        self.distributed_inference = distributed_inference

        # Sample inference config and model params dict saved above
        self.if_dict = inference_config_dict
        self.benchmark_params_dict = benchmark_params_dict

        self.job_cmd = ''
        self.job_cmd_list = []
        self.inference_result_dict = {}
        print(self.gpu_type)

        # Needed only in the case of distributed inference - placeholder for future
        # Intialize cluster stats dicts ..
        self.rdma_stats_dict_before = {}
        self.ethtool_stats_dict_before = {}
        self.rdma_stats_dict_after = {}
        self.inference_start_time = s_phdl.exec('date +"%a %b %e %H:%M"')
        self.inference_end_time = None

        self.home_dir = os.path.expanduser("~")
        self.if_dict.setdefault('container_image', 'rocm/7.0:rocm7.0_ubuntu_22.04_vllm_0.10.1_instinct_20250927_rc1')
        self.if_dict.setdefault('container_name', 'inference_max_container')
        self.if_dict.setdefault('distributed_inference', False)
        self.if_dict.setdefault('nnodes', 1)
        self.if_dict.setdefault('nic_type', 'thor2')
        self.if_dict.setdefault('nccl_ib_hca_list', 'rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7')
        self.if_dict.setdefault('nccl_ib_hca', 'rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7')
        self.if_dict.setdefault('nccl_socket_ifname', 'ens51f1np1')
        self.if_dict.setdefault('gloo_socket_ifname', 'ens51f1np1')
        self.if_dict.setdefault('nccl_ib_gid_index', '3')
        self.if_dict.setdefault('nccl_debug', 'ERROR')
        self.if_dict.setdefault('data_cache_dir', f'{self.home_dir}/cache')
        self.if_dict.setdefault('log_dir', f'{self.home_dir}/LOG_DIR')
        self.if_dict.setdefault('benchmark_script_repo', 'https://github.com/kimbochen/bench_serving.git')

        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print(f'inference_dict = {self.if_dict}')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        # Get model-specific config - check if nested under single_node/multi_node or flat
        if self.distributed_inference:
            # Multi-node structure: benchmark_params['multi_node'][model_name] or benchmark_params[model_name]
            if 'multi_node' in self.benchmark_params_dict:
                self.bp_dict = self.benchmark_params_dict['multi_node'][self.model_name]
            else:
                self.bp_dict = self.benchmark_params_dict[self.model_name]
        else:
            # Single-node structure: benchmark_params['single_node'][model_name] or benchmark_params[model_name]
            if 'single_node' in self.benchmark_params_dict:
                self.bp_dict = self.benchmark_params_dict['single_node'][self.model_name]
            else:
                self.bp_dict = self.benchmark_params_dict[self.model_name]

        # Container image can be model-specific (in bp_dict) or global (in if_dict)
        self.container_image = self.bp_dict.get('container_image', self.if_dict['container_image'])
        self.container_name = self.if_dict['container_name']

        self.nnodes = self.if_dict['nnodes']
        self.nic_type = self.if_dict['nic_type']
        self.nccl_ib_hca_list = self.if_dict['nccl_ib_hca_list']
        self.nccl_ib_hca = self.if_dict['nccl_ib_hca']
        self.nccl_socket_ifname = self.if_dict['nccl_socket_ifname']
        self.gloo_socket_ifname = self.if_dict['gloo_socket_ifname']
        self.nccl_ib_gid_index = self.if_dict['nccl_ib_gid_index']
        self.nccl_debug = self.if_dict['nccl_debug']
        self.data_cache_dir = self.if_dict['data_cache_dir']
        self.log_dir = self.if_dict['log_dir']

        # Allow derived classes to override server launch wait duration
        self.default_server_precheck_wait_time = 30
        self.default_server_wait_time = 330
        self.default_server_poll_wait_time = 60
        self.default_server_poll_count = server_launch_poll_count
        self.default_server_precheck_error_pattern = re.compile(
            'no such file or directory|command not found|cannot access|permission denied|error:|exception:|traceback|failed to start',
            re.I,
        )
        self.default_server_error_pattern_poll = re.compile(
            'failed to start|no such file or directory|command not found|cannot access', re.I
        )
        self.default_client_wait_time = 120
        self.default_client_poll_count = 20
        self.default_client_poll_wait_time = 60

        # Regex/parse defaults that derived classes may override
        self.readiness_pattern = re.compile('Application startup complete|Uvicorn running|Started server', re.I)

        # set defaults for benchmark param dict if not passed via JSON file
        self.bp_dict.setdefault('backend', 'vllm')
        self.bp_dict.setdefault('base_url', 'http://0.0.0.0')
        self.bp_dict.setdefault('dataset_name', 'sharegpt')
        self.bp_dict.setdefault('max_concurrency', '64')
        self.bp_dict.setdefault('model', 'openai/gpt-oss-120b')
        self.bp_dict.setdefault('num_prompts', '1000')
        self.bp_dict.setdefault('input_sequence_length', '8192')
        self.bp_dict.setdefault('output_sequence_length', '1024')
        self.bp_dict.setdefault('burstiness', '1.0')
        self.bp_dict.setdefault('seed', '0')
        self.bp_dict.setdefault('request_rate', 'inf')
        self.bp_dict.setdefault('max_model_length', '9216')
        self.bp_dict.setdefault('random_range_ration', '1.0')
        self.bp_dict.setdefault('random_prefix_len', '0')
        self.bp_dict.setdefault('tensor_parallelism', '1')
        self.bp_dict.setdefault('port_no', '8000')
        self.bp_dict.setdefault('tokenizer_mode', 'auto')
        self.bp_dict.setdefault('percentile_metrics', 'ttft,tpot,itl,e2el')
        self.bp_dict.setdefault('metric_percentiles', '99')

        # Set server and client scripts
        self.server_script = self.bp_dict['server_script']
        self.bench_serv_script = self.bp_dict['bench_serv_script']

        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print(f'benchmark_params_dict = {self.bp_dict}')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    # Framework-specific methods to be implemented by derived classes
    def get_server_script_directory(self):
        """Get directory where server scripts are located."""
        raise NotImplementedError("Derived class must implement get_server_script_directory()")

    def get_result_filename(self):
        """Get result filename for benchmark output."""
        raise NotImplementedError("Derived class must implement get_result_filename()")

    def get_completion_pattern(self):
        """Get regex pattern to detect benchmark completion."""
        raise NotImplementedError("Derived class must implement get_completion_pattern()")

    def get_log_subdir(self):
        """Get log subdirectory name for this framework."""
        raise NotImplementedError("Derived class must implement get_log_subdir()")

    def run_preinference_tasks(
        self,
    ):
        if self.distributed_inference is True:
            self.rdma_stats_dict_before = linux_utils.get_rdma_stats_dict(self.s_phdl)
            self.ethtool_stats_dict_before = linux_utils.get_nic_ethtool_stats_dict(self.s_phdl)

    def launch_docker_container(self, container_name, image, device_list, volume_dict, env_dict):
        if self.distributed_inference is True:
            docker_lib.launch_docker_container(self.s_phdl, container_name, image, device_list, volume_dict, env_dict)
        else:
            env_dict['NNODES'] = 1
            docker_lib.launch_docker_container(self.s_phdl, container_name, image, device_list, volume_dict, env_dict)

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

        # Run all your backend NIC related bringups for containers here ..
        if self.distributed_inference is True:
            # This is a temporary hack needed for broadcom nics to work within containers ..
            if re.search('broadcom|thor', self.nic_type, re.I):
                # override the gid_index to 3 for broadcom
                self.nccl_ib_gid_index = 3
                out_dict = self.s_phdl.exec(
                    f'docker exec {self.container_name} /bin/bash -c "sudo \
                    cp /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.host \
                    /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so; \
                    sleep 2;ibv_devinfo;sleep 2;"'
                )
                for node in out_dict.keys():
                    if not re.search('hca_id:\s+bnxt_', out_dict[node], re.I):
                        print(out_dict[node])
                        fail_test(f'Broadcom libbnxt rdma driver is not properly copied on node {node}')

    def build_server_inference_job_cmd(
        self,
    ):
        s_cmd = f'''docker exec {self.container_name} /bin/bash -c "echo '
                    export MODEL={self.bp_dict['model']}
                    export ISL={self.bp_dict['input_sequence_length']}
                    export OSL={self.bp_dict['output_sequence_length']}
                    export MAX_MODEL_LEN={self.bp_dict['max_model_length']}
                    export RANDOM_RANGE_RATIO={self.bp_dict['random_range_ratio']}
                    export TP={self.bp_dict['tensor_parallelism']}
                    export CONC={self.bp_dict['max_concurrency']}
                    export HF_TOKEN={self.hf_token}
                    export VLLM_USE_AITER_UNIFIED_ATTENTION=1
                    export VLLM_ROCM_USE_AITER_MHA=0
                    export VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4=1
                    export PORT={self.bp_dict['port_no']}'  > /tmp/server_env_script.sh"
                    '''
        time.sleep(3)
        formatted_cmd = textwrap_for_yml(s_cmd)

        self.s_phdl.exec(formatted_cmd)

        if self.distributed_inference:
            cmd_list = []
            for i in range(0, int(self.nnodes)):
                cmd = f'''docker exec {self.container_name} /bin/bash -c  "echo  '
                      export NNODES=1
                      export NODE_RANK=0
                      export NCCL_DEBUG={self.if_dict['nccl_debug']}
                      export NCCL_IB_DISABLE=1
                      export NCCL_SHM_DISABLE=0
                      export NCCL_P2P_DISABLE=0
                      export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
                      export NCCL_DEBUG={self.if_dict['nccl_debug']}
                      export NCCL_IB_HCA={self.if_dict['nccl_ib_hca']}
                      export NCCL_IB_GID_INDEX={self.if_dict['nccl_ib_gid_index']}
                      export HSA_FORCE_FINE_GRAIN_PCIE=1
                      export NCCL_SOCKET_IFNAME={self.if_dict['nccl_socket_ifname']}
                      export GLOO_SOCKET_IFNAME={self.if_dict['gloo_socket_ifname']}
                      export PORT={self.port_no}'  > /tmp/server_env_script.sh"
                    '''
                formatted_cmd = textwrap_for_yml(cmd)
                cmd_list.append(formatted_cmd)
            print(cmd_list)
            self.s_phdl.exec_cmd_list(cmd_list)

        cmd_list = []
        for i in range(0, int(self.nnodes)):
            cmd = f'''docker exec {self.container_name} /bin/bash -c "mkdir -p {self.log_dir}/{self.get_log_subdir()}/out-node{i}" '''
            cmd_list.append(cmd)
        self.s_phdl.exec_cmd_list(cmd_list)

    def clone_bench_serving_repo(self, clone_dir):
        """Clone bench_serving repository for client benchmarks."""
        cmd = f'''docker exec {self.container_name} /bin/bash -c "cd {clone_dir}; git clone {self.if_dict['benchmark_script_repo']}" '''
        out_dict = self.c_phdl.exec(cmd)
        for node in out_dict.keys():
            # Ignore "already exists" error - repo was cloned in previous test
            if re.search('error|fail', out_dict[node], re.I) and not re.search('already exists', out_dict[node], re.I):
                fail_test('Errors or failures seen in pulling bench_serving repo from Github, pls check')
        time.sleep(3)

    def launch_server(self):
        """Launch inference server."""
        script_dir = self.get_server_script_directory()
        script_name = self.server_script
        log_file = f'{self.server_script}_server.log'

        # Start the server side inference job
        cmd_list = []
        for i in range(0, int(self.nnodes)):
            cmd = f'''docker exec {self.container_name} /bin/bash -c "cd {script_dir}; source /tmp/server_env_script.sh; nohup /bin/bash {script_name} > {self.log_dir}/{self.get_log_subdir()}/out-node{i}/{log_file} 2>&1 &" '''
            cmd_list.append(cmd)
        out_dict = self.s_phdl.exec_cmd_list(cmd_list)

        # Check for immediate failures in server launch
        for node in out_dict.keys():
            if re.search(
                'No such file or directory|command not found|cannot access|Permission denied', out_dict[node], re.I
            ):
                log.error(f'FAIL - Failed to start server on node {node}: {out_dict[node]}')
                raise Exception(f'Failed to start server on node {node}: {out_dict[node]}')

    def check_server_status(self, log_file, log_subdir, error_pattern):
        """Tail recent server logs, detect launch failures, and return the output dict."""
        cmd_list = []
        for i in range(0, int(self.nnodes)):
            cmd = f'tail -30 {self.log_dir}/{log_subdir}/out-node{i}/{log_file}'
            cmd_list.append(cmd)
        out_dict = self.s_phdl.exec_cmd_list(cmd_list)

        for node, output in out_dict.items():
            if error_pattern.search(output or ''):
                error_msg = f'Failed to start server on node {node}: {output[-500:]}'
                fail_test(error_msg)
                raise Exception(error_msg)

        return out_dict

    def is_server_ready(self, out_dict, readiness_pattern):
        """Return True if all nodes show the readiness marker."""
        if not out_dict:
            return False

        node_ready = {node: bool(readiness_pattern.search(output or '')) for node, output in out_dict.items()}
        return bool(node_ready) and all(node_ready.values())

    def poll_server_startup(self):
        """Poll for server startup completion."""
        log_file = f'{self.server_script}_server.log'
        readiness_pattern = self.readiness_pattern

        # Do an early check for fast failures before the long wait
        print(f'Waiting {self.default_server_precheck_wait_time} secs for server to start writing logs...')
        time.sleep(self.default_server_precheck_wait_time)

        # Early failure detection
        cmd_list = []
        for i in range(0, int(self.nnodes)):
            cmd = f'tail -30 {self.log_dir}/{self.get_log_subdir()}/out-node{i}/{log_file}'
            cmd_list.append(cmd)
        out_dict = self.s_phdl.exec_cmd_list(cmd_list)
        for node in out_dict.keys():
            log_content = out_dict[node].lower()
            if self.default_server_precheck_error_pattern.search(log_content):
                error_msg = f'Failed to start server on node {node}: {out_dict[node][-500:]}'
                fail_test(error_msg)
                raise Exception(error_msg)

        print(
            f'No immediate errors detected. Waiting {self.default_server_wait_time} more secs for server to fully launch...'
        )
        time.sleep(self.default_server_wait_time)

        for j in range(0, self.default_server_poll_count):
            print(f'Polling for application startup complete on all nodes, iteration {j}')
            cmd_list = []
            for i in range(0, int(self.nnodes)):
                cmd = f'tail -30 {self.log_dir}/{self.get_log_subdir()}/out-node{i}/{log_file}'
                cmd_list.append(cmd)
            out_dict = self.s_phdl.exec_cmd_list(cmd_list)

            for node in out_dict.keys():
                if self.default_server_error_pattern_poll.search(out_dict[node] or ''):
                    error_msg = f'Failed to start server on node {node}: {out_dict[node][-500:]}'
                    fail_test(error_msg)
                    raise Exception(error_msg)

            if self.is_server_ready(out_dict, readiness_pattern):
                print('Server startup confirmed on all nodes')
                return

            print(f'Waiting {self.default_server_poll_wait_time} secs for next poll')
            time.sleep(self.default_server_poll_wait_time)

        error_msg = 'Server did not report readiness before timeout; aborting startup'
        fail_test(error_msg)
        raise Exception(error_msg)

    def launch_client(self):
        """Launch client benchmark."""
        clone_dir = '/app'
        backend = self.bp_dict['backend']
        result_filename = self.get_result_filename()

        # Launch client benchmark
        cmd_list = []
        for i in range(0, int(self.nnodes)):
            client_cmd = f'''source /tmp/server_env_script.sh; cd {clone_dir}; \
                    python3 bench_serving/{self.bench_serv_script} \
                    --model {self.bp_dict['model']} \
                    --backend {backend} \
                    --base-url {self.bp_dict['base_url']}:{self.bp_dict['port_no']} \
                    --dataset-name {self.bp_dict['dataset_name']} \
                    --num-prompts {self.bp_dict['num_prompts']} \
                    --random-input-len {self.bp_dict['input_sequence_length']} \
                    --random-output-len {self.bp_dict['output_sequence_length']} \
                    --max-concurrency {self.bp_dict['max_concurrency']} \
                    --request-rate {self.bp_dict['request_rate']} \
                    --burstiness {self.bp_dict['burstiness']} \
                    --tokenizer-mode {self.bp_dict['tokenizer_mode']} \
                    --seed {self.bp_dict['seed']} \
                    --random-range-ratio {self.bp_dict['random_range_ratio']} \
                    --random-prefix-len {self.bp_dict['random_prefix_len']} \
                    --percentile-metrics {self.bp_dict['percentile_metrics']} \
                    --ignore-eos \
                    --save-result \
                    --result-dir {self.log_dir}/{self.get_log_subdir()}/out-node{i} \
                    --result-filename {result_filename} \
                    > {self.log_dir}/{self.get_log_subdir()}/out-node{i}/bench_serv_script.log 2>&1 &'''
            cmd = f'''docker exec {self.container_name} /bin/bash -c "{client_cmd}" '''
            cmd_list.append(cmd)
        self.c_phdl.exec_cmd_list(cmd_list)

    def poll_client_completion(self):
        """Poll for client benchmark completion."""
        print(f'Waiting for {self.default_client_wait_time} secs for benchmark scripts to start')
        time.sleep(self.default_client_wait_time)
        for j in range(0, self.default_client_poll_count):
            print(f'Polling for Benchmark script to complete on all nodes, iteration {j}')
            cmd_list = []
            for i in range(0, int(self.nnodes)):
                cmd = f'tail -30 {self.log_dir}/{self.get_log_subdir()}/out-node{i}/bench_serv_script.log'
                cmd_list.append(cmd)
            out_dict = self.c_phdl.exec_cmd_list(cmd_list)
            for node in out_dict.keys():
                if re.search('Failed', out_dict[node], re.I):
                    fail_test(f'Failed to run benchmark script on node {node}')
                    return
                if not re.search('End-to-end Latency', out_dict[node], re.I):
                    print(f'Waiting {self.default_client_poll_wait_time} secs for next poll')
                    time.sleep(self.default_client_poll_wait_time)

    def start_inference_server_job(
        self,
    ):
        """Start inference server - launch and poll for startup."""
        print('Start Server side Inference on all Nodes')
        self.launch_server()
        self.poll_server_startup()

    def start_inference_client_job(
        self,
    ):
        print('Start Client side benchmark script on all Nodes')

        # Clone bench_serving repo to /app
        self.clone_bench_serving_repo('/app')

        if self.distributed_inference:
            print('Distributed inference - TBD')
            return

        # Launch client and poll for completion
        self.launch_client()
        self.poll_client_completion()

    def get_inference_results_dict(self, out_dict):
        self.inference_results_dict = {}
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
        print('Scan for inference errors')
        inference_pass = True

        # Build the list of commands to read each node's inference log file
        cmd_list = []

        # Execute the commands across nodes; returns a mapping of node -> command output
        for j in range(0, int(self.nnodes)):
            log_file = f'{self.server_script}_server.log'
            cmd = f"sudo cat {self.log_dir}/{self.get_log_subdir()}/out-node{j}/{log_file}"
            cmd_list.append(cmd)
        out_dict = self.s_phdl.exec_cmd_list(cmd_list)

        # Check the log content against all known inference error patterns
        for node in out_dict.keys():
            for err_key in inference_err_dict:
                if re.search(f"{inference_err_dict[err_key]}", out_dict[node]):
                    fail_test(f"ERROR {inference_err_dict[err_key]} seen in inference logs ...")
                    log.error('Aborting inference log polling')
                    inference_pass = False
        return inference_pass

    def poll_for_inference_completion(self, waittime_between_iters=60, total_timeout=3600, require_all_nodes=True):
        # Initial wait to give inference time to start logging
        time.sleep(60)

        num_prompts = self.bp_dict['num_prompts']
        # Assume 1000 prompts completes in 120 secs ..
        iterations = int(float(num_prompts) / 60)
        self.inference_poll_iterations = iterations
        completion_pattern = self.get_completion_pattern()

        # Track wall-clock timeout if specified
        start_time = time.time()

        def timed_out() -> bool:
            return total_timeout is not None and (time.time() - start_time) >= float(total_timeout)

        for itr in range(1, iterations + 1):
            print(f'Starting iteration {itr}')

            # Early abort on inference errors
            if not self.scan_for_inference_errors():
                msg = 'Failures seen in inference logs, Aborting!!!'
                fail_test(msg)
                return {"status": "error", "reason": msg}

            # Build commands to tail recent lines from each node's inference log and capture stderr as well
            cmd_list = []
            for j in range(0, int(self.nnodes)):
                cmd = f"sudo tail -2000 {self.log_dir}/{self.get_log_subdir()}/out-node{j}/bench_serv_script.log"
                cmd_list.append(cmd)

            out_dict = self.c_phdl.exec_cmd_list(cmd_list)

            # Determine completion across nodes
            node_completion = {}
            for node, output in out_dict.items():
                node_completion[node] = bool(completion_pattern.search(output))

            if require_all_nodes:
                all_complete = all(node_completion.values()) if node_completion else False
            else:
                all_complete = any(node_completion.values()) if node_completion else False

            # If not yet complete, wait and continue (subject to timeout)
            if not all_complete:
                if timed_out():
                    msg = f"Timeout while waiting for inference completion after ~{int(time.time() - start_time)}s"
                    print(msg)
                    return {"status": "timeout", "reason": msg}
                print('Training still in progress')
                # Short progress wait before the longer inter-iteration sleep
                time.sleep(30)
                time.sleep(int(waittime_between_iters))
                continue

            # Parse/store final results and report success
            self.get_inference_results_dict(out_dict)
            print('Completed Inference, returning !!!')
            return {"status": "success", "results": self.inference_result_dict}

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

    def verify_inference_results(
        self,
    ):
        print('Verify Inference Completion Msg')
        for node in self.inference_result_dict.keys():
            for metric_name in self.expected_result_dict[node].keys():
                if metric_name in expected_result_dict:
                    # latency metric, so actual should be lower than expected ..
                    if re.search('ms', metric_name, re.I):
                        if float(self.inference_result_dict[node][metric_name]) > float(
                            self.expected_result_dict[metric_name]
                        ):
                            fail_test(
                                f"FAIL - The latency metric {metric_name} actual value higher than expected \
                                Actual = {self.inference_result_dict[node][metric_name]},  \
                                Expected = {self.expected_result_dict[metric_name]}"
                            )
                    else:
                        if float(self.inference_result_dict[node][metric_name]) < float(
                            self.expected_result_dict[metric_name]
                        ):
                            fail_test(
                                f"FAIL - The latency metric {metric_name} actual value lower than expected \
                                Actual = {self.inference_result_dict[node][metric_name]}, \
                                Expected = {self.expected_result_dict[metric_name]}"
                            )

        # Scan Dmesg for errors ..
        self.inference_end_time = self.s_phdl.exec('date +"%a %b %e %H:%M"')
        time.sleep(2)
        verify_dmesg_for_errors(self.s_phdl, self.inference_start_time, self.inference_end_time)
        print(self.inference_result_dict)
