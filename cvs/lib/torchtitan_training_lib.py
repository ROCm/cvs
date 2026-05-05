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


training_err_dict = {
    'NCCL ERROR': 'NCCL ERROR|NCCL timeout|ncclRemoteError: A call failed possibly due to a network error|NCCL error:',
    'GPU HW ERROR': 'HW Exception by GPU|GPU Hang|Uncorrectable error|GPU Reset',
    'torch': 'torch.distributed.elastic.multiprocessing.errors',
}

err_counters_pattern = 'err|retransmit|drop|discard|naks|invalid|oflow|out_of_buffer|reset|fail'

# Library for building TorchTitan training jobs ..


def detect_rocm_path(phdl, config_rocm_path):
    """
    Detect the ROCm installation path, supporting both old (/opt/rocm) and
    new (/opt/rocm/core-X.Y) layouts.

    Args:
        phdl: Parallel SSH handle.
        config_rocm_path (str): Configured ROCm path from config file
                                (empty string or '<changeme>' for auto-detect).

    Returns:
        str: Detected ROCm path.
    """
    if config_rocm_path and config_rocm_path != '<changeme>':
        log.info(f'Using configured ROCm path: {config_rocm_path}')
        return config_rocm_path

    log.info('Auto-detecting ROCm path...')

    # Try new ROCm layout first (/opt/rocm/core-X.Y)
    out_dict = phdl.exec('ls -d /opt/rocm/core-* 2>/dev/null | sort -V | tail -1')
    for node, output in out_dict.items():
        if output and '/opt/rocm/core-' in output:
            rocm_path = output.strip()
            validate_dict = phdl.exec(
                f'test -d {rocm_path}/lib && ls {rocm_path}/lib/libamdhip64.so* 2>/dev/null | head -1'
            )
            for _, lib_output in validate_dict.items():
                if lib_output.strip() and 'libamdhip64.so' in lib_output:
                    log.info(f'Detected ROCm path (new layout): {rocm_path}')
                    return rocm_path

    # Fall back to legacy /opt/rocm
    out_dict = phdl.exec('test -d /opt/rocm/lib && ls /opt/rocm/lib/libamdhip64.so* 2>/dev/null | head -1')
    for node, output in out_dict.items():
        if output.strip() and 'libamdhip64.so' in output:
            log.info('Detected ROCm path (legacy layout): /opt/rocm')
            return '/opt/rocm'

    log.warning('Could not detect ROCm path, defaulting to /opt/rocm')
    return '/opt/rocm'


class TorchTitanTrainingJob:
    """
    Orchestrates a TorchTitan training job across one or more nodes.
    Supports multiple model architectures: Llama, DeepSeek-V3 (MoE), Qwen.

    TorchTitan is PyTorch's native training framework with modular composability
    of parallelism techniques (FSDP2, TP, PP, CP, EP), memory optimizations, and
    PyTorch compiler integration.

    Key differences from Megatron-LM:
      - Uses TOML config files with CLI overrides instead of shell script environment variables
      - Invokes training via 'python -m torchtitan.train' instead of bash scripts
      - Different parallelism configuration (data_parallel_shard_degree instead of FSDP flag)
      - Native PyTorch integration with torch.compile and Float8 support
      - Aggregate metrics (tokens_per_sec) instead of per-GPU metrics
      - Supports MoE models with expert parallelism (DeepSeek-V3)

    Responsibilities:
      - Normalize training configuration and model parameters (with sensible defaults).
      - Build torchrun command with TorchTitan module invocation and CLI arg overrides.
      - Optionally collect pre/post network (RDMA/ethtool) stats for validation.
      - Launch the job (single-node or distributed) inside a specified container.
      - Poll logs for completion and errors; extract performance metrics from logs.
      - Verify training results against expected thresholds and system health checks.

    Assumptions:
      - phdl provides remote execution utilities across nodes
      - Docker container with TorchTitan is pre-deployed (e.g., rocm/primus:v26.2)
      - TorchTitan is installed and accessible via 'python -m torchtitan.train'
    """

    def __init__(
        self,
        phdl,
        model_name,
        training_config_dict,
        model_params_dict,
        hf_token,
        gpu_type='mi350',
        distributed_training=True,
        tune_model_params=True,
        scripts_dir=os.path.expanduser("~") + '/SCRIPTS',
    ):
        """
        Initialize TorchTitan job configuration and resolve defaults from the provided dicts.

        Args:
          phdl: Remote execution handle for multi-node command execution.
          model_name: Canonical model name key used in model_params_dict (e.g., "llama3_1_8b").
          training_config_dict: Unstructured training config; defaults are applied here.
          model_params_dict: Parameter sets per model and topology (single/multi-node).
          hf_token: Hugging Face token passed to the job environment.
          gpu_type: GPU platform key to select model params (default: 'mi350').
          distributed_training: Whether to run distributed (multi-node) or single-node training.
          tune_model_params: If True, adjust some parameters based on cluster size.
          scripts_dir: Folder on nodes to place generated wrapper scripts.
        """

        self.phdl = phdl
        self.host_list = phdl.host_list
        self.model_name = model_name
        self.hf_token = hf_token
        self.gpu_type = gpu_type

        self.training_config_dict = training_config_dict
        self.model_params_dict = model_params_dict
        self.iterations = int(training_config_dict['training_iterations'])
        self.tune_model_params = tune_model_params

        self.scripts_dir = scripts_dir

        self.job_cmd = ''
        self.job_cmd_list = []
        self.training_results_dict = {}
        log.info("%s", self.gpu_type)

        # Initialize cluster stats dicts
        self.rdma_stats_dict_before = {}
        self.ethtool_stats_dict_before = {}
        self.rdma_stats_dict_after = {}
        self.ethtool_stats_dict_after = {}
        self.training_start_time = self.phdl.exec('date')
        self.training_end_time = None

        # Training configs - set defaults if not defined in input json file
        self.home_dir = os.path.expanduser("~")
        tdict = training_config_dict
        tdict.setdefault('container_image', 'rocm/primus:v26.2')
        tdict.setdefault('container_name', 'torchtitan_llama3.1_8b')
        tdict.setdefault('training_iterations', 10)
        tdict.setdefault('nnodes', 1)
        tdict.setdefault('nic_type', 'thor2')
        tdict.setdefault('nccl_ib_hca_list', 'bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7')
        tdict.setdefault('nccl_ib_hca', 'bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7')
        tdict.setdefault('nccl_socket_ifname', 'ensf1np1')
        tdict.setdefault('gloo_socket_ifname', 'ensf1np1')
        tdict.setdefault('nccl_ib_gid_index', '3')
        tdict.setdefault('nccl_debug', 'ERROR')
        tdict.setdefault('data_cache_dir', f'{self.home_dir}/cache')
        tdict.setdefault('log_dir', f'{self.home_dir}/LOGS')
        tdict.setdefault('master_address', '127.0.0.1')
        tdict.setdefault('verify_network_errors', 'False')
        tdict.setdefault('rocm_dir', '')

        self.container_image = tdict['container_image']
        self.container_name = tdict['container_name']
        self.distributed_training = distributed_training
        self.iterations = int(tdict['training_iterations'])
        self.nnodes = int(tdict['nnodes'])
        self.nic_type = tdict['nic_type']
        self.nccl_ib_hca_list = tdict['nccl_ib_hca_list']
        self.nccl_ib_hca = tdict['nccl_ib_hca']
        self.nccl_socket_ifname = tdict['nccl_socket_ifname']
        self.gloo_socket_ifname = tdict['gloo_socket_ifname']
        self.nccl_ib_gid_index = tdict['nccl_ib_gid_index']
        self.nccl_debug = tdict['nccl_debug']
        self.data_cache_dir = tdict['data_cache_dir']
        self.log_dir = tdict['log_dir']
        self.master_address = tdict['master_address']
        self.verify_network_errors = tdict['verify_network_errors']
        self.rocm_path = detect_rocm_path(self.phdl, tdict['rocm_dir'])

        # Get the model parameters dict
        log.info('^^^^')
        log.info("%s", self.model_params_dict)
        log.info("%s", self.model_name)
        log.info("%s", self.gpu_type)
        log.info('^^^^')
        if not self.distributed_training:
            pdict = self.model_params_dict['single_node'][self.model_name][self.gpu_type]
            self.expected_result_dict = self.model_params_dict['single_node'][self.model_name][self.gpu_type][
                'result_dict'
            ]
        else:
            pdict = self.model_params_dict['multi_node'][self.model_name][self.gpu_type]
            self.expected_result_dict = self.model_params_dict['multi_node'][self.model_name][self.gpu_type][
                'result_dict'
            ]

        # Model Params - Let us set defaults if params not defined in input json file
        pdict.setdefault('tokenizer_path', 'meta-llama/Llama-3.1-70B')
        pdict.setdefault('model_size', '70b')
        pdict.setdefault('sequence_length', '8192')
        pdict.setdefault('global_batch_size', '128')
        pdict.setdefault('micro_batch_size', '2')
        pdict.setdefault('data_parallel_shard_degree', '8')
        pdict.setdefault('tensor_parallel_degree', '1')
        pdict.setdefault('pipeline_parallel_degree', '1')
        pdict.setdefault('context_parallel_degree', '1')
        pdict.setdefault('expert_parallel_degree', '1')  # For MoE models like DeepSeek-V3
        pdict.setdefault('activation_checkpointing', 'selective')
        pdict.setdefault('compile', 'false')
        pdict.setdefault('enable_float8', 'true')

        self.tokenizer_path = pdict['tokenizer_path']
        self.model_size = pdict['model_size']
        self.sequence_length = pdict['sequence_length']
        self.global_batch_size = pdict['global_batch_size']
        self.micro_batch_size = pdict['micro_batch_size']
        self.data_parallel_shard_degree = pdict['data_parallel_shard_degree']
        self.tensor_parallel_degree = pdict['tensor_parallel_degree']
        self.pipeline_parallel_degree = pdict['pipeline_parallel_degree']
        self.context_parallel_degree = pdict['context_parallel_degree']
        self.expert_parallel_degree = pdict['expert_parallel_degree']
        self.activation_checkpointing = pdict['activation_checkpointing']
        self.compile = pdict['compile']
        self.enable_float8 = pdict['enable_float8']

        # Determine TorchTitan module and config based on model name
        if re.search('llama.*[34]', self.model_name, re.I):
            self.tt_module = 'llama3'
            # Map model size to config name
            if '70' in str(self.model_size):
                self.tt_config = 'llama3_70b'
            elif '405' in str(self.model_size):
                self.tt_config = 'llama3_405b'
            else:
                self.tt_config = 'llama3_8b'
        elif re.search('deepseek', self.model_name, re.I):
            self.tt_module = 'deepseek_v3'
            # DeepSeek-V3 16B is the main supported size
            if '16' in str(self.model_size):
                self.tt_config = 'deepseek_v3_16b'
            else:
                self.tt_config = 'deepseek_v3_16b'
        elif re.search('qwen', self.model_name, re.I):
            self.tt_module = 'qwen3'
            # Qwen3 has various sizes (4B, 7B, 14B, 32B)
            if '32' in str(self.model_size):
                self.tt_config = 'qwen3_32b'
            elif '14' in str(self.model_size):
                self.tt_config = 'qwen3_14b'
            elif '7' in str(self.model_size):
                self.tt_config = 'qwen3_7b'
            else:
                self.tt_config = 'qwen3_4b'
        else:
            # Default fallback to llama
            self.tt_module = 'llama3'
            self.tt_config = 'llama3_8b'

        # Remove and recreate the scripts dir
        self.phdl.exec(f'rm -rf {self.scripts_dir}')
        time.sleep(2)
        self.phdl.exec(f'mkdir {self.scripts_dir}')
        time.sleep(2)
        self.phdl.exec(f'sudo chmod 777 {self.scripts_dir}')

        # Adjust batch size based on cluster size if tune_model_params is set
        if self.tune_model_params and self.distributed_training:
            gpus_per_node = 8
            total_gpus = self.nnodes * gpus_per_node
            # Assuming default config was for 32 GPUs (4 nodes)
            if int(self.global_batch_size) > 32:
                if int(self.global_batch_size) % 32 == 0:
                    per_gpu_batch_size = int(self.global_batch_size) / 32
                    self.global_batch_size = str(int(per_gpu_batch_size * total_gpus))

    def run_pretraining_tasks(self):
        """Collect network stats before training starts (for distributed training)."""
        if self.distributed_training is True:
            self.rdma_stats_dict_before = linux_utils.get_rdma_stats_dict(self.phdl)
            self.ethtool_stats_dict_before = linux_utils.get_nic_ethtool_stats_dict(self.phdl)

    def exec_nic_setup_scripts(self):
        """
        Prepare backend NICs inside containers before starting distributed training.
        Applies vendor-specific workarounds (e.g., Broadcom RDMA library setup).
        """
        if self.distributed_training is True:
            # Broadcom/Thor NIC workaround for container environments
            if re.search('broadcom|thor', self.nic_type, re.I):
                # Override the gid_index to 3 for Broadcom
                self.nccl_ib_gid_index = 3
                out_dict = self.phdl.exec(
                    f'docker exec {self.container_name} /bin/bash -c "sudo \
                    cp /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.host \
                    /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so; \
                    sleep 2;ibv_devinfo;sleep 2;"'
                )
                for node in out_dict.keys():
                    if not re.search('hca_id:\s+bnxt_', out_dict[node], re.I):
                        log.info("%s", out_dict[node])
                        fail_test(f'Broadcom libbnxt rdma driver is not properly copied on node {node}')

    def build_training_job_cmd(self):
        """
        Construct the TorchTitan training command using torchrun and TOML config overrides.

        TorchTitan uses:
        - torchrun for distributed launch (instead of custom launcher scripts)
        - python -m torchtitan.train (instead of shell scripts)
        - --module and --config flags to select TOML base config
        - CLI overrides using dot notation (e.g., --training.steps 100)
        """
        cmd = ''

        # Base environment setup
        cmd = (
            cmd
            + 'cd /workspace/torchtitan; '
            + f'export HF_TOKEN="{self.hf_token}"; '
            + f'export TOKENIZERS_PARALLELISM=false; '
            + f'export LD_LIBRARY_PATH=/usr/local/lib/:{self.rocm_path}/lib:$LD_LIBRARY_PATH; '
        )

        # Add network-related environment variables for distributed training
        if self.distributed_training is True:
            cmd = (
                cmd
                + f'export NCCL_IB_HCA={self.nccl_ib_hca_list}; '
                + f'export NCCL_SOCKET_IFNAME={self.nccl_socket_ifname}; '
                + f'export GLOO_SOCKET_IFNAME={self.gloo_socket_ifname}; '
                + f'export NCCL_DEBUG={self.nccl_debug}; '
                + f'export NCCL_IB_GID_INDEX={self.nccl_ib_gid_index}; '
            )

        # Calculate total number of GPUs
        gpus_per_node = 8
        total_gpus = self.nnodes * gpus_per_node

        # Build TorchTitan CLI arguments for TOML config overrides
        tt_args = (
            f'--job.description "CVS_Training_Job" '
            + f'--training.steps {self.iterations} '
            + f'--training.global_batch_size {self.global_batch_size} '
            + f'--training.micro_batch_size {self.micro_batch_size} '
            + f'--training.seq_len {self.sequence_length} '
            + f'--model.tokenizer_path {self.tokenizer_path} '
            + f'--checkpoint.folder {self.log_dir}/checkpoints '
            + f'--metrics.log_dir {self.log_dir}/metrics '
            + f'--parallelism.data_parallel_shard_degree {self.data_parallel_shard_degree} '
            + f'--parallelism.tensor_parallel_degree {self.tensor_parallel_degree} '
            + f'--parallelism.pipeline_parallel_degree {self.pipeline_parallel_degree} '
            + f'--parallelism.context_parallel_degree {self.context_parallel_degree} '
        )

        # Add expert parallelism for MoE models (DeepSeek-V3)
        if int(self.expert_parallel_degree) > 1:
            tt_args += f'--parallelism.expert_parallel_degree {self.expert_parallel_degree} '

        tt_args += (
            f'--model.activation_checkpointing.mode {self.activation_checkpointing} '
            + f'--model.compile {self.compile} '
            + f'--model.enable_float8_linear {self.enable_float8} '
        )

        if self.distributed_training:
            # Multi-node distributed training command
            # Build per-node commands with different node_rank
            for i in range(self.nnodes):
                full_cmd = (
                    cmd
                    + f'torchrun --nnodes {self.nnodes} --nproc_per_node {gpus_per_node} '
                    + f'--node_rank {i} --master_addr {self.master_address} --master_port 29500 '
                    + f'--rdzv_backend c10d --rdzv_endpoint {self.master_address}:29500 '
                    + f'-m torchtitan.train --module {self.tt_module} --config {self.tt_config} '
                    + tt_args
                    + f' > {self.log_dir}/torchtitan-logs/out-node{i}/training.log 2>&1 &'
                )
                script_cmd = f'echo "{full_cmd}" > {self.scripts_dir}/distributed_wrapper_script_{i}.sh;chmod 777 {self.scripts_dir}/distributed_wrapper_script_{i}.sh'
                self.job_cmd_list.append(script_cmd)
        else:
            # Single node training case
            self.job_cmd = (
                cmd
                + f'torchrun --nnodes 1 --nproc_per_node {gpus_per_node} '
                + f'-m torchtitan.train --module {self.tt_module} --config {self.tt_config} '
                + tt_args
                + f' > {self.log_dir}/torchtitan-logs/out-node0/training.log 2>&1 &'
            )

    def start_training_job(self, timeout=500):
        """
        Launch the TorchTitan training job (distributed or single-node).

        Behavior:
         - Creates log directories for each node
         - Distributed mode: Creates and executes per-node wrapper scripts
         - Single-node mode: Creates and executes single wrapper script
         - Sleeps briefly to allow processes to initialize
        """
        log.info('start training job')
        log.info("%s", self.job_cmd_list)
        log.info("%s", self.job_cmd)

        # Create log directories
        cmd_list = []
        for i in range(self.nnodes):
            cmd = f'''docker exec {self.container_name} /bin/bash -c "mkdir -p {self.log_dir}/torchtitan-logs/out-node{i}"'''
            cmd_list.append(cmd)
        self.phdl.exec_cmd_list(cmd_list)

        if self.distributed_training:
            # Run NIC setup if needed
            self.exec_nic_setup_scripts()
            # Create the training scripts
            self.phdl.exec_cmd_list(self.job_cmd_list)

            # Execute the scripts inside containers
            cmd_list = []
            for i in range(self.nnodes):
                cmd = f'docker exec {self.container_name} /bin/bash {self.scripts_dir}/distributed_wrapper_script_{i}.sh'
                cmd_list.append(cmd)
            self.phdl.exec_cmd_list(cmd_list)
        else:
            # Single node execution
            self.phdl.exec(
                f'echo "{self.job_cmd}" > {self.scripts_dir}/single_node_wrapper_script.sh; '
                f'chmod 777 {self.scripts_dir}/single_node_wrapper_script.sh'
            )
            cmd_list = []
            for i in range(self.nnodes):
                cmd = f'docker exec {self.container_name} /bin/bash {self.scripts_dir}/single_node_wrapper_script.sh'
                cmd_list.append(cmd)
            self.phdl.exec_cmd_list(cmd_list)

        time.sleep(50)

    def get_training_results_dict(self):
        """
        Parse TorchTitan training log output and extract key performance metrics.

        TorchTitan log format differs from Megatron:
        - Logs: "step: X, loss: Y, tok/s: Z, mem: W GB"
        - No separate "throughput per GPU" or "tokens/GPU/s" patterns
        - Uses "tok/s" for tokens per second (aggregate across all GPUs)

        Returns:
        dict: A dictionary with lists of extracted values for each metric:
        - 'tokens_per_sec': Matches 'tok/s: <float>'
        - 'loss': Matches 'loss: <float>'
        - 'mem_usage_gb': Matches 'mem: <float> GB'
        """
        training_results_dict = {}

        # Read the training log from the last node
        last_node = self.host_list[-1]
        last_node_num = len(self.host_list) - 1
        out_dict = self.phdl.exec(f'cat {self.log_dir}/torchtitan-logs/out-node{last_node_num}/training.log | tail -20')

        output = out_dict[last_node]

        log.info('Extracting results from logs')
        log.info('#===========================#')
        log.info("%s", output)
        log.info('#===========================#')

        # Extract tokens per second (tok/s) - aggregate metric
        pattern = r'tok/s:\s+([0-9\.]+)'
        training_results_dict['tokens_per_sec'] = re.findall(pattern, output, re.I)

        # Extract loss values
        pattern = r'loss:\s+([0-9\.]+)'
        training_results_dict['loss'] = re.findall(pattern, output, re.I)

        # Extract memory usage (in GB)
        pattern = r'mem:\s+([0-9\.]+)\s+GB'
        training_results_dict['mem_usage_gb'] = re.findall(pattern, output, re.I)

        log.info("%s", training_results_dict)
        return training_results_dict

    def scan_for_training_errors(self):
        """
        Scan the training logs for known error patterns.

        Returns:
        bool: True if no error patterns are found; False otherwise.
        """
        log.info('Scan for training errors')
        training_pass = True

        last_node = self.host_list[-1]
        last_node_num = len(self.host_list) - 1

        out_dict = self.phdl.exec(f'sudo cat {self.log_dir}/torchtitan-logs/out-node{last_node_num}/training.log')
        output = out_dict[last_node]

        # Check against all known training error patterns
        for err_key in training_err_dict:
            if re.search(f'{training_err_dict[err_key]}', output):
                fail_test(f'ERROR {training_err_dict[err_key]} seen in training logs ..')
                log.error('Aborting training log polling')
                training_pass = False
        return training_pass

    def poll_for_training_completion(self, time_between_iters=120):
        """
        Periodically poll training logs to detect completion, surface errors, and validate results.

        TorchTitan completion detection:
        - Looks for "step: {iterations}" pattern indicating final step reached
        - Checks for tok/s metrics to confirm training is producing results

        Args:
        time_between_iters (int | float): Seconds to sleep between each polling iteration.
        """
        log.info('Poll for training completion ..')
        time.sleep(80)

        last_node = self.host_list[-1]
        last_node_num = len(self.host_list) - 1

        # Poll for up to iterations + 10 extra cycles to account for slower iterations
        for i in range(1, int(self.iterations) + 10):
            log.info(f'Starting Iteration {i}')

            # Check for errors first
            if not self.scan_for_training_errors():
                fail_test('Failures seen in training logs, Aborting!!!')
                return

            out_dict = self.phdl.exec(f'sudo cat {self.log_dir}/torchtitan-logs/out-node{last_node_num}/training.log')
            output = out_dict[last_node]

            # Check if training has reached the final step
            # TorchTitan logs format: "step: X, loss: Y, tok/s: Z, mem: W GB"
            final_step_pattern = f'step:\\s+{self.iterations}'

            if not re.search(final_step_pattern, output, re.I):
                log.info('Training still in progress - final step not yet reached')
            else:
                # Training completed - check for valid metrics
                if re.search(r'tok/s:\s+[NaN|Inf]', output, re.I) or re.search(r'loss:\s+[NaN|Inf]', output, re.I):
                    fail_test(f'ERROR - NaN or Inf values seen in training results {output}')
                    return
                else:
                    time.sleep(5)
                    self.training_results_dict = self.get_training_results_dict()
                    log.info('Completed Training, returning !!!')
                    return

            # Wait between iterations
            time.sleep(int(time_between_iters))

    def verify_training_results(self):
        """
        Validate collected training results and environment health after a training run.

        Checks:
        - Parsed training_results_dict for NaN/Inf values
        - Network stats (RDMA, ethtool) if distributed training
        - Kernel logs (dmesg) for errors
        - Performance results against expected thresholds
        """
        # Record training end time
        self.training_end_time = self.phdl.exec('date')

        log.info('#==================================================#')
        log.info('\t\tTraining Results')
        log.info("%s", self.training_results_dict)
        log.info('#==================================================#')

        # Check if results were populated
        if not self.training_results_dict:
            fail_test(
                'Failed to populate training results, training_results_dict is empty - please check logs for failures'
            )

        # Check for NaN/Inf in results
        for result_key in self.training_results_dict.keys():
            for result_val in self.training_results_dict[result_key]:
                if re.search('nan|inf', str(result_val), re.I):
                    fail_test(
                        f'Failures seen in training_result dict for {result_key}, numbers are either NaN or Inf - {result_val}'
                    )

        # Check network stats if distributed training
        if self.distributed_training is True and self.verify_network_errors == 'True':
            self.rdma_stats_dict_after = linux_utils.get_rdma_stats_dict(self.phdl)
            self.ethtool_stats_dict_after = linux_utils.get_nic_ethtool_stats_dict(self.phdl)

            # Compare RDMA error counters
            for node in self.rdma_stats_dict_after.keys():
                for counter_name in self.rdma_stats_dict_after[node]:
                    if re.search(f'{err_counters_pattern}', counter_name, re.I):
                        if int(self.rdma_stats_dict_after[node][counter_name]) > int(
                            self.rdma_stats_dict_before[node][counter_name]
                        ):
                            fail_test(
                                f'Error counter {counter_name} has gone up after training on node {node} '
                                f'Before = {self.rdma_stats_dict_before[node][counter_name]}, '
                                f'After = {self.rdma_stats_dict_after[node][counter_name]}'
                            )

            # Compare NIC error counters
            for node in self.ethtool_stats_dict_after.keys():
                for counter_name in self.ethtool_stats_dict_after[node]:
                    if re.search(f'{err_counters_pattern}', counter_name, re.I):
                        if int(self.ethtool_stats_dict_after[node][counter_name]) > int(
                            self.ethtool_stats_dict_before[node][counter_name]
                        ):
                            fail_test(
                                f'Error counter {counter_name} has gone up after training on node {node} '
                                f'Before = {self.ethtool_stats_dict_before[node][counter_name]}, '
                                f'After = {self.ethtool_stats_dict_after[node][counter_name]}'
                            )

        # Scan dmesg for errors
        verify_dmesg_for_errors(self.phdl, self.training_start_time, self.training_end_time)

        log.info('^^^^^^^^^^^^^^^^^^^^')
        log.info('training_results_dict')
        log.info('^^^^^^^^^^^^^^^^^^^^')
        log.info("%s", self.training_results_dict)

        # Compare performance against expected results
        for result_key in self.training_results_dict.keys():
            if result_key in self.expected_result_dict:
                log.info("%s", self.training_results_dict[result_key])
                # Check if all metrics meet expected performance
                for actual_perf in self.training_results_dict[result_key]:
                    if float(actual_perf) < float(self.expected_result_dict[result_key]):
                        fail_test(
                            f'The Training performance numbers are below expected numbers for '
                            f'{result_key}, expected = {self.expected_result_dict[result_key]}, '
                            f'actual = {actual_perf}'
                        )
            else:
                log.warning(f'Perf result key {result_key} not provided in input JSON file, so will not be checked')


# Backward compatibility alias for existing Llama test files
TorchTitanLlamaTrainingJob = TorchTitanTrainingJob
