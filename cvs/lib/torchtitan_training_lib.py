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
    Orchestrates a TorchTitan training job via AMD's Primus framework.
    Supports multiple model architectures: Llama, DeepSeek-V3 (MoE), Qwen.

    TorchTitan is PyTorch's native training framework integrated into AMD's Primus
    ecosystem with modular composability of parallelism techniques (FSDP2, TP, PP, CP, EP),
    ROCm optimizations, and PyTorch compiler integration.

    Key differences from Megatron-LM:
      - Uses torchrun with TOML configuration files (not CLI arguments)
      - Working directory: /workspace/torchtitan
      - Different parallelism configuration (data_parallel_shard_degree instead of FSDP flag)
      - Native PyTorch integration with torch.compile and Float8 support
      - Aggregate metrics (tokens_per_sec) instead of per-GPU metrics
      - Supports MoE models with expert parallelism (DeepSeek-V3)
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
        scripts_dir=None,
    ):
        self.phdl = phdl
        self.host_list = phdl.host_list
        self.model_name = model_name
        self.hf_token = hf_token
        self.gpu_type = gpu_type
        self.home_dir = os.path.expanduser("~")
        self.training_config_dict = training_config_dict
        self.model_params_dict = model_params_dict
        self.tune_model_params = tune_model_params

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
        tdict = training_config_dict
        tdict.setdefault('log_dir', f'{self.home_dir}/LOGS')
        tdict.setdefault('scripts_dir', f'{self.home_dir}/SCRIPTS')
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
        self.scripts_dir = scripts_dir or tdict['scripts_dir']
        self.master_address = tdict['master_address']
        self.verify_network_errors = tdict['verify_network_errors']
        self.rocm_path = detect_rocm_path(self.phdl, tdict['rocm_dir'])

        log.info('^^^^')
        log.info("%s", self.model_params_dict)
        log.info("%s", self.model_name)
        log.info("%s", self.gpu_type)
        log.info('^^^^')

        if not self.distributed_training:
            pdict = self.model_params_dict['single_node'][self.model_name][self.gpu_type]
            self.expected_result_dict = self.model_params_dict['single_node'][self.model_name][self.gpu_type]['result_dict']
        else:
            pdict = self.model_params_dict['multi_node'][self.model_name][self.gpu_type]
            self.expected_result_dict = self.model_params_dict['multi_node'][self.model_name][self.gpu_type]['result_dict']

        # Model params - set defaults if not defined in input json file
        pdict.setdefault('tokenizer_path', 'meta-llama/Llama-3.1-70B')
        pdict.setdefault('model_size', '70b')
        pdict.setdefault('sequence_length', '8192')
        pdict.setdefault('global_batch_size', '128')
        pdict.setdefault('micro_batch_size', '2')
        pdict.setdefault('data_parallel_shard_degree', '8')
        pdict.setdefault('tensor_parallel_degree', '1')
        pdict.setdefault('pipeline_parallel_degree', '1')
        pdict.setdefault('context_parallel_degree', '1')
        pdict.setdefault('expert_parallel_degree', '1')
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

        # Determine TorchTitan module from model name; config is always {module}_{model_size}
        if re.search('deepseek', self.model_name, re.I):
            self.tt_module = 'deepseek_v3'
        elif re.search('qwen', self.model_name, re.I):
            self.tt_module = 'qwen3'
        else:
            self.tt_module = 'llama3'
        self.tt_config = f'{self.tt_module}_{self.model_size}'

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
            if re.search('broadcom|thor', self.nic_type, re.I):
                self.nccl_ib_gid_index = 3
                out_dict = self.phdl.exec(
                    f'docker exec {self.container_name} /bin/bash -c "sudo \
                    cp /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.host \
                    /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so; \
                    sleep 2;ibv_devinfo;sleep 2;"'
                )
                for node in out_dict.keys():
                    if not re.search(r'hca_id:\s+bnxt_', out_dict[node], re.I):
                        log.info("%s", out_dict[node])
                        fail_test(f'Broadcom libbnxt rdma driver is not properly copied on node {node}')

    def build_training_job_cmd(self):
        """
        Construct native TorchTitan training command using torchrun.

        Uses torchrun to launch distributed training with TOML config.
        """
        cmd = f'cd /workspace/torchtitan; export HF_TOKEN={self.hf_token}; '
        cmd += 'export HSA_FORCE_FINE_GRAIN_PCIE=1; '
        cmd += 'export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True; '
        cmd += f'export CONFIG_FILE="./torchtitan/models/{self.tt_module}/train_configs/{self.tt_config}"; '

        if self.distributed_training is True:
            cmd += (
                f'export NCCL_IB_HCA={self.nccl_ib_hca_list}; '
                + f'export NCCL_SOCKET_IFNAME={self.nccl_socket_ifname}; '
                + f'export GLOO_SOCKET_IFNAME={self.gloo_socket_ifname}; '
                + f'export NCCL_DEBUG={self.nccl_debug}; '
                + f'export NCCL_IB_GID_INDEX={self.nccl_ib_gid_index}; '
            )

        nproc_per_node = 8

        download_tokenizer_cmd = (
            f'python scripts/download_hf_assets.py --repo_id {self.tokenizer_path} --assets --all tokenizer --hf_token={self.hf_token}'
        )

        if self.distributed_training:
            for i in range(self.nnodes):
                torchrun_cmd = (
                    f'torchrun --nnodes {self.nnodes} --node_rank={i} --nproc_per_node {nproc_per_node}'
                    f' --rdzv_id 101 --rdzv_backend c10d'
                    f' --rdzv_endpoint "{self.master_address}:29500"'
                    f' --role rank --tee 3'
                    f' -m torchtitan.train --job.config_file $CONFIG_FILE'
                )
                full_cmd = (
                    cmd + download_tokenizer_cmd + ' && ' + torchrun_cmd
                    + f' > {self.log_dir}/torchtitan-logs/out-node{i}/training.log 2>&1 &'
                )
                script_cmd = (
                    f'echo "{full_cmd}" > {self.scripts_dir}/distributed_wrapper_script_{i}.sh;'
                    f'chmod 777 {self.scripts_dir}/distributed_wrapper_script_{i}.sh'
                )
                self.job_cmd_list.append(script_cmd)
        else:
            torchrun_cmd = (
                f'torchrun --nnodes 1 --node_rank=0 --nproc_per_node {nproc_per_node}'
                f' --rdzv_id 101 --rdzv_backend c10d'
                f' --rdzv_endpoint "{self.master_address}:29500"'
                f' --role rank --tee 3'
                f' -m torchtitan.train --job.config_file $CONFIG_FILE'
            )
            self.job_cmd = (
                cmd + download_tokenizer_cmd + ' && ' + torchrun_cmd
                + f' > {self.log_dir}/torchtitan-logs/out-node0/training.log 2>&1 &'
            )

    def start_training_job(self, timeout=500):
        """
        Launch the TorchTitan training job (distributed or single-node).

        - Creates log directories for each node
        - Distributed mode: creates and executes per-node wrapper scripts
        - Single-node mode: creates and executes a single wrapper script
        """
        log.info('start training job')
        log.info("%s", self.job_cmd_list)
        log.info("%s", self.job_cmd)

        # Create log directories
        cmd_list = []
        for i in range(self.nnodes):
            cmd = f'docker exec {self.container_name} /bin/bash -c "mkdir -p {self.log_dir}/torchtitan-logs/out-node{i}"'
            cmd_list.append(cmd)
        self.phdl.exec_cmd_list(cmd_list)

        if self.distributed_training:
            self.exec_nic_setup_scripts()
            self.phdl.exec_cmd_list(self.job_cmd_list)

            cmd_list = []
            for i in range(self.nnodes):
                cmd = f'docker exec {self.container_name} /bin/bash {self.scripts_dir}/distributed_wrapper_script_{i}.sh'
                cmd_list.append(cmd)
            self.phdl.exec_cmd_list(cmd_list)
        else:
            self.phdl.exec(
                f'echo "{self.job_cmd}" > {self.scripts_dir}/single_node_wrapper_script.sh; '
                f'chmod 777 {self.scripts_dir}/single_node_wrapper_script.sh'
            )
            self.phdl.exec(f'docker exec {self.container_name} /bin/bash {self.scripts_dir}/single_node_wrapper_script.sh')

        time.sleep(50)

    def get_training_results_dict(self):
        """
        Parse TorchTitan training log output and extract key performance metrics.

        TorchTitan log format: "step: X, loss: Y, tok/s: Z, mem: W GB"

        Returns:
            dict: Extracted metric lists:
              - 'tokens_per_sec': Matches 'tok/s: <float>'
              - 'loss':           Matches 'loss: <float>'
              - 'mem_usage_gb':   Matches 'mem: <float> GB'
        """
        training_results_dict = {}

        last_node = self.host_list[-1]
        last_node_num = len(self.host_list) - 1
        out_dict = self.phdl.exec(
            f'cat {self.log_dir}/torchtitan-logs/out-node{last_node_num}/training.log | tail -20'
        )
        output = out_dict[last_node]

        log.info('Extracting results from logs')
        log.info('#===========================#')
        log.info("%s", output)
        log.info('#===========================#')

        training_results_dict['tokens_per_sec'] = re.findall(r'tok/s:\s+([0-9\.]+)', output, re.I)
        training_results_dict['loss'] = re.findall(r'loss:\s+([0-9\.]+)', output, re.I)
        training_results_dict['mem_usage_gb'] = re.findall(r'mem:\s+([0-9\.]+)\s+GB', output, re.I)

        log.info("%s", training_results_dict)
        return training_results_dict

    def scan_for_training_errors(self):
        """
        Scan training logs for known error patterns.

        Returns:
            bool: True if no errors found; False otherwise.
        """
        log.info('Scan for training errors')
        training_pass = True

        last_node = self.host_list[-1]
        last_node_num = len(self.host_list) - 1

        out_dict = self.phdl.exec(
            f'sudo cat {self.log_dir}/torchtitan-logs/out-node{last_node_num}/training.log'
        )
        output = out_dict[last_node]

        for err_key in training_err_dict:
            if re.search(f'{training_err_dict[err_key]}', output):
                fail_test(f'ERROR {training_err_dict[err_key]} seen in training logs ..')
                log.error('Aborting training log polling')
                training_pass = False
        return training_pass

    def poll_for_training_completion(self, time_between_iters=120):
        """
        Periodically poll training logs to detect completion and surface errors.

        TorchTitan completion detection looks for "step: {iterations}" in logs.
        """
        log.info('Poll for training completion ..')
        time.sleep(80)

        last_node = self.host_list[-1]
        last_node_num = len(self.host_list) - 1

        for i in range(1, int(self.iterations) + 10):
            log.info(f'Starting Iteration {i}')

            if not self.scan_for_training_errors():
                fail_test('Failures seen in training logs, Aborting!!!')
                return

            out_dict = self.phdl.exec(
                f'sudo cat {self.log_dir}/torchtitan-logs/out-node{last_node_num}/training.log'
            )
            output = out_dict[last_node]

            final_step_pattern = f'step:\\s+{self.iterations}'

            if not re.search(final_step_pattern, output, re.I):
                log.info('Training still in progress - final step not yet reached')
            else:
                if re.search(r'tok/s:\s+[NaN|Inf]', output, re.I) or re.search(r'loss:\s+[NaN|Inf]', output, re.I):
                    fail_test(f'ERROR - NaN or Inf values seen in training results {output}')
                    return
                else:
                    time.sleep(5)
                    self.training_results_dict = self.get_training_results_dict()
                    log.info('Completed Training, returning !!!')
                    return

            time.sleep(int(time_between_iters))

    def verify_training_results(self):
        """
        Validate collected training results and environment health after a training run.

        Checks:
          - training_results_dict for NaN/Inf values
          - Network stats (RDMA, ethtool) if distributed and verify_network_errors enabled
          - Kernel logs (dmesg) for errors
          - Performance results against expected thresholds
        """
        self.training_end_time = self.phdl.exec('date')

        log.info('#==================================================#')
        log.info('\t\tTraining Results')
        log.info("%s", self.training_results_dict)
        log.info('#==================================================#')

        if not self.training_results_dict:
            fail_test(
                'Failed to populate training results, training_results_dict is empty - please check logs for failures'
            )

        for result_key in self.training_results_dict.keys():
            for result_val in self.training_results_dict[result_key]:
                if re.search('nan|inf', str(result_val), re.I):
                    fail_test(
                        f'Failures seen in training_result dict for {result_key}, numbers are either NaN or Inf - {result_val}'
                    )

        if self.distributed_training is True and self.verify_network_errors == 'True':
            self.rdma_stats_dict_after = linux_utils.get_rdma_stats_dict(self.phdl)
            self.ethtool_stats_dict_after = linux_utils.get_nic_ethtool_stats_dict(self.phdl)

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

        verify_dmesg_for_errors(self.phdl, self.training_start_time, self.training_end_time)

        log.info('^^^^^^^^^^^^^^^^^^^^')
        log.info('training_results_dict')
        log.info('^^^^^^^^^^^^^^^^^^^^')
        log.info("%s", self.training_results_dict)

        for result_key in self.training_results_dict.keys():
            if result_key in self.expected_result_dict:
                log.info("%s", self.training_results_dict[result_key])
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
