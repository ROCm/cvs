'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

TorchTitan training job orchestration library.

Adapted from megatron_lib.py with TorchTitan-specific implementation:
- Uses torchrun instead of mpirun
- Generates TOML config files instead of CLI arguments
- Parses TorchTitan-specific metrics (tokens_per_sec, loss)
- Supports single-node and multi-node distributed training
'''

import re
import shlex
import time

from cvs.lib import globals
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *
from cvs.lib import linux_utils
from cvs.lib.training.torchtitan.utils.model_registry import (
    TORCHTITAN_MODELS,
    PRECISION_FLAGS,
    DEFAULT_TRAINING_PARAMS,
)

log = globals.log


training_err_dict = {
    'NCCL ERROR': 'NCCL ERROR|NCCL timeout|ncclRemoteError: A call failed possibly due to a network error|NCCL error:',
    'GPU HW ERROR': 'HW Exception by GPU|GPU Hang|Uncorrectable error|GPU Reset',
    'torch': 'torch.distributed.elastic.multiprocessing.errors',
}

err_counters_pattern = 'err|retransmit|drop|discard|naks|invalid|oflow|out_of_buffer|reset|fail'


# Ordered fallback chains for parsing TorchTitan training output
TRAINING_RESULT_PATTERNS = {
    'tokens_per_sec': [r'tps:\s+([0-9,\.]+)', r'tok/s:\s+([0-9\.]+)'],
    'loss': [r'loss:\s+([0-9\.]+)'],
    'mem_usage_gb': [r'memory:\s+([0-9\.]+)\s*GiB', r'mem:\s+([0-9\.]+)\s+GB'],
}

TRAINING_PROGRESS_PATTERNS = [
    r'step:\s+\d+',
    r'tps:\s+[0-9,\.]+',
    r'loss:\s+[0-9\.]+',
]

TRAINING_NAN_PATTERNS = [
    r'tok/s:\s+(?:NaN|Inf)',
    r'loss:\s+(?:NaN|Inf)',
]


def _parse_training_results(output):
    """Extract metric values from training-log text using ordered fallback chains."""
    out = {}
    for metric, patterns in TRAINING_RESULT_PATTERNS.items():
        out[metric] = []
        for pat in patterns:
            matches = re.findall(pat, output, re.I)
            if matches:
                # TorchTitan may emit comma-grouped numbers
                out[metric] = [m.replace(',', '') for m in matches]
                break
    return out


def _is_training_complete(output, iterations):
    """Return True if training log shows the configured final step."""
    final_step_pattern = rf'step:\s+{iterations}\b'
    return bool(re.search(final_step_pattern, output, re.I))


def _has_nan_inf_results(output):
    """Return True if training log shows NaN/Inf results."""
    return any(re.search(p, output, re.I) for p in TRAINING_NAN_PATTERNS)


def detect_rocm_path(orch, config_rocm_path):
    """
    Detect the ROCm installation path inside the container.
    """
    if config_rocm_path and config_rocm_path != '<changeme>':
        log.info(f'Using configured ROCm path: {config_rocm_path}')
        return config_rocm_path

    log.info('Auto-detecting ROCm path inside container...')

    # Try new ROCm layout first (/opt/rocm/core-X.Y)
    out_dict = orch.exec('ls -d /opt/rocm/core-* 2>/dev/null | sort -V | tail -1')
    for node, output in out_dict.items():
        if output and '/opt/rocm/core-' in output:
            rocm_path = output.strip()
            validate_dict = orch.exec(
                f'test -d {rocm_path}/lib && ls {rocm_path}/lib/libamdhip64.so* 2>/dev/null | head -1'
            )
            for _, lib_output in validate_dict.items():
                if lib_output.strip() and 'libamdhip64.so' in lib_output:
                    log.info(f'Detected ROCm path (new layout): {rocm_path}')
                    return rocm_path

    # Fall back to legacy /opt/rocm
    out_dict = orch.exec('test -d /opt/rocm/lib && ls /opt/rocm/lib/libamdhip64.so* 2>/dev/null | head -1')
    for node, output in out_dict.items():
        if output.strip() and 'libamdhip64.so' in output:
            log.info('Detected ROCm path (legacy layout): /opt/rocm')
            return '/opt/rocm'

    log.warning('Could not detect ROCm path, defaulting to /opt/rocm')
    return '/opt/rocm'


class TorchTitanTrainingJob:
    """
    Orchestrates a TorchTitan training job across one or more nodes.

    Similar to MegatronTrainingJob but adapted for TorchTitan:
    - Uses torchrun instead of mpirun
    - Generates TOML config files
    - Parses TorchTitan-specific metrics
    """

    def __init__(
        self,
        orch,
        variant_config,
        hf_token,
        micro_batch_size=None,
        global_batch_size=None,
        precision='',
        result_dict=None,
        distributed_training=True,
        tune_model_params=True,
        scripts_dir=None,
        run_label=None,
    ):
        self.orch = orch
        self.variant_config = variant_config
        self.hf_token = hf_token
        self.distributed_training = distributed_training
        self.tune_model_params = tune_model_params
        self.run_label = run_label

        self.job_cmd = ''
        self.job_cmd_list = []
        self.training_results_dict = {}
        self.local_tokenizer_path = None
        self.checkpoint_dir = None
        self.save_interval = None
        self.load_checkpoint = False

        # Get config and model params
        self.config = variant_config.config
        self.model_params = variant_config.model_params
        self.gpu_arch = variant_config.gpu_arch

        # Training configs with defaults
        self.container_image = self.config.get('container_image', 'rocm/pytorch:latest')
        self.container_name = self.config.get('container_name', 'torchtitan_training')
        self.torchtitan_root = self.config.get('torchtitan_root', '/workspace/Primus/third_party/torchtitan')
        self.iterations = int(self.config.get('training_iterations', 30))
        self.nnodes = int(self.config.get('nnodes', 1))
        self.nic_type = self.config.get('nic_type', 'thor2')
        self.hca_id_pattern = self.config.get('hca_id_pattern', 'bnxt_|rocep')
        self.nccl_ib_hca_list = self.config.get('nccl_ib_hca_list', '')
        self.nccl_ib_hca = self.config.get('nccl_ib_hca', '')
        self.nccl_socket_ifname = self.config.get('nccl_socket_ifname', '')
        self.gloo_socket_ifname = self.config.get('gloo_socket_ifname', '')
        self.nccl_ib_gid_index = self.config.get('nccl_ib_gid_index', '3')
        self.nccl_debug = self.config.get('nccl_debug', 'ERROR')
        self.data_cache_dir = self.config.get('data_cache_dir', '/tmp/cache')
        self.log_dir = self.config.get('log_dir', '/tmp/logs')
        self.scripts_dir = scripts_dir if scripts_dir is not None else self.config.get('scripts_dir', '/tmp/scripts')
        self.master_address = self.config.get('master_address', list(orch.hosts)[0] if orch.hosts else 'localhost')
        self.verify_network_errors = self.config.get('verify_network_errors', 'False')
        self.rocm_path = detect_rocm_path(self.orch, self.config.get('rocm_dir', ''))
        self.use_generated_config = self.config.get('use_generated_config', 'True') == 'True'
        self.hf_token_file = self.config.get('hf_token_file', '/tmp/.hf_token')

        # Per-combo log dir so sweep combos don't overwrite each other's training.log
        raw_label = run_label or "torchtitan_training"
        self.run_label_sanitized = re.sub(r'[^A-Za-z0-9._-]', '_', str(raw_label))
        self.combo_log_dir = f'{self.log_dir}/torchtitan-logs/{self.run_label_sanitized}'

        # Model params with defaults
        model_name = self.model_params.get('model_name', 'llama3_3_70b')
        self.model_config = TORCHTITAN_MODELS.get(model_name, TORCHTITAN_MODELS['llama3_3_70b'])
        self.model_name = model_name
        self.tt_module = self.model_config['module']
        self.model_size = self.model_config['model_size']
        self.tokenizer_path = self.model_config['tokenizer_path']

        # Override batch sizes if provided
        if micro_batch_size is not None:
            self.micro_batch_size = str(micro_batch_size)
        else:
            self.micro_batch_size = str(self.model_params.get('micro_batch_size', '1'))

        if global_batch_size is not None:
            self.global_batch_size = str(global_batch_size)
        else:
            self.global_batch_size = str(self.model_params.get('global_batch_size', '32'))

        # Precision settings
        if precision:
            self.precision = precision
        else:
            self.precision = self.model_params.get('precision', 'bf16')

        prec_flags = PRECISION_FLAGS.get(self.precision, PRECISION_FLAGS['bf16'])
        self.dtype = prec_flags['dtype']
        self.enable_float8 = prec_flags['enable_float8']
        self.converters = prec_flags['converters']

        # TorchTitan config name for fallback to canned TOMLs
        self.tt_config = f'{self.tt_module}_{self.model_size}'

        # HF assets path for model downloads
        self.hf_assets_path = self.model_params.get(
            'hf_assets_path',
            f'./assets/hf/{self.model_config["hf_assets_subdir"]}/{self.tokenizer_path.split("/")[-1]}',
        )

        # Other training params with defaults from DEFAULT_TRAINING_PARAMS
        for key, default_val in DEFAULT_TRAINING_PARAMS.items():
            setattr(self, key, self.model_params.get(key, default_val))

        # Sequence length
        self.sequence_length = str(self.model_params.get('sequence_length', '8192'))

        # Parallelism degrees
        self.data_parallel_shard_degree = str(self.model_params.get('data_parallel_shard_degree', '8'))
        self.tensor_parallel_degree = str(self.model_params.get('tensor_parallel_degree', '1'))
        self.pipeline_parallel_degree = str(self.model_params.get('pipeline_parallel_degree', '1'))
        self.context_parallel_degree = str(self.model_params.get('context_parallel_degree', '1'))
        self.expert_parallel_degree = str(self.model_params.get('expert_parallel_degree', '1'))
        self.enable_async_tensor_parallel = str(self.model_params.get('enable_async_tensor_parallel', 'false')).lower()
        self.precompute_float8_dynamic_scale_for_fsdp = str(
            self.model_params.get('precompute_float8_dynamic_scale_for_fsdp', 'false')
        ).lower()

        # Result expectations
        self.expected_result_dict = result_dict or {}

        # Initialize stats dicts
        self.rdma_stats_dict_before = {}
        self.ethtool_stats_dict_before = {}
        self.rdma_stats_dict_after = {}
        self.ethtool_stats_dict_after = {}
        self.training_start_time = None
        self.training_end_time = None

        # Create scripts directory (owner-only for security - contains HF tokens)
        self.orch.exec(f'rm -rf {self.scripts_dir}')
        time.sleep(1)
        self.orch.exec(f'mkdir -p {self.scripts_dir}')
        time.sleep(1)
        self.orch.exec(f'chmod 700 {self.scripts_dir}')

        # Adjust batch size for distributed if needed
        if self.tune_model_params and self.distributed_training:
            gpus_per_node = 8
            total_gpus = self.nnodes * gpus_per_node
            if int(self.global_batch_size) > 32:
                if int(self.global_batch_size) % 32 == 0:
                    per_gpu_batch_size = int(self.global_batch_size) / 32
                    self.global_batch_size = str(int(per_gpu_batch_size * total_gpus))

    def run_pretraining_tasks(self):
        """Snapshot network stats before training (distributed only)."""
        if self.distributed_training:
            self.rdma_stats_dict_before = linux_utils.get_rdma_stats_dict(self.orch)
            self.ethtool_stats_dict_before = linux_utils.get_nic_ethtool_stats_dict(self.orch)

    def download_hf_assets(self):
        """Download HuggingFace model assets if needed.

        Uses TorchTitan's download_hf_assets.py script to fetch model weights
        and tokenizers from HuggingFace Hub. Idempotent - skips if already present.
        """
        if not self.use_generated_config:
            # Canned configs expect assets in ./assets/hf/
            local_dir = './assets/hf/'
        else:
            # Generated configs use hf_assets_path, but download needs base dir only
            # (download script adds repo name automatically)
            local_dir = f'./assets/hf/{self.model_config["hf_assets_subdir"]}'

        log.info(f'Downloading HF assets for {self.tokenizer_path} to {local_dir}')

        download_cmd = (
            f'cd {self.torchtitan_root}; '
            f'export HF_TOKEN={self.hf_token}; '
            f'python scripts/download_hf_assets.py --repo_id {self.tokenizer_path} '
            f'--local_dir {local_dir} --all'
        )

        out_dict = self.orch.exec(download_cmd)
        for node, output in out_dict.items():
            if 'error' in (output or '').lower():
                log.warning(f'Potential download error on {node}: {output}')

    def exec_nic_setup_scripts(self):
        """Setup NICs for distributed training (Broadcom/Thor only)."""
        if not self.distributed_training:
            return

        if re.search('broadcom|thor', self.nic_type, re.I):
            self.nccl_ib_gid_index = '3'
            out_dict = self.orch.exec(
                'sudo cp /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.host '
                '/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so && '
                'sleep 2 && ibv_devinfo'
            )

            segments = [re.escape(s.strip()) for s in self.hca_id_pattern.split('|') if s.strip()]
            if not segments:
                fail_test(f'hca_id_pattern invalid: {self.hca_id_pattern}')

            hca_id_regex = rf'hca_id:\s+({"|".join(segments)})'
            for node, output in out_dict.items():
                if not re.search(hca_id_regex, output or '', re.I):
                    fail_test(f'Broadcom RDMA device not detected on {node}')

    def _build_toml_config(self):
        """Generate TorchTitan TOML configuration."""
        # Use hf_assets_path for model location
        hf_path = self.hf_assets_path

        # Build quantization converters list
        self.converters if isinstance(self.converters, str) else '[]'

        lines = [
            "[model]",
            f'name = "{self.tt_module}"',
            f'flavor = "{self.model_size.upper()}"',
            f'hf_assets_path = "{hf_path}"',
            "",
            "[training]",
            f'dataset = "{self.dataset}"',
            f'local_batch_size = {self.micro_batch_size}',
            f'global_batch_size = {self.global_batch_size}',
            f'seq_len = {self.sequence_length}',
            f'steps = {self.iterations}',
            f'dtype = "{self.dtype}"',
            "",
            "[optimizer]",
            f'lr = {self.lr}',
            "",
            "[lr_scheduler]",
            f'warmup_steps = {self.warmup_steps}',
            "",
            "[parallelism]",
            f'data_parallel_shard_degree = {self.data_parallel_shard_degree}',
            f'tensor_parallel_degree = {self.tensor_parallel_degree}',
            f'pipeline_parallel_degree = {self.pipeline_parallel_degree}',
            f'context_parallel_degree = {self.context_parallel_degree}',
            f'expert_parallel_degree = {self.expert_parallel_degree}',
            f'enable_async_tensor_parallel = {self.enable_async_tensor_parallel}',
            "",
            "[activation_checkpoint]",
            f'mode = "{self.activation_checkpointing}"',
            "",
            "[compile]",
            f'enable = {self.compile}',
            "",
            "[quantize.linear.float8]",
            f'enable_fsdp_float8_all_gather = {str(self.enable_float8).lower()}',
            f'precompute_float8_dynamic_scale_for_fsdp = {self.precompute_float8_dynamic_scale_for_fsdp}',
            # converters not supported in this TorchTitan version
            # f'converters = {converters_str}',
            'filter_fqns = ["output"]',
            "",
            "[comm]",
            'init_timeout_seconds = 3600',
        ]
        return "\n".join(lines) + "\n"

    def _write_generated_toml(self, dest_path):
        """Write TOML config to destination path on all nodes."""
        toml_content = self._build_toml_config()
        log.info('Generated TorchTitan TOML config')

        # Use printf to write multi-line content
        escaped = toml_content.replace('\\', '\\\\').replace('$', '\\$').replace('"', '\\"')
        write_cmd = f'printf "%s" "{escaped}" > {dest_path}'
        self.orch.exec(write_cmd)

    def build_training_job_cmd(self):
        """Build torchrun commands for training."""
        # Base environment setup
        cmd = f'cd {self.torchtitan_root}; '
        cmd += f'export HF_TOKEN={self.hf_token}; '
        cmd += 'export HSA_FORCE_FINE_GRAIN_PCIE=1; '
        cmd += 'export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True; '
        # Add TorchTitan to PYTHONPATH so it can be imported as a module
        cmd += f'export PYTHONPATH={self.torchtitan_root}:$PYTHONPATH; '

        # Config file path - supports both generated and canned TOMLs
        if self.use_generated_config:
            config_file_path = f'{self.scripts_dir}/run_config.toml'
            self._write_generated_toml(config_file_path)
        else:
            # Fallback to canned TOML shipped with TorchTitan
            config_file_path = f'./train_configs/{self.tt_config}.toml'
            log.info(f'Using canned TOML config: {config_file_path}')

        # Distributed env vars
        if self.distributed_training:
            cmd += f'export NCCL_IB_HCA={self.nccl_ib_hca_list}; '
            cmd += f'export NCCL_SOCKET_IFNAME={self.nccl_socket_ifname}; '
            cmd += f'export GLOO_SOCKET_IFNAME={self.gloo_socket_ifname}; '
            cmd += f'export NCCL_DEBUG={self.nccl_debug}; '
            cmd += f'export NCCL_IB_GID_INDEX={self.nccl_ib_gid_index}; '

        nproc_per_node = 8

        if self.distributed_training:
            for i in range(self.nnodes):
                torchrun_cmd = (
                    f'torchrun --nnodes {self.nnodes} --node_rank={i} --nproc_per_node {nproc_per_node} '
                    f'--rdzv_id 101 --rdzv_backend c10d '
                    f'--rdzv_endpoint "{self.master_address}:29500" '
                    f'--role rank --tee 3 '
                    f'--module torchtitan.train --job.config_file {config_file_path}'
                )

                # Add checkpoint args if configured
                if self.checkpoint_dir:
                    torchrun_cmd += f' --checkpoint.folder {self.checkpoint_dir}'
                    torchrun_cmd += f' --checkpoint.interval {self.save_interval}'
                    torchrun_cmd += ' --checkpoint.enable_checkpoint true'
                if self.load_checkpoint:
                    torchrun_cmd += ' --checkpoint.load_step -1'

                log_path = f'{self.combo_log_dir}/out-node{i}/training.log'
                self.orch.exec(f'mkdir -p $(dirname {log_path})')

                full_cmd = cmd + f': > {log_path}; nohup {torchrun_cmd} > {log_path} 2>&1 & disown'

                script_cmd = (
                    f"cat > {self.scripts_dir}/distributed_wrapper_script_{i}.sh << 'WRAPPER_EOF'\n"
                    f"#!/bin/bash\n{full_cmd}\nWRAPPER_EOF\n; "
                    f'chmod 600 {self.scripts_dir}/distributed_wrapper_script_{i}.sh'
                )
                self.job_cmd_list.append(script_cmd)
        else:
            torchrun_cmd = (
                f'torchrun --nnodes 1 --node_rank=0 --nproc_per_node {nproc_per_node} '
                f'--rdzv_id 101 --rdzv_backend c10d '
                f'--rdzv_endpoint "{self.master_address}:29500" '
                f'--role rank --tee 3 '
                f'--module torchtitan.train --job.config_file {config_file_path}'
            )

            # Add checkpoint args if configured
            if self.checkpoint_dir:
                torchrun_cmd += f' --checkpoint.folder {self.checkpoint_dir}'
                torchrun_cmd += f' --checkpoint.interval {self.save_interval}'
                torchrun_cmd += ' --checkpoint.enable_checkpoint true'
            if self.load_checkpoint:
                torchrun_cmd += ' --checkpoint.load_step -1'

            log_path = f'{self.combo_log_dir}/out-node0/training.log'
            self.orch.exec(f'mkdir -p $(dirname {log_path})')

            self.job_cmd = cmd + f': > {log_path}; nohup {torchrun_cmd} > {log_path} 2>&1 & disown'

    def start_training_job(self, timeout=500):
        """Launch the training job."""
        # Capture start time for dmesg verification
        self.training_start_time = self.orch.exec('date')

        if self.distributed_training:
            for i, cmd in enumerate(self.job_cmd_list):
                log.info(f'Writing wrapper script for node {i}')
                self.orch.exec(cmd)

            time.sleep(2)

            for i in range(self.nnodes):
                script_path = f'{self.scripts_dir}/distributed_wrapper_script_{i}.sh'
                log.info(f'Launching training on node {i}')
                self.orch.exec(f'bash {script_path}', hosts=[list(self.orch.hosts)[i]])
                time.sleep(1)
        else:
            log.info('Launching single-node training')
            self.orch.exec(f'bash -c {shlex.quote(self.job_cmd)}')

    def get_training_results_dict(self):
        """Parse training results from logs."""
        if self.distributed_training:
            log_files = [f'{self.log_dir}/torchtitan-logs/out-node{i}/training.log' for i in range(self.nnodes)]
        else:
            log_files = [f'{self.log_dir}/torchtitan-logs/out-node0/training.log']

        all_results = {}
        for log_file in log_files:
            out_dict = self.orch.exec(f'cat {log_file}')
            for host, output in out_dict.items():
                if output:
                    parsed = _parse_training_results(output)
                    for metric, values in parsed.items():
                        if metric not in all_results:
                            all_results[metric] = []
                        all_results[metric].extend(values)

        return all_results

    def scan_for_training_errors(self):
        """Scan training logs for known error patterns."""
        if self.distributed_training:
            log_files = [f'{self.log_dir}/torchtitan-logs/out-node{i}/training.log' for i in range(self.nnodes)]
        else:
            log_files = [f'{self.log_dir}/torchtitan-logs/out-node0/training.log']

        for log_file in log_files:
            out_dict = self.orch.exec(f'tail -1000 {log_file}')
            for host, output in out_dict.items():
                if not output:
                    continue
                for err_type, pattern in training_err_dict.items():
                    if re.search(pattern, output, re.I):
                        fail_test(f'{err_type} detected in training log on {host}')

    def poll_for_training_completion(self, time_between_iters=120):
        """Poll training logs until completion."""
        max_iters = 60

        if self.distributed_training:
            log_file = f'{self.log_dir}/torchtitan-logs/out-node0/training.log'
        else:
            log_file = f'{self.log_dir}/torchtitan-logs/out-node0/training.log'

        for iteration in range(max_iters):
            time.sleep(time_between_iters)
            log.info(f'Polling iteration {iteration + 1}/{max_iters}')

            out_dict = self.orch.exec(f'tail -500 {log_file}')
            for host, output in out_dict.items():
                if output and _is_training_complete(output, self.iterations):
                    log.info(f'Training completed on {host}')
                    return

                if output and _has_nan_inf_results(output):
                    fail_test(f'NaN/Inf detected in training output on {host}')

        fail_test(f'Training did not complete within {max_iters * time_between_iters} seconds')

    def verify_training_results(self):
        """Verify training results meet expectations."""
        # Capture end time for dmesg verification
        self.training_end_time = self.orch.exec('date')

        self.training_results_dict = self.get_training_results_dict()
        log.info(f'Training results: {self.training_results_dict}')

        # Scan for errors
        self.scan_for_training_errors()

        # Check for NaN/Inf in results
        for metric, values in self.training_results_dict.items():
            for val in values:
                try:
                    float_val = float(val)
                    if str(float_val).lower() in ['nan', 'inf', '-inf']:
                        fail_test(f'Invalid value {val} for metric {metric}')
                except ValueError:
                    fail_test(f'Cannot parse value {val} for metric {metric}')

        # Check network errors if requested
        if self.distributed_training and self.verify_network_errors == 'True':
            self.rdma_stats_dict_after = linux_utils.get_rdma_stats_dict(self.orch)
            self.ethtool_stats_dict_after = linux_utils.get_nic_ethtool_stats_dict(self.orch)

            # Compare RDMA error counters; fail if any error counter increased
            for node in self.rdma_stats_dict_after.keys():
                for counter_name in self.rdma_stats_dict_after[node]:
                    if re.search(err_counters_pattern, counter_name, re.I):
                        if int(self.rdma_stats_dict_after[node][counter_name]) > int(
                            self.rdma_stats_dict_before[node][counter_name]
                        ):
                            fail_test(
                                f'Error counter {counter_name} has gone up after training on node {node} '
                                f'Before = {self.rdma_stats_dict_before[node][counter_name]}, '
                                f'After = {self.rdma_stats_dict_after[node][counter_name]}'
                            )

            # Compare NIC error counters; fail if any error counter increased
            for node in self.ethtool_stats_dict_after.keys():
                for counter_name in self.ethtool_stats_dict_after[node]:
                    if re.search(err_counters_pattern, counter_name, re.I):
                        if int(self.ethtool_stats_dict_after[node][counter_name]) > int(
                            self.ethtool_stats_dict_before[node][counter_name]
                        ):
                            fail_test(
                                f'Error counter {counter_name} has gone up after training on node {node} '
                                f'Before = {self.ethtool_stats_dict_before[node][counter_name]}, '
                                f'After = {self.ethtool_stats_dict_after[node][counter_name]}'
                            )

        # Scan dmesg for errors during training window
        verify_dmesg_for_errors(self.orch, self.training_start_time, self.training_end_time, till_end_flag=False)

        update_test_result()

    def _needs_local_tokenizer(self):
        """TorchTitan uses HF download script, so always downloads tokenizer.

        Returns False since we use download_hf_assets() instead of download_tokenizer_model().
        """
        return False

    def download_tokenizer_model(self):
        """Download tokenizer model (wrapper for download_hf_assets).

        For compatibility with test suite expecting this method name.
        """
        self.download_hf_assets()
        self.local_tokenizer_path = self.hf_assets_path

    def stop_training_processes(self):
        """Check GPU VRAM after a training combo and free memory if any processes remain.

        After normal training completion VRAM% is 0 and no KFD PIDs are present —
        returns immediately in that case. If processes are still holding GPU memory
        (crash or hang), extracts their PIDs from rocm-smi --showpids, kills them
        with SIGKILL, then waits and verifies VRAM is clear before the next combo.
        """
        log.info('Checking GPU memory state after training combo')
        out_dict = self.orch.exec('rocm-smi --showpids 2>/dev/null')

        has_pids = False
        for node, output in (out_dict or {}).items():
            if 'No KFD PIDs currently running' in (output or ''):
                log.info('Node %s: VRAM already free, no GPU processes running', node)
            else:
                log.warning('Node %s: GPU processes still holding VRAM, will kill', node)
                has_pids = True

        if not has_pids:
            return

        # Extract PIDs (lines starting with a number) and SIGKILL on all nodes
        self.orch.exec(
            "rocm-smi --showpids 2>/dev/null "
            "| awk '/^[0-9]+[[:space:]]/{print $1}' "
            "| xargs -r kill -9 2>/dev/null || true; "
            "sleep 10"
        )

        # Verify VRAM is now free
        out_dict = self.orch.exec('rocm-smi --showpids 2>/dev/null')
        for node, output in (out_dict or {}).items():
            if 'No KFD PIDs currently running' in (output or ''):
                log.info('Node %s: VRAM successfully freed', node)
            else:
                log.warning('Node %s: GPU processes may still be running after kill attempt', node)

    def _parse_step_losses(self, log_text):
        """Parse step-to-loss mapping from TorchTitan training log.

        Args:
            log_text (str): Full training log text.

        Returns:
            dict: {step: loss} mapping for all logged steps.
        """
        losses = {}
        # TorchTitan pattern: step: N | loss: X.XX
        pattern = re.compile(r'step:\s+(\d+)[^\n]*?loss:\s+([0-9.eE+\-]+)', re.I)
        for m in pattern.finditer(log_text):
            step = int(m.group(1))
            loss = float(m.group(2))
            losses[step] = loss
        return losses

    def _read_last_node_log(self, tail_lines=0):
        """Read the training log from the last node and return its output.

        Args:
            tail_lines (int): If > 0, only the last N lines of the log are read.

        Returns:
            str: Log text from the last node.
        """
        n = len(self.orch.hosts)
        last_host = self.orch.hosts[-1]
        tail_suffix = f' | tail -{tail_lines}' if tail_lines > 0 else ''
        log_path = f'{self.combo_log_dir}/out-node{n - 1}/training.log'
        out_dict = self.orch.exec(f'cat {log_path}{tail_suffix}', hosts=[last_host])
        return out_dict.get(last_host) or ''
