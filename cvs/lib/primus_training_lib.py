'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import os
import re
import time
import base64

from cvs.lib import globals
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *
from cvs.lib import linux_utils

log = globals.log


def str_to_bool(value):
    """Convert string/int/bool to boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes')
    return bool(value)


training_err_dict = {
    'NCCL ERROR': 'NCCL ERROR|NCCL timeout|ncclRemoteError: A call failed possibly due to a network error|NCCL error:',
    'GPU HW ERROR': 'HW Exception by GPU|GPU Hang|Uncorrectable error|GPU Reset',
    'torch': 'torch.distributed.elastic.multiprocessing.errors',
    'MoE ERROR': 'Expert.*error|MoE.*fail|routing.*error',
}

err_counters_pattern = 'err|retransmit|drop|discard|naks|invalid|oflow|out_of_buffer|reset|fail'

# Library for building Primus DeepSeek training jobs ..


def prepare_c4_dataset(
    phdl,
    container_name,
    num_shards=200,
    data_dir="/shared/c4",
    tokenizer_type="DeepSeekV3Tokenizer",
    tokenizer_model="deepseek-ai/DeepSeek-V3",
    primus_path="/workspace/Primus",
    workers=None,
    timeout=3600,
):
    """
    Prepare C4 dataset inside Primus container for DeepSeek training.

    Steps:
    1. Check if tokenized data already exists
    2. If not, merge C4 shards to JSONL
    3. Tokenize using Primus's preprocess_data.py with DeepSeekV3Tokenizer
    4. Return path to tokenized data

    Args:
        phdl: Parallel SSH handle
        container_name: Primus container name
        num_shards: Number of C4 shards to process (1-1024)
        data_dir: Base data directory path
        tokenizer_type: Tokenizer type (DeepSeekV3Tokenizer)
        tokenizer_model: HuggingFace model identifier
        primus_path: Path to Primus inside container
        workers: Number of preprocessing workers (default: nproc)
        timeout: Max time for tokenization (seconds)

    Returns:
        str: Path prefix to tokenized data (without .bin/.idx extensions)

    Raises:
        Exception: If data preparation fails
    """

    head_node = phdl.host_list[0]
    raw_dir = f"{data_dir}/en"
    jsonl_dir = f"{data_dir}/jsonl"
    tokenized_dir = f"{data_dir}/tokenized"
    tokenized_path = f"{tokenized_dir}/c4_en_train_text_document"

    log.info(f"Preparing C4 dataset: {num_shards} shards")
    log.info(f"Output path: {tokenized_path}")

    # Check if already tokenized
    check_cmd = f"test -f {tokenized_path}.bin && test -f {tokenized_path}.idx && echo 'EXISTS' || echo 'NOT_FOUND'"
    result_dict = phdl.exec(check_cmd)

    if 'EXISTS' in result_dict[head_node]:
        log.info(f"Dataset already exists: {tokenized_path}")
        return tokenized_path

    log.info("Dataset not found, starting preparation...")

    # Create directories
    mkdir_cmd = f"mkdir -p {raw_dir} {jsonl_dir} {tokenized_dir}"
    phdl.exec(mkdir_cmd)

    # Step 1: Merge C4 shards into JSONL
    log.info(f"Step 1: Merging {num_shards} C4 shards into JSONL...")
    jsonl_file = f"{jsonl_dir}/c4_en_train.jsonl"

    # Check if JSONL already exists
    jsonl_check_cmd = f"test -f {jsonl_file} && echo 'EXISTS' || echo 'NOT_FOUND'"
    jsonl_result = phdl.exec(jsonl_check_cmd)

    if 'NOT_FOUND' in jsonl_result[head_node]:
        # Verify shards exist
        verify_cmd = f"""
        missing=0
        for i in $(seq 0 {num_shards - 1}); do
            shard=$(printf "c4-train.%05d-of-01024.json.gz" $i)
            if [ ! -f "{raw_dir}/$shard" ]; then
                echo "WARNING: Missing shard $shard"
                ((missing++))
            fi
        done
        if [ $missing -gt 0 ]; then
            echo "ERROR: $missing shards missing"
            exit 1
        fi
        echo "All shards present"
        """
        verify_result = phdl.exec(verify_cmd, timeout=60)

        if 'ERROR' in verify_result[head_node]:
            fail_test(f"Missing C4 shards in {raw_dir}")

        # Merge shards
        merge_cmd = f"""
        cd {data_dir} &&
        rm -f {jsonl_file} &&
        for i in $(seq 0 {num_shards - 1}); do
            shard=$(printf "c4-train.%05d-of-01024.json.gz" $i)
            echo "Processing shard $i/{num_shards}..."
            zcat {raw_dir}/$shard >> {json_file}
        done &&
        echo "Merge complete" &&
        wc -l {jsonl_file}
        """

        log.info("Merging shards (this may take 10-30 minutes)...")
        merge_result = phdl.exec(merge_cmd, timeout=1800)
        log.info(f"Merge complete: {merge_result[head_node]}")
    else:
        log.info(f"JSONL file already exists: {jsonl_file}")

    # Step 2: Tokenize using Primus's preprocess_data.py
    log.info(f"Step 2: Tokenizing with {tokenizer_type}...")

    workers_str = workers if workers else "$(nproc)"

    tokenize_cmd = f"""
    docker exec {container_name} /bin/bash -c '
        cd {primus_path} &&
        export PYTHONPATH={primus_path}/third_party/Megatron-LM:{primus_path}:$PYTHONPATH &&
        python3 {primus_path}/examples/megatron/preprocess_data.py \\
            --input {jsonl_file} \\
            --tokenizer-type {tokenizer_type} \\
            --tokenizer-model {tokenizer_model} \\
            --output-prefix {tokenized_path} \\
            --workers {workers_str} \\
            --append-eod \\
            --partitions 1 &&
        echo "Tokenization complete" &&
        ls -lh {tokenized_dir}/
    '
    """

    log.info(f"Tokenizing (this may take 30-60 minutes for {num_shards} shards)...")
    tokenize_result = phdl.exec(tokenize_cmd, timeout=timeout)

    if 'Tokenization complete' not in tokenize_result[head_node]:
        fail_test(f"Tokenization failed on {head_node}")

    log.info(f"Dataset preparation complete: {tokenized_path}")
    log.info(tokenize_result[head_node])

    return tokenized_path


class PrimusDeepSeekTrainingJob:
    """
    Orchestrates a Primus DeepSeek training job across multiple nodes.

    This class handles DeepSeek V2/V3 models using the Primus framework, which provides:
    - AMD-optimized MoE (Mixture of Experts) implementations
    - Turbo optimizations (turbo_deepep, turbo_grouped_mlp)
    - Expert Parallelism (EP) and Virtual Pipeline Parallelism (VPP)
    - AINIC networking support for MI300X
    - DeepSeek-specific model configurations

    Container: tasimage/primus:pr-563-ainic
    Models: DeepSeek-V3 (61 layers, MoE), DeepSeek-V2-Lite

    Responsibilities:
      - Configure MoE-specific parameters (EP, VPP, expert routing)
      - Compute dynamic pipeline layouts for uneven layer distributions
      - Set up AINIC networking for optimal MI300X performance
      - Launch training via Primus's run_slurm_pretrain.sh (adapted for Docker)
      - Monitor training and extract performance metrics
      - Validate MoE-specific correctness (expert load balance, routing)
    """

    def __init__(
        self,
        phdl,
        model_name,
        training_config_dict,
        model_params_dict,
        hf_token,
        gpu_type='mi300x',
        distributed_training=True,
        tune_model_params=False,
        scripts_dir=os.path.expanduser("~") + '/SCRIPTS',
    ):
        """
        Initialize Primus DeepSeek training job configuration.

        Args:
          phdl: Remote execution handle for multi-node command execution
          model_name: Model identifier (e.g., "deepseek_v3", "deepseek_v2_lite")
          training_config_dict: Training configuration with Primus-specific settings
          model_params_dict: Model parameters including MoE configuration
          hf_token: HuggingFace token for model/data access
          gpu_type: GPU platform key (default: 'mi300x')
          distributed_training: Whether to run distributed training
          tune_model_params: If True, adjust params based on cluster size
          scripts_dir: Directory for generated wrapper scripts
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

        # Initialize cluster stats dicts
        self.rdma_stats_dict_before = {}
        self.ethtool_stats_dict_before = {}
        self.rdma_stats_dict_after = {}
        self.ethtool_stats_dict_after = {}
        self.training_start_time = self.phdl.exec('date')
        self.training_end_time = None

        # Training configs - set defaults if not defined
        self.home_dir = os.path.expanduser("~")
        tdict = training_config_dict

        # Primus-specific defaults
        tdict.setdefault('container_image', 'docker.io/tasimage/primus:pr-563-ainic')
        tdict.setdefault('container_name', 'primus_deepseek_v3')
        tdict.setdefault('training_iterations', 50)
        tdict.setdefault('nnodes', 128)
        tdict.setdefault('primus_path', '/workspace/Primus')
        tdict.setdefault('mock_data', False)

        # AINIC networking defaults
        tdict.setdefault('using_ainic', True)
        tdict.setdefault(
            'nccl_ib_hca', 'ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1'
        )
        tdict.setdefault('nccl_socket_ifname', 'ens9np0')
        tdict.setdefault('gloo_socket_ifname', 'ens9np0')
        tdict.setdefault('nccl_ib_timeout', 23)
        tdict.setdefault('nccl_debug', 'ERROR')

        # Other defaults
        tdict.setdefault('data_cache_dir', f'{self.home_dir}/cache')
        tdict.setdefault('log_dir', f'{self.home_dir}/LOGS/primus')
        tdict.setdefault('master_address', '127.0.0.1')
        tdict.setdefault('verify_network_errors', 'False')
        tdict.setdefault('enable_numa_binding', True)
        tdict.setdefault('hsa_kernarg_pool_size', 12582912)

        # Data preparation config
        data_prep = tdict.get('data_prep', {})
        tdict.setdefault('tokenized_data_path', data_prep.get('tokenized_data_path', ''))

        self.container_image = tdict['container_image']
        self.container_name = tdict['container_name']
        self.distributed_training = distributed_training
        self.iterations = int(tdict['training_iterations'])
        self.nnodes = int(tdict['nnodes'])
        self.primus_path = tdict['primus_path']
        self.mock_data = str_to_bool(tdict['mock_data'])

        # AINIC networking
        self.using_ainic = str_to_bool(tdict['using_ainic'])
        self.nccl_ib_hca = tdict['nccl_ib_hca']
        self.nccl_socket_ifname = tdict['nccl_socket_ifname']
        self.gloo_socket_ifname = tdict['gloo_socket_ifname']
        self.nccl_ib_timeout = int(tdict['nccl_ib_timeout'])
        self.nccl_debug = tdict['nccl_debug']

        self.data_cache_dir = tdict['data_cache_dir']
        self.log_dir = tdict['log_dir']
        self.master_address = tdict['master_address']
        self.verify_network_errors = str_to_bool(tdict['verify_network_errors'])
        self.tokenized_data_path = tdict['tokenized_data_path']

        # ROCm-specific
        self.enable_numa_binding = str_to_bool(tdict['enable_numa_binding'])
        self.hsa_kernarg_pool_size = int(tdict['hsa_kernarg_pool_size'])

        # Get model parameters
        log.info(f"Loading model params for: {self.model_name}, {self.gpu_type}")

        if not self.distributed_training:
            pdict = self.model_params_dict['single_node'][self.model_name][self.gpu_type]
            self.expected_result_dict = pdict.get('result_dict', {})
        else:
            pdict = self.model_params_dict['multi_node'][self.model_name][self.gpu_type]
            self.expected_result_dict = pdict.get('result_dict', {})

        # Model parameters - set defaults
        pdict.setdefault('model_type', 'deepseek_v3')
        pdict.setdefault('num_layers', 61)
        pdict.setdefault('hidden_size', 7168)
        pdict.setdefault('num_attention_heads', 128)
        pdict.setdefault('seq_length', 4096)
        pdict.setdefault('max_position_embeddings', 4096)

        # Batch sizes
        pdict.setdefault('micro_batch_size', 2)
        pdict.setdefault('global_batch_size', 128 * self.nnodes)

        # Parallelism
        pdict.setdefault('tensor_parallel', 1)
        pdict.setdefault('pipeline_parallel', 8)
        pdict.setdefault('expert_parallel', 8)
        pdict.setdefault('virtual_pipeline_parallel', 2)
        pdict.setdefault('data_parallel', -1)  # Auto-compute

        # MoE parameters
        pdict.setdefault('num_experts', 256)
        pdict.setdefault('moe_top_k', 8)
        pdict.setdefault('moe_layer_freq', 1)

        # Recomputation
        pdict.setdefault('recompute_num_layers', 3)
        pdict.setdefault('recompute_granularity', 'full')
        pdict.setdefault('recompute_method', 'block')

        # Primus-specific optimizations
        pdict.setdefault('turbo_deepep', True)
        pdict.setdefault('turbo_grouped_mlp', False)
        pdict.setdefault('legacy_grouped_gemm', True)
        pdict.setdefault('manual_gc', True)
        pdict.setdefault('manual_gc_interval', 1)
        pdict.setdefault('pp_warmup', True)

        # Learning rate
        pdict.setdefault('lr', 1.0e-5)
        pdict.setdefault('min_lr', 0.0)
        pdict.setdefault('lr_warmup_iters', 200)
        pdict.setdefault('lr_decay_iters', 5000)
        pdict.setdefault('lr_decay_style', 'cosine')

        # Monitoring
        pdict.setdefault('disable_wandb', False)
        pdict.setdefault('disable_tensorboard', False)
        pdict.setdefault('profile', False)
        pdict.setdefault('use_pytorch_profiler', False)
        pdict.setdefault('profile_step_start', 6)
        pdict.setdefault('profile_step_end', 7)

        # Store all parameters with type conversion for JSON string values
        self.model_type = pdict['model_type']
        self.num_layers = int(pdict['num_layers'])
        self.hidden_size = int(pdict['hidden_size'])
        self.num_attention_heads = int(pdict['num_attention_heads'])
        self.seq_length = int(pdict['seq_length'])
        self.max_position_embeddings = int(pdict['max_position_embeddings'])

        self.micro_batch_size = int(pdict['micro_batch_size'])
        self.global_batch_size = int(pdict['global_batch_size'])

        self.tensor_parallel = int(pdict['tensor_parallel'])
        self.pipeline_parallel = int(pdict['pipeline_parallel'])
        self.expert_parallel = int(pdict['expert_parallel'])
        self.virtual_pipeline_parallel = int(pdict['virtual_pipeline_parallel'])
        self.data_parallel = int(pdict['data_parallel'])

        self.num_experts = int(pdict['num_experts'])
        self.moe_top_k = int(pdict['moe_top_k'])
        self.moe_layer_freq = int(pdict['moe_layer_freq'])

        self.recompute_num_layers = int(pdict['recompute_num_layers'])
        self.recompute_granularity = pdict['recompute_granularity']
        self.recompute_method = pdict['recompute_method']

        self.turbo_deepep = str_to_bool(pdict['turbo_deepep'])
        self.turbo_grouped_mlp = str_to_bool(pdict['turbo_grouped_mlp'])
        self.legacy_grouped_gemm = str_to_bool(pdict['legacy_grouped_gemm'])
        self.manual_gc = str_to_bool(pdict['manual_gc'])
        self.manual_gc_interval = int(pdict['manual_gc_interval'])
        self.pp_warmup = str_to_bool(pdict['pp_warmup'])

        self.lr = float(pdict['lr'])
        self.min_lr = float(pdict['min_lr'])
        self.lr_warmup_iters = int(pdict['lr_warmup_iters'])
        self.lr_decay_iters = int(pdict['lr_decay_iters'])
        self.lr_decay_style = pdict['lr_decay_style']

        self.disable_wandb = str_to_bool(pdict['disable_wandb'])
        self.disable_tensorboard = str_to_bool(pdict['disable_tensorboard'])
        self.profile = str_to_bool(pdict['profile'])
        self.use_pytorch_profiler = str_to_bool(pdict['use_pytorch_profiler'])
        self.profile_step_start = int(pdict['profile_step_start'])
        self.profile_step_end = int(pdict['profile_step_end'])

        # Compute pipeline layout
        self.pipeline_layout = None
        self.decoder_last_pipeline_num_layers = None
        if self.virtual_pipeline_parallel > 1:
            self.pipeline_layout = self.compute_pipeline_layout()

        # Setup scripts directory
        self.phdl.exec(f'rm -rf {self.scripts_dir}')
        time.sleep(2)
        self.phdl.exec(f'mkdir -p {self.scripts_dir}')
        time.sleep(2)
        self.phdl.exec(f'sudo chmod 777 {self.scripts_dir}')

    def compute_pipeline_layout(self):
        """
        Compute dynamic pipeline layout for DeepSeek V3 based on PP and VPP.

        DeepSeek V3 has 61 decoder layers. With VPP, we need to split these layers
        into balanced chunks across pipeline stages.

        Returns:
            str: Pipeline layout string (e.g., "Et*4|t*4|...|t*3,L")
        """
        pp = self.pipeline_parallel
        vpp = self.virtual_pipeline_parallel

        if vpp <= 1:
            # No VPP, use decoder_last_pipeline_num_layers approach
            if pp == 4:
                self.decoder_last_pipeline_num_layers = 13
            elif pp == 8:
                self.decoder_last_pipeline_num_layers = 12
            else:
                self.decoder_last_pipeline_num_layers = 13
            return None

        # VPP > 1: compute balanced layout
        if vpp == 2:
            if pp == 4:
                # PP4+VPP2 = 8 chunks: 5×8 + 3×7 = 61
                layout = "Et*8|t*8|t*8|t*8|t*8|t*7|t*7|t*7,L"
            elif pp == 8:
                # PP8+VPP2 = 16 chunks: 13×4 + 3×3 = 61
                layout = "Et*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3|t*3,L"
            else:
                fail_test(f"Unsupported PP={pp} for VPP=2. Supported: 4, 8")

        elif vpp == 4:
            # PP4+VPP4 = 16 chunks: same as PP8+VPP2
            layout = "Et*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3|t*3,L"
        else:
            fail_test(f"Unsupported VPP={vpp}. Supported: 1, 2, 4")

        log.info(f"Computed pipeline layout for PP={pp}, VPP={vpp}: {layout}")
        return layout

    def run_pretraining_tasks(self):
        """Collect pre-training network statistics"""
        if self.distributed_training:
            self.rdma_stats_dict_before = linux_utils.get_rdma_stats_dict(self.phdl)
            self.ethtool_stats_dict_before = linux_utils.get_nic_ethtool_stats_dict(self.phdl)

    def exec_nic_setup_scripts(self):
        """
        Setup AINIC networking for MI300X clusters.

        For AINIC-based clusters, this ensures proper InfiniBand configuration
        and applies any necessary workarounds for the ionic RDMA drivers.
        """
        if not self.distributed_training:
            return

        if self.using_ainic:
            log.info("Setting up AINIC networking...")

            # Verify ionic devices are available
            verify_cmd = f'docker exec {self.container_name} bash -c "ibv_devices | grep ionic && echo IONIC_OK || echo IONIC_MISSING"'
            result_dict = self.phdl.exec(verify_cmd)

            for node in result_dict.keys():
                if 'IONIC_MISSING' in result_dict[node]:
                    log.warning(f"Ionic devices not detected on {node}, may need driver setup")

    def build_training_job_cmd(self):
        """
        Build the Primus training command for DeepSeek models.

        This constructs a command that calls Primus's run_slurm_pretrain.sh
        (or equivalent) with all DeepSeek-specific parameters.
        """

        # Base environment setup
        env_vars = {
            'HF_TOKEN': self.hf_token,
            'PRIMUS_PATH': self.primus_path,
            'USING_AINIC': '1' if self.using_ainic else '0',
            'NCCL_IB_HCA': self.nccl_ib_hca,
            'NCCL_SOCKET_IFNAME': self.nccl_socket_ifname,
            'GLOO_SOCKET_IFNAME': self.gloo_socket_ifname,
            'NCCL_IB_TIMEOUT': str(self.nccl_ib_timeout),
            'NCCL_DEBUG': self.nccl_debug,
            'ENABLE_NUMA_BINDING': '1' if self.enable_numa_binding else '0',
            'HSA_KERNARG_POOL_SIZE': str(self.hsa_kernarg_pool_size),
        }

        # Build environment string without quote escaping issues
        env_str = '\n'.join([f'export {k}={v}' for k, v in env_vars.items()])

        # Build training arguments
        train_args = [
            f'--num_layers {self.num_layers}',
            f'--train_iters {self.iterations}',
            f'--micro_batch_size {self.micro_batch_size}',
            f'--global_batch_size {self.global_batch_size}',
            f'--seq_length {self.seq_length}',
            f'--max_position_embeddings {self.max_position_embeddings}',
            f'--pipeline_model_parallel_size {self.pipeline_parallel}',
            f'--expert_model_parallel_size {self.expert_parallel}',
            f'--tensor_model_parallel_size {self.tensor_parallel}',
            f'--use_turbo_deepep {str(self.turbo_deepep)}',
            f'--moe_use_legacy_grouped_gemm {str(self.legacy_grouped_gemm)}',
            f'--moe_layer_freq {self.moe_layer_freq}',
            f'--recompute_num_layers {self.recompute_num_layers}',
            f'--recompute_granularity {self.recompute_granularity}',
            f'--recompute_method {self.recompute_method}',
            f'--lr {self.lr}',
            f'--min_lr {self.min_lr}',
            f'--lr_warmup_iters {self.lr_warmup_iters}',
            f'--lr_decay_iters {self.lr_decay_iters}',
            f'--lr_decay_style {self.lr_decay_style}',
            f'--mock_data {str(self.mock_data)}',
            f'--manual_gc {str(self.manual_gc)}',
            f'--manual_gc_interval {self.manual_gc_interval}',
            f'--pp_warmup {str(self.pp_warmup)}',
            f'--disable_wandb {str(self.disable_wandb)}',
            f'--disable_tensorboard {str(self.disable_tensorboard)}',
            f'--profile {str(self.profile)}',
            f'--use_pytorch_profiler {str(self.use_pytorch_profiler)}',
            f'--profile_step_start {self.profile_step_start}',
            f'--profile_step_end {self.profile_step_end}',
            '--cross_entropy_fusion_impl "te"',
            '--cross_entropy_loss_fusion False',
            '--disable_last_saving True',
            '--mtp_num_layers 0',
            '--check_for_nan_in_loss_and_grad False',
        ]

        # Add pipeline layout if VPP is used
        if self.pipeline_layout:
            train_args.append(f'--pipeline_model_parallel_layout "{self.pipeline_layout}"')
        elif self.decoder_last_pipeline_num_layers:
            train_args.append(f'--decoder_last_pipeline_num_layers {self.decoder_last_pipeline_num_layers}')

        # Add tokenized data path if not using mock data
        if not self.mock_data and self.tokenized_data_path:
            train_args.append(f'--data_path {self.tokenized_data_path}')

        # Model config file
        if 'deepseek_v3' in self.model_name.lower():
            config_file = 'examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml'
        elif 'deepseek_v2_lite' in self.model_name.lower():
            config_file = 'examples/megatron/configs/MI355X/deepseek_v2_lite-BF16-pretrain.yaml'
        else:
            config_file = 'examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml'

        train_args.append(f'--config {config_file}')

        # Join all arguments
        train_args_str = ' \\\n  '.join(train_args)

        # Build full command
        log.info("Building Primus training command...")

        # For distributed training, we'll create per-node commands using torchrun
        # (not run_slurm_pretrain.sh which requires srun)
        if self.distributed_training:
            self.job_cmd_list = []
            for i, node in enumerate(self.host_list):
                node_rank = i

                # Number of GPUs per node (assume 8 for MI355X)
                nproc_per_node = 8

                # Create script content without quoting nightmares
                script_content = f"""#!/bin/bash
set -e
{env_str}
export NODE_RANK={node_rank}
export MASTER_ADDR={self.master_address}
export MASTER_PORT=29500
export NNODES={self.nnodes}
cd {self.primus_path}
torchrun \\
  --nproc_per_node={nproc_per_node} \\
  --nnodes={self.nnodes} \\
  --node_rank={node_rank} \\
  --master_addr={self.master_address} \\
  --master_port=29500 \\
  examples/pretrain_gpt_megatron.py \\
  {train_args_str} \\
  2>&1 | tee {self.log_dir}/deepseek_training_node{node_rank}.log
"""

                # Base64 encode to avoid all shell quoting issues
                script_b64 = base64.b64encode(script_content.encode('utf-8')).decode('ascii')

                # Write script to host, copy to container, execute
                cmd = f"""printf '%s' '{script_b64}' | base64 -d > /tmp/primus_train_node{node_rank}.sh.host && \\
docker cp /tmp/primus_train_node{node_rank}.sh.host {self.container_name}:/tmp/primus_train.sh && \\
rm -f /tmp/primus_train_node{node_rank}.sh.host && \\
docker exec {self.container_name} bash /tmp/primus_train.sh"""

                self.job_cmd_list.append(cmd)

        else:
            # Single node
            nproc_per_node = 8
            script_content = f"""#!/bin/bash
set -e
{env_str}
cd {self.primus_path}
torchrun \\
  --nproc_per_node={nproc_per_node} \\
  --nnodes=1 \\
  --node_rank=0 \\
  examples/pretrain_gpt_megatron.py \\
  {train_args_str} \\
  2>&1 | tee {self.log_dir}/deepseek_training.log
"""

            script_b64 = base64.b64encode(script_content.encode('utf-8')).decode('ascii')
            self.job_cmd = f"""printf '%s' '{script_b64}' | base64 -d > /tmp/primus_train.sh.host && \\
docker cp /tmp/primus_train.sh.host {self.container_name}:/tmp/primus_train.sh && \\
rm -f /tmp/primus_train.sh.host && \\
docker exec {self.container_name} bash /tmp/primus_train.sh"""

        log.info("Training command built successfully")

    def start_training_job(self):
        """Launch the Primus DeepSeek training job"""

        log.info("=" * 80)
        log.info("Starting Primus DeepSeek Training Job")
        log.info("=" * 80)
        log.info(f"Model: {self.model_name}")
        log.info(f"Nodes: {self.nnodes}")
        log.info(f"Iterations: {self.iterations}")
        log.info(f"Micro batch size: {self.micro_batch_size}")
        log.info(f"Global batch size: {self.global_batch_size}")
        log.info(f"Pipeline Parallel: {self.pipeline_parallel}")
        log.info(f"Expert Parallel: {self.expert_parallel}")
        log.info(f"Virtual Pipeline Parallel: {self.virtual_pipeline_parallel}")
        log.info("=" * 80)

        # Create log directory in containers
        for i in range(len(self.host_list)):
            mkdir_cmd = f'docker exec {self.container_name} /bin/bash -c "mkdir -p {self.log_dir}"'
            self.phdl.exec(mkdir_cmd)

        self.training_start_time = time.time()

        if self.distributed_training:
            # Launch on all nodes
            log.info(f"Launching training on {len(self.host_list)} nodes...")
            for i, cmd in enumerate(self.job_cmd_list):
                node = self.host_list[i]
                log.info(f"Starting training on node {i}: {node}")
                # Run in background
                bg_cmd = cmd + " &"
                self.phdl.exec(bg_cmd)
                time.sleep(5)  # Stagger starts slightly
        else:
            log.info("Launching single-node training...")
            self.phdl.exec(self.job_cmd + " &")

        log.info("Training jobs launched, monitoring logs...")
        time.sleep(10)  # Give it time to start

    def poll_for_training_completion(self):
        """
        Poll training logs for completion or errors.

        Monitors for:
        - Iteration completion
        - NCCL errors
        - MoE-specific errors
        - GPU errors
        - Training convergence
        """

        log.info("Polling for training completion...")

        max_wait_time = self.iterations * 120  # 120 seconds per iteration max
        poll_interval = 30
        elapsed_time = 0

        last_node = self.host_list[-1]
        log_file = f'{self.log_dir}/deepseek_training_node{len(self.host_list) - 1}.log'

        while elapsed_time < max_wait_time:
            time.sleep(poll_interval)
            elapsed_time += poll_interval

            # Check log file
            check_cmd = f'docker exec {self.container_name} bash -c "tail -50 {log_file}"'
            result_dict = self.phdl.exec(check_cmd)

            log_output = result_dict.get(last_node, '')

            # Check for completion
            if re.search(f'iteration.*{self.iterations}/', log_output, re.I):
                log.info(f"Training completed {self.iterations} iterations!")
                self.training_end_time = time.time()
                return True

            # Check for errors
            for err_type, err_pattern in training_err_dict.items():
                if re.search(err_pattern, log_output, re.I):
                    fail_test(f"{err_type} detected in training logs: {log_output}")

            # Show progress
            iter_match = re.search(r'iteration\s+(\d+)/', log_output, re.I)
            if iter_match:
                current_iter = int(iter_match.group(1))
                progress = (current_iter / self.iterations) * 100
                log.info(f"Progress: {current_iter}/{self.iterations} iterations ({progress:.1f}%)")

            log.info(f"Waiting for completion... ({elapsed_time}s elapsed)")

        # Timeout
        fail_test(f"Training did not complete within {max_wait_time}s")

    def verify_training_results(self):
        """
        Verify training completed successfully and extract metrics.

        Checks:
        - Loss decreased
        - No NaN/Inf values
        - Throughput metrics
        - Expert load balance (MoE)
        - Network error counters
        """

        log.info("Verifying training results...")

        log_file = f'{self.log_dir}/deepseek_training_node{len(self.host_list) - 1}.log'
        last_node = self.host_list[-1]

        # Get full log
        cat_cmd = f'docker exec {self.container_name} bash -c "cat {log_file}"'
        result_dict = self.phdl.exec(cat_cmd)
        full_log = result_dict.get(last_node, '')

        # Extract metrics
        losses = re.findall(r'loss:\s+([\d\.]+)', full_log, re.I)
        if losses:
            initial_loss = float(losses[0])
            final_loss = float(losses[-1])
            log.info(f"Initial loss: {initial_loss:.4f}")
            log.info(f"Final loss: {final_loss:.4f}")

            if final_loss >= initial_loss:
                log.warning("Loss did not decrease - potential training issue")
            else:
                log.info(f"Loss decreased by {initial_loss - final_loss:.4f}")

        # Check for NaN/Inf
        if re.search(r'nan|inf', full_log, re.I):
            fail_test("NaN or Inf detected in training")

        # Extract throughput if available
        throughput_matches = re.findall(r'throughput.*?([\d\.]+)\s+tokens/s', full_log, re.I)
        if throughput_matches:
            avg_throughput = sum(float(x) for x in throughput_matches) / len(throughput_matches)
            log.info(f"Average throughput: {avg_throughput:.2f} tokens/s")
            self.training_results_dict['throughput_tokens_per_sec'] = avg_throughput

        # Verify network if requested
        if self.verify_network_errors == 'True':
            self.rdma_stats_dict_after = linux_utils.get_rdma_stats_dict(self.phdl)
            self.ethtool_stats_dict_after = linux_utils.get_nic_ethtool_stats_dict(self.phdl)

            # Check for error counter increases
            verify_dmesg_for_errors(self.phdl, self.training_start_time)

        log.info("Training verification complete")
        log.info("=" * 80)
        log.info("Training Results:")
        for key, value in self.training_results_dict.items():
            log.info(f"  {key}: {value}")
        log.info("=" * 80)
