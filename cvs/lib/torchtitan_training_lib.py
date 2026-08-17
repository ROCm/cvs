'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import os
import re
import textwrap
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


# Ordered fallback chains for parsing TorchTitan training output.
# Each chain is tried in order; first non-empty match wins. Single-entry
# lists today, but the structure matches megatron_training_lib.py so adding
# alternate log formats later is a one-line change.
TRAINING_RESULT_PATTERNS = {
    'tokens_per_sec': [r'tps:\s+([0-9,\.]+)', r'tok/s:\s+([0-9\.]+)'],
    'loss': [r'loss:\s+([0-9\.]+)'],
    'mem_usage_gb': [r'memory:\s+([0-9\.]+)\s*GiB', r'mem:\s+([0-9\.]+)\s+GB'],
}

# Completion indicator: TorchTitan emits `step: <N>` where N is the configured
# final iteration count. Populated per-instance because it depends on
# self.iterations; see _is_training_complete.
TRAINING_PROGRESS_PATTERNS_TEMPLATE = [r'step:\s+{iterations}']

# NaN/Inf detection on result lines. NOTE: `[NaN|Inf]` is a character class
# (matches any one of N,a,I,n,f,|) — preserved verbatim from the prior inline
# regex to honor "don't change functionality". Switch to `(NaN|Inf)` to make
# this check actually fire.
TRAINING_NAN_PATTERNS = [
    r'tok/s:\s+[NaN|Inf]',
    r'loss:\s+[NaN|Inf]',
]


def _parse_training_results(output):
    """Extract metric values from training-log text using ordered fallback chains.

    For each metric in TRAINING_RESULT_PATTERNS, try each pattern in order and
    return the first non-empty list of matches. If no pattern matches, the
    metric maps to an empty list.

    Args:
        output (str): Raw training-log text to parse.

    Returns:
        dict: {metric_name: list[str]} for every key in TRAINING_RESULT_PATTERNS.
    """
    out = {}
    for metric, patterns in TRAINING_RESULT_PATTERNS.items():
        out[metric] = []
        for pat in patterns:
            matches = re.findall(pat, output, re.I)
            if matches:
                # TorchTitan emits comma-grouped numbers (e.g. "tps: 4,716");
                # strip so downstream float() doesn't choke.
                out[metric] = [m.replace(',', '') for m in matches]
                break
    return out


def _is_training_complete(output, iterations):
    """Return True if the training-log text shows the configured final step
    matching any pattern in TRAINING_PROGRESS_PATTERNS_TEMPLATE."""
    patterns = [p.format(iterations=iterations) for p in TRAINING_PROGRESS_PATTERNS_TEMPLATE]
    return any(re.search(p, output, re.I) for p in patterns)


def _has_nan_inf_results(output):
    """Return True if the training-log text shows a NaN/Inf result line
    matching any pattern in TRAINING_NAN_PATTERNS."""
    return any(re.search(p, output, re.I) for p in TRAINING_NAN_PATTERNS)


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
        """
        Initialize job configuration and resolve defaults from the provided dicts.

        - Normalizes training_config_dict and model_params_dict; applies defaults
          if fields are missing.
        - Builds paths and internal state used later to construct torchrun commands.

        Args:
          phdl: Remote execution handle for multi-node command execution.
          model_name: Canonical model name key used in model_params_dict
            (e.g., "llama3_1_8b", "deepseek_v3", "qwen3").
          training_config_dict: Unstructured training config; defaults are applied here.
          model_params_dict: Parameter sets per model and topology (single/multi-node).
          hf_token: Hugging Face token passed to the job environment.
          gpu_type: GPU platform key to select model params (default: 'mi350').
          distributed_training: If True, build multi-node torchrun launchers.
          tune_model_params: If True, adjust global_batch_size based on cluster size.
          scripts_dir: Optional override for the per-node folder where generated
            wrapper scripts are placed. When None (default), the value is read
            from training_config_dict['scripts_dir'], which itself defaults to
            ``{self.home_dir}/SCRIPTS``.
        """
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
        tdict.setdefault('hca_id_pattern', 'bnxt_|rocep')
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
        tdict.setdefault('torchtitan_root', '/workspace/torchtitan')
        # 'True'  -> generate a TOML from JSON model_params and point torchrun at it
        # 'False' -> use the canned TOML shipped with TorchTitan at
        #            ./torchtitan/models/{tt_module}/train_configs/{tt_config}
        tdict.setdefault('use_generated_config', 'True')

        self.container_image = tdict['container_image']
        self.container_name = tdict['container_name']
        self.distributed_training = distributed_training
        self.iterations = int(tdict['training_iterations'])
        self.nnodes = int(tdict['nnodes'])
        self.nic_type = tdict['nic_type']
        self.hca_id_pattern = tdict['hca_id_pattern']
        self.nccl_ib_hca_list = tdict['nccl_ib_hca_list']
        self.nccl_ib_hca = tdict['nccl_ib_hca']
        self.nccl_socket_ifname = tdict['nccl_socket_ifname']
        self.gloo_socket_ifname = tdict['gloo_socket_ifname']
        self.nccl_ib_gid_index = tdict['nccl_ib_gid_index']
        self.nccl_debug = tdict['nccl_debug']
        self.data_cache_dir = tdict['data_cache_dir']
        self.log_dir = tdict['log_dir']
        # kwarg wins over training_dict so direct callers (tests, custom harnesses)
        # can still override; falls back to the setdefault'd dict value otherwise.
        self.scripts_dir = scripts_dir if scripts_dir is not None else tdict['scripts_dir']
        self.master_address = tdict['master_address']
        self.verify_network_errors = tdict['verify_network_errors']
        self.rocm_path = detect_rocm_path(self.phdl, tdict['rocm_dir'])
        self.use_generated_config = tdict['use_generated_config']
        self.torchtitan_root = tdict['torchtitan_root']

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

        # Model params - set defaults if not defined in input json file
        pdict.setdefault('tokenizer_path', 'meta-llama/Llama-3.1-8B')
        pdict.setdefault('model_size', '8b')
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
        # New fields driving the generated TOML (all values stored as strings per
        # CVS config convention; lib classifies emit kind per field).
        pdict.setdefault('hf_assets_path', './assets/hf')
        pdict.setdefault('converters', '["float8"]')
        pdict.setdefault('dataset', 'c4')
        pdict.setdefault('lr', '8e-5')
        pdict.setdefault('warmup_steps', '600')
        pdict.setdefault('enable_async_tensor_parallel', 'true')
        pdict.setdefault('precompute_float8_dynamic_scale_for_fsdp', 'true')

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
        self.hf_assets_path = pdict['hf_assets_path']
        self.converters = pdict['converters']
        self.dataset = pdict['dataset']
        self.lr = pdict['lr']
        self.warmup_steps = pdict['warmup_steps']
        self.enable_async_tensor_parallel = pdict['enable_async_tensor_parallel']
        self.precompute_float8_dynamic_scale_for_fsdp = pdict['precompute_float8_dynamic_scale_for_fsdp']

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
        """
        Snapshot per-node RDMA and ethtool counters before training starts.

        Behavior:
          - Only runs when distributed_training is True (single-node skips network
            stat collection entirely).
          - Stores pre-training counters on self.rdma_stats_dict_before and
            self.ethtool_stats_dict_before for later diff in verify_training_results.

        Assumptions:
          - linux_utils.get_rdma_stats_dict / get_nic_ethtool_stats_dict return
            {node: {counter: value}} mappings.
        """
        if self.distributed_training is True:
            self.rdma_stats_dict_before = linux_utils.get_rdma_stats_dict(self.phdl)
            self.ethtool_stats_dict_before = linux_utils.get_nic_ethtool_stats_dict(self.phdl)

    def exec_nic_setup_scripts(self):
        """
        Prepare backend NICs inside containers before starting distributed training.

        Behavior:
          - Only runs when distributed_training is True.
          - If nic_type indicates Broadcom/Thor, it:
            * Forces NCCL GID index to 3 (common Broadcom requirement).
            * Copies the host-side libbnxt_re library into the container's ibverbs path.
            * Runs ibv_devinfo to verify the RDMA device enumerates against hca_id_pattern.
            * Fails the test if the expected device string is not detected on any node.

        Assumptions:
          - self.phdl provides exec(...) to run commands on all nodes/hosts.
          - Docker is installed and the container is already running on each node.
          - fail_test(...) is available in scope to abort on setup failures.
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
                # Treat `hca_id_pattern` as a `|`-separated list of literal
                # NIC-name prefixes. Each segment is `re.escape`d so users
                # can't accidentally inject regex syntax (e.g. `mlx5+` is a
                # literal 5-char prefix, not `mlx` + `5+` quantifier).
                segments = [re.escape(s.strip()) for s in self.hca_id_pattern.split('|') if s.strip()]
                if not segments:
                    fail_test(
                        f'hca_id_pattern parsed to zero non-empty segments, got: {self.hca_id_pattern!r}. '
                        f'Expected a `|`-separated list of NIC-name prefixes, e.g. "bnxt_|rocep".'
                    )
                hca_id_regex = rf'hca_id:\s+({"|".join(segments)})'
                for node in out_dict.keys():
                    if not re.search(hca_id_regex, out_dict[node], re.I):
                        log.info("%s", out_dict[node])
                        fail_test(f'Broadcom libbnxt rdma driver is not properly copied on node {node}')

    def _build_generated_toml(self):
        """
        Render a TorchTitan job-config TOML from JSON model_params.

        Section/key layout follows the upstream TorchTitan preset; values come
        from self.* (sourced from JSON model_params) or are kept as TorchTitan
        defaults where the JSON doesn't expose a knob (e.g. dtype, name,
        filter_fqns, comm timeout).

        Extras beyond the upstream preset (kept for CVS reproducibility):
          - [model]    tokenizer_path
          - [training] global_batch_size
          - [parallelism] data_parallel_shard_degree, expert_parallel_degree

        Booleans (compile, enable_float8, enable_async_tensor_parallel,
        precompute_float8_dynamic_scale_for_fsdp) are stored as 'true'/'false'
        strings in JSON and lowercased here for TOML literal form.
        """
        compile_lower = str(self.compile).lower()
        float8_lower = str(self.enable_float8).lower()
        async_tp_lower = str(self.enable_async_tensor_parallel).lower()
        float8_precompute_lower = str(self.precompute_float8_dynamic_scale_for_fsdp).lower()
        return textwrap.dedent(f"""\
            [model]
            name = "{self.tt_module}"
            flavor = "{self.model_size.upper()}"
            hf_assets_path = "{self.hf_assets_path}/{self.tokenizer_path.rsplit('/', 1)[-1]}"
           
            

            [training]
            dataset = "{self.dataset}"
            local_batch_size = {self.micro_batch_size}
            global_batch_size = {self.global_batch_size}
            seq_len = {self.sequence_length}
            steps = {self.iterations}
            dtype = "bfloat16"

            [optimizer]
            lr = {self.lr}

            [lr_scheduler]
            warmup_steps = {self.warmup_steps}

            [parallelism]
            data_parallel_shard_degree = {self.data_parallel_shard_degree}
            tensor_parallel_degree = {self.tensor_parallel_degree}
            pipeline_parallel_degree = {self.pipeline_parallel_degree}
            context_parallel_degree = {self.context_parallel_degree}
            expert_parallel_degree = {self.expert_parallel_degree}
            enable_async_tensor_parallel = {async_tp_lower}

            [activation_checkpoint]
            mode = "{self.activation_checkpointing}"

            [compile]
            enable = {compile_lower}

            [quantize.linear.float8]
            enable_fsdp_float8_all_gather = {float8_lower}
            precompute_float8_dynamic_scale_for_fsdp = {float8_precompute_lower}
            filter_fqns = ["output"]

            [comm]
            init_timeout_seconds = 600
            """)

    def _write_generated_toml(self, dest_path):
        """
        Push the generated TOML to `dest_path` on every node via a quoted
        heredoc (no shell expansion). Relies on scripts_dir being on a
        bind-mounted host path so the same path is visible inside the container.
        """
        toml_str = self._build_generated_toml()
        log.info('Generated TorchTitan TOML config:\n%s', toml_str)
        # 'CVS_TOML_EOF' quoted -> bash does not expand $vars or backticks.
        # Sentinel chosen to not collide with TOML content.
        write_cmd = f"cat > {dest_path} <<'CVS_TOML_EOF'\n{toml_str}\nCVS_TOML_EOF"
        self.phdl.exec(write_cmd)

    def download_hf_assets(self):
        """
        Download Hugging Face assets (tokenizer + model) synchronously on
        every node. Blocks until the download finishes on all nodes, so the
        subsequent torchrun launch can assume assets are in place.

        Path mirrors what build_training_job_cmd would have used inline:
          - use_generated_config == 'True'  -> --local_dir self.hf_assets_path
          - use_generated_config == 'False' -> --local_dir ./assets/hf/
        The download script itself appends a per-model subdir
        (e.g. './assets/hf/Qwen3-32B'); see _build_generated_toml for the
        matching path used in the generated TOML.

        Runs per-node (paths are container-local, not NFS-shared in the
        current setup); idempotent — download_hf_assets.py skips files
        already present, so re-runs are cheap.
        """
        local_dir = self.hf_assets_path if self.use_generated_config == 'True' else './assets/hf/'
        download_cmd = (
            f'docker exec {self.container_name} /bin/bash -c '
            f'"cd {self.torchtitan_root} && '
            f'python scripts/download_hf_assets.py --repo_id {self.tokenizer_path} '
            f'--assets tokenizer --all --hf_token={self.hf_token} --local_dir {local_dir}"'
        )
        log.info('Downloading HF assets for %s (may take a while for large models)...', self.tokenizer_path)
        self.phdl.exec(download_cmd)
        log.info('HF asset download complete')

    def build_training_job_cmd(self):
        """
        Construct native TorchTitan training command using torchrun.

        Two job-config sources, selected by self.use_generated_config:
          - 'True'  -> generate a TOML from JSON model_params (written to
            {scripts_dir}/run_config.toml on every node) and pass it to
            --job.config_file. All run-tuning knobs come from JSON.
          - 'False' -> use the canned TOML shipped with TorchTitan at
            ./torchtitan/models/{tt_module}/train_configs/{tt_config}.
            Customer tunes by editing that TOML directly.
        """
        cmd = f'cd {self.torchtitan_root}; export HF_TOKEN={self.hf_token}; '
        cmd += 'export HSA_FORCE_FINE_GRAIN_PCIE=1; '
        cmd += 'export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True; '
        # Path is inlined into torchrun below (not exported as $CONFIG_FILE).
        # Reason: start_training_job uses double-quoted `echo` to write the
        # wrapper, which would expand `$CONFIG_FILE` against the SSH session's
        # env at echo-time -- silently overriding our export if the host
        # environment already has CONFIG_FILE set (e.g. baked into the
        # container image or the user's shell init).
        if self.use_generated_config == 'True':
            config_file_path = f'{self.scripts_dir}/run_config.toml'
            self._write_generated_toml(config_file_path)
        else:
            config_file_path = f'./torchtitan/models/{self.tt_module}/train_configs/{self.tt_config}.toml'

        if self.distributed_training is True:
            cmd += (
                f'export NCCL_IB_HCA={self.nccl_ib_hca_list}; '
                + f'export NCCL_SOCKET_IFNAME={self.nccl_socket_ifname}; '
                + f'export GLOO_SOCKET_IFNAME={self.gloo_socket_ifname}; '
                + f'export NCCL_DEBUG={self.nccl_debug}; '
                + f'export NCCL_IB_GID_INDEX={self.nccl_ib_gid_index}; '
            )

        nproc_per_node = 8

        if self.distributed_training:
            for i in range(self.nnodes):
                torchrun_cmd = (
                    f'torchrun --nnodes {self.nnodes} --node_rank={i} --nproc_per_node {nproc_per_node}'
                    f' --rdzv_id 101 --rdzv_backend c10d'
                    f' --rdzv_endpoint "{self.master_address}:29500"'
                    f' --role rank --tee 3'
                    f' -m torchtitan.train --job.config_file {config_file_path}'
                )
                # Truncate any stale training.log from a prior run before
                # torchrun opens it. download_hf_assets() runs synchronously
                # before this script executes, so torchrun is the only thing
                # writing to log_path now.
                log_path = f'{self.log_dir}/torchtitan-logs/out-node{i}/training.log'
                # Wrap the chain in `nohup sh -c '...' & disown` so it
                # survives the wrapper bash exiting. Without this, the
                # backgrounded chain dies when `docker exec` returns because
                # its parent bash also exits, leaving the chain unreparented
                # and reaped.
                inner_chain = f'{torchrun_cmd} > {log_path} 2>&1'
                full_cmd = (
                    cmd + f': > {log_path}; ' + f"nohup sh -c '{inner_chain}' </dev/null >/dev/null 2>&1 & disown"
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
                f' -m torchtitan.train --job.config_file {config_file_path}'
            )
            log_path = f'{self.log_dir}/torchtitan-logs/out-node0/training.log'
            inner_chain = f'{torchrun_cmd} > {log_path} 2>&1'
            self.job_cmd = (
                cmd + f': > {log_path}; ' + f"nohup sh -c '{inner_chain}' </dev/null >/dev/null 2>&1 & disown"
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
            cmd = (
                f'docker exec {self.container_name} /bin/bash -c "mkdir -p {self.log_dir}/torchtitan-logs/out-node{i}"'
            )
            cmd_list.append(cmd)
        self.phdl.exec_cmd_list(cmd_list)

        if self.distributed_training:
            self.exec_nic_setup_scripts()
            self.phdl.exec_cmd_list(self.job_cmd_list)

            cmd_list = []
            for i in range(self.nnodes):
                cmd = (
                    f'docker exec {self.container_name} /bin/bash {self.scripts_dir}/distributed_wrapper_script_{i}.sh'
                )
                cmd_list.append(cmd)
            self.phdl.exec_cmd_list(cmd_list)
        else:
            self.phdl.exec(
                f'echo "{self.job_cmd}" > {self.scripts_dir}/single_node_wrapper_script.sh; '
                f'chmod 777 {self.scripts_dir}/single_node_wrapper_script.sh'
            )
            self.phdl.exec(
                f'docker exec {self.container_name} /bin/bash {self.scripts_dir}/single_node_wrapper_script.sh'
            )

        time.sleep(50)

    def get_training_results_dict(self):
        """
        Parse the last node's training log and extract TorchTitan metrics.

        TorchTitan log format: "step: X, loss: Y, tok/s: Z, mem: W GB"

        Returns:
            dict: Extracted metric lists (one entry per regex match):
              - 'tokens_per_sec': Matches 'tok/s: <float>'
              - 'loss':           Matches 'loss: <float>'
              - 'mem_usage_gb':   Matches 'mem: <float> GB'

        Behavior:
          - Reads the tail of the training log on the last node (authoritative
            per project convention).
          - Delegates regex extraction to _parse_training_results so the metric
            set is config-table driven.

        Assumptions:
          - self.phdl.exec(cmd) returns {host: stdout_str}.
          - self.host_list is non-empty; the last entry contains the final log.
        """
        last_node = self.host_list[-1]
        last_node_num = len(self.host_list) - 1
        out_dict = self.phdl.exec(f'cat {self.log_dir}/torchtitan-logs/out-node{last_node_num}/training.log | tail -20')
        output = out_dict[last_node]

        log.info('Extracting results from logs')
        log.info('#===========================#')
        log.info("%s", output)
        log.info('#===========================#')

        training_results_dict = _parse_training_results(output)

        log.info("%s", training_results_dict)
        return training_results_dict

    def scan_for_training_errors(self):
        """
        Scan the consolidated training logs for known error patterns.

        Returns:
            bool: True if no error patterns are found; False otherwise.

        Behavior:
          - Reads the training log file from the last node (authoritative).
          - Iterates through regex patterns in training_err_dict and searches
            the log content.
          - On first match: calls fail_test, logs an abort message, and sets
            training_pass to False. Does not short-circuit; continues scanning
            so all matched patterns get recorded.

        Assumptions:
          - self.phdl.exec(cmd) returns {host: stdout_str}.
          - training_err_dict is a module-level dict of name -> regex.
          - sudo can read the training log without an interactive prompt.

        Notes:
          - Regex search is case-sensitive as written.
        """
        log.info('Scan for training errors')
        training_pass = True

        last_node = self.host_list[-1]
        last_node_num = len(self.host_list) - 1

        out_dict = self.phdl.exec(f'sudo cat {self.log_dir}/torchtitan-logs/out-node{last_node_num}/training.log')
        output = out_dict[last_node]

        for err_key in training_err_dict:
            if re.search(f'{training_err_dict[err_key]}', output):
                fail_test(f'ERROR {training_err_dict[err_key]} seen in training logs ..')
                log.error('Aborting training log polling')
                training_pass = False
        return training_pass

    def poll_for_training_completion(self, time_between_iters=120):
        """
        Periodically poll training logs to detect completion, surface errors,
        and validate results.

        Args:
            time_between_iters (int | float): Seconds to sleep between each
                polling iteration.

        Behavior:
          - Waits an initial 80s to allow training to start producing logs.
          - For up to self.iterations + 10 loops:
            * Invokes scan_for_training_errors(); aborts if it flags errors.
            * Reads the consolidated training log from the last node.
            * Checks for completion via _is_training_complete (looks for
              `step: {self.iterations}` line).
          - If not seen: logs in-progress and sleeps time_between_iters.
          - If seen: verifies via _has_nan_inf_results that metric lines are
            not NaN/Inf, then parses and stores results.

        Notes:
          - TorchTitan completion detection looks for `step: {iterations}`.
          - The NaN/Inf check uses the legacy `[NaN|Inf]` character-class
            pattern preserved verbatim from prior behavior (see
            TRAINING_NAN_PATTERNS comment).
        """
        log.info('Poll for training completion ..')
        # Download is now done synchronously in download_hf_assets() before
        # the wrapper script runs, so torchrun starts immediately. Short
        # warmup gives torchrun time to write its first log line.
        time.sleep(30)

        last_node = self.host_list[-1]
        last_node_num = len(self.host_list) - 1

        for i in range(1, int(self.iterations) + 10):
            log.info(f'Starting Iteration {i}')

            if not self.scan_for_training_errors():
                fail_test('Failures seen in training logs, Aborting!!!')
                return

            out_dict = self.phdl.exec(f'sudo cat {self.log_dir}/torchtitan-logs/out-node{last_node_num}/training.log')
            output = out_dict[last_node]

            if not _is_training_complete(output, self.iterations):
                log.info('Training still in progress - final step not yet reached')
            else:
                if _has_nan_inf_results(output):
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
        Validate collected training results and environment health after a run.

        Behavior:
          - Records the training end time for dmesg time-bounded scanning.
          - Fails if training_results_dict is empty.
          - Scans every parsed metric for NaN/Inf and fails on any match.
          - Distributed only: collects post-training RDMA + ethtool stats and
            fails if any error counter (matching err_counters_pattern) went up
            vs. the pre-training baseline.
          - Scans dmesg between training_start_time and training_end_time.
          - Compares each observed metric against the threshold in
            self.expected_result_dict; fails on any node below threshold.

        Assumptions:
          - self.phdl.exec returns {node: stdout_str}.
          - self.verify_network_errors is the string 'True' or 'False' (per
            CVS config convention).
          - self.expected_result_dict supplies numeric thresholds keyed by
            the same names as TRAINING_RESULT_PATTERNS.

        Side effects:
          - Accumulates failures via fail_test; does not raise.
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
