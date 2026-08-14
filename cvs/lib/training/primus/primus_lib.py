'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import os
import re
import shlex
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
    'primus': r'primus launcher exited with code [^0]',
}

err_counters_pattern = 'err|retransmit|drop|discard|naks|invalid|oflow|out_of_buffer|reset|fail'


# Primus iteration logs have two formats depending on the iteration:
#   - Warmup  (iter 1-2): single value  "throughput per GPU (TFLOP/s/GPU): 72.7"
#   - Steady  (iter 3+):  current/avg   "throughput per GPU (TFLOP/s/GPU): 491.8/492.9"
#
# Primary patterns capture the running-average (second number in X/Y) from the last
# steady-state iteration — this avg is computed by Primus across all steady-state
# iterations and naturally excludes warmup, so `float(v[-1])` in test_metric gives
# the best single summary value.
# Single-value fallback handles runs that have only warmup lines in the tail.
TRAINING_RESULT_PATTERNS = {
    'throughput_per_gpu': [
        r'throughput per GPU \(TFLOP/s/GPU\):\s+[0-9.]+/([0-9.]+)',   # avg from X/Y (iter 3+)
        r'throughput per GPU \(TFLOP/s/GPU\):\s+([0-9.]+)(?:\s|$|\|)',  # single value (iter 1-2)
    ],
    'tokens_per_gpu': [
        r'tokens/s/GPU inst/harmonic mean:\s+[0-9.]+/([0-9.]+)',      # harmonic mean from X/Y (iter 3+)
    ],
    'elapsed_time_per_iteration': [
        r'elapsed time per iteration \(ms\):\s+[0-9.]+/([0-9.]+)',    # avg from X/Y (iter 3+)
        r'elapsed time per iteration \(ms\):\s+([0-9.]+)(?:\s|$|\|)',   # single value (iter 1-2)
    ],
    'lm_loss': [
        r'lm loss:\s+([0-9.E+\-]+)',
    ],
    'grad_norm': [
        r'grad norm:\s+([0-9.]+)',
    ],
    'hip_mem_usage_ratio': [
        r'hip mem usage/free/total/usage_ratio:\s+[0-9.]+GB/[0-9.]+GB/[0-9.]+GB/([0-9.]+)%',
    ],
    'rocm_mem_usage_ratio': [
        r'rocm mem usage/free/total/usage_ratio:\s+[0-9.]+GB/[0-9.]+GB/[0-9.]+GB/([0-9.]+)%',
    ],
}

# Per-iteration patterns for computing mean across all steady-state iterations
# from the full log. Used as fallback in _parse_training_results when primary
# tail patterns return no match.
TRAINING_ITERATION_PATTERNS = {
    'throughput_per_gpu': r'throughput per GPU \(TFLOP/s/GPU\):\s+([0-9.]+)',
    'tokens_per_gpu': r'tokens/s/GPU inst/harmonic mean:\s+([0-9.]+)',
    'elapsed_time_per_iteration': r'elapsed time per iteration \(ms\):\s+([0-9.]+)',
    'lm_loss': r'lm loss:\s+([0-9.E+\-]+)',
    'grad_norm': r'grad norm:\s+([0-9.]+)',
    'hip_mem_usage_ratio': r'hip mem usage/free/total/usage_ratio:\s+[0-9.]+GB/[0-9.]+GB/[0-9.]+GB/([0-9.]+)%',
    'rocm_mem_usage_ratio': r'rocm mem usage/free/total/usage_ratio:\s+[0-9.]+GB/[0-9.]+GB/[0-9.]+GB/([0-9.]+)%',
}

# Matches both log formats for progress detection (no capture group needed).
TRAINING_PROGRESS_PATTERNS = [
    r'throughput per GPU \(TFLOP/s/GPU\):\s+[0-9.]+(?:/[0-9.]+)?',
    r'tokens/s/GPU inst/harmonic mean:\s+[0-9.]+(?:/[0-9.]+)?',
]

TRAINING_NAN_PATTERNS = [
    r'throughput per GPU \(TFLOP/s/GPU\):\s+(?:NaN|Inf)',
    r'tokens/s/GPU inst/harmonic mean:\s+(?:NaN|Inf)',
    r'lm loss:\s+(?:NaN|Inf)',
]


def _parse_mean_from_iterations(log_text, pattern, skip_warmup=True):
    """Parse per-iteration metric values from full log and return their mean.

    Args:
        log_text:     Full training log text.
        pattern:      Regex with one capture group for the numeric value.
        skip_warmup:  If True and more than one value found, drop the first match
                      (iteration 1 is artificially slow due to JIT compilation).

    Returns:
        Mean value as a string, or None if no values were found.
    """
    matches = re.findall(pattern, log_text, re.I)
    if not matches:
        return None
    values = [float(m) for m in matches]
    if skip_warmup and len(values) > 1:
        values = values[1:]
    return str(sum(values) / len(values))


def _parse_training_results(output, full_log=None):
    """Extract metric values from Primus training-log text.

    Primary: tries each pattern in TRAINING_RESULT_PATTERNS against `output`
    (tail of the log). For steady-state iterations the patterns capture the
    running-average column (X/Y → Y) from the last iteration line.

    Fallback: when a metric is still empty and `full_log` is provided, computes
    the mean of all per-iteration values from the full log using
    TRAINING_ITERATION_PATTERNS, skipping the first (warmup) iteration.

    Args:
        output (str):        Tail of the training log.
        full_log (str|None): Full training log for per-iteration fallback.

    Returns:
        dict: {metric_name: list[str]} for every key in TRAINING_RESULT_PATTERNS.
    """
    out = {}
    for metric, patterns in TRAINING_RESULT_PATTERNS.items():
        out[metric] = []
        for pat in patterns:
            matches = re.findall(pat, output, re.I)
            if matches:
                out[metric] = matches
                break
        if not out[metric] and full_log is not None:
            pattern = TRAINING_ITERATION_PATTERNS.get(metric)
            if pattern:
                mean = _parse_mean_from_iterations(full_log, pattern, skip_warmup=True)
                if mean:
                    out[metric] = [mean]
                    log.info('per-iteration fallback: %s = %s', metric, mean)
    return out


def _is_training_complete(output, total_iters):
    """Return True when training is complete.

    Two signals are checked (either is sufficient):
      1. Final iteration N/N line emitted by the megatron/torchtitan backend.
      2. "torchrun finished successfully" written by primus-cli after torchrun exits.
         This is the definitive end-of-run marker and always appears last in the log.
    """
    n = int(total_iters)
    if re.search(rf'iteration\s+{n}\s*/\s*{n}\b', output):
        return True
    if re.search(r'torchrun finished successfully', output, re.I):
        return True
    return False


def _has_nan_inf_results(output):
    """Return True if the training log contains any NaN/Inf result line."""
    return any(re.search(p, output, re.I) for p in TRAINING_NAN_PATTERNS)


class PrimusTrainingJob:
    """
    Orchestrates a Primus training job (megatron / torchtitan / jax backend)
    across one or more nodes.

    Primus launch sequence per node:
        cd <primus_root> &&
        bash runner/primus-cli direct \\
          --log_file <combo_log_dir>/out-node{rank}/training.log \\
          -- train pretrain \\
          --config examples/<primus_framework>/configs/<gpu_arch>/<model_name>-<precision>-pretrain.yaml

    The EXP config path is auto-constructed from variant_config fields:
        primus_framework  — config key, e.g. "torchtitan", "megatron", "jax"
        gpu_arch          — variant_config.gpu_arch, e.g. "MI355X"
        model_name        — variant_config.model_params["model_name"], e.g. "llama3.1_8B"
        precision         — sweep combo precision, e.g. "FP8"

    All model hyperparameters (MBS, GBS, TP, PP, etc.) live inside the Primus
    EXP YAML — they are NOT injected as env vars by this class.
    The sweep-level micro_batch_size / global_batch_size / precision parameters
    are accepted for interface compatibility and used only in run_label and
    result verification, not in the launch command.

    Required config keys (under variant_config.config):
        primus_framework  (str)  Backend framework: "torchtitan", "megatron", or "jax".

    Optional config keys:
        primus_root  (str)  Path to Primus workspace inside container.
                            Default: /workspace/Primus
        primus_cli   (str)  Path to primus-cli relative to primus_root.
                            Default: runner/primus-cli

    All other keys (log_dir, nic_type, nccl_*, ...) are identical to
    MegatronTrainingJob.
    """

    def __init__(
        self,
        orch,
        variant_config,
        hf_token,
        micro_batch_size,
        global_batch_size,
        primus_framework=None,
        precision=None,
        distributed_training=False,
        tune_model_params=False,
        scripts_dir=None,
        run_label=None,
    ):
        self.orch = orch
        self.model_name = variant_config.model_params["model_name"]
        self.gpu_arch = variant_config.gpu_arch
        self.hf_token = hf_token
        self.tune_model_params = tune_model_params

        self.job_cmd = ''
        self.job_cmd_list = []
        self.training_results_dict = {}
        self.local_tokenizer_path = None

        self.rdma_stats_dict_before = {}
        self.ethtool_stats_dict_before = {}
        self.rdma_stats_dict_after = {}
        self.ethtool_stats_dict_after = {}
        self.training_start_time = self.orch.all.exec('date')
        self.training_end_time = None

        self.home_dir = os.path.expanduser("~")
        tdict = dict(variant_config.config)
        tdict.setdefault('training_iterations', 10)
        tdict.setdefault('nnodes', '1')
        tdict.setdefault('nic_type', 'thor2')
        tdict.setdefault('hca_id_pattern', 'bnxt_|rocep')
        tdict.setdefault('nccl_ib_hca_list', 'bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7')
        tdict.setdefault('nccl_ib_hca', 'bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7')
        tdict.setdefault('nccl_socket_ifname', 'enp49s0f1np1')
        tdict.setdefault('gloo_socket_ifname', 'enp49s0f1np1')
        tdict.setdefault('nccl_ib_gid_index', '3')
        tdict.setdefault('nccl_debug', 'ERROR')
        tdict.setdefault('data_cache_dir', f'{self.home_dir}/cache')
        tdict.setdefault('log_dir', f'{self.home_dir}/LOGS')
        tdict.setdefault('scripts_dir', f'{self.home_dir}/SCRIPTS')
        tdict.setdefault('master_address', '127.0.0.1')
        tdict.setdefault('verify_network_errors', 'False')
        tdict.setdefault('primus_root', '/workspace/Primus')
        tdict.setdefault('primus_cli', 'runner/primus-cli')

        # primus_framework is always derived from variant_config.framework by the
        # factory (topology suffix stripped). Direct callers must pass it explicitly.
        if not primus_framework:
            raise ValueError(
                "primus_framework is required. Use create_training_job() which "
                "derives it automatically from variant_config.framework, or pass "
                "primus_framework= directly when constructing PrimusTrainingJob."
            )
        resolved_framework = primus_framework

        self.container_image = orch.container_config["image"]
        self.distributed_training = distributed_training
        self.iterations = int(tdict['training_iterations'])
        self.nnodes = str(tdict['nnodes'])
        if int(self.nnodes) != len(orch.hosts):
            log.warning(
                f"config nnodes={self.nnodes} does not match cluster host count={len(orch.hosts)}; "
                f"using cluster host count"
            )
            self.nnodes = str(len(orch.hosts))
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
        self.scripts_dir = scripts_dir if scripts_dir is not None else tdict['scripts_dir']
        self.master_address = tdict['master_address']
        self.verify_network_errors = tdict['verify_network_errors']
        self.primus_root = tdict['primus_root']
        self.primus_cli = tdict['primus_cli']
        self.primus_framework = resolved_framework

        # Model params — kept for run_label, exp path construction, and result
        # verification; NOT injected into the launch command (all live in the EXP YAML).
        pdict = dict(variant_config.model_params)
        pdict['micro_batch_size'] = micro_batch_size
        pdict['global_batch_size'] = global_batch_size
        if precision:
            pdict['precision'] = precision
        pdict.pop('model_name', None)
        pdict.setdefault('precision', 'FP8')

        self.micro_batch_size = pdict['micro_batch_size']
        self.global_batch_size = pdict['global_batch_size']
        self.precision = pdict['precision']

        raw_label = run_label or f"{self.model_name}_mbs{micro_batch_size}_gbs{global_batch_size}_{self.precision}"
        self.run_label = re.sub(r'[^A-Za-z0-9._-]', '_', str(raw_label))
        self.combo_log_dir = f'{self.log_dir}/primus-logs/{self.run_label}'

        self.orch.all.exec(f'rm -rf {self.scripts_dir}')
        self.orch.all.exec(f'mkdir -p {self.scripts_dir}')
        self.orch.all.exec(f'sudo chmod 777 {self.scripts_dir}')

    def _needs_local_tokenizer(self):
        """Primus reads tokenizer directly from the HF repo — no local file needed."""
        return False

    def _batch_size_args(self):
        """Return batch size CLI args for the primus-cli command.

        megatron:   --micro_batch_size <mbs> --global_batch_size <gbs>
        torchtitan: --training.local_batch_size=<mbs>
        jax:        --per_device_batch_size <mbs>
        """
        if self.primus_framework == 'megatron':
            return f'--micro_batch_size {self.micro_batch_size} --global_batch_size {self.global_batch_size}'
        if self.primus_framework == 'torchtitan':
            return f'--training.local_batch_size={self.micro_batch_size}'
        if self.primus_framework == 'jax':
            return f'--per_device_batch_size {self.micro_batch_size}'
        return ''

    def _exp_config_path(self):
        """Return the EXP config path for this sweep combo.

        Pattern: examples/<primus_framework>/configs/<gpu_arch>/<model_name>-<precision>-pretrain.yaml
        """
        return (
            f'examples/{self.primus_framework}/configs/'
            f'{self.gpu_arch}/{self.model_name}-{self.precision}-pretrain.yaml'
        )

    def run_pretraining_tasks(self):
        if self.distributed_training is True:
            self.rdma_stats_dict_before = linux_utils.get_rdma_stats_dict(self.orch.all)
            self.ethtool_stats_dict_before = linux_utils.get_nic_ethtool_stats_dict(self.orch.all)

    def stop_training_processes(self):
        """Check GPU VRAM after a training combo and free memory if any processes remain.

        After normal completion VRAM% is 0 and no KFD PIDs are present — returns
        immediately in that case. If processes are still holding GPU memory (crash
        or hang), kills them with SIGKILL then verifies VRAM is clear.
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

        self.orch.exec(
            "rocm-smi --showpids 2>/dev/null "
            "| awk '/^[0-9]+[[:space:]]/{print $1}' "
            "| xargs -r kill -9 2>/dev/null || true; "
            "sleep 10"
        )

        out_dict = self.orch.exec('rocm-smi --showpids 2>/dev/null')
        for node, output in (out_dict or {}).items():
            if 'No KFD PIDs currently running' in (output or ''):
                log.info('Node %s: VRAM successfully freed', node)
            else:
                log.warning('Node %s: GPU processes may still be running after kill attempt', node)

    def exec_nic_setup_scripts(self):
        """Apply in-container NIC setup steps for distributed Primus runs.

        Mirrors MegatronTrainingJob.exec_nic_setup_scripts — Broadcom/Thor
        workaround: copies the host-side libbnxt_re library into the container
        and verifies the RDMA device enumerates correctly.
        """
        if self.distributed_training is True:
            if re.search('broadcom|thor', self.nic_type, re.I):
                self.nccl_ib_gid_index = 3
                out_dict = self.orch.exec(
                    'sudo cp /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.host '
                    '/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so; '
                    'sleep 2;ibv_devinfo;sleep 2;'
                )
                segments = [re.escape(s.strip()) for s in self.hca_id_pattern.split('|') if s.strip()]
                if not segments:
                    fail_test(
                        f'hca_id_pattern parsed to zero non-empty segments, got: {self.hca_id_pattern!r}. '
                        f'Expected a `|`-separated list of NIC-name prefixes, e.g. "bnxt_|rocep".'
                    )
                    return False
                hca_id_regex = rf'hca_id:\s+({"|".join(segments)})'
                for node in out_dict.keys():
                    if not re.search(hca_id_regex, out_dict[node], re.I):
                        log.info('%s', out_dict[node])
                        fail_test(f'Broadcom libbnxt rdma driver is not properly copied on node {node}')
                        return False
        return True

    def build_training_job_cmd(self):
        """Construct the Primus launch command per node.

        Final command structure (distributed, per node i):

            export HF_TOKEN="..."; export LOG_DIR=...; export NCCL_*=...;
            cd <primus_root> &&
            NNODES=<n> NODE_RANK=<i> MASTER_ADDR=<addr>
            nohup bash runner/primus-cli direct
              --log_file <combo_log_dir>/out-node<i>/training.log
              -- train pretrain
              --config examples/<framework>/configs/<gpu_arch>/<model>-<precision>-pretrain.yaml &

        Single-node omits NNODES/NODE_RANK/MASTER_ADDR and NCCL exports.
        Model hyperparameters are not injected — they live in the EXP YAML.
        """
        exp_path = self._exp_config_path()
        log.info(f'Primus EXP config path: {exp_path}')

        env_exports = (
            f'export HF_TOKEN="{self.hf_token}"; '
            f'export LOG_DIR={self.log_dir}; '
            f'export NCCL_SOCKET_IFNAME={self.nccl_socket_ifname}; '
            f'export GLOO_SOCKET_IFNAME={self.gloo_socket_ifname}; '
        )

        if re.search(r'MI3(00|25)X', self.gpu_arch, re.I):
            env_exports += (
                'export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1; '
                'export NVTE_CK_IS_V3_ATOMIC_FP32=1; '
            )

        if self.distributed_training:
            env_exports += (
                f'export NCCL_IB_HCA_LIST={self.nccl_ib_hca_list}; '
                f'export NCCL_IB_HCA={self.nccl_ib_hca_list}; '
                f'export NCCL_DEBUG={self.nccl_debug}; '
                f'export NCCL_IB_GID_INDEX={self.nccl_ib_gid_index}; '
            )

            for i in range(len(self.orch.hosts)):
                log_file = f'{self.combo_log_dir}/out-node{i}/training.log'
                batch_args = self._batch_size_args()
                full_cmd = (
                    env_exports
                    + f'cd {self.primus_root} && '
                    + f'NNODES={self.nnodes} NODE_RANK={i} MASTER_ADDR={self.master_address} '
                    + f'nohup bash {self.primus_cli} direct '
                    + f'--log_file {log_file} '
                    + f'-- train pretrain '
                    + f'--config {exp_path} '
                    + (f'{batch_args} ' if batch_args else '')
                    + '&'
                )
                script_cmd = (
                    f'echo {shlex.quote(full_cmd)} > '
                    f'{self.scripts_dir}/distributed_wrapper_script_{i}.sh '
                    f'&& chmod 777 {self.scripts_dir}/distributed_wrapper_script_{i}.sh'
                )
                self.job_cmd_list.append(script_cmd)
        else:
            log_file = f'{self.combo_log_dir}/out-node0/training.log'
            batch_args = self._batch_size_args()
            self.job_cmd = (
                env_exports
                + f'cd {self.primus_root} && '
                + f'nohup bash {self.primus_cli} direct '
                + f'--log_file {log_file} '
                + f'-- train pretrain '
                + f'--config {exp_path} '
                + (f'{batch_args} ' if batch_args else '')
                + '&'
            )

    def start_training_job(self, timeout=500):
        """Launch the Primus training job (distributed or single-node).

        Primus writes structured rank-prefixed logs to --log_file. The wrapper
        script's own stdout/stderr is appended to the same file so any startup
        errors (e.g. config file not found) are also captured there.
        """
        log.info('start training job')
        log.info('%s', self.job_cmd_list)
        log.info('%s', self.job_cmd)
        n = len(self.orch.hosts)

        self.orch.exec_cmd_list([
            f'mkdir -p {self.combo_log_dir}/out-node{i}'
            for i in range(n)
        ])

        if self.distributed_training:
            if not self.exec_nic_setup_scripts():
                return

            # Write per-node wrapper scripts on the bare host (volume-mounted path)
            self.orch.all.exec_cmd_list(self.job_cmd_list)

            # Launch wrapper scripts inside container on all nodes; append stdout
            # to the same training.log that --log_file writes to
            self.orch.exec_cmd_list([
                f'nohup {self.scripts_dir}/distributed_wrapper_script_{i}.sh '
                f'>> {self.combo_log_dir}/out-node{i}/training.log 2>&1 &'
                for i in range(n)
            ])
        else:
            self.orch.all.exec(
                f'echo {shlex.quote(self.job_cmd)} > {self.scripts_dir}/single_node_wrapper_script.sh '
                f'&& chmod 777 {self.scripts_dir}/single_node_wrapper_script.sh'
            )
            # Append stdout to the same log file that --log_file writes to
            self.orch.exec(
                f'nohup {self.scripts_dir}/single_node_wrapper_script.sh '
                f'>> {self.combo_log_dir}/out-node0/training.log 2>&1 &'
            )
        time.sleep(50)

    def _read_all_node_logs(self, tail_lines=0):
        """Read training logs from every node and return per-node output dict.

        Primus emits iteration stats from the last rank, which lives on the last
        node. Scanning all nodes ensures errors on any node are caught.

        Args:
            tail_lines (int): If > 0, only the last N lines of each log are read.

        Returns:
            dict: {host: log_text} for each node in orch.hosts order.
        """
        n = len(self.orch.hosts)
        tail_suffix = f' | tail -{tail_lines}' if tail_lines > 0 else ''
        return self.orch.exec_cmd_list([
            f'cat {self.combo_log_dir}/out-node{i}/training.log{tail_suffix}'
            for i in range(n)
        ])

    def _read_last_node_log(self, tail_lines=0):
        """Read the training log from the last node using orch.exec() and return its text.

        Uses orch.exec() (not exec_cmd_list) so the output is returned as a string,
        not just streamed to the logger. Iteration stats are emitted by the last rank
        which runs on the last node, so this is the authoritative log for completion
        detection and metric parsing.

        Args:
            tail_lines (int): If > 0, only the last N lines of the log are read.

        Returns:
            str: Log text from the last node.
        """
        n = len(self.orch.hosts)
        tail_suffix = f' | tail -{tail_lines}' if tail_lines > 0 else ''
        out_dict = self.orch.exec(
            f'cat {self.combo_log_dir}/out-node{n - 1}/training.log{tail_suffix}'
        )
        return list(out_dict.values())[-1] if out_dict else ''

    def get_training_results_dict(self):
        """Parse training log from the last node and extract performance metrics.

        Reads both a tail (summary/last-iteration values) and the full log.
        The tail primary patterns capture the running-average column from the
        last iteration line. The full log is passed as fallback so that
        _parse_training_results can compute per-iteration means for any metric
        whose primary pattern produces no match.

        Returns:
            dict: {metric_name: list[str]} — see TRAINING_RESULT_PATTERNS for keys.
        """
        tail_output = self._read_last_node_log(tail_lines=30)

        log.info('Extracting results from logs')
        log.info('#===========================#')
        log.info('%s', tail_output)
        log.info('#===========================#')

        full_log = self._read_last_node_log()
        training_results_dict = _parse_training_results(tail_output, full_log)
        log.info('%s', training_results_dict)
        return training_results_dict

    def scan_for_training_errors(self):
        """Scan the last node's training log for known error patterns.

        Returns:
            tuple[bool, str]: (True if no errors found, log text from last node).
        """
        log.info('Scan for training errors')
        training_pass = True

        output = self._read_last_node_log()
        for err_key in training_err_dict:
            if re.search(training_err_dict[err_key], output):
                fail_test(f'ERROR {training_err_dict[err_key]} seen in training logs ..')
                log.error('Aborting training log polling')
                training_pass = False
        return training_pass, output

    def poll_for_training_completion(self, time_between_iters=120):
        """Periodically poll logs to detect completion, surface errors, and
        extract results.

        scan_for_training_errors() reads the last node's log via
        _read_last_node_log() and returns the text alongside the pass/fail
        result. The same log text is reused for completion and NaN detection,
        avoiding a redundant second read.

        Completion is detected by the final "iteration N/N" line in the log
        (emitted by the last rank on the last node).

        Args:
            time_between_iters (int): Seconds to sleep between polling loops.
        """
        log.info('Poll for training completion ..')
        time.sleep(80)

        for i in range(1, int(self.iterations) + 10):
            log.info(f'Starting Iteration {i}')
            training_pass, output = self.scan_for_training_errors()
            if not training_pass:
                fail_test('Failures seen in training logs, Aborting!!!')
                return

            if not _is_training_complete(output, self.iterations):
                log.info('Training still in progress')
            else:
                if _has_nan_inf_results(output):
                    fail_test(f'ERROR - NaN or Inf values seen in training results {output}')
                    return
                time.sleep(30)
                self.training_results_dict = self.get_training_results_dict()
                log.info('Completed Training, returning !!!')
                return

            time.sleep(int(time_between_iters))

    def verify_training_results(self):
        """Validate collected training results and environment health.

        Checks for NaN/Inf values, optionally verifies RDMA/NIC error counters
        (distributed only), and scans dmesg. Threshold evaluation is handled by
        test_metric via the threshold JSON files.
        """
        self.training_end_time = self.orch.all.exec('date')

        log.info('#==================================================#')
        log.info('\t\tTraining Results')
        log.info('%s', self.training_results_dict)
        log.info('#==================================================#')

        if not self.training_results_dict:
            fail_test(
                'Failed to populate training results, training_results_dict is empty '
                '- please check logs for failures'
            )
            return

        for result_key in self.training_results_dict.keys():
            for result_val in self.training_results_dict[result_key]:
                if re.search('nan|inf', result_val, re.I):
                    fail_test(
                        f'Failures seen in training_result dict for {result_key}, '
                        f'numbers are either NaN or Inf - {result_val}'
                    )

        if self.distributed_training is True:
            if self.verify_network_errors.lower() == 'true':
                self.rdma_stats_dict_after = linux_utils.get_rdma_stats_dict(self.orch.all)
                self.ethtool_stats_dict_after = linux_utils.get_nic_ethtool_stats_dict(self.orch.all)

                for node in self.rdma_stats_dict_after.keys():
                    for counter_nam in self.rdma_stats_dict_after[node]:
                        if re.search(err_counters_pattern, counter_nam, re.I):
                            if int(self.rdma_stats_dict_after[node][counter_nam]) > int(
                                self.rdma_stats_dict_before[node][counter_nam]
                            ):
                                fail_test(
                                    f'Error counter {counter_nam} has gone up after training on node {node} '
                                    f'Before = {self.rdma_stats_dict_before[node][counter_nam]}, '
                                    f'After = {self.rdma_stats_dict_after[node][counter_nam]}'
                                )

                for node in self.ethtool_stats_dict_after.keys():
                    for counter_nam in self.ethtool_stats_dict_after[node]:
                        if re.search(err_counters_pattern, counter_nam, re.I):
                            if int(self.ethtool_stats_dict_after[node][counter_nam]) > int(
                                self.ethtool_stats_dict_before[node][counter_nam]
                            ):
                                fail_test(
                                    f'Error counter {counter_nam} has gone up after training on node {node} '
                                    f'Before = {self.ethtool_stats_dict_before[node][counter_nam]}, '
                                    f'After = {self.ethtool_stats_dict_after[node][counter_nam]}'
                                )

        verify_dmesg_for_errors(self.orch.all, self.training_start_time, self.training_end_time, till_end_flag=False)