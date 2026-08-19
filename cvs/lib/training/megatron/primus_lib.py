'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Primus-Megatron training job — standalone module for Primus-specific logic.

PrimusTrainingJob has the same public interface as MegatronTrainingJob and can
be used as a drop-in replacement when the container image is Primus-based:
  build_training_job_cmd()
  start_training_job()
  poll_for_training_completion()
  verify_training_results()
  stop_training_processes()
  download_tokenizer_model()   (always a no-op; Primus uses HF repo IDs)
  _needs_local_tokenizer()     (always False)
  _read_last_node_log()
'''

import os
import re
import shlex
import time

from cvs.lib import globals
from cvs.lib import linux_utils
from cvs.lib.utils_lib import fail_test
from cvs.lib.verify_lib import verify_dmesg_for_errors

log = globals.log

# ---------------------------------------------------------------------------
# Primus-Megatron log patterns
# ---------------------------------------------------------------------------

PRIMUS_RESULT_PATTERNS = {
    'throughput_per_gpu': [
        r'throughput per GPU \(TFLOP/s/GPU\):\s+[0-9.]+/([0-9.]+)',
        r'throughput per GPU \(TFLOP/s/GPU\):\s+([0-9.]+)(?:\s|$|\|)',
    ],
    'tokens_per_gpu': [
        r'tokens/s/GPU inst/harmonic mean:\s+[0-9.]+/([0-9.]+)',
        r'tokens per GPU \(tokens/s/GPU\):\s+[0-9.]+/([0-9.]+)',
    ],
    'elapsed_time_per_iteration': [
        r'elapsed time per iteration \(ms\):\s+[0-9.]+/([0-9.]+)',
        r'elapsed time per iteration \(ms\):\s+([0-9.]+)(?:\s|$|\|)',
    ],
}

PRIMUS_ITERATION_PATTERNS = {
    'throughput_per_gpu': r'throughput per GPU \(TFLOP/s/GPU\):\s+[0-9.]+/([0-9.]+)',
    'tokens_per_gpu': r'(?:tokens/s/GPU inst/harmonic mean|tokens per GPU \(tokens/s/GPU\)):\s+[0-9.]+/([0-9.]+)',
    'elapsed_time_per_iteration': r'elapsed time per iteration \(ms\):\s+[0-9.]+/([0-9.]+)',
    'lm_loss': r'lm loss:\s+([0-9.E+\-]+)',
    'grad_norm': r'grad norm:\s+([0-9.]+)',
    'hip_mem_usage_ratio': r'hip mem usage/free/total/usage_ratio:\s+[0-9.]+GB/[0-9.]+GB/[0-9.]+GB/([0-9.]+)%',
    'rocm_mem_usage_ratio': r'rocm mem usage/free/total/usage_ratio:\s+[0-9.]+GB/[0-9.]+GB/[0-9.]+GB/([0-9.]+)%',
}

PRIMUS_NAN_PATTERNS = [
    r'throughput per GPU \(TFLOP/s/GPU\):\s+[0-9.]+/(?:NaN|Inf)',
    r'lm loss:\s+(?:NaN|Inf)',
]

def _is_training_complete(output, total_iters):
    """Return True only when the final Primus iteration line is present.

    Primus emits `iteration <cur>/<total>` on every step, so `<total>/<total>`
    marks true completion.  The `torchrun finished successfully` line is a
    secondary signal for runs where the iteration counter format differs.
    Checking for any per-iteration throughput line instead would fire after
    step 1 and cut slow runs short.
    """
    n = int(total_iters)
    return bool(re.search(rf'iteration\s+{n}\s*/\s*{n}\b', output)) or bool(
        re.search(r'torchrun finished successfully', output, re.I)
    )


_training_err_dict = {
    'NCCL ERROR': 'NCCL ERROR|NCCL timeout|ncclRemoteError: A call failed possibly due to a network error|NCCL error:',
    'GPU HW ERROR': 'HW Exception by GPU|GPU Hang|Uncorrectable error|GPU Reset',
    'torch': 'torch.distributed.elastic.multiprocessing.errors',
}

_err_counters_pattern = 'err|retransmit|drop|discard|naks|invalid|oflow|out_of_buffer|reset|fail'


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_mean_from_iterations(log_text, pattern, skip_warmup=True):
    """Parse per-iteration metric values and return their mean as a string."""
    matches = re.findall(pattern, log_text, re.I)
    if not matches:
        return None
    values = [float(m) for m in matches]
    if skip_warmup and len(values) > 1:
        values = values[1:]
    return str(sum(values) / len(values))


def _parse_step_losses(log_text):
    """Return {step: lm_loss} from a Primus training log.

    Each iteration line looks like:
      iteration  42/  200 | ... | lm loss: 2.3456 | ...
    """
    result = {}
    for m in re.finditer(r'iteration\s+(\d+)/\s*\d+.*?lm loss:\s+([0-9.E+\-]+)', log_text, re.I):
        result[int(m.group(1))] = float(m.group(2))
    return result


# ---------------------------------------------------------------------------
# PrimusTrainingJob
# ---------------------------------------------------------------------------

class PrimusTrainingJob:
    """Manages a single Primus-Megatron training run (single-node or distributed).

    Provides the same public interface as MegatronTrainingJob so it can be used
    interchangeably in test files.
    """

    def __init__(
        self,
        orch,
        variant_config,
        hf_token,
        micro_batch_size,
        global_batch_size,
        precision=None,
        distributed_training=False,
        tune_model_params=False,
        scripts_dir=None,
        run_label=None,
    ):
        self.orch = orch
        self.model_name = variant_config.model_params['model_name']
        self.hf_token = hf_token
        self.tune_model_params = tune_model_params
        self.distributed_training = distributed_training

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

        self.home_dir = os.path.expanduser('~')
        tdict = dict(variant_config.config)
        tdict.setdefault('training_iterations', 10)
        tdict.setdefault('nnodes', '1')
        tdict.setdefault('nccl_socket_ifname', 'ensf1np1')
        tdict.setdefault('gloo_socket_ifname', 'ensf1np1')
        tdict.setdefault('nccl_ib_hca_list', 'bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7')
        tdict.setdefault('nccl_ib_gid_index', '3')
        tdict.setdefault('nccl_debug', 'ERROR')
        tdict.setdefault('data_cache_dir', f'{self.home_dir}/cache')
        tdict.setdefault('log_dir', f'{self.home_dir}/LOGS')
        tdict.setdefault('scripts_dir', f'{self.home_dir}/SCRIPTS')
        tdict.setdefault('master_address', '127.0.0.1')
        tdict.setdefault('verify_network_errors', 'False')
        tdict.setdefault('primus_root', '/workspace/Primus')
        tdict.setdefault('primus_cli', 'runner/primus-cli')

        self.iterations = int(tdict['training_iterations'])
        self.checkpoint_dir = None
        self.save_interval = None
        self.load_checkpoint = False

        self.nnodes = str(tdict['nnodes'])
        if int(self.nnodes) != len(orch.hosts):
            log.warning(
                'config nnodes=%s does not match cluster host count=%d; using cluster host count',
                self.nnodes, len(orch.hosts),
            )
            self.nnodes = str(len(orch.hosts))

        self.nccl_socket_ifname = tdict['nccl_socket_ifname']
        self.gloo_socket_ifname = tdict['gloo_socket_ifname']
        self.nccl_ib_hca_list = tdict['nccl_ib_hca_list']
        self.nccl_ib_gid_index = tdict['nccl_ib_gid_index']
        self.nccl_debug = tdict['nccl_debug']
        self.data_cache_dir = tdict['data_cache_dir']
        self.log_dir = tdict['log_dir']
        self.scripts_dir = scripts_dir if scripts_dir is not None else tdict['scripts_dir']
        self.master_address = tdict['master_address']
        self.verify_network_errors = tdict['verify_network_errors']
        self.primus_root = tdict['primus_root']
        self.primus_cli = tdict['primus_cli']
        self.gpu_arch = getattr(variant_config, 'gpu_arch', '')

        pdict = dict(variant_config.model_params)
        pdict['micro_batch_size'] = micro_batch_size
        pdict['global_batch_size'] = global_batch_size
        if precision:
            pdict['precision'] = precision
        pdict.pop('model_name', None)
        pdict.setdefault('tokenizer_model', 'meta-llama/Llama-3.1-70B')
        pdict.setdefault('precision', 'BF16')
        pdict.setdefault('micro_batch_size', '2')
        pdict.setdefault('global_batch_size', '128')

        self.tokenizer_model = pdict['tokenizer_model']
        self.precision = pdict['precision']
        self.micro_batch_size = pdict['micro_batch_size']
        self.global_batch_size = pdict['global_batch_size']

        raw_label = run_label or f"{self.model_name}_mbs{micro_batch_size}_gbs{global_batch_size}_{self.precision}"
        self.run_label = re.sub(r'[^A-Za-z0-9._-]', '_', str(raw_label))
        self.combo_log_dir = f'{self.log_dir}/primus-logs/{self.run_label}'

        self.orch.all.exec(f'rm -rf {self.scripts_dir}')
        self.orch.all.exec(f'mkdir -p {self.scripts_dir}')
        self.orch.all.exec(f'sudo chmod 700 {self.scripts_dir}')

    # ------------------------------------------------------------------
    # Interface required by test files
    # ------------------------------------------------------------------

    def _needs_local_tokenizer(self):
        return False

    def download_tokenizer_model(self):
        """No-op: Primus accepts HF repo IDs directly."""

    def build_training_job_cmd(self):
        self._build_primus_cmd()

    def start_training_job(self):
        self._start_primus_job()

    def stop_training_processes(self):
        """Kill any GPU processes still holding VRAM after a training combo."""
        log.info('Checking GPU memory state after training combo')
        out_dict = self.orch.exec('rocm-smi --showpids 2>/dev/null')

        has_pids = False
        for node, output in (out_dict or {}).items():
            if 'No KFD PIDs currently running' in (output or ''):
                log.info('Node %s: VRAM already free', node)
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

    def poll_for_training_completion(self, time_between_iters=120):
        """Poll training logs until completion, NaN, or iteration budget exhausted."""
        log.info('Poll for training completion ..')
        time.sleep(80)

        n = int(self.iterations)
        for i in range(1, n + 10):
            log.info('Starting Iteration %d', i)
            if not self._scan_for_training_errors():
                fail_test('Failures seen in training logs, Aborting!!!')
                return
            output = self._read_last_node_log()
            complete = _is_training_complete(output, n)
            if not complete:
                log.info('Training still in progress')
            else:
                if any(re.search(p, output, re.I) for p in PRIMUS_NAN_PATTERNS):
                    fail_test(f'ERROR - NaN or Inf values seen in training results {output}')
                    return
                time.sleep(30)
                self.training_results_dict = self._get_training_results_dict()
                log.info('Completed Training, returning !!!')
                return
            time.sleep(int(time_between_iters))

    def verify_training_results(self):
        """Validate training results and network health after a run."""
        self.training_end_time = self.orch.all.exec('date')

        log.info('#==================================================#')
        log.info('\t\tTraining Results')
        log.info('%s', self.training_results_dict)
        log.info('#==================================================#')

        if not self.training_results_dict:
            fail_test(
                'Failed to populate training results, training_results_dict is empty'
                ' - please check logs for failures'
            )
            return

        for result_key, result_vals in self.training_results_dict.items():
            for val in result_vals:
                if re.search('nan|inf', val, re.I):
                    fail_test(
                        f'Failures seen in training_result dict for {result_key},'
                        f' numbers are either NaN or Inf - {val}'
                    )

        if self.distributed_training and self.verify_network_errors.lower() == 'true':
            self.rdma_stats_dict_after = linux_utils.get_rdma_stats_dict(self.orch.all)
            self.ethtool_stats_dict_after = linux_utils.get_nic_ethtool_stats_dict(self.orch.all)

            for node, counters in self.rdma_stats_dict_after.items():
                for name, val in counters.items():
                    if re.search(_err_counters_pattern, name, re.I):
                        before = int(self.rdma_stats_dict_before.get(node, {}).get(name, 0))
                        if int(val) > before:
                            fail_test(
                                f'Error counter {name} has gone up after training on node {node} '
                                f'Before = {before}, After = {val}'
                            )

            for node, counters in self.ethtool_stats_dict_after.items():
                for name, val in counters.items():
                    if re.search(_err_counters_pattern, name, re.I):
                        before = int(self.ethtool_stats_dict_before.get(node, {}).get(name, 0))
                        if int(val) > before:
                            fail_test(
                                f'Error counter {name} has gone up after training on node {node} '
                                f'Before = {before}, After = {val}'
                            )

        verify_dmesg_for_errors(self.orch.all, self.training_start_time, self.training_end_time, till_end_flag=False)

        log.info('^^^^^^^^^^^^^^^^^^^^')
        log.info('training_results_dict')
        log.info('^^^^^^^^^^^^^^^^^^^^')
        log.info('%s', self.training_results_dict)

    def run_pretraining_tasks(self):
        if self.distributed_training:
            self.rdma_stats_dict_before = linux_utils.get_rdma_stats_dict(self.orch.all)
            self.ethtool_stats_dict_before = linux_utils.get_nic_ethtool_stats_dict(self.orch.all)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _exp_config_path(self):
        """Return the Primus EXP YAML path relative to primus_root."""
        return (
            f'examples/megatron/configs/'
            f'{self.gpu_arch}/{self.model_name}-{self.precision}-pretrain.yaml'
        )

    def _build_primus_cmd(self):
        """Build the primus-cli launch command per node."""
        exp_path = self._exp_config_path()
        log.info('Primus EXP config path: %s', exp_path)

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

        batch_args = (
            f'--micro_batch_size {self.micro_batch_size} '
            f'--global_batch_size {self.global_batch_size} '
            f'--train_iters {self.iterations}'
        )
        if self.checkpoint_dir:
            batch_args += (
                f' --save {self.checkpoint_dir}'
                f' --save_interval {self.save_interval or self.iterations}'
            )
        if self.load_checkpoint and self.checkpoint_dir:
            batch_args += f' --load {self.checkpoint_dir}'

        if self.distributed_training:
            env_exports += (
                f'export NCCL_IB_HCA_LIST={self.nccl_ib_hca_list}; '
                f'export NCCL_IB_HCA={self.nccl_ib_hca_list}; '
                f'export NCCL_DEBUG={self.nccl_debug}; '
                f'export NCCL_IB_GID_INDEX={self.nccl_ib_gid_index}; '
            )
            for i in range(len(self.orch.hosts)):
                log_file = f'{self.combo_log_dir}/out-node{i}/training.log'
                full_cmd = (
                    env_exports
                    + f'cd {self.primus_root} && '
                    + f'NNODES={self.nnodes} NODE_RANK={i} MASTER_ADDR={self.master_address} '
                    + f'nohup bash {self.primus_cli} direct '
                    + f'--log_file {log_file} '
                    + f'-- train pretrain --config {exp_path} {batch_args} &'
                )
                script_cmd = (
                    f'echo {shlex.quote(full_cmd)} > '
                    f'{self.scripts_dir}/distributed_wrapper_script_{i}.sh '
                    f'&& chmod 777 {self.scripts_dir}/distributed_wrapper_script_{i}.sh'
                )
                self.job_cmd_list.append(script_cmd)
        else:
            log_file = f'{self.combo_log_dir}/out-node0/training.log'
            self.job_cmd = (
                env_exports
                + f'cd {self.primus_root} && '
                + f'nohup bash {self.primus_cli} direct '
                + f'--log_file {log_file} '
                + f'-- train pretrain --config {exp_path} {batch_args} &'
            )

    def _start_primus_job(self):
        """Launch the Primus training job (distributed or single-node)."""
        log.info('start primus training job')
        log.info('%s', self.job_cmd_list)
        log.info('%s', self.job_cmd)

        exp_full_path = f'{self.primus_root}/{self._exp_config_path()}'
        out_dict = self.orch.exec(f'test -f {exp_full_path} && echo EXISTS || echo MISSING')
        output = list((out_dict or {}).values())[0] if out_dict else ''

        if 'MISSING' in output and re.search(r'MI325X', self.gpu_arch, re.I):
            fallback_arch = 'MI300X'
            log.warning(
                'EXP config not found for %s, retrying with fallback arch %s',
                self.gpu_arch, fallback_arch,
            )
            self.gpu_arch = fallback_arch
            self.job_cmd = ''
            self.job_cmd_list = []
            self._build_primus_cmd()
            exp_full_path = f'{self.primus_root}/{self._exp_config_path()}'
            out_dict = self.orch.exec(f'test -f {exp_full_path} && echo EXISTS || echo MISSING')
            output = list((out_dict or {}).values())[0] if out_dict else ''

        if 'MISSING' in output:
            msg = f'Primus EXP config not found in container: {exp_full_path}'
            log.error('Primus EXP config path: %s', exp_full_path)
            fail_test(msg)
            raise FileNotFoundError(msg)

        n = len(self.orch.hosts)
        self.orch.exec_cmd_list([f'mkdir -p {self.combo_log_dir}/out-node{i}' for i in range(n)])

        if self.distributed_training:
            self.orch.all.exec_cmd_list(self.job_cmd_list)
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
            self.orch.exec(
                f'nohup {self.scripts_dir}/single_node_wrapper_script.sh '
                f'>> {self.combo_log_dir}/out-node0/training.log 2>&1 &'
            )
        time.sleep(50)

    def _read_last_node_log(self, tail_lines=0):
        """Read training log from the last node."""
        n = len(self.orch.hosts)
        last_host = self.orch.hosts[-1]
        tail_suffix = f' | tail -{tail_lines}' if tail_lines > 0 else ''
        out_dict = self.orch.exec(
            f'cat {self.combo_log_dir}/out-node{n - 1}/training.log{tail_suffix}',
            hosts=[last_host],
        )
        return out_dict.get(last_host) or ''

    def _get_training_results_dict(self):
        """Parse training log and extract Primus metrics."""
        tail_output = self._read_last_node_log(tail_lines=15)

        log.info('Extracting results from logs')
        log.info('#===========================#')
        log.info('%s', tail_output)
        log.info('#===========================#')

        full_log = self._read_last_node_log()
        return self._parse_primus_results(tail_output, full_log)

    def _parse_primus_results(self, output, full_log=None):
        """Extract metrics from a Primus-Megatron training log.

        For x/y format metrics (throughput, tokens, elapsed): search full_log and
        take matches[-1] which is the final step's overall harmonic mean (y value).
        For single-value metrics (lm_loss, grad_norm, mem): compute mean across all steps.
        """
        log_text = full_log if full_log is not None else output
        out = {}
        for metric, patterns in PRIMUS_RESULT_PATTERNS.items():
            out[metric] = []
            for pat in patterns:
                matches = re.findall(pat, log_text, re.I)
                if matches:
                    out[metric] = [matches[-1]]
                    break
        for metric, pattern in PRIMUS_ITERATION_PATTERNS.items():
            if not out.get(metric):
                mean = _parse_mean_from_iterations(log_text, pattern, skip_warmup=True)
                if mean:
                    out[metric] = [mean]
                    log.info('primus per-iteration mean: %s = %s', metric, mean)
        return out

    def _scan_for_training_errors(self):
        """Scan training log for known error patterns.

        Returns:
            bool: True if no errors found.
        """
        output = self._read_last_node_log()
        training_pass = True
        for err_key, pattern in _training_err_dict.items():
            if re.search(pattern, output):
                fail_test(f'ERROR {pattern} seen in training logs ..')
                log.error('Aborting training log polling')
                training_pass = False
        return training_pass
