'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import os
import re
import time
import statistics
from cvs.lib import globals
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *
from cvs.lib import linux_utils
from cvs.lib import docker_lib

log = globals.log


training_err_dict = {
    'cache_err': 'Unable to save MockGPTDataset indexes because path_to_cache is None',
    'NCCL ERROR': 'NCCL ERROR|NCCL timeout|local work queue catastrophic error',
    'GPU HW ERROR': 'HW Exception by GPU|GPU Hang|Uncorrectable error|GPU Reset',
    'AssertionError': 'AssertionError|ValueError:|JaxStackTrace|During handling of the above exception|triggered the following exception',
    'rocm Err': 'FAILED_PRECONDITION: No visible GPU devices|failed call to hipInit: HIP_ERROR_NoDevice|librocm reported version is: NOT_FOUND',
    'python err': 'ModuleNotFoundError: No module named|Fatal Python error:',
    'tensorflow': 'tensorflow.CoordinationServiceError|tensorflow.BarrierError|CoordinationServiceError',
    'resource': 'RESOURCE_EXHAUSTED: Out of memory|failed: RESOURCE_EXHAUSTED',
}

err_counters_pattern = 'err|retransmit|drop|discard|naks|invalid|oflow|out_of_buffer|reset|fail'


def textwrap_for_yml(msg_string):
    return '\n'.join([m.lstrip() for m in msg_string.split('\n')])


class JaxTrainingJob:
    """
    JAX training class for running training of popular models like Llama on AMD GPU cluster.

    Responsibilities:
      - Store user-provided training and model parameters.
      - Initialize defaults for missing training config entries (container, networking, etc.).
      - Capture cluster context (host list), job command stubs, and timing metadata.
      - Prepare pre-training network/GPU stats containers and log directories.

    Args:
      phdl: Process/host handle abstraction; expected to expose:
            - host_list: list of cluster nodes
            - exec(cmd: str) -> dict[node: str]: run command on nodes, return stdout per node
      model_name (str): Identifier for the model to be trained (e.g., "llama2-7b").
      training_config_dict (dict): Training configuration parameters (may be partial).
      model_params_dict (dict): Model-specific parameters/hyperparameters (may be tuned).
      hf_token (str): Hugging Face token for model/dataset access.
      gpu_type (str): GPU type selector (e.g., "mi300"). Defaults to "mi300".
      tune_model_params (bool): Whether to allow auto-tuning of model params. Defaults to True.
      scripts_dir (str): Path to scripts directory. Defaults to ~/SCRIPTS.

    Attributes:
      phdl, host_list, model_name, hf_token, gpu_type
      training_config_dict, tc_dict (alias), model_params_dict, tune_model_params
      scripts_dir, job_cmd, job_cmd_list, training_result_dict
      rdma_stats_dict_before, ethtool_stats_dict_before, rdma_stats_dict_after
      training_start_time, training_end_time
      home_dir, container_image, container_name
      (plus defaulted entries placed into self.tc_dict, such as networking/NCCL knobs)
    """

    def __init__(
        self,
        phdl,
        model_name,
        training_config_dict,
        model_params_dict,
        hf_token,
        gpu_type='mi300',
        distributed_training=True,
        tune_model_params=True,
        scripts_dir=os.path.expanduser("~") + '/SCRIPTS',
    ):
        self.phdl = phdl
        self.host_list = phdl.host_list
        self.model_name = model_name
        self.hf_token = hf_token
        self.gpu_type = gpu_type

        # Sample training config and model params dict saved above
        self.training_config_dict = training_config_dict
        self.tc_dict = training_config_dict
        self.model_params_dict = model_params_dict
        self.tune_model_params = tune_model_params

        self.scripts_dir = scripts_dir

        self.job_cmd = ''
        self.job_cmd_list = []
        self.training_result_dict = {}
        log.info("%s", self.gpu_type)

        # Intialize cluster stats dicts ..
        self.rdma_stats_dict_before = {}
        self.ethtool_stats_dict_before = {}
        self.rdma_stats_dict_after = {}
        self.training_start_time = phdl.exec('date +"%a %b %e %H:%M"')
        self.training_end_time = None

        # Training configs - let us set defaults if not defined in input json file
        self.home_dir = os.path.expanduser("~")
        self.tc_dict.setdefault('container_image', 'rocm/jax-training:maxtext-v25.5')
        self.tc_dict.setdefault('container_name', 'jax_container')
        self.tc_dict.setdefault('distributed_training', True)
        self.tc_dict.setdefault('training_steps', 10)
        self.tc_dict.setdefault('nnodes', 2)
        self.tc_dict.setdefault('nic_type', 'thor2')
        self.tc_dict.setdefault('nccl_ib_hca_list', 'rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7')
        self.tc_dict.setdefault('nccl_ib_hca', 'rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7')
        self.tc_dict.setdefault('nccl_socket_ifname', 'ens51f1np1')
        self.tc_dict.setdefault('gloo_socket_ifname', 'ens51f1np1')
        self.tc_dict.setdefault('data_cache_dir', f'{self.home_dir}/cache')
        self.tc_dict.setdefault('log_dir', f'{self.home_dir}/LOG_DIR')

        self.tc_dict.setdefault(
            'env_vars',
            {
                'GPU_MAX_HW_QUEUES': '2',
                'HSA_FORCE_FINE_GRAIN_PCIE': '1',
                'HIP_FORCE_DEV_KERNARG': '1',
                'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.93',
                'NCCL_DEBUG': 'ERROR',
                'NCCL_PROTO': 'Simple',
                'NCCL_IB_TC': '41',
                'NCCL_IB_SL': '0',
                'NCCL_IB_GID_INDEX': '3',
                'NCCL_CHECKS_DISABLE': '1',
                'NCCL_CROSS_NIC': '0',
                'NVTE_ALLOW_NONDETERMINISTIC_ALGO': '1',
                'NVTE_USE_HIPBLASLT': '1',
                'NVTE_FUSED_ATTN': '1',
                'NVTE_CK_USES_BWD_V3': '1',
                'NVTE_CK_USES_FWD_V3': '1',
                'NVTE_CK_IS_V3_ATOMIC_FP32': '0',
                'NVTE_CK_HOW_V3_BF16_CVT': '2',
                'NVTE_FUSED_ATTN_CK': '1',
                'NVTE_FUSED_ATTN_AOTRITON': '0',
            },
        )

        self.tc_dict.setdefault(
            'xla_flags',
            {
                'xla_gpu_enable_latency_hiding_scheduler': 'True',
                'xla_gpu_enable_triton_gemm': 'False',
                'xla_gpu_memory_limit_slop_factor': '95',
                'xla_gpu_enable_command_buffer': "''",
                'xla_gpu_enable_cublaslt': 'True',
                'xla_gpu_autotune_level': '0',
                'xla_gpu_enable_reduce_scatter_combine_by_dim': 'false',
                'xla_gpu_reduce_scatter_combine_threshold_bytes': '8589934592',
                'xla_gpu_all_reduce_combine_threshold_bytes': '8589934592',
                'xla_gpu_all_gather_combine_threshold_bytes': '8589934592',
                'xla_gpu_enable_all_gather_combine_by_dim': 'FALSE',
            },
        )

        self.tc_dict.setdefault(
            'rdma_lib',
            {
                'host_source_file': '/usr/local/lib/libbnxt_re-rdmav34.so',
                'container_mount_file': '/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.host',
                'container_dest_file': '/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so',
            },
        )

        self.container_image = self.tc_dict['container_image']
        self.container_name = self.tc_dict['container_name']

        self.distributed_training = distributed_training

        self.training_steps = int(self.tc_dict['training_steps'])
        self.nnodes = self.tc_dict['nnodes']
        self.nic_type = self.tc_dict['nic_type']
        self.nccl_ib_hca_list = self.tc_dict['nccl_ib_hca_list']
        self.nccl_ib_hca = self.tc_dict['nccl_ib_hca']
        self.nccl_socket_ifname = self.tc_dict['nccl_socket_ifname']
        self.gloo_socket_ifname = self.tc_dict['gloo_socket_ifname']
        self.data_cache_dir = self.tc_dict['data_cache_dir']
        self.log_dir = self.tc_dict['log_dir']
        self.coordinator_ip = self.tc_dict.get('coordinator_ip', 'localhost')
        self.coordinator_port = (
            self.tc_dict.get('container_config', {}).get('env_dict', {}).get('JAX_COORDINATOR_PORT', '1234')
        )

        # Let us assume 1 training step can complete in 10 polling iterations
        self.training_poll_iterations = int(self.training_steps * 10)

        # Get the model parameters dict
        if not self.distributed_training:
            self.mp_dict = self.model_params_dict['single_node'][self.model_name][self.gpu_type]
            self.expected_result_dict = self.model_params_dict['single_node'][self.model_name][self.gpu_type][
                'result_dict'
            ]
        else:
            self.mp_dict = self.model_params_dict['multi_node'][self.model_name][self.gpu_type]
            self.expected_result_dict = self.model_params_dict['multi_node'][self.model_name][self.gpu_type][
                'result_dict'
            ]

        self.mp_dict.setdefault('hardware', 'gpu')
        self.mp_dict.setdefault('packing', 'True')
        self.mp_dict.setdefault('enable_goodput_recording', 'False')
        self.mp_dict.setdefault('monitor_goodput', 'False')
        self.mp_dict.setdefault('optimizer_memory_host_offload', 'False')
        self.mp_dict.setdefault('param_scan_axis', '1')
        self.mp_dict.setdefault('shardy', 'False')

        # Remove and recreate the scripts dir
        self.phdl.exec(f'rm -rf {self.scripts_dir}')
        time.sleep(2)
        self.phdl.exec(f'mkdir {self.scripts_dir}')
        time.sleep(2)
        self.phdl.exec(f'sudo chmod 777 {self.scripts_dir}')

        # Let us override some of the params based on number of nodes and platform
        # if override flag set ..
        if self.tune_model_params:
            # Assuming the training json configs were built with 4 nodes = 32 gpus
            if int(self.batch_size) > 32:
                if int(self.batch_size) % 32 == 0:
                    per_gpu_batch_size = int(self.batch_size) / 32
                    self.batch_size = per_gpu_batch_size * int(self.nnodes) * 8

    def get_maxtext_paths(self, container_name):
        """
        Auto-detect MaxText installation and return relevant paths.

        Args:
            container_name (str): Name of the Docker container

        Returns:
            dict: {'base': str, 'config': str, 'reference_config': str}
        """
        # (base_path, config_subdir, reference_config, train_script_path)
        path_configs = [
            (
                '/workspace/maxtext/src/maxtext',
                'configs/models',
                'llama3-70b.yml',
                '/workspace/maxtext/src/maxtext/trainers/pre_train/train.py',
            ),
            (
                '/workspace/maxtext/src/MaxText',
                'configs/models',
                'llama3-70b.yml',
                '/workspace/maxtext/src/MaxText/train.py',
            ),
            ('/workspace/maxtext/MaxText', 'configs', 'llama2_70b_gpu_bs7.yml', '/workspace/maxtext/MaxText/train.py'),
        ]

        for base, config_subdir, ref_config, train_script in path_configs:
            if docker_lib.path_exists_in_container(self.phdl, container_name, base):
                return {
                    'base': base,
                    'config': f"{base}/{config_subdir}",
                    'reference_config': ref_config,
                    'train_script': train_script,
                }

        raise RuntimeError(f"MaxText not found in container '{container_name}'")

    def run_pretraining_tasks(
        self,
    ):
        if self.distributed_training is True:
            self.rdma_stats_dict_before = linux_utils.get_rdma_stats_dict(self.phdl)
            self.ethtool_stats_dict_before = linux_utils.get_nic_ethtool_stats_dict(self.phdl)

    def launch_docker_container(self, container_name, image, device_list, volume_dict, env_dict):
        if self.distributed_training is True:
            docker_lib.launch_docker_container(self.phdl, container_name, image, device_list, volume_dict, env_dict)
        else:
            env_dict['JAX_COORDINATOR_IP'] = 'localhost'
            env_dict['NNODES'] = 1
            docker_lib.launch_docker_container(self.phdl, container_name, image, device_list, volume_dict, env_dict)

    def exec_nic_setup_scripts(
        self,
    ):
        """
        Execute NIC-related setup steps inside the training container.

        Behavior:
        - Only runs for distributed training.
        - If NIC type appears to be Broadcom/Thor, applies a temporary workaround:
          * Copies the bnxt RDMA library from the host-named file to the container?s expected path.
          * Verifies that ibv_devinfo shows a bnxt_ HCA (to confirm RDMA is wired correctly).
        - NCCL_IB_GID_INDEX is set via env_vars (NCCL knobs), not here.

        Assumptions:
        - self.phdl.exec runs a shell command and returns a dict: {node: stdout}.
        - sudo is non-interactive within the container.
        - The bnxt library file paths exist in the container base image.
        """

        # Run all your backend NIC related bringups for containers here ..
        if self.distributed_training is True:
            # This is a temporary hack needed for broadcom nics to work within containers ..
            if re.search('broadcom|thor', self.nic_type, re.I):
                rdma_lib = self.tc_dict['rdma_lib']
                out_dict = self.phdl.exec(
                    f'docker exec {self.container_name} /bin/bash -c "sudo \
                    cp {rdma_lib["container_mount_file"]} {rdma_lib["container_dest_file"]}; \
                    sleep 2;ibv_devinfo;sleep 2;"'
                )
                for node in out_dict.keys():
                    if not re.search(r'hca_id:\s+(bnxt_|rocep|rdma)', out_dict[node], re.I):
                        log.info("%s", out_dict[node])
                        fail_test(f'Broadcom libbnxt rdma driver is not properly copied on node {node}')

    def build_training_job_cmd(
        self,
    ):
        """
        Generate MaxText YAML config files and environment variables inside the container,
        then generate a per-node training wrapper script and logging directories.

        Steps:
        1) Emit training_config_for_jax.yml (runtime/training knobs).
        2) Emit model_config_for_jax.yml (model architecture knobs).
        3) Create XLA_FLAGS file and fix quoting.
        4) Append per-node environment exports into maxtext_env.sh (NCCL/IB/GPU env).
        5) Create log directories per node.
        6) Create training_wrapper_script.sh to launch training (reference config).
        7) Patch wrapper script to use the generated training_config_for_jax.yml.

        Assumptions:
        - self.phdl.exec executes commands on all nodes, returning {node: stdout}.
        - textwrap_for_yml ensures the multi-line echo is properly formatted for YAML.
        - self.mp_dict and self.tc_dict contain all referenced keys.
        """

        # Auto-detect MaxText paths in the container
        maxtext_paths = self.get_maxtext_paths(self.container_name)

        # Keys in mp_dict that are CVS internal metadata, not MaxText YAML params
        mp_skip_keys = {
            'result_dict',
            'tokenizer_model',
            'model_size',
            'batch_size',
            'micro_batch_size',
            'precision',
            'sequence_length',
            'tensor_parallelism',
            'pipeline_parallelism',
            'recompute',
            'fsdp',
        }

        # Header params that come from training_config_dict or self, not mp_dict
        yml_lines = [
            'base_config: base.yml',
            f'run_name: {self.model_name}-job',
            f'steps: {self.training_config_dict["training_steps"]}',
            f'model_name: {self.model_name}',
            f'enable_checkpointing: {self.training_config_dict["enable_checkpointing"]}',
        ]

        # Loop over mp_dict: skip CVS-internal keys, dicts, and underscore-prefixed keys
        for k, v in self.mp_dict.items():
            if k in mp_skip_keys or k.startswith('_') or isinstance(v, dict):
                continue
            yml_lines.append(f'{k}: {v}')

        yml_content = '\n'.join(yml_lines)
        cmd = f'''docker exec {self.container_name} /bin/bash -c "cat > {maxtext_paths['config']}/training_config_for_jax.yml <<'YMLEOF'\n{yml_content}\nYMLEOF"'''
        self.phdl.exec(cmd)

        # Print the generated training config for verification
        out_dict = self.phdl.exec(
            f'docker exec {self.container_name} cat {maxtext_paths["config"]}/training_config_for_jax.yml'
        )
        for node in out_dict:
            log.info("training_config_for_jax.yml on %s:\n%s", node, out_dict[node])

        time.sleep(5)

        # Build XLA_FLAGS and env var exports from config
        xla_dict = self.tc_dict['xla_flags']
        xla_flags_str = ' '.join(f'--{k}={v}' for k, v in xla_dict.items())
        env_vars = self.tc_dict['env_vars']
        env_exports = '\n'.join(f'export {k}={v}' for k, v in env_vars.items())
        env_exports += f'\nexport XLA_FLAGS=\\"{xla_flags_str}\\"'

        # Need check for Single Node vs Double node ..
        if self.distributed_training is not True:
            cmd_list = []
            for i in range(0, int(self.nnodes)):
                env_content = f"""export NNODES=1
export NODE_RANK=0
export JAX_COORDINATOR_ADDRESS=localhost
export JAX_COORDINATOR_IP=localhost
export JAX_COORDINATOR_PORT={self.coordinator_port}
export LD_LIBRARY_PATH=/opt/rocm/lib:\\$LD_LIBRARY_PATH
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_P2P_DISABLE=0
{env_exports}"""
                # heredoc avoids quoting issues with '' inside XLA_FLAGS
                cmd = f'''docker exec {self.container_name} /bin/bash -c "cat >> /workspace/maxtext/maxtext_env.sh <<'ENVEOF'\n{env_content}\nENVEOF"'''
                cmd_list.append(cmd)
        else:
            cmd_list = []
            for i in range(0, int(self.nnodes)):
                env_content = f"""export NNODES={int(self.nnodes)}
export NODE_RANK={i}
export JAX_COORDINATOR_ADDRESS={self.coordinator_ip}
export JAX_COORDINATOR_IP={self.coordinator_ip}
export JAX_COORDINATOR_PORT={self.coordinator_port}
export HF_HOME={self.data_cache_dir}
export LD_LIBRARY_PATH=/opt/rocm/lib:\\$LD_LIBRARY_PATH
export NCCL_IB_HCA={self.tc_dict['nccl_ib_hca']}
export NCCL_SOCKET_IFNAME={self.tc_dict['nccl_socket_ifname']}
export GLOO_SOCKET_IFNAME={self.tc_dict['gloo_socket_ifname']}
{env_exports}"""
                # heredoc avoids quoting issues with '' inside XLA_FLAGS (e.g. --xla_gpu_enable_command_buffer='')
                cmd = f'''docker exec {self.container_name} /bin/bash -c "cat >> /workspace/maxtext/maxtext_env.sh <<'ENVEOF'\n{env_content}\nENVEOF"'''
                cmd_list.append(cmd)
        log.info("%s", cmd_list)
        self.phdl.exec_cmd_list(cmd_list)

        # Print the generated env file for verification
        out_dict = self.phdl.exec(f'docker exec {self.container_name} cat /workspace/maxtext/maxtext_env.sh')
        for node in out_dict:
            log.info("maxtext_env.sh on %s:\n%s", node, out_dict[node])

        cmd_list = []
        for i in range(0, int(self.nnodes)):
            cmd = f'''docker exec {self.container_name} /bin/bash -c "mkdir -p {self.log_dir}/jax-logs/out-node{i}"'''
            cmd_list.append(cmd)
        self.phdl.exec_cmd_list(cmd_list)

        cmd_list = []
        for i in range(0, int(self.nnodes)):
            # Take a reference config yml like llama2_70b
            cmd = f'''docker exec {self.container_name} /bin/bash -c "echo '
                      mkdir -p {self.log_dir}/jax-logs/out-node{i}
                      export PYTHONPATH=$PYTHONPATH:/workspace/maxtext/; cd /workspace/maxtext; python {maxtext_paths['train_script']} {maxtext_paths['config']}/{maxtext_paths['reference_config']} enable_goodput_recording=false monitor_goodput=false shardy=False base_output_directory={self.tc_dict['log_dir']} 2>&1 | tee >(grep -v 'external/xla/xla/') > {self.log_dir}/jax-logs/out-node{i}/training.log' > /workspace/maxtext/training_wrapper_script.sh"'''
            formatted_cmd = textwrap_for_yml(cmd)
            cmd_list.append(formatted_cmd)
        self.phdl.exec_cmd_list(cmd_list)

        # Replace with the config.yml we generated
        cmd = f'''docker exec {self.container_name} /bin/bash -c 'sed -i -e "s/{maxtext_paths['reference_config']}/training_config_for_jax.yml/g"  /workspace/maxtext/training_wrapper_script.sh'  '''
        self.phdl.exec(cmd)

    def start_training_job(
        self,
    ):
        """
        Launch the JAX training job inside the container on all nodes.

        Behavior:
        - Builds a list of docker exec commands, one per node rank, that:
          * source the MaxText environment (maxtext_env.sh),
          * run the training wrapper script in the background (nohup),
          * redirect output to a per-node log file under log_dir/jax-logs/out-node<i>.
        - Executes the list of commands across nodes via phdl.exec_cmd_list.
        - Sleeps for 60 seconds to give processes time to initialize and start logging.

        Assumptions:
        - self.nnodes is an int-compatible value (number of nodes/ranks to launch).
        - self.container_name and self.tc_dict['log_dir'] are correctly set.
        - maxtext_env.sh and training_wrapper_script.sh exist in /workspace/maxtext inside the container.
        - self.phdl.exec_cmd_list(cmd_list) executes each command across the appropriate nodes.
        - Docker and the container are available and running on each node.

        """

        log.info('Start Training on all Nodes')
        cmd_list = []

        # Build a docker exec command per node/rank
        for i in range(0, int(self.nnodes)):
            # Source the env file, then launch training in background with nohup, redirecting output to a per-node log
            cmd = f'''docker exec {self.container_name} /bin/bash -c "source /workspace/maxtext/maxtext_env.sh && nohup bash /workspace/maxtext/training_wrapper_script.sh > {self.log_dir}/jax-logs/out-node{i}/training_redirect_logs"'''
            cmd_list.append(cmd)

        # Execute the launch commands (expected to run across the cluster)
        self.phdl.exec_cmd_list(cmd_list)

        # Allow time for processes to come up and begin writing logs
        time.sleep(60)

    def check_deviation_from_median(self, training_results_dict, metric_name, percentage_off):
        """
        Validate that a per-step training metric stays within a percentage band around the median.

        Args:
        training_results_dict (dict): Mapping of step index -> dict of metrics for that step.
                                    Example: { 0: {"loss": 3.2, "tflops_per_sec_per_gpu": 120.5}, ... }
        metric_name (str): The metric key to check (e.g., "loss", "tflops_per_sec_per_gpu").
        percentage_off (float): Allowed deviation as a fraction (e.g., 0.10 for ±10%).

        Behavior:
        - Ignores the first two steps to avoid warmup noise/outliers.
        - Computes the median of the selected metric over steps [2, self.training_steps).
        - Calculates acceptable lower/upper bounds as median ± (median * percentage_off).
        - For each checked step, prints a confirmation if within bounds; otherwise records a failure via fail_test.

        Assumptions:
        - training_results_dict contains entries for all steps in [0, self.training_steps).
        - training_results_dict[i][metric_name] is numeric (int/float).
        - self.training_steps >= 3 to ensure at least one value when skipping first two steps.
        - statistics module is imported (statistics.median).
        - fail_test is available in scope to record failures.
        """

        list_of_values = []
        # To avoid any outliers, we exclude the first couple of steps ..
        for i in range(2, self.training_steps):
            list_of_values.append(float(training_results_dict[i][metric_name]))

        # Compute the central tendency of the metric across the stable range of steps
        median_value = statistics.median(list_of_values)
        log.info(f'%%%% median_value = {median_value}')

        # Define acceptable bounds around the median based on the allowed percentage deviation
        upper_bound = median_value * (1 + percentage_off)
        lower_bound = median_value * (1 - percentage_off)
        for i in range(2, self.training_steps):
            if lower_bound <= float(training_results_dict[i][metric_name]) <= upper_bound:
                log.info(f'Training step {i} training metric {metric_name} is in expected range')
            else:
                fail_test(
                    f'FAIL Training step {i} training metric {metric_name} is over {percentage_off}% from median value {median_value} - actual value {training_results_dict[i][metric_name]}'
                )

    def get_training_results_dict(
        self,
    ):
        """
        Parse per-step training metrics from the aggregated training log and run sanity checks.

        Returns:
        dict: A dictionary keyed by step index with extracted metrics per step, for example:
            {
              0: {
                "time_elapsed": "<str seconds>",
                "tflops_per_sec_per_gpu": "<str float>",
                "tokens_per_sec_per_gpu": "<str float>",
                "total_weights": "<str float>",
                "loss": "<str float>"
              },
              1: { ... },
              ...
            }

        Behavior:
        - Reads the consolidated training log from the last node in self.host_list.
        - For each step from 0..self.training_steps-1:
          * Extracts time, TFLOP/s/device, Tokens/s/device, total_weights, loss using a regex.
          * Stores them as strings under the step index in the result dict.
        - Invokes check_deviation_from_median on several metrics to ensure stability.

        Assumptions:
        - self.phdl.exec(cmd) returns a dict mapping node -> stdout string.
        - The training log format contains a line per step that matches the regex provided.
        - self.training_steps is an int >= 1.
        - self.check_deviation_from_median exists and accepts (results_dict, metric_name, percentage_off).

        """

        training_results_dict = {}
        percentage_off = 10

        # Read the training log output from the "last" node (assumed authoritative)
        last_node = self.host_list[len(self.host_list) - 1]
        last_node_num = len(self.host_list) - 1
        out_dict = self.phdl.exec(f'cat {self.log_dir}/jax-logs/out-node{last_node_num}/training.log')
        output = out_dict[last_node]

        # Parse metrics for each step based on expected log line structure
        for i in range(0, self.training_steps):
            training_results_dict[i] = {}

            pattern = f'completed step:\s+{i},\s+seconds:\s+([0-9\.]+),\s+TFLOP\/s\/device:\s+([0-9\.]+),\s+Tokens\/s\/device:\s+([0-9\.]+),\s+total_weights:\s+([0-9\\.]+),\s+loss:\s+([0-9\.]+|nan|inf|-inf)'
            match = re.search(pattern, output, re.I)

            # Guard against missing or malformed lines to avoid AttributeError on match.group(...)
            if not match:
                fail_test(f'Missing or malformed metrics line for step {i} in training logs on node {last_node}')
                continue

            # Store extracted values as strings; convert to float later if needed
            training_results_dict[i]['time_elapsed'] = match.group(1)
            training_results_dict[i]['tflops_per_sec_per_gpu'] = match.group(2)
            training_results_dict[i]['tokens_per_sec_per_gpu'] = match.group(3)
            training_results_dict[i]['total_weights'] = match.group(4)
            training_results_dict[i]['loss'] = match.group(5)

        # Check if the Loss function is not growing by over 10%
        self.check_deviation_from_median(training_results_dict, 'tflops_per_sec_per_gpu', percentage_off)
        self.check_deviation_from_median(training_results_dict, 'tokens_per_sec_per_gpu', percentage_off)
        # self.check_deviation_from_median( training_results_dict, 'loss', percentage_off )

        log.info("%s", training_results_dict)
        return training_results_dict

    def scan_for_training_errors(
        self,
    ):
        """
        Scan training logs for known error patterns and report status.

        Returns:
        bool: True if no error patterns are found; False otherwise.

        Behavior:
        - Constructs a list of commands to read each node's training.log.
        - Executes the commands across nodes using phdl.exec_cmd_list.
        - Checks the last node's log output against regex patterns in training_err_dict.
        - On first match:
          * Calls fail_test with a descriptive message.
          * Logs an error to indicate polling should stop.
          * Sets training_pass to False.
        - Returns training_pass at the end.

        Assumptions:
        - self.nnodes is an integer-compatible value.
        - self.tc_dict['log_dir'] points to the base directory containing jax-logs/out-node<rank>/training.log.
        - self.phdl.exec_cmd_list(cmd_list) executes each command and returns a dict-like mapping
          from node to command output. This method uses the output from the "last" node only.
        - training_err_dict is a dict of error-name -> regex pattern (strings).
        - re, log, and fail_test are available in scope.

        """

        log.info('Scan for training errors')
        training_pass = True

        # Identify which node's output will be used (currently "last" node only)
        last_node = self.host_list[len(self.host_list) - 1]

        # Build the list of commands to read each node's training log file
        cmd_list = []

        # Execute the commands across nodes; returns a mapping of node -> command output
        for j in range(0, int(self.nnodes)):
            cmd = f"sudo cat {self.log_dir}/jax-logs/out-node{j}/training.log"
            cmd_list.append(cmd)
        out_dict = self.phdl.exec_cmd_list(cmd_list)

        output = out_dict[last_node]

        # Check the log content against all known training error patterns
        for err_key in training_err_dict:
            if re.search(f'{training_err_dict[err_key]}', output):
                fail_test(f'ERROR {training_err_dict[err_key]} seen in training logs ..')
                log.error('Aborting training log polling')
                training_pass = False
        return training_pass

    def poll_for_training_completion(self, waittime_between_iters=60, total_timeout=3600, require_all_nodes=True):
        """
        Periodically poll training logs to detect completion or failure, with robust checks and a hard timeout.

        Improvements implemented (based on previous suggestions):
        - Scan all nodes' logs (not just the last node) for completion and invalid values (NaN/Inf).
        - Fix regex checks to use proper alternation for NaN/Inf: (NaN|Inf) instead of [NaN|Inf].
        - Add a hard wall-clock timeout via total_timeout; return a structured status report.
        - Capture both stdout and stderr when tailing logs (2>&1).
        - Return a structured dict with status, reason, and optional results.

        Args:
        waittime_between_iters: Seconds to sleep between polling iterations (in addition to short waits while in progress).
        total_timeout: Maximum wall-clock seconds to poll before returning a timeout status. If None, no wall-clock limit.
        require_all_nodes: If True, require the "final step completed" pattern to be present on every node;
                         if False, proceed when any node shows final step completed.

        Returns:
        dict: A structured status report with keys:
        - "status": "success" | "error" | "timeout" | "in_progress"
        - "reason": Optional short message on why we exited early or failed
        - "results": Optionally, the parsed training results dict when status == "success"

        Assumptions:
        - self.nnodes, self.training_steps, self.training_poll_iterations are valid integers.
        - self.tc_dict['log_dir'] contains per-node paths: .../jax-logs/out-node<rank>/training.log
        - self.phdl.exec_cmd_list(cmd_list) executes commands across the cluster and returns a mapping of node -> concatenated stdout (and now stderr due to 2>&1).
        - self.scan_for_training_errors() returns True when OK, False when any error pattern is detected.
        - self.get_training_results_dict() parses final metrics and populates self.training_result_dict.

        Notes:
        - Completion criterion:
          final_step = self.training_steps - 1
          We look for 'completed step:\s+<final_step>,' on logs.
        - NaN/Inf check uses alternation on key metrics to detect invalid numeric outputs.
        - If require_all_nodes=True and at least one node lags behind, we continue polling (unless timeout).
        """

        log.info('Poll for training completion')

        # Initial wait to give training time to start logging
        time.sleep(60)

        # Track wall-clock timeout if specified
        start_time = time.time()

        def timed_out() -> bool:
            return total_timeout is not None and (time.time() - start_time) >= float(total_timeout)

        final_step = int(self.training_steps) - 1
        completed_step_pattern = re.compile(rf'completed step:\s+{final_step},', re.I)
        tflops_pattern = re.compile(r'TFLOPS\/s\/device:\s+(NaN|Inf)', re.I)
        tokens_pattern = re.compile(r'Tokens\/s\/device:\s+(NaN|Inf)', re.I)

        # Poll up to a maximum iteration count (safety cap) as well as wall-clock (if provided)
        for itr in range(1, int(self.training_poll_iterations) + 1):
            log.info(f'Starting iteration {itr}')

            # Early abort on training errors
            if not self.scan_for_training_errors():
                msg = 'Failures seen in training logs, Aborting!!!'
                fail_test(msg)
                return {"status": "error", "reason": msg}

            # Build commands to tail recent lines from each node's training log and capture stderr as well
            cmd_list = []
            for j in range(0, int(self.nnodes)):
                cmd = f"sudo tail -2000 {self.log_dir}/jax-logs/out-node{j}/training.log 2>&1"
                cmd_list.append(cmd)

            # Execute across nodes; out_dict is expected to be a mapping of node -> tailed output
            out_dict = self.phdl.exec_cmd_list(cmd_list)

            # Determine completion across nodes
            node_completion = {}
            for node, output in out_dict.items():
                node_completion[node] = bool(completed_step_pattern.search(output))

            if require_all_nodes:
                all_complete = all(node_completion.values()) if node_completion else False
            else:
                all_complete = any(node_completion.values()) if node_completion else False

            # If not yet complete, wait and continue (subject to timeout)
            if not all_complete:
                if timed_out():
                    msg = f"Timeout while waiting for training completion after ~{int(time.time() - start_time)}s"
                    log.warning("%s", msg)
                    return {"status": "timeout", "reason": msg}
                log.info('Training still in progress')
                # Short progress wait before the longer inter-iteration sleep
                time.sleep(30)
                time.sleep(int(waittime_between_iters))
                continue

            # If complete (per the requirement), validate that reported metrics are not NaN/Inf on any node
            invalid_nodes = []
            for node, output in out_dict.items():
                if tflops_pattern.search(output) or tokens_pattern.search(output):
                    invalid_nodes.append(node)

            if invalid_nodes:
                msg = f"ERROR - NaN or Inf values seen in training results on node(s): {', '.join(map(str, invalid_nodes))}"
                fail_test(f'{msg}\nLast outputs: { {n: out_dict[n] for n in invalid_nodes} }')
                return {"status": "error", "reason": msg}

            # Parse/store final results and report success
            self.training_result_dict = self.get_training_results_dict()
            log.info('Completed Training, returning !!!')
            return {"status": "success", "results": self.training_result_dict}

            # If we reached here, it means poll for training completion failed

        # If we exhaust the iteration cap without completing, treat as timeout (or in_progress if no wall-clock limit)
        if timed_out():
            msg = f"Timeout after maximum iterations ({self.training_poll_iterations}) and ~{int(time.time() - start_time)}s"
            log.warning("%s", msg)
            return {"status": "timeout", "reason": msg}
        else:
            # If no wall-clock timeout was set and we hit the iteration cap, report in-progress
            msg = f"Reached iteration cap ({self.training_poll_iterations}) without completion; still in progress"
            log.warning("%s", msg)
            return {"status": "stuck_in_progress", "reason": msg}

    def verify_training_results(
        self,
    ):
        log.info('Verify Training Completion Msg')
        last_node = self.host_list[len(self.host_list) - 1]
        last_node_num = len(self.host_list) - 1
        out_dict = self.phdl.exec(f'cat {self.log_dir}/jax-logs/out-node{last_node_num}/training.log')

        # Check for proper shutdown first
        if re.search('Distributed task shutdown result: OK', out_dict[last_node], re.I):
            log.info("Clean shutdown detected")
        else:
            # Fallback: check if all training steps completed successfully
            # This handles the known race condition where PJRT cleanup runs after coordination service exits
            final_step = self.training_steps - 1
            if re.search(f'completed step:\s+{final_step},', out_dict[last_node], re.I):
                log.info(
                    f"Training completed all {self.training_steps} steps successfully. Shutdown message missing but acceptable due to coordination service cleanup race condition."
                )
            else:
                fail_test('JAX training did not complete all steps properly')

        log.info("%s", list(self.training_result_dict.keys()))
        for i in self.training_result_dict.keys():
            log.info(
                'Actual Step %s TFLOPS/s/device = %s',
                i,
                self.training_result_dict[i]['tflops_per_sec_per_gpu'],
            )
            log.info(
                'Expected Step %s TFLOPS/s/device = %s',
                i,
                self.expected_result_dict['tflops_per_sec_per_gpu'],
            )
            if int(i) > 5:
                if float(self.training_result_dict[i]['tflops_per_sec_per_gpu']) < float(
                    self.expected_result_dict['tflops_per_sec_per_gpu']
                ):
                    fail_test(
                        f"FAIL - The TFLOPS/s/device is not matching expected number. \
                            Expected = {self.expected_result_dict['tflops_per_sec_per_gpu']}, \
                            Actual = {self.training_result_dict[i]['tflops_per_sec_per_gpu']}"
                    )
                else:
                    log.info(
                        f"Step {i} TFLOPS/s/device = {self.training_result_dict[i]['tflops_per_sec_per_gpu']} is above expected threshold {self.expected_result_dict['tflops_per_sec_per_gpu']}"
                    )
                if float(self.training_result_dict[i]['tokens_per_sec_per_gpu']) < float(
                    self.expected_result_dict['tokens_per_sec_per_gpu']
                ):
                    fail_test(
                        f"FAIL - The tokens_per_sec_per_gpu is not matching expected number. \
                            Expected = {self.expected_result_dict['tokens_per_sec_per_gpu']} \
                            Actual = {self.training_result_dict[i]['tokens_per_sec_per_gpu']}"
                    )
                else:
                    log.info(
                        f"Step {i} Tokens/s/device = {self.training_result_dict[i]['tokens_per_sec_per_gpu']} is above expected threshold {self.expected_result_dict['tokens_per_sec_per_gpu']}"
                    )

        # Scan Dmesg for errors ..
        self.training_end_time = self.phdl.exec('date +"%a %b %e %H:%M"')
        time.sleep(2)
        verify_dmesg_for_errors(self.phdl, self.training_start_time, self.training_end_time)
        log.info("%s", self.training_result_dict)
