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

# Library for building Megatron training jobs ..


class MegatronLlamaTrainingJob:
    """
    Orchestrates a Megatron-LM Llama training job across one or more nodes.

    Responsibilities:
      - Normalize training configuration and model parameters (with sensible defaults).
      - Prepare per-node wrapper scripts and environment variables for distributed runs.
      - Optionally collect pre/post network (RDMA/ethtool) stats for validation.
      - Launch the job (single-node or distributed) inside a specified container.
      - Poll logs for completion and errors; extract performance metrics from logs.
      - Verify training results against expected thresholds and system health checks.

    Assumptions:
      - phdl provides remote execution utilities across nodes:
          - phdl.host_list (list of nodes)
          - phdl.exec(cmd: str) -> Dict[node, str] or str, depending on implementation
          - phdl.exec_cmd_list(cmd_list: List[str]) -> Dict[node, str]
      - Docker container is pre-deployed and accessible on each node.
      - Training scripts exist under /workspace/Megatron-LM/examples/llama/.
      - External helpers referenced in the methods are available in scope:
          - linux_utils.get_rdma_stats_dict, linux_utils.get_nic_ethtool_stats_dict
          - json_to_dict, fail_test, verify_dmesg_for_errors, log, training_err_dict
          - err_counters_pattern
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
        """
        Initialize job configuration and resolve defaults from the provided dicts.

        - Normalizes training_config_dict and model_params_dict; applies defaults
          if fields are missing.
        - Builds paths and internal state used later to construct job commands.

        Args:
          phdl: Remote execution handle for multi-node command execution.
          model_name: Canonical model name key used in model_params_dict (e.g., "llama3.1_8b").
          training_config_dict: Unstructured training config; defaults are applied here.
          model_params_dict: Parameter sets per model and topology (single/multi-node).
          hf_token: Hugging Face token passed to the job environment.
          gpu_type: GPU platform key to select model params (default: 'mi300').
          tune_model_params: If True, adjust some parameters based on cluster size.
          scripts_dir: Folder on nodes to place generated wrapper scripts.
        """

        self.phdl = phdl
        self.host_list = phdl.host_list
        self.model_name = model_name
        self.hf_token = hf_token
        self.gpu_type = gpu_type

        # Sample training config and model params dict saved above
        # User-supplied config/params and derived fields
        self.training_config_dict = training_config_dict
        self.model_params_dict = model_params_dict
        self.iterations = int(training_config_dict['training_iterations'])
        self.tune_model_params = tune_model_params

        self.scripts_dir = scripts_dir

        self.job_cmd = ''
        self.job_cmd_list = []
        self.training_results_dict = {}
        print(self.gpu_type)

        # Intialize cluster stats dicts ..
        self.rdma_stats_dict_before = {}
        self.ethtool_stats_dict_before = {}
        self.rdma_stats_dict_after = {}
        self.ethtool_stats_dict_after = {}
        self.training_start_time = self.phdl.exec('date')
        self.training_end_time = None

        # Training configs - let us set defaults if not defined in input json file
        self.home_dir = os.path.expanduser("~")
        tdict = training_config_dict
        tdict.setdefault('container_image', 'rocm/megatron-lm:v25.5_py312')
        tdict.setdefault('container_name', 'megatron_llama3.1_8b')
        tdict.setdefault('training_iterations', 10)
        tdict.setdefault('nnodes', 2)
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

        self.container_image = tdict['container_image']
        self.container_name = tdict['container_name']
        self.distributed_training = distributed_training
        self.iterations = int(tdict['training_iterations'])
        self.nnodes = tdict['nnodes']
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

        # Get the model parameters dict
        print('^^^^')
        print(self.model_params_dict)
        print(self.model_name)
        print(self.gpu_type)
        print('^^^^')
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

        # Training configs - let us set defaults if not defined in input json file
        # Model Params - Let us set defaults if params not defined in input json file
        pdict.setdefault('tokenizer_model', 'meta-llama/Llama-3.1-70B')
        pdict.setdefault('model_size', 70)
        pdict.setdefault('sequence_length', '8192')
        pdict.setdefault('batch_size', '128')
        pdict.setdefault('micro_batch_size', '2')
        pdict.setdefault('fsdp', '0')
        pdict.setdefault('tensor_parallelism', '1')
        pdict.setdefault('pipeline_parallelism', '1')
        pdict.setdefault('recompute', '0')
        pdict.setdefault('precision', 'TE_FP8')

        self.tokenizer_model = pdict['tokenizer_model']
        self.model_size = pdict['model_size']
        self.sequence_length = pdict['sequence_length']
        self.batch_size = pdict['batch_size']
        self.micro_batch_size = pdict['micro_batch_size']
        self.fsdp = pdict['fsdp']
        self.tensor_parallelism = pdict['tensor_parallelism']
        self.pipeline_parallelism = pdict['pipeline_parallelism']
        self.recompute = pdict['recompute']
        self.precision = pdict['precision']

        if re.search('llama-3', self.tokenizer_model, re.I):
            self.training_script = '/workspace/Megatron-LM/examples/llama/train_llama3.sh'
        elif re.search('llama-2', self.tokenizer_model, re.I):
            self.training_script = '/workspace/Megatron-LM/examples/llama/train_llama2.sh'

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

    def run_pretraining_tasks(
        self,
    ):
        if self.distributed_training is True:
            self.rdma_stats_dict_before = linux_utils.get_rdma_stats_dict(self.phdl)
            self.ethtool_stats_dict_before = linux_utils.get_nic_ethtool_stats_dict(self.phdl)

    def exec_nic_setup_scripts(
        self,
    ):
        """
        Prepare backend NICs inside containers before starting distributed training.

        This method applies in-container NIC setup steps when running distributed jobs.
        It currently implements a Broadcom-specific workaround to ensure the RDMA
        provider library (bnxt_re) is correctly available inside the container.

        Behavior:
         - Only runs when distributed_training is True.
         - If nic_type indicates Broadcom/Thor, it:
          * Forces NCCL GID index to 3 (common Broadcom requirement).
          * Copies the host-side libbnxt_re library into the container?s ibverbs path.
          * Runs ibv_devinfo to verify the RDMA device enumerates as bnxt_.
          * Fails the test if the expected device string is not detected.

        Assumptions:
          - self.phdl provides exec(...) to run commands on all nodes/hosts.
          - Docker is installed and the container is already running on each node.
          - The source and destination library paths are correct for the target image.
          - fail_test(...) is available in scope to abort on setup failures.

        """
        # Run all your backend NIC related bringups for containers here ..
        if self.distributed_training is True:
            # This is a temporary hack needed for broadcom nics to work within containers ..
            if re.search('broadcom|thor', self.nic_type, re.I):
                # override the gid_index to 3 for broadcom
                self.nccl_ib_gid_index = 3
                out_dict = self.phdl.exec(
                    f'docker exec {self.container_name} /bin/bash -c "sudo \
                    cp /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.host \
                    /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so; \
                    sleep 2;ibv_devinfo;sleep 2;"'
                )
                for node in out_dict.keys():
                    if not re.search('hca_id:\s+bnxt_', out_dict[node], re.I):
                        print(out_dict[node])
                        fail_test(f'Broadcom libbnxt rdma driver is not properly copied on node {node}')

    def build_training_job_cmd(
        self,
    ):
        # Construct the main megatron training command
        # Compute the batch size and mini batch size based on the cluster size
        # Add NIC and Socket details for distributed training ..
        cmd = ''

        # cmd = f'docker exec {self.container_name} /bin/bash -c """'

        cmd = (
            cmd
            + 'cd /workspace/Megatron-LM; export MOCK_DATA=1; '
            + f'export IMAGE={self.container_image}; '
            + f'export HF_TOKEN="{self.hf_token}"; '
            + f'export DATA_CACHE_PATH={self.data_cache_dir}; '
            + f'export TOKENIZER_MODEL={self.tokenizer_model}; '
            + 'export LD_LIBRARY_PATH=/usr/local/lib/:/opt/rocm/lib:$LD_LIBRARY_PATH; '
            + f'export LOG_DIR={self.log_dir}; '
            + 'export EXP_NAME="megatron_training"; '
        )

        if self.distributed_training is True:
            # Add the backend network related environment variables ..
            cmd = (
                cmd
                + f'export NCCL_IB_HCA_LIST={self.nccl_ib_hca_list}; '
                + f'export NCCL_IB_HCA={self.nccl_ib_hca_list}; '
                + f'export NCCL_SOCKET_IFNAME={self.nccl_socket_ifname}; '
                + f'export GLOO_SOCKET_IFNAME={self.gloo_socket_ifname}; '
                + f'export NCCL_DEBUG={self.nccl_debug}; '
                + f'export NCCL_IB_GID_INDEX={self.nccl_ib_gid_index}; '
            )

        if self.distributed_training is True:
            # Build CMD List as every node has different node rank
            cmd = (
                cmd
                + f'cd /workspace/Megatron-LM; RECOMPUTE={self.recompute} '
                + f'SEQ_LENGTH={self.sequence_length} '
                + f'MBS={self.micro_batch_size} BS={self.batch_size} '
                + f'TP={self.tensor_parallelism} '
                + f'PP={self.pipeline_parallelism} FSDP={self.fsdp} '
                + f'MODEL_SIZE={self.model_size} TOTAL_ITERS={self.iterations} '
            )
            if self.precision == "TE_BF16":
                cmd = cmd + 'TE_FP8=0 TE_BF16=1 '
            elif self.precision == "TE_F16":
                cmd = cmd + 'TE_FP8=0 TE_F16=1 '
            else:
                cmd = cmd + 'TE_FP8=1 '

            cmd = cmd + f'MASTER_ADDR={self.master_address} NNODES={self.nnodes} '

            for i in range(0, int(self.nnodes)):
                full_cmd = cmd + f'NODE_RANK={i} nohup bash {self.training_script} &'
                script_cmd = f'echo "{full_cmd}" > {self.scripts_dir}/distributed_wrapper_script_{i}.sh;chmod 777 {self.scripts_dir}/distributed_wrapper_script_{i}.sh'
                self.job_cmd_list.append(script_cmd)
        else:
            # Single node training case, run same cmd on all nodes.
            cmd = (
                cmd
                + f'cd /workspace/Megatron-LM; RECOMPUTE={self.recompute} '
                + f'SEQ_LENGTH={self.sequence_length} '
                + f'MBS={self.micro_batch_size} BS={self.batch_size} '
                + f'TP={self.tensor_parallelism} '
                + f'PP={self.pipeline_parallelism} FSDP={self.fsdp} '
                + f'MODEL_SIZE={self.model_size} TOTAL_ITERS={self.iterations} '
            )
            if self.precision == "TE_BF16":
                cmd = cmd + 'TE_FP8=0 TE_BF16=1 '
            elif self.precision == "TE_F16":
                cmd = cmd + 'TE_FP8=0 TE_F16=1 '
            else:
                cmd = cmd + 'TE_FP8=1 '
            self.job_cmd = cmd + f'nohup bash {self.training_script} &'

    def start_training_job(self, timeout=500):
        """
        Launch the Megatron-LM training job (distributed or single-node).

        Behavior:
         - Prints debug information about the prepared commands.
         - Distributed mode:
         * Runs NIC setup workarounds (if any).
         * Creates per-node distributed wrapper scripts across nodes via phdl.exec_cmd_list.
         * Executes those scripts inside each node's container using docker exec.
        - Single-node mode:
         * Writes a single wrapper script locally.
         * Executes the script inside the container.
         * Ensures the training log file is writable.
        - Sleeps for a short period to allow processes to initialize before polling.

        Args:
          timeout (int): Reserved for future use (e.g., health checks). Not currently used.

        Assumptions:
          - self.phdl provides:
          * exec(cmd: str) -> per-node or local command execution
          * exec_cmd_list(cmd_list: List[str]) -> parallel per-node execution
          - Docker is installed and container self.container_name is available on each node.
          - self.job_cmd_list (distributed) or self.job_cmd (single-node) has been populated
           by build_training_job_cmd() prior to invocation.
        """
        print('start training job')
        print(self.job_cmd_list)
        print(self.job_cmd)
        cmd_list = []
        for i in range(0, int(self.nnodes)):
            cmd = f'''docker exec {self.container_name} /bin/bash -c "mkdir -p {self.log_dir}/megatron-logs/out-node{i}"'''
            cmd_list.append(cmd)
        self.phdl.exec_cmd_list(cmd_list)

        cmd_list = []
        for i in range(0, int(self.nnodes)):
            result_training_log = f'{self.log_dir}/megatron-logs/out-node{i}/training.log'
            cmd = f'''docker exec {self.container_name} /bin/bash -c 'sed -i  "/^TRAIN_LOG=/c\TRAIN_LOG={result_training_log}" /workspace/Megatron-LM/examples/llama/train_llama3.sh' '''
            cmd_list.append(cmd)
        self.phdl.exec_cmd_list(cmd_list)

        if self.distributed_training:
            # Run any required NIC setup steps inside containers (e.g., Broadcom workaround)
            self.exec_nic_setup_scripts()
            # Following creates the training script
            out_dict = self.phdl.exec_cmd_list(self.job_cmd_list)

            cmd_list = []
            for i in range(0, int(self.nnodes)):
                cmd = f'docker exec {self.container_name} /bin/bash -c "nohup \
                      {self.scripts_dir}/distributed_wrapper_script_{i}.sh > \
                      {self.log_dir}/megatron-logs/out-node{i}/training.log 2>&1 &"'
                cmd_list.append(cmd)
            out_dict = self.phdl.exec_cmd_list(cmd_list)
        else:
            out_dict = self.phdl.exec(
                f'echo "{self.job_cmd}" > \
               {self.scripts_dir}/single_node_wrapper_script.sh; \
               chmod 777 {self.scripts_dir}/single_node_wrapper_script.sh'
            )
            cmd_list = []
            for i in range(0, int(self.nnodes)):
                cmd = f'docker exec {self.container_name} /bin/bash -c "nohup \
                      {self.scripts_dir}/single_node_wrapper_script.sh > \
                      {self.log_dir}/megatron-logs/out-node{i}/training.log 2>&1 &"'
                cmd_list.append(cmd)
            out_dict = self.phdl.exec_cmd_list(cmd_list)
        time.sleep(50)

    def get_training_results_dict(
        self,
    ):
        """
        Parse training log output from the last node and extract key performance metrics.

        Returns:
        dict: A dictionary with lists of extracted values (strings) for each metric:
        - 'throughput_per_gpu': Matches 'throughput per GPU: <float>'
        - 'tokens_per_gpu': Matches 'tokens/GPU/s: <int>'
        - 'mem_usage': Matches 'mem usages: <float>'
        - 'elapsed_time_per_iteration': Matches 'elapsed time per iteration: <float>'

        Behavior:
        - Reads the consolidated training log file from self.home_dir/training_logs on all hosts.
        - Selects the output from the last host in self.host_list (assumes that node has the final log).
        - Applies regex searches to extract metrics and returns them in a dictionary.
        - Prints the dictionary for quick visibility.

        Assumptions:
        - self.phdl.exec(cmd) returns a dict mapping host -> command output (string).
        - self.host_list is a non-empty list of hosts; the last entry contains the desired log.
        - The training log contains lines that match the expected regex patterns.
        - re is imported in the module scope.
        """

        training_results_dict = {}

        # Read the training log output from the "last" node (assumed authoritative)
        last_node = self.host_list[len(self.host_list) - 1]
        last_node_num = len(self.host_list) - 1
        out_dict = self.phdl.exec(f'cat {self.log_dir}/megatron-logs/out-node{last_node_num}/training.log | tail -15')

        # Select the log content from the last node
        output = out_dict[last_node]

        print('Extracting results from logs')
        print('#===========================#')
        print(output)
        print('#===========================#')

        # Extract throughput per GPU as a list of numbers (strings), if multiple occurrences exist
        # pattern = f'throughput per GPU \(TFLOP/s/GPU\):\s+([0-9\.]+)'
        pattern = 'throughput per GPU:\s+([0-9\.]+)'
        training_results_dict['throughput_per_gpu'] = re.findall(pattern, output, re.I)

        pattern = 'tokens\/GPU\/s:\s+([0-9]+)'
        # Extract tokens per GPU per second (integers as strings)
        training_results_dict['tokens_per_gpu'] = re.findall(pattern, output, re.I)

        # Extract memory usage values (floats as strings)
        pattern = 'mem usages:\s+([0-9\.]+)'
        training_results_dict['mem_usage'] = re.findall(pattern, output, re.I)

        # Extract elapsed time per iteration (floats as strings)
        pattern = 'elapsed time per iteration \(ms\):\s+([0-9\.]+)'
        training_results_dict['elapsed_time_per_iteration'] = re.findall(pattern, output, re.I)

        print(training_results_dict)
        return training_results_dict

    def scan_for_training_errors(
        self,
    ):
        """
        Scan the consolidated training logs for known error patterns and report status.

        Returns:
        bool: True if no error patterns are found; False otherwise.

        Behavior:
        - Reads the training log file from self.home_dir/training_logs via sudo on all hosts.
        - Selects the output from the last host in self.host_list (assumes it has the final log).
        - Iterates through regex patterns in training_err_dict and searches the log content.
        - On first match:
          * Calls fail_test with a descriptive message.
          * Logs an error indicating polling should stop.
          * Marks training_pass as False.
        - Returns training_pass.

        Assumptions:
        - self.phdl.exec(cmd) returns a dict mapping host -> command stdout (string).
        - self.host_list is non-empty and its last element contains the relevant log.
        - training_err_dict is a dict of name -> regex pattern available in scope.
        - re, log, and fail_test are imported/available in scope.
        - sudo can read the training_logs file without interactive prompts.

        Notes:
        - Regex search is case-sensitive as written; pass re.I to re.search for case-insensitive matching.
        - If multiple error patterns are present, all matches will trigger fail_test, but the function
          does not short-circuit; it continues scanning and ultimately returns False.
        - Consider logging or returning which specific patterns matched for better diagnostics.
        """

        print('Scan for training errors')
        training_pass = True  # Default to pass; flip to False if any error pattern is detected

        # Identify the node whose log we treat as authoritative (last in the host list)
        last_node = self.host_list[len(self.host_list) - 1]
        last_node_num = len(self.host_list) - 1

        # Read training logs across nodes; select the last node's output for scanning
        out_dict = self.phdl.exec(f'sudo cat {self.log_dir}/megatron-logs/out-node{last_node_num}/training.log')

        output = out_dict[last_node]

        # Check the log content against all known training error patterns
        for err_key in training_err_dict:
            if re.search(f'{training_err_dict[err_key]}', output):
                # Record failure and log an error for visibility
                fail_test(f'ERROR {training_err_dict[err_key]} seen in training logs ..')
                log.error('Aborting training log polling')
                training_pass = False
        return training_pass

    def poll_for_training_completion(self, time_between_iters=120):
        """
        Periodically poll training logs to detect completion, surface errors, and validate results.

        Args:
        time_between_iters (int | float): Seconds to sleep between each polling iteration.

        Behavior:
        - Waits an initial 60s to allow training to start producing logs.
        - For up to `self.iterations` loops:
          * Invokes self.scan_for_training_errors(); aborts if it flags errors.
          * Reads the consolidated training log from the "last" node in self.host_list.
          * Checks for completion indicators (throughput per GPU or tokens/GPU/s).
        - If not seen, prints a status and sleeps before next iteration.
        - If seen, verifies that metrics do not contain NaN/Inf values.
        - Fails on invalid values, else parses and stores results via get_training_results_dict().
        - Returns on success or failure (no explicit return value).
        - Sleeps `time_between_iters` seconds between iterations (except when it early-sleeps 30s on in-progress).


        Assumptions:
        - self.host_list is non-empty; last node contains authoritative training logs.
        - self.phdl.exec(cmd) returns {node: stdout_str}.
        - self.scan_for_training_errors() returns True when OK, False on error patterns.
        - self.get_training_results_dict() parses known metrics into self.training_results_dict.
        - re, time, and fail_test are available in scope.

        Notes:
        - The regex '[NaN|Inf]' uses a character class and will not match "NaN" or "Inf" as intended.
        Consider using '(NaN|Inf)' to check for either token.
        - The progress regex for tokens/GPU/s lacks a colon; consider 'tokens\\/GPU\\/s:\\s+[0-9]+'.
        """

        print('Poll for training completion ..')
        time.sleep(80)
        last_node = self.host_list[len(self.host_list) - 1]

        last_node = self.host_list[len(self.host_list) - 1]
        last_node_num = len(self.host_list) - 1

        # 10 additional iterations in case time per iteration is longer ..
        for i in range(1, int(self.iterations) + 10):
            print(f'Starting Iteration {i}')
            if not self.scan_for_training_errors():
                fail_test('Failures seen in training logs, Aborting!!!')
                return
            out_dict = self.phdl.exec(f'sudo cat {self.log_dir}/megatron-logs/out-node{last_node_num}/training.log')
            output = out_dict[last_node]

            if not re.search('throughput per GPU:|tokens\/GPU\/s\s+[0-9]+', output, re.I):
                print('Training still in progress')
            else:
                if (
                    re.search('throughput per GPU:\s+[NaN|Inf]', output, re.I)
                    or re.search('tokens\/GPU\/s:\s+[NaN|Inf]', output, re.I)
                    or re.search('mem usages:\s+[NaN|Inf]', output, re.I)
                ):
                    fail_test(f'ERROR - NaN or Inf values seen in training results {output}')
                    return
                else:
                    time.sleep(5)
                    self.training_results_dict = self.get_training_results_dict()
                    print('Completed Training, returning !!!')
                    return
            # Wait secs between every iteration
            time.sleep(int(time_between_iters))

    def verify_training_results(
        self,
    ):
        """
        Validate collected training results and environment health after a training run.

        Behavior:
        - Records end time of training for later log scanning.
        - Scans parsed training_results_dict for NaN/Inf values in any reported metric.
        - If distributed training is enabled:
          * Collects RDMA and NIC (ethtool) stats after training.
          * Verifies selected error counters did not increase vs. their pre-training baselines.
        - Scans kernel logs (dmesg) between training start and end for known error patterns.
        - Compares observed performance results against expected thresholds provided in
          self.expected_result_dict and flags deviations.

        Assumptions:
        - self.phdl.exec(cmd) returns a mapping of node -> command output (string).
        - self.training_results_dict is populated before calling this method and structured
          as: { metric_key: <iterable of metric lists/values> }.
        - self.distributed_training indicates whether to collect/compare network-related stats.
        - linux_utils.get_rdma_stats_dict and linux_utils.get_nic_ethtool_stats_dict return
          per-node dictionaries of counters where values are numeric strings or ints.
        - err_counters_pattern is a regex pattern for error counters to check.
        - verify_dmesg_for_errors(phdl, start_time_dict, end_time_dict) is available and
          scans logs between provided timestamps mapped by node.
        - self.training_start_time and self.training_end_time are dicts keyed by node with
          human-readable timestamps, compatible with verify_dmesg_for_errors.
        - self.expected_result_dict contains numeric thresholds as strings or numbers.

        Side effects:
        - Calls fail_test(...) to report errors and accumulate failure messages.
        - Logs warnings for missing expected result keys.

        Returns:
        None. Uses fail_test to record failures.
        """

        # across nodes what numbers we are getting - median variance, per iteration variance.
        # Network errors

        # Record the training end time; used later for dmesg time-bounded scanning
        self.training_end_time = self.phdl.exec('date')

        print('#==================================================#')
        print('\t\tTraining Results')
        print(self.training_results_dict)
        print('#==================================================#')
        # Check the parsed training results for invalid numeric values (NaN/Inf)
        if not self.training_results_dict:
            fail_test(
                'Failed to populate training results, training_results_dict is empty - please check logs for failures'
            )

        for result_key in self.training_results_dict.keys():
            for result_list in self.training_results_dict[result_key]:
                for result_val in result_list:
                    # Search for 'nan' or 'inf' (case-sensitive as written; add re.I if desired)
                    if re.search('nan|inf', result_val):
                        fail_test(
                            f'Failures seen in training_result dict for {result_key}, numbers are either NaN or Inf - f{result_val}'
                        )

        # Check if RDMA and Ethtool stats have errors ..
        if self.distributed_training is True:
            if self.verify_network_errors is True:
                self.rdma_stats_dict_after = linux_utils.get_rdma_stats_dict(self.phdl)
                self.ethtool_stats_dict_after = linux_utils.get_nic_ethtool_stats_dict(self.phdl)

                # Compare RDMA error counters; fail if any error counter increased
                for node in self.rdma_stats_dict_after.keys():
                    for counter_nam in self.rdma_stats_dict_after[node]:
                        if re.search(f'{err_counters_pattern}', counter_nam, re.I):
                            if int(self.rdma_stats_dict_after[node][counter_nam]) > int(
                                self.rdma_stats_dict_before[node][counter_nam]
                            ):
                                fail_test(
                                    f'Error counter {counter_nam} has gone up after training on node {node} \
                                 Before = {self.rdma_stats_dict_before[node][counter_nam]}, \
                                 After = {self.rdma_stats_dict_after[node][counter_nam]}'
                                )

                # Compare NIC error counters; fail if any error counter increased
                for node in self.ethtool_stats_dict_after.keys():
                    for counter_nam in self.ethtool_stats_dict_after[node]:
                        if re.search(f'{err_counters_pattern}', counter_nam, re.I):
                            if int(self.ethtool_stats_dict_after[node][counter_nam]) > int(
                                self.ethtool_stats_dict_before[node][counter_nam]
                            ):
                                fail_test(
                                    f'Error counter {counter_nam} has gone up after training on node {node} \
                                  Before = {self.ethtool_stats_dict_before[node][counter_nam]}, \
                                  After = {self.ethtool_stats_dict_after[node][counter_nam]}'
                                )

        # Scan Dmesg for errors ..
        verify_dmesg_for_errors(self.phdl, self.training_start_time, self.training_end_time)

        print('^^^^^^^^^^^^^^^^^^^^')
        print('training_results_dict')
        print('^^^^^^^^^^^^^^^^^^^^')
        print(self.training_results_dict)
        # Compare perf expected numbers from input JSON file ..
        for result_key in self.training_results_dict.keys():
            if result_key in self.expected_result_dict:
                print(self.training_results_dict[result_key])
                # check if all nodes have met the expected perf numbers
                for actual_perf in self.training_results_dict[result_key]:
                    if float(actual_perf) < float(self.expected_result_dict[result_key]):
                        fail_test(
                            f'The Training performance numbers are below expected numbers for \
                           {result_key}, expected = {self.expected_result_dict[result_key]}, \
                           actual = {actual_perf}'
                        )
            else:
                log.warn(f'Perf result key {result_key} not provided in input JSON file, so will not be checked')
