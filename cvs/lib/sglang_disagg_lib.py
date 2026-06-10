'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json
import os
import re
import time

from cvs.lib import globals
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *


log = globals.log

inference_err_dict = {
    'NCCL ERROR': 'NCCL ERROR|NCCL timeout|local work queue catastrophic error',
    'GPU HW ERROR': 'HW Exception by GPU|GPU Hang|Uncorrectable error|GPU Reset',
    'AssertionError': 'AssertionError|ValueError:|During handling of the above exception|triggered the following exception|RuntimeError|Python error: Aborted',
    'rocm Err': 'FAILED_PRECONDITION: No visible GPU devices|failed call to hipInit: HIP_ERROR_NoDevice|librocm reported version is: NOT_FOUND',
    'python err': 'ModuleNotFoundError: No module named|Fatal Python error:',
    'resource': 'RESOURCE_EXHAUSTED: Out of memory|failed: RESOURCE_EXHAUSTED|urllib.error.URLError|ConnectionRefusedError,HSA_STATUS_ERROR_OUT_OF_RESOURCES',
    'app_err': 'Service Unavailable|No decode workers available|No prefill workers available|Please check if decode servers are configured and healthy|Please check if prefill servers are configured and healthy|Cannot access gated repo|You must have access to it and be authenticated',
}

err_counters_pattern = 'err|retransmit|drop|discard|naks|invalid|oflow|out_of_buffer|reset|fail'


def textwrap_for_yml(msg_string):
    return '\n'.join([m.lstrip() for m in msg_string.split('\n')])


def _is_lower_better(metric_name):
    """Return True if a smaller value is better for the given metric.

    Latency-style metrics (anything ending in `_ms` or mentioning latency /
    ttft / tpot / itl / e2el) are better when lower; throughput, count and
    accuracy metrics are better when higher. This reproduces the historical
    comparison direction used by verify_inference_results while being explicit.
    """
    return bool(re.search(r'_ms$|latency|ttft|tpot|itl|e2el', metric_name, re.I))


def parse_gsm8k_metrics(text):
    """Extract gsm8k bench_sglang.py metrics from one node's stdout.

    The throughput is stored under `tokens_per_sec` so it lines up with the
    gsm8k `expected_results` threshold key.

    Returns a dict with any of: accuracy, invalid, latency_s, tokens_per_sec.
    """
    patterns = {
        'accuracy': r'Accuracy:\s+([0-9.]+)',
        'invalid': r'Invalid:\s+([0-9.]+)',
        'latency_s': r'Latency:\s+([0-9.]+)\s*s',
        'tokens_per_sec': r'Output throughput:\s+([0-9.]+)\s+token',
    }
    metrics = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.I)
        if match:
            metrics[key] = match.group(1)
    return metrics


def format_metrics_table(test_name, results_dict, expected_dict=None):
    """Render a uniform, greppable per-node metrics block.

    Args:
        test_name: label for the block header (e.g. 'gsm8k').
        results_dict: {node: {metric: value}}.
        expected_dict: optional {metric: threshold}. When provided, each
            metric that has a threshold gets a `[expected <op> <t>]  <verdict>`
            suffix (PASS/FAIL, or UNPARSED if the value isn't numeric).

    Returns a multi-line string (one block per node).
    """
    expected_dict = expected_dict or {}
    lines = []
    for node in results_dict.keys():
        metrics = results_dict[node]
        lines.append(f'================ METRICS [{test_name}] node={node} ================')
        width = max((len(m) for m in metrics), default=0)
        for metric in metrics:
            value = metrics[metric]
            line = f'  {metric:<{width}} = {value}'
            if metric in expected_dict:
                lower_better = _is_lower_better(metric)
                op = '<=' if lower_better else '>='
                try:
                    actual_f = float(value)
                    exp_f = float(expected_dict[metric])
                    ok = actual_f <= exp_f if lower_better else actual_f >= exp_f
                    verdict = 'PASS' if ok else 'FAIL'
                except (TypeError, ValueError):
                    verdict = 'UNPARSED'
                line += f'   [expected {op} {expected_dict[metric]}]  {verdict}'
            lines.append(line)
        lines.append('=' * 66)
    return '\n'.join(lines)


# Ordered (label, key) pairs for the full sglang.bench_serving "Serving
# Benchmark Result" block. Longer labels that are prefixes of shorter ones are
# listed first so the generic `<label>:` match doesn't cross-match. Keys for
# gated metrics (output_throughput_per_sec, mean_ttft_ms, mean_tpot_ms) are
# preserved so threshold checks keep working.
_BENCH_SERV_FIELDS = [
    ('Successful requests', 'successful_requests'),
    ('Benchmark duration (s)', 'benchmark_duration_s'),
    ('Total input text tokens', 'total_input_text_tokens'),
    ('Total input tokens', 'total_input_tokens'),
    ('Total generated tokens (retokenized)', 'total_generated_tokens_retokenized'),
    ('Total generated tokens', 'total_generated_tokens'),
    ('Request throughput (req/s)', 'request_throughput_req_per_sec'),
    ('Input token throughput (tok/s)', 'input_token_throughput_per_sec'),
    ('Output token throughput (tok/s)', 'output_throughput_per_sec'),
    ('Peak output token throughput (tok/s)', 'peak_output_throughput_per_sec'),
    ('Peak concurrent requests', 'peak_concurrent_requests'),
    ('Total token throughput (tok/s)', 'total_token_throughput_per_sec'),
    ('Concurrency', 'concurrency'),
    ('Mean E2E Latency (ms)', 'mean_e2e_ms'),
    ('Median E2E Latency (ms)', 'median_e2e_ms'),
    ('P90 E2E Latency (ms)', 'p90_e2e_ms'),
    ('P99 E2E Latency (ms)', 'p99_e2e_ms'),
    ('Mean TTFT (ms)', 'mean_ttft_ms'),
    ('Median TTFT (ms)', 'median_ttft_ms'),
    ('P99 TTFT (ms)', 'p99_ttft_ms'),
    ('Mean TPOT (ms)', 'mean_tpot_ms'),
    ('Median TPOT (ms)', 'median_tpot_ms'),
    ('P99 TPOT (ms)', 'p99_tpot_ms'),
    ('Mean ITL (ms)', 'mean_itl_ms'),
    ('Median ITL (ms)', 'median_itl_ms'),
    ('P95 ITL (ms)', 'p95_itl_ms'),
    ('P99 ITL (ms)', 'p99_itl_ms'),
    ('Max ITL (ms)', 'max_itl_ms'),
]


def parse_bench_serv_metrics(text):
    """Parse the full sglang.bench_serving 'Serving Benchmark Result' block.

    Uses a generic `<escaped label>:\\s+<number>` match for every known field,
    which avoids the historical unescaped-paren bug (e.g. `Median TTFT (ms):`)
    and the `E2EL`-vs-`E2E Latency` mismatch. Returns {key: value_str}.
    """
    metrics = {}
    for label, key in _BENCH_SERV_FIELDS:
        match = re.search(re.escape(label) + r':\s+([0-9.]+)', text)
        if match:
            metrics[key] = match.group(1)
    return metrics


def parse_bench_serv_per_request(jsonl_text):
    """Parse the per-request arrays from a bench_serving --output-details JSONL.

    The file (one JSON object per line) carries parallel arrays input_lens,
    output_lens, ttfts, itls, errors. Returns a list of per-request dicts.
    """
    lines = [ln for ln in jsonl_text.strip().split('\n') if ln.strip().startswith('{')]
    if not lines:
        return []
    data = json.loads(lines[-1])
    input_lens = data.get('input_lens', [])
    output_lens = data.get('output_lens', [])
    ttfts = data.get('ttfts', [])
    itls = data.get('itls', [])
    errors = data.get('errors', [])
    rows = []
    for i in range(len(output_lens)):
        ttft = ttfts[i] if i < len(ttfts) else None
        err = errors[i] if i < len(errors) else ''
        rows.append(
            {
                'req': i,
                'input_len': input_lens[i] if i < len(input_lens) else '',
                'output_len': output_lens[i],
                'ttft_ms': round(ttft * 1000, 2) if isinstance(ttft, (int, float)) else '',
                'gen_tokens': (len(itls[i]) + 1) if i < len(itls) and isinstance(itls[i], list) else '',
                'error': (err or '')[:60],
            }
        )
    return rows


def parse_gsm8k_per_question(jsonl_text):
    """Parse the gsm8k --raw-result-file dump (one {prompt_id,prompt,output,
    correct} JSON per line). Returns a list of per-question dicts (the bulky
    prompt/output text is dropped; a short output preview is kept)."""
    rows = []
    for line in jsonl_text.strip().split('\n'):
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        rows.append(
            {
                'q': d.get('prompt_id'),
                'correct': d.get('correct'),
                'output': (d.get('output') or '').replace('\n', ' ')[:60],
            }
        )
    return rows


def format_per_request_table(test_name, rows, columns, limit=None):
    """Render a compact per-request / per-question table (one row per item).

    limit caps the number of displayed rows (None = all); a trailing note shows
    how many were elided.
    """
    total = len(rows)
    shown = rows if limit in (None, 0) else rows[:limit]
    lines = [f'================ PER-ITEM [{test_name}] ({total} items) ================']
    widths = {c: max(len(c), *(len(str(r.get(c, ''))) for r in shown)) if shown else len(c) for c in columns}
    lines.append('  ' + '  '.join(f'{c:<{widths[c]}}' for c in columns))
    for r in shown:
        lines.append('  ' + '  '.join(f'{str(r.get(c, "")):<{widths[c]}}' for c in columns))
    if limit not in (None, 0) and total > limit:
        lines.append(f'  ... ({total - limit} more; full detail in the run log / node artifact)')
    lines.append('=' * 66)
    return '\n'.join(lines)


class SglangDisaggPD:
    def __init__(
        self,
        model_name,
        inference_config_dict,
        benchmark_params_dict,
        hf_token,
        p_phdl=None,
        d_phdl=None,
        r_phdl=None,
        b_phdl=None,
        gpu_type='mi300',
        user_name=None,
        priv_key_file=None,
    ):
        """
        Initialize a Disaggregated Prefill/Decode (PD) inference controller
        for SGLang.

        This class encapsulates:
          - Cluster topology (prefill, decode, proxy, benchmark nodes)
          - SSH-based remote execution (via Pssh handlers)
          - Inference configuration (networking, containers, env vars)
          - Benchmark configuration (load, concurrency, prompt sizes)

        Args:
            model_name (str): HuggingFace or local model identifier
            inference_config_dict (dict): Cluster and runtime configuration
            benchmark_params_dict (dict): Benchmark workload parameters
            hf_token (str): HuggingFace access token
            p_phdl, d_phdl, r_phdl, b_phdl: Optional pre-created SSH handlers
            gpu_type (str): GPU type (e.g., mi300, mi325)
            user_name (str): SSH username for remote nodes
            priv_key_file (str): SSH private key file
        """

        # ------------------------------------------------------------------
        # Basic identity and authentication parameters
        # ------------------------------------------------------------------
        self.user_name = user_name
        self.priv_key_file = priv_key_file
        self.model_name = model_name
        self.hf_token = hf_token
        self.gpu_type = gpu_type

        # ------------------------------------------------------------------
        # Store inference and benchmark configuration dictionaries
        # These are typically loaded from a JSON/YAML configuration file
        # ------------------------------------------------------------------
        self.inf_dict = inference_config_dict
        self.bp_dict = benchmark_params_dict

        self.model_name = model_name
        self.hf_token = hf_token
        self.gpu_type = gpu_type

        # ------------------------------------------------------------------
        # Extract cluster topology for disaggregated inference
        #
        # Prefill nodes  : Handle prompt ingestion + KV cache creation
        # Decode nodes   : Handle token generation
        # Proxy node     : Routes requests between prefill/decode
        # Benchmark node : Generates inference load
        # ------------------------------------------------------------------
        self.prefill_node_list = self.inf_dict['prefill_node_list']
        self.decode_node_list = self.inf_dict['decode_node_list']
        self.prefill_nnodes = len(self.prefill_node_list)
        self.decode_nnodes = len(self.decode_node_list)

        self.proxy_node = list(self.inf_dict['proxy_router_node'])
        self.benchmark_serv_node = list(self.inf_dict['benchmark_serv_node'])

        # ------------------------------------------------------------------
        # SSH handlers for each node group
        #
        # p_phdl : Prefill nodes
        # d_phdl : Decode nodes
        # r_phdl : Proxy/router node
        # b_phdl : Benchmark client node
        # ------------------------------------------------------------------
        self.p_phdl = p_phdl
        self.d_phdl = d_phdl
        self.r_phdl = r_phdl
        self.b_phdl = b_phdl

        if self.p_phdl is None:
            self.p_phdl = Pssh(log, self.prefill_node_list, user=self.user_name, pkey=self.priv_key_file)

        if self.d_phdl is None:
            self.d_phdl = Pssh(log, self.decode_node_list, user=self.user_name, pkey=self.priv_key_file)

        if self.r_phdl is None:
            self.r_phdl = Pssh(log, self.proxy_node, user=self.user_name, pkey=self.priv_key_file)

        if self.b_phdl is None:
            self.b_phdl = Pssh(log, self.benchmark_serv_node, user=self.user_name, pkey=self.priv_key_file)

        self.job_cmd = ''
        self.job_cmd_list = []
        self.inference_results_dict = {}
        log.info("%s", self.gpu_type)

        # ------------------------------------------------------------------
        # Extract commonly used inference parameters for convenience
        # ------------------------------------------------------------------
        # Needed only in the case of distributed inference - placeholder for future
        # Intialize cluster stats dicts ..
        self.rdma_stats_dict_before = {}
        self.ethtool_stats_dict_before = {}
        self.rdma_stats_dict_after = {}
        self.inference_start_time = p_phdl.exec('date +"%a %b %e %H:%M"')
        self.inference_end_time = None

        # ------------------------------------------------------------------
        # Set default benchmark parameters if not provided
        # These control request generation and performance measurement
        # ------------------------------------------------------------------
        self.home_dir = os.path.expanduser("~")
        self.inf_dict.setdefault('container_image', 'lmsysorg/sglang:dev')
        self.inf_dict.setdefault('container_name', 'sglang_container')
        self.inf_dict.setdefault('nic_type', 'ainic')
        self.inf_dict.setdefault('nccl_ib_hca_list', 'rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7')
        self.inf_dict.setdefault('nccl_ib_hca', 'rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7')
        self.inf_dict.setdefault('nccl_socket_ifname', 'eno0')
        self.inf_dict.setdefault('gloo_socket_ifname', 'eno0')
        self.inf_dict.setdefault('nccl_ib_gid_index', '1')
        self.inf_dict.setdefault('nccl_debug', 'ERROR')
        self.inf_dict.setdefault('data_cache_dir', f'{self.home_dir}/cache')
        self.inf_dict.setdefault('log_dir', f'{self.home_dir}/LOG_DIR')
        self.inf_dict.setdefault('max_concurrent_requests', '-1')
        self.inf_dict.setdefault('queue_size', '100')
        self.inf_dict.setdefault('queue_timeout_secs', '60')
        self.inf_dict.setdefault('max_retries', '5')

        log.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        log.info(f'inference_dict = {self.inf_dict}')
        log.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        self.container_image = self.inf_dict['container_image']
        self.container_name = self.inf_dict['container_name']

        self.nic_type = self.inf_dict['nic_type']
        self.nccl_ib_hca_list = self.inf_dict['nccl_ib_hca_list']
        self.nccl_ib_hca = self.inf_dict['nccl_ib_hca']
        self.nccl_socket_ifname = self.inf_dict['nccl_socket_ifname']
        self.gloo_socket_ifname = self.inf_dict['gloo_socket_ifname']
        self.nccl_ib_gid_index = self.inf_dict['nccl_ib_gid_index']
        self.nccl_debug = self.inf_dict['nccl_debug']
        self.data_cache_dir = self.inf_dict['data_cache_dir']
        self.log_dir = self.inf_dict['log_dir']

        # set defaults for benchmark param dict if not passed via JSON file
        self.bp_dict.setdefault('backend', 'sglang')
        self.bp_dict.setdefault('dataset_name', 'sharegpt')
        self.bp_dict.setdefault('max_concurrency', '64')
        self.bp_dict.setdefault('model', 'openai/gpt-oss-120b')
        self.bp_dict.setdefault('num_prompts', '1000')
        self.bp_dict.setdefault('input_sequence_length', '8192')
        self.bp_dict.setdefault('burstiness', '1.0')
        self.bp_dict.setdefault('seed', '0')
        self.bp_dict.setdefault('request_rate', 'inf')
        self.bp_dict.setdefault('max_model_length', '9216')
        self.bp_dict.setdefault('random_range_ration', '1.0')
        self.bp_dict.setdefault('random_prefix_len', '0')
        self.bp_dict.setdefault('tensor_parallelism', '8')
        self.bp_dict.setdefault('port_no', '8000')
        self.bp_dict.setdefault('tokenizer_mode', 'auto')
        self.bp_dict.setdefault('percentile_metrics', 'ttft,tpot,itl,e2el')
        self.bp_dict.setdefault('metric_percentiles', '99')
        self.bp_dict.setdefault('inference_poll_iterations', '16')

        self.inference_poll_iterations = self.bp_dict['inference_poll_iterations']

        log.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        log.info(f'benchmark_params_dict = {self.bp_dict}')
        log.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    def install_container_packages(
        self,
    ):
        """
        Install required system networking utilities inside inference containers.

        Purpose:
        --------
        This method prepares the container environment for distributed inference
        by installing basic networking and diagnostic tools that are commonly
        needed for:
        - Connectivity validation between nodes
        - Debugging network paths (ping, ip route, ifconfig)
        - Verifying NIC and routing configuration
        - Troubleshooting NCCL/Gloo/RDMA-related issues

        These tools are installed inside the running container on:
        - Prefill nodes
        - Decode nodes
        - Proxy/router nodes
        """

        log.info('Run pre inference tasks')
        # Install ip tools
        cmd = f'docker exec {self.container_name} /bin/bash -c " \
            sudo apt -y update; \
            sudo apt install -y iputils-ping; \
            sudo apt install -y iproute2; \
            sudo apt install -y net-tools" '
        self.p_phdl.exec(cmd)
        self.d_phdl.exec(cmd)
        self.r_phdl.exec(cmd)

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

        # This is a temporary hack needed for broadcom nics to work within containers ..
        if re.search('broadcom|thor', self.nic_type, re.I):
            # override the gid_index to 3 for broadcom
            self.nccl_ib_gid_index = 3
            # Copy the host bnxt_re userspace driver into place, then run
            # ibv_devinfo to confirm RDMA devices enumerate. HCA names are
            # platform-dependent (e.g. bnxt_re0 or rocepXXs0), so verify that
            # devices are present rather than matching a fixed name prefix.
            cmd = f'docker exec {self.container_name} /bin/bash -c "sudo \
                    cp /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.host \
                    /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so; \
                    ibv_devinfo;" '
            pout_dict = self.p_phdl.exec(cmd)
            dout_dict = self.d_phdl.exec(cmd)
            for out_dict in (pout_dict, dout_dict):
                for node in out_dict.keys():
                    if re.search('No IB devices found', out_dict[node], re.I) or not re.search(
                        'hca_id:', out_dict[node], re.I
                    ):
                        log.info("%s", out_dict[node])
                        fail_test(f'RDMA devices not visible after bnxt driver copy on node {node}')

    def check_ibv_devices(
        self,
    ):
        """
        Verify that InfiniBand / RDMA devices are visible inside the container
        on all relevant nodes.

        Purpose:
        --------
        This method ensures that RDMA-capable devices (e.g., InfiniBand HCAs)
        are correctly exposed inside the container environment. This is a
        critical prerequisite for:
        - NCCL / RCCL over RDMA
        - High-performance distributed inference
        - Low-latency, high-bandwidth GPU communication

        The check is performed on:
        - Prefill nodes
        - Decode nodes

        Proxy and benchmark nodes typically do not require RDMA access.
        """
        for hdl in [self.p_phdl, self.d_phdl]:
            cmd = f'''docker exec {self.container_name} /bin/bash -c "ibv_devinfo" '''
            out_dict = hdl.exec(cmd)
            for node in out_dict.keys():
                if re.search('No IB devices found', out_dict[node], re.I):
                    fail_test(f'IB devices not seen inside the container for node {node}')

    def setup_prefill_container_env(
        self,
    ):
        # Env setup for Prefill Nodes ..
        p_cmd = f'''docker exec {self.container_name} /bin/bash -c "echo '

                    export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
                    export NCCL_DEBUG={self.inf_dict['nccl_debug']}
                    export NCCL_IB_HCA={self.inf_dict['nccl_ib_hca']}
                    export NCCL_IB_GID_INDEX={self.inf_dict['nccl_ib_gid_index']}
                    export HSA_FORCE_FINE_GRAIN_PCIE=1
                    export NCCL_SOCKET_IFNAME={self.inf_dict['nccl_socket_ifname']}
                    export GLOO_SOCKET_IFNAME={self.inf_dict['gloo_socket_ifname']}
                    export GLOO_TCP_IFNAME={self.inf_dict['gloo_socket_ifname']}
                    export HSA_FORCE_FINE_GRAIN_PCIE=1

                    export MASTER_PREFILL_ADDR={self.inf_dict['prefill_coordinator_addr']}
                    export MASTER_PREFILL_PORT={self.inf_dict['prefill_coordinator_port']}

                    export MODEL={self.bp_dict['model']}
                    export TP={self.bp_dict['tensor_parallelism']}
                    export HF_TOKEN={self.hf_token}
                    '  > /tmp/prefill_env_script.sh"
                    '''
        time.sleep(3)
        formatted_p_cmd = textwrap_for_yml(p_cmd)
        self.p_phdl.exec(formatted_p_cmd)
        cmd = f'''docker exec {self.container_name} /bin/bash -c " \
                   chmod 755 /tmp/prefill_env_script.sh; /tmp/prefill_env_script.sh" '''
        self.p_phdl.exec(cmd)

    def setup_decode_container_env(
        self,
    ):
        # Env setup for Decode Nodes ..
        d_cmd = f'''docker exec {self.container_name} /bin/bash -c "echo '

                    export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
                    export NCCL_DEBUG={self.inf_dict['nccl_debug']}
                    export NCCL_IB_HCA={self.inf_dict['nccl_ib_hca']}
                    export NCCL_IB_GID_INDEX={self.inf_dict['nccl_ib_gid_index']}
                    export HSA_FORCE_FINE_GRAIN_PCIE=1
                    export NCCL_SOCKET_IFNAME={self.inf_dict['nccl_socket_ifname']}
                    export GLOO_SOCKET_IFNAME={self.inf_dict['gloo_socket_ifname']}
                    export GLOO_TCP_IFNAME={self.inf_dict['gloo_socket_ifname']}
                    export HSA_FORCE_FINE_GRAIN_PCIE=1

                    export MASTER_DECODE_ADDR={self.inf_dict['decode_coordinator_addr']}
                    export MASTER_DECODE_PORT={self.inf_dict['decode_coordinator_port']}

                    export MODEL={self.bp_dict['model']}
                    export TP={self.bp_dict['tensor_parallelism']}
                    export HF_TOKEN={self.hf_token}
                    '  > /tmp/decode_env_script.sh"
                    '''
        time.sleep(3)
        formatted_d_cmd = textwrap_for_yml(d_cmd)
        self.d_phdl.exec(formatted_d_cmd)
        cmd = f'''docker exec {self.container_name} /bin/bash -c " \
                   chmod 755 /tmp/decode_env_script.sh; /tmp/decode_env_script.sh" '''
        self.d_phdl.exec(cmd)

    def setup_proxy_router_container_env(
        self,
    ):
        # Env setup for Proxy Router Node ..
        r_cmd = f'''docker exec {self.container_name} /bin/bash -c "echo '

                    export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
                    export NCCL_DEBUG={self.inf_dict['nccl_debug']}
                    export NCCL_IB_HCA={self.inf_dict['nccl_ib_hca']}
                    export NCCL_IB_GID_INDEX={self.inf_dict['nccl_ib_gid_index']}
                    export HSA_FORCE_FINE_GRAIN_PCIE=1
                    export NCCL_SOCKET_IFNAME={self.inf_dict['nccl_socket_ifname']}
                    export GLOO_SOCKET_IFNAME={self.inf_dict['gloo_socket_ifname']}
                    export GLOO_TCP_IFNAME={self.inf_dict['gloo_socket_ifname']}
                    export HSA_FORCE_FINE_GRAIN_PCIE=1

                    export HF_TOKEN={self.hf_token}
                    '  > /tmp/router_env_script.sh"
                    '''
        time.sleep(3)
        formatted_r_cmd = textwrap_for_yml(r_cmd)
        self.r_phdl.exec(formatted_r_cmd)
        cmd = f'''docker exec {self.container_name} /bin/bash -c " \
                   chmod 755 /tmp/router_env_script.sh; /tmp/router_env_script.sh" '''
        self.r_phdl.exec(cmd)

    def setup_benchmark_serv_container_env(
        self,
    ):
        # Env setup for Benchserv node ..
        b_cmd = f'''docker exec {self.container_name} /bin/bash -c "echo '

                    export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
                    export NCCL_DEBUG={self.inf_dict['nccl_debug']}
                    export NCCL_IB_HCA={self.inf_dict['nccl_ib_hca']}
                    export NCCL_IB_GID_INDEX={self.inf_dict['nccl_ib_gid_index']}
                    export HSA_FORCE_FINE_GRAIN_PCIE=1
                    export NCCL_SOCKET_IFNAME={self.inf_dict['nccl_socket_ifname']}
                    export GLOO_SOCKET_IFNAME={self.inf_dict['gloo_socket_ifname']}
                    export GLOO_TCP_IFNAME={self.inf_dict['gloo_socket_ifname']}
                    export HSA_FORCE_FINE_GRAIN_PCIE=1
                    export HF_TOKEN={self.hf_token}
                    '  > /tmp/benchmark_env_script.sh"
                    '''
        time.sleep(3)
        formatted_b_cmd = textwrap_for_yml(b_cmd)
        self.b_phdl.exec(formatted_b_cmd)
        cmd = f'''docker exec {self.container_name} /bin/bash -c " \
                   chmod 755 /tmp/benchmark_env_script.sh; /tmp/benchmark_env_script.sh" '''
        self.b_phdl.exec(cmd)
        time.sleep(5)

    def run_test_rmsnorm(self, max_jobs=192):
        """
        Run RMSNorm 2D operator tests inside the SGLang container across
        relevant nodes and validate correctness.

        Purpose:
        --------
        This method executes the AITER RMSNorm 2D operator test, which validates:
        - Correctness of RMSNorm kernel implementation
        - Stability under high parallel job execution
        - GPU kernel behavior under concurrent workloads

        The test is executed on:
        - Prefill nodes
        - Decode nodes
        - Proxy/router nodes

        Args:
        max_jobs (int): Maximum number of concurrent jobs to launch within
                        the RMSNorm test to stress the kernel.
        """
        log.info('#================ * * * =========================#')
        log.info('Run rmsnorm2d')
        log.info('#================ * * * =========================#')
        # ------------------------------------------------------------------
        # Construct command to run RMSNorm test inside the container
        #
        # Details:
        #   - MAX_JOBS controls parallelism inside the test
        #   - Output is redirected to a per-container log file
        #   - Command is executed in the background to allow parallel execution
        # ------------------------------------------------------------------
        cmd = f'''docker exec {self.container_name} /bin/bash -c  "MAX_JOBS={max_jobs} \
                python /sgl-workspace/aiter/op_tests/test_rmsnorm2d.py > /tmp/rsmnorm_test.log 2>&1 &" '''
        for hdl in [self.p_phdl, self.d_phdl, self.r_phdl]:
            out_dict = hdl.exec(cmd)
        log.info('Wait 180 secs for tests to complete')
        time.sleep(180)
        for hdl in [self.p_phdl, self.d_phdl, self.r_phdl]:
            cmd = f'''docker exec {self.container_name} /bin/bash -c  "cat /tmp/rsmnorm_test.log" '''
            out_dict = hdl.exec(cmd)
            for node in out_dict.keys():
                if re.search('fail', out_dict[node], re.I):
                    log.warning(f'Some failures observed in test rmsnorm on node {node}')
                    fail_test(f'Some failures observed in test rmsnorm on node {node}')

    # supported --dtype {auto,half,float16,bfloat16,float,float32}
    # supported --kv-cache-dtype {auto,fp8_e5m2,fp8_e4m3,bf16,bfloat16,fp4_e2m1}
    def launch_prefill_servers(self, dtype='auto', kv_cache_dtype='auto'):
        """
        Generate and stage Prefill server launch scripts on all Prefill nodes
        for SGLang disaggregated inference.

        Purpose:
        --------
        This method prepares the launch script for SGLang Prefill servers.
        In disaggregated PD (Prefill / Decode) mode:
        - Prefill servers are responsible for processing input prompts
        - They generate KV cache entries
        - KV cache is later consumed by Decode servers

        This method:
        - Creates one launch script per Prefill node
        - Sets distributed environment variables (NNODES, NODE_RANK)
        - Configures SGLang for Prefill-only execution
        - Does NOT start the servers yet; it stages the script for later execution

        Args:
        dtype (str): Model compute datatype (e.g., fp16, bf16, auto)
        kv_cache_dtype (str): KV cache datatype (e.g., fp16, bf16, auto)
        """
        log.info('#================ * * * =========================#')
        log.info('Create Prefill launch script on Prefill nodes')
        log.info('#================ * * * =========================#')

        cmd_list = []
        prefill_node_list = self.inf_dict['prefill_node_list']
        log.info('%%%% self.prefill_nnodes {}'.format(self.prefill_nnodes))
        for i in range(0, int(self.prefill_nnodes)):
            cmd = f'''docker exec {self.container_name} /bin/bash -c  "echo  '
                      export NNODES={self.prefill_nnodes}
                      export NODE_RANK={i}
                      export SGLANG_USE_AITER=1
                      python3 -m sglang.launch_server --model {self.bp_dict['model']} \
                              --disaggregation-mode prefill \
                              --disaggregation-ib-device {self.inf_dict['nccl_ib_hca']} \
                              --host {prefill_node_list[i]} \
                              --port {self.inf_dict['prefill_serv_port']} \
                              --dtype {dtype} \
                              --kv-cache-dtype {kv_cache_dtype} \
                              --trust-remote-code \
                              --tp {self.bp_dict['tensor_parallelism']} \
                              --disable-radix-cache --disable-cuda-graph \
                              --mem-fraction-static {self.bp_dict['memory_fraction']} \
                              --attention-backend aiter \
                              --log-level {self.inf_dict['log_level']}' > /tmp/prefill_launch_script.sh" '''
            formatted_cmd = textwrap_for_yml(cmd)
            cmd_list.append(formatted_cmd)
        log.info('%%%%%%%%%%%%%%%%%%%')
        log.info("%s", cmd_list)
        log.info('%%%%%%%%%%%%%%%%%%%')
        self.p_phdl.exec_cmd_list(cmd_list)
        log.info('#================ * * * =========================#')
        log.info('Launching Prefill servers on Prefill nodes')
        log.info('#================ * * * =========================#')
        cmd_list = []
        for i in range(0, int(self.prefill_nnodes)):
            cmd = f'''docker exec {self.container_name} /bin/bash -c " \
                   chmod 755 /tmp/prefill_launch_script.sh; \
                   mkdir -p {self.log_dir}/prefill_node{i}; \
                   source /tmp/prefill_env_script.sh && \
                   nohup /tmp/prefill_launch_script.sh > \
                   {self.log_dir}/prefill_node{i}/prefill_server.log 2>&1 &" '''
            formatted_cmd = textwrap_for_yml(cmd)
            cmd_list.append(formatted_cmd)
        self.p_phdl.exec_cmd_list(cmd_list)
        time.sleep(5)

    def launch_decode_servers(self, dtype='auto', kv_cache_dtype='auto'):
        """
        Generate and deploy Decode server launch scripts on all Decode nodes
        for SGLang disaggregated inference.

        Purpose:
        --------
        In disaggregated PD (Prefill / Decode) inference:
        - Decode servers are responsible for token generation
        - They consume KV cache generated by Prefill servers
        - They perform the latency- and throughput-critical decode loop

        This method:
        - Creates one Decode launch script per Decode node
        - Sets distributed environment variables (NNODES, NODE_RANK)
        - Configures SGLang for Decode-only execution
        - Deploys the scripts to Decode nodes for later execution

        Args:
        dtype (str): Model compute datatype (e.g., fp16, bf16, auto)
        kv_cache_dtype (str): KV cache datatype (e.g., fp16, bf16, auto)
        """
        log.info('#================ * * * =========================#')
        log.info('Create Decode launch script on Decode nodes')
        log.info('#================ * * * =========================#')
        cmd_list = []
        decode_node_list = self.inf_dict['decode_node_list']
        log.info('%%%% self.decode_nnodes {}'.format(self.decode_nnodes))
        for i in range(0, int(self.decode_nnodes)):
            # ------------------------------------------------------------------
            # Construct a command that writes a Decode server launch script
            # into /tmp/decode_launch_script.sh inside the container
            #
            # Key configuration details:
            #   - NNODES / NODE_RANK: Distributed topology for SGLang
            #   - disaggregation-mode decode: Run in Decode-only mode
            #   - disaggregation-ib-device: RDMA device used for KV transfers
            #   - host / port: Network endpoint for this Decode server
            #   - dtype / kv-cache-dtype: Compute and KV precision
            #   - tensor parallelism: Model sharding across GPUs
            #   - aiter backend: Optimized attention backend for AMD GPUs
            #   - memory fraction: Static GPU memory reservation
            #
            # NOTE:
            #   The script is written (echo > file), not executed here.
            #   Execution is handled by a separate orchestration step.
            # ------------------------------------------------------------------
            cmd = f'''docker exec {self.container_name} /bin/bash -c  "echo  '
                      export NNODES={self.decode_nnodes}
                      export NODE_RANK={i}
                      export SGLANG_USE_AITER=1
                      python3 -m sglang.launch_server --model {self.bp_dict['model']} \
                              --disaggregation-mode decode \
                              --disaggregation-ib-device {self.inf_dict['nccl_ib_hca']} \
                              --host {decode_node_list[i]} \
                              --port {self.inf_dict['decode_serv_port']} \
                              --trust-remote-code \
                              --dtype {dtype} \
                              --kv-cache-dtype {kv_cache_dtype} \
                              --tp {self.bp_dict['tensor_parallelism']} \
                              --disable-radix-cache --disable-cuda-graph \
                              --mem-fraction-static {self.bp_dict['memory_fraction']} \
                              --attention-backend aiter \
                              --log-level {self.inf_dict['log_level']}' > /tmp/decode_launch_script.sh" '''
            formatted_cmd = textwrap_for_yml(cmd)
            cmd_list.append(formatted_cmd)
        log.info('%%%%%%%%%%%%%%%%%%%')
        log.info("%s", cmd_list)
        log.info('%%%%%%%%%%%%%%%%%%%')
        self.d_phdl.exec_cmd_list(cmd_list)
        log.info('#================ * * * =========================#')
        log.info('Launching Decode servers on Decode nodes')
        log.info('#================ * * * =========================#')
        cmd_list = []
        for i in range(0, int(self.decode_nnodes)):
            cmd = f'''docker exec {self.container_name} /bin/bash -c " \
                   chmod 755 /tmp/decode_launch_script.sh; \
                   mkdir -p {self.log_dir}/decode_node{i}; \
                   source /tmp/decode_env_script.sh && \
                   nohup bash /tmp/decode_launch_script.sh > \
                   {self.log_dir}/decode_node{i}/decode_server.log 2>&1 &" '''
            formatted_cmd = textwrap_for_yml(cmd)
            cmd_list.append(formatted_cmd)
        self.d_phdl.exec_cmd_list(cmd_list)

    def poll_and_check_server_ready(
        self,
    ):
        """
        Wait for Prefill and Decode servers to initialize and verify that they
        are fully ready to accept inference requests.

        Purpose:
        --------
        After launching Prefill and Decode server scripts, the servers require
        time to:
        - Initialize Python runtime
        - Load model weights
        - Allocate GPU memory
        - Initialize RDMA / NCCL / Gloo communication
        - Bind to network ports

        This method enforces a startup delay and then actively polls each server
        to confirm readiness before inference traffic is sent.
        """
        log.info('Waiting 120 secs after launching decode script')
        time.sleep(120)
        for node_no in range(0, self.prefill_nnodes):
            self.poll_for_server_ready(node_no, 'prefill')
        for node_no in range(0, self.decode_nnodes):
            self.poll_for_server_ready(node_no, 'decode')

    def launch_proxy_router(
        self,
    ):
        """
        Generate and launch the SGLang Proxy Router for disaggregated
        Prefill/Decode (PD) inference.

        Purpose:
        --------
        The Proxy Router is the control-plane and data-plane entry point for
        inference traffic in a disaggregated PD deployment.

        Responsibilities:
        - Accept incoming inference requests
        - Route prefill requests to Prefill servers
        - Route decode requests to Decode servers
        - Coordinate Prefill ? Decode handoff

        This method:
        - Builds routing configuration dynamically based on cluster topology
        - Creates a launch script on the Proxy Router node
        - Launches the router as a background service
        """

        # ------------------------------------------------------------------
        # Build Prefill endpoint arguments for the router
        #
        # Each Prefill server is specified as:
        #   --prefill http://<host>:<port>
        # ------------------------------------------------------------------
        prefill_str = ''
        for prefill_node in self.prefill_node_list:
            prefill_str = prefill_str + f"--prefill http://{prefill_node}:{self.inf_dict['prefill_serv_port']} "
        # ------------------------------------------------------------------
        # Build Decode endpoint arguments for the router
        #
        # Each Decode server is specified as:
        #   --decode http://<host>:<port>
        # ------------------------------------------------------------------
        decode_str = ''
        for decode_node in self.decode_node_list:
            decode_str = decode_str + f"--decode http://{decode_node}:{self.inf_dict['decode_serv_port']} "
        log.info('#================ * * * =========================#')
        log.info('Create Proxy Router launch script on Proxy Router nodes')
        log.info('#================ * * * =========================#')

        # ------------------------------------------------------------------
        # Create the Proxy Router launch script
        #
        # Key flags:
        #   --pd-disaggregation : Enable Prefill/Decode disaggregation
        #   --prefill / --decode: Upstream Prefill and Decode endpoints
        #   --host 0.0.0.0      : Listen on all interfaces
        #   --port              : External router port
        #   --log-dir           : Directory for router logs
        #
        # NOTE:
        #   The script is written to disk but not executed here.
        # ------------------------------------------------------------------
        cmd = f'''docker exec {self.container_name} /bin/bash -c  "echo  '
                      python3 -m sglang_router.launch_router \
                              --pd-disaggregation \
                              {prefill_str} \
                              {decode_str} \
                              --host 0.0.0.0 \
                              --port {self.inf_dict['proxy_router_port']} \
                              --log-dir {self.inf_dict['log_dir']} \
                      '  > /tmp/proxy_router_launch_script.sh"
                    '''
        formatted_cmd = textwrap_for_yml(cmd)
        self.r_phdl.exec(formatted_cmd)
        log.info('#================ * * * =========================#')
        log.info('Launch Proxy Router script on Proxy Router nodes')
        log.info('#================ * * * =========================#')
        cmd = f'''docker exec {self.container_name} /bin/bash -c " \
                   chmod 755 /tmp/proxy_router_launch_script.sh; \
                   mkdir -p {self.log_dir}/proxy_router_node; \
                   source /tmp/router_env_script.sh && \
                   nohup bash /tmp/proxy_router_launch_script.sh > \
                   {self.log_dir}/proxy_router_node/proxy_router.log 2>&1 &" '''
        formatted_cmd = textwrap_for_yml(cmd)
        self.r_phdl.exec(formatted_cmd)
        log.info('Waiting 120 secs after launching proxy router script')
        time.sleep(120)

    def run_gsm8k_benchmark_test(self, d_type='auto'):
        """
        Run the GSM8K inference benchmark against the SGLang disaggregated
        Prefill/Decode deployment and validate throughput.

        Purpose:
        --------
        This method executes a real-world inference workload (GSM8K question
        answering) to:
        - Validate end-to-end correctness of the inference pipeline
        - Measure sustained output token throughput
        - Ensure performance meets expected SLA thresholds

        The benchmark traffic is sent to the Proxy Router, which:
        - Routes requests to Prefill servers
        - Coordinates Decode servers for token generation
        """
        log.info('#================ * * * =========================#')
        log.info('Create Benchmark script')
        log.info('#================ * * * =========================#')

        i_dict = self.bp_dict['inference_tests']['gsm8k']
        # ------------------------------------------------------------------
        # Construct command to run GSM8K benchmark inside the container
        #
        # Key steps:
        #   - Create a directory to store benchmark logs
        #   - Navigate to the GSM8K benchmark directory
        #   - Source environment variables required for benchmark execution
        #   - Launch the benchmark using nohup to allow async execution
        #
        # Benchmark parameters:
        #   --num-questions : Total GSM8K questions to run
        #   --parallel      : Maximum concurrent inference requests
        #   --host / --port : Proxy Router endpoint for inference
        # ------------------------------------------------------------------
        cmd = f'''docker exec {self.container_name} /bin/bash -c  "
                      mkdir -p {self.log_dir}/benchmark_node; \
                      cd /sgl-workspace/sglang/benchmark/gsm8k; \
                      source /tmp/benchmark_env_script.sh && \
                      nohup python3 ./bench_sglang.py \
                      --num-questions {i_dict['num_questions']} \
                      --parallel {i_dict['max_concurrency']} \
                      --raw-result-file {self.log_dir}/benchmark_node/gsm8k_per_question.jsonl \
                      --host http://{self.inf_dict['proxy_router_node']} --port {self.inf_dict['proxy_router_serv_port']}" '''
        formatted_cmd = textwrap_for_yml(cmd)
        out_dict = self.b_phdl.exec(formatted_cmd, timeout=800)
        time.sleep(5)
        for node in out_dict.keys():
            if not re.search('Output throughput', out_dict[node], re.I):
                fail_test(f'Benchmark test did not complete properly on node {node}, no throughput pattern seen')
        # Parse the gsm8k stdout into structured metrics, display them, then
        # gate on the configured threshold (tokens_per_sec) via the shared
        # verifier. dmesg is left to the bench_serv test at suite end.
        self.get_gsm8k_results_dict(out_dict)
        expected = i_dict['expected_results'][d_type]
        self.verify_inference_results('gsm8k', expected, check_dmesg=False)
        # Per-question detail (prompt_id, correct, output preview).
        self.log_per_item_detail(
            'gsm8k',
            f'{self.log_dir}/benchmark_node/gsm8k_per_question.jsonl',
            parse_gsm8k_per_question,
            ['q', 'correct', 'output'],
        )

    def benchserv_test_random(self, d_type='auto'):
        """
        Run SGLang serving benchmark using a synthetic random dataset and
        validate inference performance and correctness.

        Purpose:
        --------
        This benchmark exercises the inference serving stack using randomly
        generated input/output sequences to:
        - Stress-test request scheduling and batching
        - Evaluate sustained throughput under synthetic load
        - Validate end-to-end serving stability independent of real datasets

        The benchmark targets the Proxy Router endpoint, ensuring that
        Prefill, Decode, and routing logic work together correctly.

        Args:
        d_type (str): Data type identifier used to select expected
                      performance thresholds (e.g., fp16, bf16, auto).
        """
        log.info('#================ * * * =========================#')
        log.info('Benchmark Random Dataset')
        log.info('#================ * * * =========================#')
        i_dict = self.bp_dict['inference_tests']['bench_serv_random']
        # Optional local dataset path (e.g. a pre-staged ShareGPT corpus) so the
        # random dataset sampler doesn't try to download from HF Hub - required
        # when the container runs with HF_HUB_OFFLINE.
        dataset_path_arg = ''
        if self.inf_dict.get('bench_dataset_path'):
            dataset_path_arg = f"--dataset-path {self.inf_dict['bench_dataset_path']}"
        # Bound in-flight requests so the benchmark does not flood the
        # deployment (request_rate defaults to inf). Falls back to the model's
        # top-level max_concurrency.
        max_conc = i_dict.get('max_concurrency', self.bp_dict.get('max_concurrency', '64'))
        # ------------------------------------------------------------------
        # Construct command to run sglang.bench_serving with random dataset
        #
        # Key parameters:
        #   --dataset-name random     : Use synthetic random prompts
        #   --num-prompts             : Total number of inference requests
        #   --random-input            : Input token length per request
        #   --random-output           : Output token length per request
        #   --random-range-ratio      : Variability in input/output lengths
        #   --host / --port           : Proxy Router endpoint
        #
        # Output is redirected to a log file for later inspection.
        # ------------------------------------------------------------------
        cmd = f'''docker exec {self.container_name} /bin/bash -c  "
                      mkdir -p {self.log_dir}/benchmark_node; \
                      source /tmp/benchmark_env_script.sh && \
                      PYTHONPATH=/sgl-workspace/sglang/python python3 -m sglang.bench_serving --backend {i_dict['backend']} \
                      --dataset-name random \
                      {dataset_path_arg} \
                      --max-concurrency {max_conc} \
                      --num-prompts {i_dict['num_prompts']} \
                      --random-input {i_dict['input_length']} \
                      --random-output {i_dict['output_length']} \
                      --random-range-ratio {i_dict['random_range_ratio']} \
                      --output-file {self.log_dir}/benchmark_node/bench_serv_details.jsonl --output-details \
                      --host {self.inf_dict['proxy_router_node']} --port {self.inf_dict['proxy_router_serv_port']} \
                      > {self.log_dir}/benchmark_node/benchmark_results.log" '''
        formatted_cmd = textwrap_for_yml(cmd)
        self.b_phdl.exec(formatted_cmd, timeout=500)
        time.sleep(5)
        self.poll_for_inference_completion(iterations=10, waittime_between_iters=60)
        self.verify_inference_results('bench_serv', i_dict['expected_results'][d_type])
        # Per-request detail (input/output lengths, TTFT, generated tokens, errors).
        self.log_per_item_detail(
            'bench_serv',
            f'{self.log_dir}/benchmark_node/bench_serv_details.jsonl',
            parse_bench_serv_per_request,
            ['req', 'input_len', 'output_len', 'ttft_ms', 'gen_tokens', 'error'],
        )

    def poll_for_server_ready(self, node_no, sglang_function, no_of_iterations=16):
        """
        Poll SGLang Prefill or Decode server logs to determine when the server
        is ready to accept inference traffic.

        Readiness definition:
        ---------------------
        A server is considered "ready" when its log shows successful HTTP
        requests (HTTP 200 OK), indicating that:
        - The server process has started
        - The model is loaded
        - Network endpoints are listening
        - Request handling is functional

        Assumptions:
        ------------
        - Log directory is located on a shared filesystem (e.g., NFS)
        - Logs are accessible from a designated head node
        - Each server writes logs to a predictable per-node path

        Args:
        node_no (int): Index of the Prefill or Decode node being checked
        sglang_function (str): Server role ('prefill' or 'decode')
        no_of_iterations (int): Maximum number of polling attempts before
                                declaring failure
        """
        # ------------------------------------------------------------------
        # Prefill server readiness check
        # ------------------------------------------------------------------
        if re.search('prefill', sglang_function):
            head_node = self.prefill_node_list[0]
            for j in range(1, no_of_iterations):
                log.info(f'Starting poll iteration {j}')
                out_dict = self.p_phdl.exec(
                    f'grep -B 20 -A 20 "200 OK" {self.log_dir}/prefill_node{node_no}/prefill_server.log'
                )
                if re.search('GET|POST', out_dict[head_node], re.I):
                    log.info('Wait 60 secs to start serving traffic')
                    time.sleep(60)
                    # if re.search('fired up and ready to roll', out_dict[head_node], re.I ):
                    #    print('Prefill server {node_no} ready to serve')
                    return
                else:
                    log.info('Wait for 120 secs and continue polling')
                    time.sleep(120)
            head_node = self.prefill_node_list[0]
            log.warning(f'Prefill node {node_no} did not get to ready to serve 200 OK state in {j} iterations')
            fail_test(f'Prefill node {node_no} did not get to ready to serve 200 OK state in {j} iterations')
        # ------------------------------------------------------------------
        # Decode server readiness check
        # ------------------------------------------------------------------
        elif re.search('decode', sglang_function):
            head_node = self.decode_node_list[0]
            for j in range(1, no_of_iterations):
                log.info(f'Starting poll iteration {j}')
                out_dict = self.d_phdl.exec(
                    f'grep -B 20 -A 20 "200 OK" {self.log_dir}/decode_node{node_no}/decode_server.log'
                )
                if re.search('GET|POST', out_dict[head_node]):
                    log.info('Wait 60 secs to start serving traffic')
                    time.sleep(60)
                    # if re.search('fired up and ready to roll', out_dict[head_node], re.I ):
                    #    print('Decode server {node_no} ready to serve')
                    return
                else:
                    log.info('Wait for 120 secs and continue polling')
                    time.sleep(120)
            log.warning(f'Decode node {node_no} did not get to ready to serve 200 OK state in {j} iterations')
            fail_test(f'Decode node {node_no} did not get to ready to serve 200 OK state in {j} iterations')

    def get_gsm8k_results_dict(self, out_dict):
        """
        Parse gsm8k bench_sglang.py stdout into the structured results dict.

        Fills self.inference_results_dict keyed per node with accuracy,
        invalid, latency_s and tokens_per_sec (see parse_gsm8k_metrics).
        """
        self.inference_results_dict = {node: parse_gsm8k_metrics(out_dict[node]) for node in out_dict.keys()}
        log.info("%s", self.inference_results_dict)
        return self.inference_results_dict

    def log_metrics(self, test_name, expected_result_dict=None):
        """
        Log a uniform, greppable metrics table for self.inference_results_dict,
        annotating each thresholded metric with its expected bound and verdict.
        """
        table = format_metrics_table(test_name, self.inference_results_dict, expected_result_dict)
        for line in table.split('\n'):
            log.info("%s", line)

    def log_per_item_detail(self, test_name, remote_path, parser_fn, columns):
        """
        Fetch a per-item detail file (gsm8k per-question or bench_serv
        per-request) from the benchmark node, parse it, log a one-line summary
        and a compact per-item table.

        The number of displayed rows is capped by inf_dict['per_item_display_limit']
        (unset/0 => show all). The full file remains on the benchmark node as an
        artifact regardless of the display cap.
        """
        out_dict = self.b_phdl.exec(f'cat {remote_path} 2>/dev/null')
        limit = self.inf_dict.get('per_item_display_limit')
        limit = int(limit) if limit not in (None, '') else None
        for node in out_dict.keys():
            rows = parser_fn(out_dict[node])
            if not rows:
                log.warning(f'No per-item {test_name} detail parsed from {remote_path} on node {node}')
                continue
            # Summary line: correctness for gsm8k, error count for bench_serv.
            if any('correct' in r for r in rows):
                n_correct = sum(1 for r in rows if r.get('correct') is True)
                log.info(f'{test_name} per-item summary [node={node}]: {n_correct}/{len(rows)} correct')
            if any('error' in r for r in rows):
                n_err = sum(1 for r in rows if r.get('error'))
                log.info(
                    f'{test_name} per-item summary [node={node}]: {len(rows) - n_err}/{len(rows)} succeeded ({n_err} errored)'
                )
            for line in format_per_request_table(test_name, rows, columns, limit=limit).split('\n'):
                log.info("%s", line)

    def get_inference_results_dict(self, out_dict):
        """
        Parse inference benchmark output logs and extract key performance metrics
        into a structured dictionary.

        Purpose:
        --------
        This method processes raw text output generated by inference benchmarks
        (e.g., sglang.bench_serving) and extracts important metrics such as:
        - Request counts
        - Token throughput
        - Latency statistics (TTFT, TPOT)
        - Benchmark duration

        The extracted metrics are stored per node in:
        self.inference_results_dict

        Args:
        out_dict (dict):
            Dictionary keyed by node identifier, where each value is the
            raw stdout/stderr text produced by the benchmark on that node.
        """
        self.inference_results_dict = {node: parse_bench_serv_metrics(out_dict[node]) for node in out_dict.keys()}
        log.info("%s", self.inference_results_dict)
        return self.inference_results_dict

    def scan_for_inference_errors(
        self,
    ):
        """
        Scan Prefill and Decode server logs for known inference error patterns
        and fail the test if any are detected.

        Purpose:
        --------
        This method performs a post-inference health check by scanning
        server logs for known error signatures that indicate:
        - Runtime failures
        - Communication errors (RDMA/NCCL)
        - Out-of-memory conditions
        - Kernel or backend crashes
        - Fatal exceptions during inference

        The method ensures that even if benchmarks complete, silent or
        non-fatal errors do not go unnoticed.
        """
        log.info('Scan for inference errors')
        inference_pass = True

        # Build the list of commands to read each node's inference log file
        cmd_list = []

        # Scan all prefill nodes
        for j in range(0, int(self.prefill_nnodes)):
            cmd = f"sudo cat {self.log_dir}/prefill_node{j}/prefill_server.log"
            cmd_list.append(cmd)
        out_dict = self.p_phdl.exec_cmd_list(cmd_list)

        # Check the log content against all known inference error patterns
        for node in out_dict.keys():
            for err_key in inference_err_dict:
                if re.search(f'{inference_err_dict[err_key]}', out_dict[node]):
                    fail_test(f'ERROR {inference_err_dict[err_key]} seen in inference logs ..')
                    log.error('Aborting inference log polling')
                    inference_pass = False

        # Scan all decode nodes
        for j in range(0, int(self.decode_nnodes)):
            cmd = f"sudo cat {self.log_dir}/decode_node{j}/decode_server.log"
            cmd_list.append(cmd)
        out_dict = self.d_phdl.exec_cmd_list(cmd_list)

        # Check the log content against all known inference error patterns
        for node in out_dict.keys():
            for err_key in inference_err_dict:
                if re.search(f'{inference_err_dict[err_key]}', out_dict[node]):
                    fail_test(f'ERROR {inference_err_dict[err_key]} seen in inference logs ..')
                    log.error('Aborting inference log polling')
                    inference_pass = False

        return inference_pass

    def poll_for_inference_completion(
        self, iterations=10, waittime_between_iters=60, total_timeout=3600, require_all_nodes=True
    ):
        """
        Poll benchmark logs to detect inference completion and extract results.

        Purpose:
        --------
        This method monitors inference progress by periodically inspecting
        benchmark output logs. It determines when inference has completed,
        detects early failures, and enforces a global timeout.

        Completion criteria:
        --------------------
        Inference is considered complete when the benchmark output contains
        the pattern 'Serving Benchmark Result'.

        Failure criteria:
        -----------------
        Any known inference error detected in Prefill or Decode logs
        immediately aborts the process.

        Args:
        iterations (int):
            Maximum number of polling iterations.
        waittime_between_iters (int):
            Time (seconds) to wait between polling attempts.
        total_timeout (int or None):
            Maximum wall-clock time (seconds) allowed for inference.
        require_all_nodes (bool):
            If True, all nodes must report completion.
            If False, completion by any node is sufficient.
        """
        # Initial wait to give inference time to start logging
        time.sleep(60)

        # Track wall-clock timeout if specified
        start_time = time.time()

        def timed_out() -> bool:
            return total_timeout is not None and (time.time() - start_time) >= float(total_timeout)

        completed_pattern = re.compile('Serving Benchmark Result', re.I)
        # ------------------------------------------------------------------
        # Poll loop: periodically inspect benchmark logs for completion
        # ------------------------------------------------------------------
        for itr in range(1, iterations + 1):
            log.info(f'Starting iteration {itr}')

            # --------------------------------------------------------------
            # Early exit if any inference errors are detected
            #
            # This scans Prefill and Decode logs for known failure patterns
            # (e.g., OOM, RDMA failures, backend crashes).
            # --------------------------------------------------------------
            # Early abort on inference errors
            if not self.scan_for_inference_errors():
                msg = 'Failures seen in inference logs, Aborting!!!'
                fail_test(msg)
                return {"status": "error", "reason": msg}

            # --------------------------------------------------------------
            # Read the most recent benchmark output
            #
            # Tail only the last 1000 lines to reduce I/O and parsing cost.
            # --------------------------------------------------------------
            cmd = f"sudo tail -1000 {self.log_dir}/benchmark_node/benchmark_results.log"

            out_dict = self.b_phdl.exec(cmd)

            # Determine completion across nodes
            node_completion = {}
            for node, output in out_dict.items():
                node_completion[node] = bool(completed_pattern.search(output))

            # --------------------------------------------------------------
            # Determine overall completion based on policy
            #
            # - require_all_nodes=True  ? all nodes must complete
            # - require_all_nodes=False ? any node completing is sufficient
            # --------------------------------------------------------------
            if require_all_nodes:
                all_complete = all(node_completion.values()) if node_completion else False
            else:
                all_complete = any(node_completion.values()) if node_completion else False

            # --------------------------------------------------------------
            # If inference is still running, wait and retry
            # --------------------------------------------------------------
            if not all_complete:
                if timed_out():
                    msg = f"Timeout while waiting for inference completion after ~{int(time.time() - start_time)}s"
                    log.warning("%s", msg)
                    return {"status": "timeout", "reason": msg}
                log.info('Inference still in progress')
                # Short progress wait before the longer inter-iteration sleep
                time.sleep(30)
                time.sleep(int(waittime_between_iters))
                continue

            # --------------------------------------------------------------
            # Inference completed successfully
            #
            # Parse benchmark results and return structured output.
            # --------------------------------------------------------------
            self.get_inference_results_dict(out_dict)
            log.info('Completed Inference, returning !!!')
            return {"status": "success", "results": self.inference_results_dict}

            # If we reached here, it means poll for inference completion failed

        # If we exhaust the iteration cap without completing, treat as timeout (or in_progress if no wall-clock limit)
        if timed_out():
            msg = f"Timeout after maximum iterations ({self.inference_poll_iterations}) and ~{int(time.time() - start_time)}s"
            log.warning("%s", msg)
            return {"status": "timeout", "reason": msg}
        else:
            # If no wall-clock timeout was set and we hit the iteration cap, report in-progress
            msg = f"Reached iteration cap ({self.inference_poll_iterations}) without completion; still in progress"
            log.warning("%s", msg)
            return {"status": "stuck_in_progress", "reason": msg}

    def verify_inference_results(self, test_name, expected_result_dict, check_dmesg=True):
        """
        Validate inference benchmark results against expected performance
        thresholds, display a structured metrics table, and optionally check
        for system-level (dmesg) errors.

        Purpose:
        --------
        This method verifies that:
        - Performance metrics meet or exceed expected baselines
        - Latency metrics stay below defined thresholds
        - No kernel-level (dmesg) errors occurred during inference (when
          check_dmesg is True)

        It acts as the final gate for inference validation.

        Args:
        test_name (str): label used in the metrics table header.
        expected_result_dict (dict): {metric: threshold} to gate on.
        check_dmesg (bool): run the per-node dmesg sweep when True.
        """
        # ------------------------------------------------------------------
        # Display a uniform metrics table (with per-metric verdicts).
        # ------------------------------------------------------------------
        self.log_metrics(test_name, expected_result_dict)

        # ------------------------------------------------------------------
        # Validate thresholded metrics on a per-node basis. Direction is
        # explicit: latency-style metrics fail when higher than expected,
        # throughput/count metrics fail when lower than expected.
        # ------------------------------------------------------------------
        for node in self.inference_results_dict.keys():
            for metric_name in expected_result_dict.keys():
                if metric_name not in self.inference_results_dict[node]:
                    log.warning(f'Expected metric {metric_name} not present in {test_name} results for node {node}')
                    continue
                actual = self.inference_results_dict[node][metric_name]
                expected = expected_result_dict[metric_name]
                lower_better = _is_lower_better(metric_name)
                op = '<=' if lower_better else '>='
                failed = float(actual) > float(expected) if lower_better else float(actual) < float(expected)
                log.info(
                    f'metric {metric_name}: actual={actual} expected {op} {expected} -> {"FAIL" if failed else "PASS"}'
                )
                if failed:
                    fail_test(
                        f"FAIL - metric {metric_name} actual={actual} violates expected {op} {expected} "
                        f"on node {node} ({test_name})"
                    )

        # ------------------------------------------------------------------
        # Perform kernel-level (dmesg) error checks
        #
        # This ensures no silent hardware or driver errors occurred during
        # inference (e.g., GPU resets, RDMA failures, IOMMU errors).
        # ------------------------------------------------------------------
        if check_dmesg:
            self.inference_end_time = self.p_phdl.exec('date +"%a %b %e %H:%M"')
            time.sleep(2)
            verify_dmesg_for_errors(self.p_phdl, self.inference_start_time, self.inference_end_time)
            verify_dmesg_for_errors(self.d_phdl, self.inference_start_time, self.inference_end_time)
            verify_dmesg_for_errors(self.r_phdl, self.inference_start_time, self.inference_end_time)
            verify_dmesg_for_errors(self.b_phdl, self.inference_start_time, self.inference_end_time)
        log.info("%s", self.inference_results_dict)
