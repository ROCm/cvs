'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import math
import re
import shlex
import time
from typing import List, Dict, Tuple

from cvs.lib import env_lib, globals
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *

log = globals.log


HEADER_MAP = {
    "MsgSize (B)": "MsgSize_B",
    "BatchSize": "BatchSize",
    "TotalSize (MB)": "TotalSize_MB",
    "Max BW (GB/s)": "Max_BW_GBps",
    "Avg Bw (GB/s)": "Avg_BW_GBps",
    "Min Lat (us)": "Min_Lat_us",
    "Avg Lat (us)": "Avg_Lat_us",
}

# MORI changed "Avg Bw" to "Avg BW" in 2026-01; match headers case-insensitively.
_HEADER_MAP_CI = {k.lower(): v for k, v in HEADER_MAP.items()}

IO_EXIT_MARKER = 'MORI_IO_EXIT_CODE='
REMOTE_TIMEOUT_MARKER = 'MORI_REMOTE_TIMEOUT after '

# Seconds precision: verify_dmesg_for_errors' node-scraper path treats the end time as an exclusive cutoff.
DMESG_DATE_CMD = 'date +"%a %b %e %H:%M:%S"'


def _convert_value(val: str):
    """Convert string to int or float."""
    val = val.strip()
    if "." in val:
        return float(val)
    return int(val)


def parse_pretty_tables_multi_rank(text: str) -> dict:
    """
    Parse multiple pretty-printed tables (one per Initiator Rank)
    into a single JSON file.

    Args:
        text (str): Raw benchmark output (multiple ranks concatenated).
    """

    lines = [line.rstrip() for line in text.splitlines() if line.strip()]

    results: Dict[int, Dict[str, List[Dict]]] = {}
    i = 0

    while i < len(lines):
        line = lines[i]
        # -----------------------------
        # Detect rank header
        # -----------------------------
        if "Initiator Rank" in line:
            match = re.search(r"Initiator Rank\s+(\d+)", line)
            if not match:
                i += 1
                continue

            rank = int(match.group(1))
            results[rank] = {"rows": []}

            # -----------------------------
            # Find header row
            # -----------------------------
            while i < len(lines) and not lines[i].startswith("| MsgSize"):
                i += 1
            headers = [h.strip() for h in lines[i].strip("|").split("|")]
            json_headers = [_HEADER_MAP_CI[h.lower()] for h in headers]

            # Skip separator line
            i += 2

            # -----------------------------
            # Parse data rows
            # -----------------------------
            while i < len(lines):
                row_line = lines[i]

                if row_line.startswith("+"):
                    break

                if not row_line.startswith("|"):
                    i += 1
                    continue

                values = [v.strip() for v in row_line.strip("|").split("|")]

                if len(values) == len(json_headers):
                    row = {key: _convert_value(val) for key, val in zip(json_headers, values)}
                    results[rank]["rows"].append(row)

                i += 1

        i += 1

    # -----------------------------
    # Write to dict
    # -----------------------------
    output_dict = {"ranks": results}
    log.info("%s", output_dict)
    return output_dict


def parse_ibgda_output(text: str) -> Tuple[Dict, List[Dict]]:
    metadata = {}
    results = []

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

    # ---- Parse metadata line ----
    # Example:
    # Blocks: 4, Threads: 256, Iterations: 10, QPs:4
    meta_pattern = re.compile(r"Blocks:\s*(\d+),\s*Threads:\s*(\d+),\s*Iterations:\s*(\d+),\s*QPs:\s*(\d+)")

    for line in lines:
        match = meta_pattern.search(line)
        if match:
            metadata = {
                "blocks": int(match.group(1)),
                "threads": int(match.group(2)),
                "iterations": int(match.group(3)),
                "qps": int(match.group(4)),
            }
            break

    # ---- Parse table rows ----
    # Expected columns:
    # Index Size(B) bw(GB) Time(ms) Rate(Mpps)
    row_pattern = re.compile(
        r"^\d+\s+"
        r"(\d+)\s+"  # Size(B)
        r"([\d.]+)\s+"  # bw(GB)
        r"([\d.]+)\s+"  # Time(ms)
        r"([\d.]+)$"  # Rate(Mpps)
    )

    for line in lines:
        if line.startswith("Index") or line.startswith("IBGDA"):
            continue

        match = row_pattern.match(line)
        if match:
            results.append(
                {
                    "size_bytes": int(match.group(1)),
                    "bandwidth_gb": float(match.group(2)),
                    "time_ms": float(match.group(3)),
                    "rate_mpps": float(match.group(4)),
                }
            )

    return metadata, results


class MoriBenchmark:
    """Runs the MORI benchmarks through an orchestrator.

    Every command goes through ``orch.exec`` / ``exec_cmd_list``, so it runs
    inside the container under the container orchestrator and on the host
    under baremetal. Per-test output is tee'd under ``log_dir/<run_id>`` so it
    outlives a ``per_run`` container.
    """

    SETUP_TIMEOUT = 120
    PIP_TIMEOUT = 600
    SHMEM_TIMEOUT = 900
    CONCURRENT_PUT_TIMEOUT = 300
    IBGDA_TIMEOUT = 600
    IO_LAUNCH_TIMEOUT = 60
    IO_POLL_TIMEOUT = 600
    IO_POLL_INTERVAL = 10
    # Floor for a poll's exec timeout near the deadline; a normal poll takes milliseconds.
    IO_POLL_MIN_TIMEOUT = 10
    # A CVS-side exec timeout only stops waiting; the remote command keeps running.
    # The remote bound (exec timeout - margin, then SIGKILL after kill_after) must
    # expire first so the command's process group is gone before CVS gives up.
    REMOTE_TIMEOUT_MARGIN = 30
    REMOTE_KILL_AFTER = 10

    def __init__(self, orch, mori_dict):
        self.orch = orch
        self.is_container = orch.orchestrator_type == 'container'
        self.host_list = list(orch.hosts)
        self.nnodes = len(self.host_list)
        self.mori_dict = mori_dict
        self.master_addr = self.mori_dict.get('master_addr') or orch.head_node
        self.master_port = self.mori_dict['master_port']
        self.mori_dir = self.mori_dict['mori_dir']
        self.env = dict(self.mori_dict.get('env') or {})
        self.mori_device_list = self.env.get('MORI_RDMA_DEVICES', '')
        self.log_dir = self.mori_dict['log_dir']
        self.run_log_dir = f"{self.log_dir}/{time.strftime('%Y%m%d_%H%M%S')}"
        self.expected_results_dict = self.mori_dict['expected_results']
        # Exported inline rather than via container.env so the same prefix works under baremetal.
        # benchmark.py imports tests.python.* from mori_dir, so that PYTHONPATH entry is exported
        # last: it then stacks on any PYTHONPATH from env instead of being replaced by it.
        exports = [
            env_lib.build_env_prefix(self.env),
            env_lib.build_env_prefix({'PYTHONPATH': f'{self.mori_dir}:$PYTHONPATH'}),
        ]
        self.env_prefix = ' ; '.join(e for e in exports if e) + '; '

    def _bounded(self, cmd, timeout):
        """Wrap ``cmd`` in coreutils ``timeout`` sized to expire before the CVS-side ``timeout``.

        Without ``--foreground``, ``timeout`` signals its whole process group, which
        reaches pytest's spawned ranks and mpiexec. On expiry it prints
        ``REMOTE_TIMEOUT_MARKER`` so the cause shows up in the output and log.
        """
        if timeout <= self.REMOTE_TIMEOUT_MARGIN:
            raise ValueError(f'timeout {timeout}s must exceed REMOTE_TIMEOUT_MARGIN {self.REMOTE_TIMEOUT_MARGIN}s')
        remote = timeout - self.REMOTE_TIMEOUT_MARGIN
        return (
            f'{{ timeout -k {self.REMOTE_KILL_AFTER} {remote} bash -c {shlex.quote(cmd)}; rc=$?; '
            f'if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then echo "{REMOTE_TIMEOUT_MARKER}{remote}s rc=$rc"; fi; }}'
        )

    def _mori_cmd(self, cmd, timeout, log_name=None):
        """Prefix the bounded ``cmd`` with the MORI env and ``cd mori_dir``; optionally tee it to a per-host log.

        The ``tee`` sits outside the bounded process group, so output written before
        a remote timeout is kept.
        """
        full = f'{self.env_prefix}cd {shlex.quote(self.mori_dir)} && {self._bounded(cmd, timeout)}'
        if log_name:
            full += f' 2>&1 | tee {self.run_log_dir}/{log_name}_$(hostname).log'
        return full

    def _exec_cmd_list(self, cmd_list, timeout=None, print_console=True):
        # BaremetalOrchestrator has no exec_cmd_list; its host handle does.
        if self.is_container:
            return self.orch.exec_cmd_list(cmd_list, timeout=timeout, print_console=print_console)
        return self.orch.all.exec_cmd_list(cmd_list, timeout=timeout, print_console=print_console)

    def create_run_log_dir(self):
        out_dict = self.orch.exec(f'mkdir -p {self.run_log_dir}', timeout=self.SETUP_TIMEOUT, detailed=True)
        for node, res in out_dict.items():
            if res.get('exit_code') != 0:
                fail_test(f'ERROR - could not create log dir {self.run_log_dir} on node {node}: {res.get("output")}')

    def install_packages(self):
        """pip-install ``prettytable`` inside the container. Returns False (no-op) under baremetal,
        where it would modify the host's system Python."""
        if not self.is_container:
            log.info('baremetal: not installing packages on the host; prettytable must already be installed')
            return False
        self.orch.exec(self._bounded('pip3 install prettytable', self.PIP_TIMEOUT), timeout=self.PIP_TIMEOUT)
        return True

    def check_ibv_devices(
        self,
    ):
        if not self.mori_device_list:
            fail_test('ERROR - env.MORI_RDMA_DEVICES is not set in the mori config')
            return
        out_dict = self.orch.exec('ibv_devinfo', timeout=self.SETUP_TIMEOUT)
        for node in out_dict.keys():
            dev_list = self.mori_device_list.split(',')
            for dev_nam in dev_list:
                if not re.search(f'{dev_nam}', out_dict[node], re.I):
                    fail_test(f'ERROR - MORI device {dev_nam} not showing up in ibv_devinfo on node {node}')

    def run_shmem_apitest(
        self,
    ):
        cmd = self._mori_cmd(
            'pytest -vvv ./tests/python/shmem/test_api.py', log_name='shmem_api', timeout=self.SHMEM_TIMEOUT
        )
        out_dict = self.orch.exec(cmd, timeout=self.SHMEM_TIMEOUT)
        for node in out_dict.keys():
            if not re.search('PASSED', out_dict[node], re.I):
                fail_test(f'ERROR - shmem test_api.py did not run properly on node {node}, no PASSED test results seen')
            if re.search('FAIL', out_dict[node]):
                fail_test(f'ERROR - one or more shmem test_api.py tests failed on node {node}')

    def run_ibgda_dist_write(self, no_of_procs=2, min_val=2, max_val='16m', ctas=2, threads=256, qp_count=4, iters=1):
        cmd = self._mori_cmd(
            f'mpiexec --allow-run-as-root -x MORI_GLOBAL_LOG_LEVEL=TRACE '
            f'-np {no_of_procs} ./build/examples/dist_write -c {ctas} -t {threads} '
            f'-b {min_val} -e {max_val} -f 2 -q {qp_count} -n {iters}',
            log_name='ibgda_dist_write',
            timeout=self.IBGDA_TIMEOUT,
        )
        out_dict = self.orch.exec(cmd, timeout=self.IBGDA_TIMEOUT)
        exp_res_dict = self.expected_results_dict['ibgda_write']
        for node in out_dict.keys():
            if not re.search(r'Index\s+Size', out_dict[node], re.I):
                fail_test(f'ERROR - dist_write did not complete properly on node {node} - results not seen')
            else:
                meta_data, results = parse_ibgda_output(out_dict[node])
                log.info("%s", results)
                m_key = f'''PROCS:{no_of_procs},CTAS:{ctas},THREADS:{threads},QP_COUNT:{qp_count}'''
                if m_key in exp_res_dict.keys():
                    for row_dict in results:
                        act_msg_size = row_dict['size_bytes']
                        actual_bw = row_dict['bandwidth_gb']
                        log.info(f'exp_res_dict = {exp_res_dict}')
                        log.info(f'exp_res_dict = {exp_res_dict[m_key].keys()}')
                        log.info("%s", act_msg_size)
                        for exp_msg_size in list(exp_res_dict[m_key].keys()):
                            exp_bw = exp_res_dict[m_key][exp_msg_size]['max_bw']
                            if int(act_msg_size) == int(exp_msg_size):
                                if float(actual_bw) < float(exp_bw):
                                    fail_test(
                                        f'IBGDA Mori BW less than expected for  \
                                      PROCS:{no_of_procs},CTAS:{ctas},THREADS:{threads},QP_COUNT:{qp_count} \
                                      expected = {exp_bw}, actual = {actual_bw}'
                                    )
                                else:
                                    log.info(
                                        f'IBGDA Mori BW is as expected for  \
                                      PROCS:{no_of_procs},CTAS:{ctas},THREADS:{threads},QP_COUNT:{qp_count} \
                                      expected = {exp_bw}, actual = {actual_bw}'
                                    )

    def run_dispatch_combine(
        self,
    ):
        cmd = self._mori_cmd(
            'pytest -vvv ./tests/python/ops/test_dispatch_combine.py',
            log_name='dispatch_combine',
            timeout=self.SHMEM_TIMEOUT,
        )
        out_dict = self.orch.exec(cmd, timeout=self.SHMEM_TIMEOUT)
        for node in out_dict.keys():
            if not re.search('PASSED', out_dict[node], re.I):
                fail_test('ERROR - test_dispatch_combine.py did not run properly, no PASSED test results seen')
            if re.search('FAIL', out_dict[node], re.I):
                fail_test('ERROR - one or more test_dispatch_combine.py tests failed')

    def run_bench_dispatch_combine(
        self,
    ):
        cmd = self._mori_cmd(
            'pytest -vvv ./tests/python/ops/bench_dispatch_combine.py',
            log_name='bench_dispatch_combine',
            timeout=self.SHMEM_TIMEOUT,
        )
        out_dict = self.orch.exec(cmd, timeout=self.SHMEM_TIMEOUT)
        for node in out_dict.keys():
            if not re.search('PASSED', out_dict[node], re.I):
                fail_test('ERROR - bench_dispatch_combine.py did not run properly, no PASSED test results seen')
            if re.search('FAIL', out_dict[node], re.I):
                fail_test('ERROR - one or more bench_dispatch_combine.py tests failed')

    def run_concurrent_put_threads(
        self,
    ):
        cmd = self._mori_cmd(
            'mpiexec --allow-run-as-root -np 2 ./build/examples/concurrent_put_thread',
            log_name='concurrent_put_thread',
            timeout=self.CONCURRENT_PUT_TIMEOUT,
        )
        out_dict = self.orch.exec(cmd, timeout=self.CONCURRENT_PUT_TIMEOUT)
        for node in out_dict.keys():
            # NOTE: concurrent_put_thread / concurrent_put_imm_thread perform NO data
            # verification - they launch the kernel, barrier, and print "test done!".
            # They never emit "PASSED" (unlike concurrent_put_signal_thread, which does
            # validate and prints "...tests passed!"). Accepting the completion marker
            # makes this a SMOKE TEST: it proves the binary ran to completion without
            # crashing, NOT that the transferred data is correct.
            if not re.search(r'PASSED|test done!', out_dict[node], re.I):
                fail_test('ERROR - test concurrent_put_thread did not run properly, no PASSED test results seen')
            if re.search('FAIL', out_dict[node], re.I):
                fail_test('ERROR - one or more concurrent_put_thread tests failed')

    def run_concurrent_put_imm_threads(
        self,
    ):
        cmd = self._mori_cmd(
            'mpiexec --allow-run-as-root -np 2 ./build/examples/concurrent_put_imm_thread',
            log_name='concurrent_put_imm_thread',
            timeout=self.CONCURRENT_PUT_TIMEOUT,
        )
        out_dict = self.orch.exec(cmd, timeout=self.CONCURRENT_PUT_TIMEOUT)
        for node in out_dict.keys():
            # NOTE: concurrent_put_thread / concurrent_put_imm_thread perform NO data
            # verification - they launch the kernel, barrier, and print "test done!".
            # They never emit "PASSED" (unlike concurrent_put_signal_thread, which does
            # validate and prints "...tests passed!"). Accepting the completion marker
            # makes this a SMOKE TEST: it proves the binary ran to completion without
            # crashing, NOT that the transferred data is correct.
            if not re.search(r'PASSED|test done!', out_dict[node], re.I):
                fail_test('ERROR - test concurrent_put_imm_thread did not run properly, no PASSED test results seen')
            if re.search('FAIL', out_dict[node], re.I):
                fail_test('ERROR - one or more concurrent_put_imm_thread tests failed')

    def run_concurrent_put_signal_thread(
        self,
    ):
        cmd = self._mori_cmd(
            'mpiexec --allow-run-as-root -np 2 ./build/examples/concurrent_put_signal_thread',
            log_name='concurrent_put_signal_thread',
            timeout=self.CONCURRENT_PUT_TIMEOUT,
        )
        out_dict = self.orch.exec(cmd, timeout=self.CONCURRENT_PUT_TIMEOUT)
        for node in out_dict.keys():
            if not re.search('PASSED', out_dict[node], re.I):
                fail_test('ERROR - test concurrent_put_signal_thread did not run properly, no PASSED test results seen')
            if re.search('FAIL', out_dict[node], re.I):
                fail_test('ERROR - one or more concurrent_put_signal_thread tests failed')

    def io_rank_hosts(self):
        """Hosts in torchrun node-rank order. ``master_addr`` must be node rank 0: that node
        hosts the rendezvous store and, as the MORI-IO initiator, prints the result tables."""
        return [self.master_addr] + [h for h in self.host_list if h != self.master_addr]

    def build_io_launch_cmds(self, case_name, cmd_opts):
        """Return ``(cmd_list, log_paths)`` for one torchrun per host.

        ``cmd_list`` is in ``orch.hosts`` order because ``exec_cmd_list`` maps
        commands to hosts by position; each command carries that host's node
        rank from ``io_rank_hosts``. Each torchrun runs detached with all fds
        redirected, so the launching exec returns at once, and appends
        ``IO_EXIT_MARKER<rc>`` to its log when it exits.
        """
        rank_hosts = self.io_rank_hosts()
        cmd_list = []
        log_paths = {}
        for host in self.host_list:
            rank = rank_hosts.index(host)
            log_path = f'{self.run_log_dir}/{case_name}_node{rank}.log'
            log_paths[host] = log_path
            torchrun = (
                f'torchrun --nnodes={self.nnodes} --node_rank={rank} --nproc_per_node=1 '
                f'--master_addr={self.master_addr} --master_port={self.master_port} '
                f'{self.mori_dir}/tests/python/io/benchmark.py --host={host} --all {cmd_opts}'
            )
            inner = f'{torchrun}; echo {IO_EXIT_MARKER}$?'
            cmd_list.append(
                f'mkdir -p {self.run_log_dir}; {self.env_prefix}'
                f'cd {shlex.quote(self.mori_dir)} || exit 1; '
                f'export NNODES={self.nnodes} NODE_RANK={rank}; '
                f'nohup bash -c {shlex.quote(inner)} > {log_path} 2>&1 < /dev/null &'
            )
        return cmd_list, log_paths

    def poll_io_completion(self, log_paths, timeout=None, interval=None):
        """Poll every host's log for ``IO_EXIT_MARKER`` until all hosts report or ``timeout`` expires.

        Each poll's exec timeout and each sleep are clamped to the remaining budget, so
        a hung poll overruns ``timeout`` by at most ``IO_POLL_MIN_TIMEOUT`` per host the
        transport reads serially (SSH), not ``SETUP_TIMEOUT``.

        Returns ``{host: exit_code}`` for the hosts that finished; a host missing
        from the result did not finish in time.
        """
        timeout = self.IO_POLL_TIMEOUT if timeout is None else timeout
        interval = self.IO_POLL_INTERVAL if interval is None else interval
        deadline = time.monotonic() + timeout
        exit_codes = {}
        cmd_list = [f"grep -ho '{IO_EXIT_MARKER}[0-9]*' {log_paths[h]} 2>/dev/null | tail -1" for h in self.host_list]
        while True:
            remaining = deadline - time.monotonic()
            poll_timeout = min(self.SETUP_TIMEOUT, max(self.IO_POLL_MIN_TIMEOUT, math.ceil(remaining)))
            out_dict = self._exec_cmd_list(cmd_list, timeout=poll_timeout, print_console=False)
            for host in self.host_list:
                match = re.search(rf'{IO_EXIT_MARKER}(\d+)', out_dict.get(host) or '')
                if match:
                    exit_codes[host] = int(match.group(1))
            remaining = deadline - time.monotonic()
            if len(exit_codes) == len(self.host_list) or remaining <= 0:
                return exit_codes
            time.sleep(min(interval, remaining))

    def kill_io_processes(self):
        # The [x] bracket keeps pkill -f from matching the shell that runs this command.
        self.orch.exec(
            f"pkill -9 -f '[t]ests/python/io/benchmark.py'; "
            f"pkill -9 -f '[t]orchrun .*--master_port={self.master_port}'; true",
            timeout=self.SETUP_TIMEOUT,
        )

    def run_mori_torch_io_test(
        self,
        op_type='read',
        enable_sess=True,
        buffer_size=32768,
        transfer_batch_size=128,
        no_of_qp_per_transfer=1,
        no_of_initiators=8,
        no_of_targets=8,
    ):
        """
        Run MORI Torch-based IO benchmark across multiple nodes using torchrun,
        collect performance results, parse them, and validate against expected
        bandwidth and latency thresholds.

        This test:
        - Launches one detached torchrun process per node (inside the container,
          or on the host under baremetal)
        - Runs a distributed IO benchmark
        - Polls until every node's torchrun exits, then kills any leftovers
        - Parses per-rank performance tables
        - Verifies bandwidth and latency against expected baselines

        Args:
        op_type (str): IO operation type ('read' or 'write'), used to select
                       expected results.
        enable_sess (bool): Enable MORI session mode if True.
        buffer_size (int): Size of the IO buffer in bytes.
        transfer_batch_size (int): Number of transfers batched together.
        no_of_qp_per_transfer (int): Number of Queue Pairs per transfer.
        no_of_initiators (int): Number of initiator devices.
        no_of_targets (int): Number of target devices.
        """
        if self.master_addr not in self.host_list:
            fail_test(f'master_addr {self.master_addr} is not one of the cluster hosts {self.host_list}')
            return

        # Build command-line options for the MORI benchmark
        # These options control IO batching, buffer sizing, and device topology
        cmd_opts = f'--enable-batch-transfer --buffer-size {buffer_size} '
        cmd_opts = cmd_opts + f' --transfer-batch-size {transfer_batch_size}'
        cmd_opts = cmd_opts + f' --num-qp-per-transfer {no_of_qp_per_transfer}'
        cmd_opts = cmd_opts + f' --num-initiator-dev {no_of_initiators}'
        cmd_opts = cmd_opts + f' --num-target-dev {no_of_targets}'
        if enable_sess:
            cmd_opts = cmd_opts + ' --enable-sess'

        case_name = f'io_{op_type}_{buffer_size}_{transfer_batch_size}_{no_of_qp_per_transfer}'
        cmd_list, log_paths = self.build_io_launch_cmds(case_name, cmd_opts)
        try:
            self._exec_cmd_list(cmd_list, timeout=self.IO_LAUNCH_TIMEOUT)
            exit_codes = self.poll_io_completion(log_paths)
        finally:
            # fail_test records without raising, so a launch/poll exception already in flight stays the reported one.
            try:
                self.kill_io_processes()
            except Exception as e:
                fail_test(f'could not kill MORI-IO processes: {e}')

        for host in self.host_list:
            if host not in exit_codes:
                fail_test(f'MORI IO benchmark did not finish within {self.IO_POLL_TIMEOUT}s on node {host}')
            elif exit_codes[host] != 0:
                fail_test(f'MORI IO benchmark exited with code {exit_codes[host]} on node {host}')

        # ------------------------------------------------------------------
        # Collect benchmark output from the rank-0 (master) node log
        # ------------------------------------------------------------------
        out_dict = self.orch.exec(f'cat {log_paths[self.master_addr]}', hosts=[self.master_addr], timeout=60)
        script_out = out_dict.get(self.master_addr) or ''
        if not re.search('Max BW', script_out, re.I):
            fail_test('MORI benchmark test did not succeed, no Bandwidth numbers seen')
            return
        # ------------------------------------------------------------------
        # Parse benchmark output into structured per-rank results
        #
        # Expected format:
        # {
        #   "ranks": {
        #     rank_id: {
        #       "rows": [ { MsgSize_B, Avg_BW_GBps, Avg_Lat_us, ... }, ... ]
        #     }
        #   }
        # }
        # ------------------------------------------------------------------
        # act_res_dict = parse_pretty_table_to_dict( script_out )
        act_res_dict = parse_pretty_tables_multi_rank(script_out)
        if op_type == "read":
            op_key = "io_read"
        elif op_type == "write":
            op_key = "io_write"

        log.info('^^^^^^^^^^^^^^^^^^^^')
        log.info("%s", list(self.expected_results_dict.keys()))
        log.info('^^^^^^^^^^^^^^^^^^^^')
        exp_res_dict = self.expected_results_dict[op_key]
        m_key = (
            'BUFF_SIZE:'
            + str(buffer_size)
            + ','
            + 'TRANSFER_SIZE:'
            + str(transfer_batch_size)
            + ','
            + 'QP_COUNT:'
            + str(no_of_qp_per_transfer)
        )
        # ------------------------------------------------------------------
        # Validate actual results against expected thresholds
        # ------------------------------------------------------------------
        for rank_no in act_res_dict['ranks'].keys():
            for row_dict in act_res_dict['ranks'][rank_no]['rows']:
                # Only validate if expected results exist for this configuration
                if m_key in exp_res_dict:
                    # Match by message size
                    for msg_size in exp_res_dict[m_key].keys():
                        exp_max_bw = exp_res_dict[m_key][msg_size]['max_bw']
                        exp_avg_lat = exp_res_dict[m_key][msg_size]['avg_lat']
                        # print(f'############ {row_dict['MsgSize_B']}, {msg_size}' )
                        if int(row_dict['MsgSize_B']) == int(msg_size):
                            # Validate bandwidth: actual must be >= expected
                            if float(row_dict['Avg_BW_GBps']) < float(exp_max_bw):
                                fail_test(f'''BW is lower than expected for \
                                      rank {rank_no}, Msg size {msg_size}, \
                                      actual BW {row_dict['Avg_BW_GBps']}, \
                                      expected BW {exp_max_bw}, \
                                      buffer_size,transfer_batch_size,no_of_qp_per_transfer = \
                                      {m_key}''')
                            else:
                                log.info(f'''BW is as expected for \
                                      rank {rank_no}, Msg size {msg_size}, \
                                      actual BW {row_dict['Avg_BW_GBps']}, \
                                      expected BW {exp_max_bw}, \
                                      buffer_size,transfer_batch_size,no_of_qp_per_transfer = \
                                      {m_key}''')
                            # Validate latency: actual must be <= expected
                            if float(row_dict['Avg_Lat_us']) > float(exp_avg_lat):
                                fail_test(f'''Latency is higher than expected for \
                                      rank {rank_no}, Msg size {msg_size}, \
                                      actual Avg Lat {row_dict['Avg_Lat_us']}, \
                                      expected Avg Lat {exp_avg_lat}, \
                                      buffer_size,transfer_batch_size,no_of_qp_per_transfer = \
                                      {m_key}''')
                            else:
                                log.info(f'''Latency is as expected for \
                                      rank {rank_no}, Msg size {msg_size}, \
                                      actual Avg Lat {row_dict['Avg_Lat_us']}, \
                                      expected Avg Lat {exp_avg_lat}, \
                                      buffer_size,transfer_batch_size,no_of_qp_per_transfer = \
                                      {m_key}''')
