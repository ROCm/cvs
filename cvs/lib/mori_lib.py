'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import re
import time
from typing import List, Dict, Tuple

from cvs.lib import globals
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *

log = globals.log


def textwrap_for_cmd(msg_string):
    return '\n'.join([m.lstrip() for m in msg_string.split('\n')])


HEADER_MAP = {
    "MsgSize (B)": "MsgSize_B",
    "BatchSize": "BatchSize",
    "TotalSize (MB)": "TotalSize_MB",
    "Max BW (GB/s)": "Max_BW_GBps",
    "Avg Bw (GB/s)": "Avg_BW_GBps",
    "Min Lat (us)": "Min_Lat_us",
    "Avg Lat (us)": "Avg_Lat_us",
}


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
            json_headers = [HEADER_MAP[h] for h in headers]

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
    print(output_dict)
    return output_dict





def parse_ibgda_output(text: str) -> Tuple[Dict, List[Dict]]:
    metadata = {}
    results = []

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

    # ---- Parse metadata line ----
    # Example:
    # Blocks: 4, Threads: 256, Iterations: 10, QPs:4
    meta_pattern = re.compile(
        r"Blocks:\s*(\d+),\s*Threads:\s*(\d+),\s*Iterations:\s*(\d+),\s*QPs:\s*(\d+)"
    )

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
        r"(\d+)\s+"          # Size(B)
        r"([\d.]+)\s+"       # bw(GB)
        r"([\d.]+)\s+"       # Time(ms)
        r"([\d.]+)$"         # Rate(Mpps)
    )

    for line in lines:
        if line.startswith("Index") or line.startswith("IBGDA"):
            continue

        match = row_pattern.match(line)
        if match:
            results.append({
                "size_bytes": int(match.group(1)),
                "bandwidth_gb": float(match.group(2)),
                "time_ms": float(match.group(3)),
                "rate_mpps": float(match.group(4)),
            })

    return metadata, results




class MoriBenchmark:
    def __init__(self, phdl, mori_dict, gpu_type):
        self.phdl = phdl
        self.host_list = phdl.host_list
        self.nnodes = len(self.host_list)
        self.mori_dict = mori_dict
        self.master_addr = self.mori_dict['master_addr']
        self.master_port = self.mori_dict['master_port']
        self.container_image = self.mori_dict['container_image']
        self.container_name = self.mori_dict['container_name']
        self.oob_port = self.mori_dict['oob_port']
        self.more_device_list = self.mori_dict['mori_device_list']
        self.gpu_type = gpu_type
        self.torchlib_dir = self.mori_dict['torchlib_dir']
        self.mori_dir = self.mori_dict['mori_dir']
        self.mori_device_list = self.mori_dict['mori_device_list']
        self.nic_type = self.mori_dict['nic_type']
        self.log_dir = self.mori_dict['log_dir']
        self.expected_results_dict = self.mori_dict['expected_results']

    def create_env_script(
        self,
    ):
        cmd = f'''docker exec {self.container_name} /bin/bash -c "echo '
              export PYTHONPATH={self.mori_dir}:$PYTHONPATH
              #export LD_LIBRARY_PATH={self.torchlib_dir}:$LD_LIBRARY_PATH
              export NCCL_SOCKET_IFNAME={self.oob_port}
              export GLOO_SOCKET_IFNAME={self.oob_port}
              export GLOO_TCP_IFNAME={self.oob_port}
              export GAI_PREFER_IPV4=1
              export MORI_RDMA_DEVICES={self.mori_device_list}
              ' > /tmp/mori_env_script.sh"
              '''
        formatted_cmd = textwrap_for_cmd(cmd)
        print(f'%%%%%%% formatted_cmd = {formatted_cmd}')
        self.phdl.exec(formatted_cmd)
        cmd = f'''docker exec {self.container_name} /bin/bash -c "
              mkdir -p {self.log_dir}"'''
        self.phdl.exec(formatted_cmd)

    def exec_nic_setup_scripts(
        self,
    ):
        # This is a temporary hack needed for broadcom nics to work within containers ..
        if re.search('broadcom|thor', self.nic_type, re.I):
            # override the gid_index to 3 for broadcom
            self.nccl_ib_gid_index = 3
            cmd = f'docker exec {self.container_name} /bin/bash -c "sudo \
                    cp /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.host \
                    /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so;" '
            pout_dict = self.phdl.exec(cmd)
            for node in pout_dict.keys():
                if not re.search('hca_id:\s+bnxt_', out_dict[node], re.I):
                    print(pout_dict[node])

    def install_packages(
        self,
    ):
        cmd = f'''docker exec {self.container_name} /bin/bash -c "
              pip3 install prettytable
              " '''
        formatted_cmd = textwrap_for_cmd(cmd)
        self.phdl.exec(formatted_cmd)

    def check_ibv_devices(
        self,
    ):
        cmd = f'''docker exec {self.container_name} /bin/bash -c "ibv_devinfo" '''
        out_dict = self.phdl.exec(cmd)
        for node in out_dict.keys():
            dev_list = self.mori_device_list.split(',')
            for dev_nam in dev_list:
                if not re.search(f'{dev_nam}', out_dict[node], re.I):
                    fail_test(
                        f'ERROR - MORI device {dev_nam} not showing up under container \
                            {self.container_name} on node {node}'
                    )


    def run_shmem_apitest(
        self,
        ):
        cmd = f'''docker exec {self.container_name} /bin/bash -c  " \
                    source /tmp/mori_env_script.sh && \
                    cd {self.mori_dir} && \
                    pytest -vvv ./tests/python/shmem/test_api.py" '''
        out_dict = self.phdl.exec(cmd)
        for node in out_dict.keys():
            if not re.search( 'PASSED', out_dict[node], re.I ):
                fail_test(f'ERROR - shmem test_api.py did not run properly, no PASSED test results seen')
            if re.search( 'FAIL', out_dict[node] ):
                fail_test(f'ERROR - one or more shmem test_api.py tests failed')


    def run_ibgda_dist_write(
        self,
        no_of_procs=2,
        min_val=2,
        max_val='16m',
        ctas=2,
        threads=256,
        qp_count=4,
        iters=1
        ):
        cmd = f'''docker exec {self.container_name} /bin/bash -c  " \
                    source /tmp/mori_env_script.sh && \
                    cd {self.mori_dir} && \
                    mpiexec --allow-run-as-root -x MORI_GLOBAL_LOG_LEVEL=TRACE \
                    -np {no_of_procs} ./build/examples/dist_write -c {ctas} -t {threads} \
                    -b {min_val} -e {max_val} -f 2 -q {qp_count} -n {iters}" '''
        out_dict = self.phdl.exec(cmd)
        exp_res_dict = self.expected_results_dict['ibgda_write']
        for node in out_dict.keys():
            if not re.search( 'Index\s+Size', out_dict[node], re.I ):
                fail_test(f'ERROR - dist_write did not complete properly - results not seen')
            else:
                meta_data, results = parse_ibgda_output( out_dict[node] )
                print(results)
                m_key = f'''PROCS:{no_of_procs},CTAS:{ctas},THREADS:{threads},QP_COUNT:{qp_count}'''
                if m_key in exp_res_dict.keys():
                    for row_dict in results:
                        act_msg_size = row_dict['size_bytes']
                        actual_bw = row_dict['bandwidth_gb']
                        print(f'exp_res_dict = {exp_res_dict}')
                        print(f'exp_res_dict = {exp_res_dict[m_key].keys()}')
                        print(act_msg_size)
                        for exp_msg_size in list(exp_res_dict[m_key].keys()):
                            exp_bw = exp_res_dict[m_key][exp_msg_size]['max_bw']
                            if int(act_msg_size) == int(exp_msg_size):
                                if float(actual_bw) < float(exp_bw):
                                    fail_test(f'IBGDA Mori BW less than expected for  \
                                      PROCS:{no_of_procs},CTAS:{ctas},THREADS:{threads},QP_COUNT:{qp_count} \
                                      expected = {exp_bw}, actual = {actual_bw}')
                                else:
                                    print(f'IBGDA Mori BW is as expected for  \
                                      PROCS:{no_of_procs},CTAS:{ctas},THREADS:{threads},QP_COUNT:{qp_count} \
                                      expected = {exp_bw}, actual = {actual_bw}')




    def run_dispatch_combine(
        self,
        ):
        cmd = f'''docker exec {self.container_name} /bin/bash -c  " \
                    source /tmp/mori_env_script.sh && \
                    cd {self.mori_dir} && \
                    pytest -vvv ./tests/python/ops/test_dispatch_combine.py" '''
        out_dict = self.phdl.exec(cmd)
        for node in out_dict.keys():
            if not re.search( 'PASSED', out_dict[node], re.I ):
                fail_test(f'ERROR - test_dispatch_combine.py did not run properly, no PASSED test results seen')
            if re.search( 'FAIL', out_dict[node], re.I ):
                fail_test(f'ERROR - one or more test_dispatch_combine.py tests failed') 



    def run_bench_dispatch_combine(
        self,
        ):
        cmd = f'''docker exec {self.container_name} /bin/bash -c  " \
                    source /tmp/mori_env_script.sh && \
                    cd {self.mori_dir} && \
                    pytest -vvv ./tests/python/ops/bench_dispatch_combine.py" '''
        out_dict = self.phdl.exec(cmd)
        for node in out_dict.keys():
            if not re.search( 'PASSED', out_dict[node], re.I ):
                fail_test(f'ERROR - bench_dispatch_combine.py did not run properly, no PASSED test results seen')
            if re.search( 'FAIL', out_dict[node], re.I ):
                fail_test(f'ERROR - one or more bench_dispatch_combine.py tests failed') 



    def run_concurrent_put_threads(
        self,
        ):
        cmd = f'''docker exec {self.container_name} /bin/bash -c  " \
                    source /tmp/mori_env_script.sh && \
                    cd {self.mori_dir} && \
                    mpiexec --allow-run-as-root -np 2 ./build/examples/concurrent_put_thread" '''
        out_dict = self.phdl.exec(cmd)
        for node in out_dict.keys():
            if not re.search( 'PASSED', out_dict[node], re.I ):
                fail_test(f'ERROR - test concurrent_put_thread did not run properly, no PASSED test results seen')
            if re.search( 'FAIL', out_dict[node], re.I ):
                fail_test(f'ERROR - one or more concurrent_put_thread tests failed') 


    def run_concurrent_put_imm_threads(
        self,
        ):
        cmd = f'''docker exec {self.container_name} /bin/bash -c  " \
                    source /tmp/mori_env_script.sh && \
                    cd {self.mori_dir} && \
                    mpiexec --allow-run-as-root -np 2 ./build/examples/concurrent_put_imm_thread" '''
        out_dict = self.phdl.exec(cmd)
        for node in out_dict.keys():
            if not re.search( 'PASSED', out_dict[node], re.I ):
                fail_test(f'ERROR - test concurrent_put_imm_thread did not run properly, no PASSED test results seen')
            if re.search( 'FAIL', out_dict[node], re.I ):
                fail_test(f'ERROR - one or more concurrent_put_imm_thread tests failed') 


    def run_concurrent_put_signal_thread(
        self,
        ):
        cmd = f'''docker exec {self.container_name} /bin/bash -c  " \
                    source /tmp/mori_env_script.sh && \
                    cd {self.mori_dir} && \
                    mpiexec --allow-run-as-root -np 2 ./build/examples/concurrent_put_signal_thread" '''
        out_dict = self.phdl.exec(cmd)
        for node in out_dict.keys():
            if not re.search( 'PASSED', out_dict[node], re.I ):
                fail_test(f'ERROR - test concurrent_put_signal_thread did not run properly, no PASSED test results seen')
            if re.search( 'FAIL', out_dict[node], re.I ):
                fail_test(f'ERROR - one or more concurrent_signal_imm_thread tests failed') 




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
        - Launches one torchrun process per node (inside a container)
        - Runs a distributed IO benchmark
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
        cmd_list = []

        # ------------------------------------------------------------------
        # Construct and launch one torchrun command per node
        # ------------------------------------------------------------------
        for i in range(0, int(self.nnodes)):
            # Build command-line options for the MORI benchmark
            # These options control IO batching, buffer sizing, and device topology
            cmd_opts = f'--enable-batch-transfer --buffer-size {buffer_size} '
            cmd_opts = cmd_opts + f' --transfer-batch-size {transfer_batch_size}'
            cmd_opts = cmd_opts + f' --num-qp-per-transfer {no_of_qp_per_transfer}'
            cmd_opts = cmd_opts + f' --num-initiator-dev {no_of_initiators}'
            cmd_opts = cmd_opts + f' --num-target-dev {no_of_targets}'
            if enable_sess:
                cmd_opts = cmd_opts + ' --enable-sess'

            # ------------------------------------------------------------------
            # Compose docker + torchrun command
            #
            # Key details:
            # - One torchrun process per node
            # - torchrun coordinates across NNODES
            # - NODE_RANK determines the rank of the node
            # - Output is redirected to /tmp/torch_cmd.log
            # - Command runs in background (nohup + &)
            # ------------------------------------------------------------------
            cmd = f'''docker exec {self.container_name} /bin/bash -c  " \
                    source /tmp/mori_env_script.sh && \
                    cd {self.mori_dir} && \
                    export NNODES={self.nnodes}
                    export NODE_RANK={i}
                    nohup torchrun --nnodes={self.nnodes} --node_rank={i} --nproc_per_node=1 \
                    --master_addr={self.master_addr} \
                    --master_port={self.master_port} \
                    {self.mori_dir}/tests/python/io/benchmark.py --host={self.host_list[i]} \
                    --all {cmd_opts} > /tmp/torch_cmd.log 2>&1 &"'''
            formatted_cmd = textwrap_for_cmd(cmd)
            cmd_list.append(formatted_cmd)
        # ------------------------------------------------------------------
        # Execute all launch commands in parallel (one per node)
        # ------------------------------------------------------------------
        out_dict = self.phdl.exec_cmd_list(cmd_list)
        time.sleep(20)
        # ------------------------------------------------------------------
        # Collect benchmark output from master node log
        # ------------------------------------------------------------------
        cmd = f'''docker exec {self.container_name} /bin/bash -c  "cat /tmp/torch_cmd.log"'''
        out_dict = self.phdl.exec(cmd)
        script_out = out_dict[self.master_addr]
        if not re.search('Max BW', script_out, re.I):
            fail_test('MORI benchmark test did not succeed, no Bandwidth numbers seen')
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

        print('^^^^^^^^^^^^^^^^^^^^')
        print(self.expected_results_dict.keys())
        print('^^^^^^^^^^^^^^^^^^^^')
        exp_res_dict = self.expected_results_dict[op_key]
        m_key = 'BUFF_SIZE:' + str(buffer_size) + ',' + \
                'TRANSFER_SIZE:' + str(transfer_batch_size) + ',' + \
                'QP_COUNT:' + str(no_of_qp_per_transfer)
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
                        #print(f'############ {row_dict['MsgSize_B']}, {msg_size}' )
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
                                print(f'''BW is as expected for \
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
                                print(f'''Latency is as expected for \
                                      rank {rank_no}, Msg size {msg_size}, \
                                      actual Avg Lat {row_dict['Avg_Lat_us']}, \
                                      expected Avg Lat {exp_avg_lat}, \
                                      buffer_size,transfer_batch_size,no_of_qp_per_transfer = \
                                      {m_key}''')
