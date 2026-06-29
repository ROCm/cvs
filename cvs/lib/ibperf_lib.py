'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import logging
import re
import time
import xlsxwriter

from cvs.lib.utils_lib import fail_test

log = logging.getLogger(__name__)

# Cached one-time probe results for the current process (per binary / ROCm config key).
_dmabuf_support_cache = {}
_rocm_path_cache = {}


def _log_bw_summary(msg_size, res_dict, instance_no=None):
    if not res_dict:
        return
    bws = []
    for node in sorted(res_dict):
        bw = res_dict[node]['bw']
        pps = res_dict[node]['pps']
        bws.append(float(bw))
        log.info('Node %s BW - %s, MPPS - %s', node, bw, pps)
    if instance_no is not None:
        log.debug(
            'BW msg_size=%s gpu=%s: avg=%.2f Gbps across %d nodes',
            msg_size,
            instance_no,
            sum(bws) / len(bws),
            len(bws),
        )


def _log_lat_summary(msg_size, res_dict, instance_no=None):
    if not res_dict:
        return
    avgs = []
    parts = []
    for node in sorted(res_dict):
        lat = res_dict[node]['t_avg']
        avgs.append(float(lat))
        parts.append(f'{node}={lat}')
        log.debug('Node %s msg_size %s: avg latency=%s us', node, msg_size, lat)
    inst_label = f' gpu={instance_no}' if instance_no is not None else ''
    log.info(
        'Latency msg_size=%s%s: avg=%.2f us [%s]',
        msg_size,
        inst_label,
        sum(avgs) / len(avgs),
        ', '.join(parts),
    )


def _log_bw_run_done(bw_test, msg_size, qp_count, result_dict):
    all_bws = []
    for node in result_dict:
        for inst in result_dict.get(node, {}):
            all_bws.append(float(result_dict[node][inst]['bw']))
    if all_bws:
        log.info(
            '%s: msg_size=%s qp=%s done, BW avg=%.2f Gbps (%d GPUs)',
            bw_test,
            msg_size,
            qp_count,
            sum(all_bws) / len(all_bws),
            len(all_bws),
        )


def _log_lat_run_done(lat_test, msg_size, result_dict):
    all_lats = []
    for node in result_dict:
        for inst in result_dict.get(node, {}):
            all_lats.append(float(result_dict[node][inst]['t_avg']))
    if all_lats:
        log.info(
            '%s: msg_size=%s done, latency avg=%.2f us (%d GPUs)',
            lat_test,
            msg_size,
            sum(all_lats) / len(all_lats),
            len(all_lats),
        )


def detect_rocm_path(phdl, config_rocm_path):
    """
    Detect the ROCm installation path, supporting both old (/opt/rocm) and new
    (/opt/rocm/core-X.Y) layouts.

    Args:
        phdl: Parallel SSH handle
        config_rocm_path (str): Configured ROCm path from config file
                                (empty string or '<changeme>' for auto-detect)

    Returns:
        str: Detected ROCm path
    """
    cache_key = config_rocm_path if config_rocm_path else '<auto>'
    if cache_key in _rocm_path_cache:
        log.debug('Using cached ROCm path for %s: %s', cache_key, _rocm_path_cache[cache_key])
        return _rocm_path_cache[cache_key]

    if config_rocm_path and config_rocm_path != '<changeme>':
        out_dict = phdl.exec(
            f'test -d {config_rocm_path}/lib && ls {config_rocm_path}/lib/libamdhip64.so* 2>/dev/null | head -1',
            print_console=False,
        )
        for node, output in out_dict.items():
            if output.strip() and 'libamdhip64.so' in output:
                log.info('ROCm path: %s (configured)', config_rocm_path)
                _rocm_path_cache[cache_key] = config_rocm_path
                return config_rocm_path
            else:
                log.warning(
                    f'Configured ROCm path {config_rocm_path} does not contain required libraries, will auto-detect'
                )

    # Try new ROCm 7.x structure first (/opt/rocm/core-X.Y)
    out_dict = phdl.exec('ls -d /opt/rocm/core-* 2>/dev/null | sort -V | tail -1', print_console=False)
    for node, output in out_dict.items():
        if output and '/opt/rocm/core-' in output:
            rocm_path = output.strip()
            validate_dict = phdl.exec(
                f'test -d {rocm_path}/lib && ls {rocm_path}/lib/libamdhip64.so* 2>/dev/null | head -1',
                print_console=False,
            )
            for _, lib_output in validate_dict.items():
                if lib_output.strip() and 'libamdhip64.so' in lib_output:
                    log.info('ROCm path: %s (auto-detected, new layout)', rocm_path)
                    _rocm_path_cache[cache_key] = rocm_path
                    return rocm_path

    # Fall back to legacy /opt/rocm
    out_dict = phdl.exec(
        'test -d /opt/rocm/lib && ls /opt/rocm/lib/libamdhip64.so* 2>/dev/null | head -1',
        print_console=False,
    )
    for node, output in out_dict.items():
        if output.strip() and 'libamdhip64.so' in output:
            log.info('ROCm path: /opt/rocm (auto-detected, legacy layout)')
            _rocm_path_cache[cache_key] = '/opt/rocm'
            return '/opt/rocm'

    log.warning('Could not detect ROCm path with required libraries, defaulting to /opt/rocm')
    _rocm_path_cache[cache_key] = '/opt/rocm'
    return '/opt/rocm'


def get_ib_bw_pps(phdl, msg_size, cmd, instance_no=None):
    res_dict = {}

    # Run some of the standard verifications
    out_dict = phdl.exec(cmd, print_console=False)
    err_pattern = (
        "Couldn't initialize ROCm device|Failed to init|Unable to open file descriptor|ERROR|FAIL|Segmentation fault"
    )
    for node in out_dict.keys():
        if not re.search('bytes of GPU buffer', out_dict[node], re.I):
            fail_test(f'GPU Buffer allocation failed or Connection not setup for IB Test on node {node}')

        if re.search(err_pattern, out_dict[node], re.I):
            fail_test(f'IB Test failed - Error patterns seen on node {node}')

    # Collect the BW, PPS numbers
    for i in range(1, 10):
        log.debug('BW collection iteration %d for msg_size %s', i, msg_size)
        out_dict = phdl.exec(cmd, print_console=False)
        for node in out_dict.keys():
            res_dict[node] = {}
            pattern = r"{}\s+\d+\s+[0-9\.]+\s+([0-9\.]+)\s+([0-9\.]+)".format(msg_size)
            if re.search(pattern, out_dict[node]):
                match = re.search(pattern, out_dict[node])
                res_dict[node]['bw'] = match.group(1)
                res_dict[node]['pps'] = match.group(2)
                continue
            else:
                log.debug('Node %s: results not ready, sleeping 10s (iteration %d)', node, i)
                time.sleep(10)
            fail_test(
                f'ERROR !!! on node {node} Client did not complete even after max iterations for msg size {msg_size}'
            )
            fail_test(f'ERROR !!! pls check log file for errors on node {node}')
    _log_bw_summary(msg_size, res_dict, instance_no)
    return res_dict


def get_ib_lat_numb(phdl, msg_size, cmd, instance_no=None):
    res_dict = {}

    # Run some of the standard verifications
    out_dict = phdl.exec(cmd, print_console=False)
    err_pattern = (
        "Couldn't initialize ROCm device|Failed to init|Unable to open file descriptor|ERROR|FAIL|Segmentation fault"
    )
    for node in out_dict.keys():
        res_dict[node] = {}
        if not re.search('bytes of GPU buffer', out_dict[node], re.I):
            fail_test(f'GPU Buffer allocation failed or Connection not setup for IB Test on node {node}')

        if re.search(err_pattern, out_dict[node], re.I):
            fail_test(f'IB Test failed - Error patterns seen on node {node}')

    # Collect the latency numbers
    for i in range(1, 4):
        log.debug('Latency collection iteration %d for msg_size %s', i, msg_size)
        out_dict = phdl.exec(cmd, print_console=False)
        for node in out_dict.keys():
            pattern = "{}[\t\s]+[0-9]+[\t\s]+([0-9\.]+)[\t\s]+([0-9\.]+)[\t\s]+([0-9\.]+)[\t\s]+([0-9\.]+)[\t\s]+([0-9\.]+)[\t\s]+([0-9\.]+)[\t\s]+([0-9\.]+)".format(
                msg_size
            )
            if re.search(pattern, out_dict[node]):
                match = re.search(pattern, out_dict[node])
                res_dict[node]['t_min'] = match.group(1)
                res_dict[node]['t_max'] = match.group(2)
                res_dict[node]['t_typical'] = match.group(3)
                res_dict[node]['t_avg'] = match.group(4)
                res_dict[node]['t_stdev'] = match.group(5)
                res_dict[node]['t_99_pct'] = match.group(6)
                res_dict[node]['t_99_9_pct'] = match.group(7)
                continue
            else:
                log.debug('Node %s: latency results not ready, sleeping 10s (iteration %d)', node, i)
                time.sleep(10)
            fail_test(
                f'ERROR !!! on node {node} Client did not complete even after max iterations for msg size {msg_size}'
            )
            fail_test(f'ERROR !!! pls check log file for errors on node {node}')
    _log_lat_summary(msg_size, res_dict, instance_no)
    return res_dict


def verify_expected_bw(bw_test, msg_size, qp_count, res_dict, expected_res):
    log.info('Verifying expected BW for test %s msg_size %s QP count %s', bw_test, msg_size, qp_count)
    log.info("%s", list(expected_res.keys()))
    log.info("%s", res_dict)
    if bw_test in expected_res.keys():
        log.info("%s", list(expected_res[bw_test].keys()))
        if msg_size in expected_res[bw_test].keys():
            if qp_count in expected_res[bw_test][msg_size].keys():
                for node in res_dict.keys():
                    if float(res_dict[node]['bw']) <= float(expected_res[bw_test][msg_size]):
                        fail_test(
                            f"Actual BW {res_dict[node]['bw']} less than the expected BW {expected_res[bw_test][msg_size]} for test {bw_test} on node {node}"
                        )


def verify_expected_lat(lat_test, msg_size, res_dict, expected_res):
    log.info('Verifying expected latency for test %s msg_size %s', lat_test, msg_size)
    log.info("%s", list(expected_res.keys()))
    log.info("%s", res_dict)
    if lat_test in expected_res.keys():
        log.info("%s", list(expected_res[lat_test].keys()))
        if msg_size in expected_res[lat_test].keys():
            for node in res_dict.keys():
                if float(res_dict[node]['lat']) >= float(expected_res[lat_test][msg_size]):
                    fail_test(
                        f"Actual BW {res_dict[node]['lat']} greater than the expected BW {expected_res[lat_test][msg_size]} for test {lat_test} on node {node}"
                    )


def run_ib_perf_bw_test(
    phdl,
    bw_test,
    gpu_numa_dict,
    gpu_nic_dict,
    bck_nic_dict,
    app_path,
    msg_size,
    gid_index,
    qp_count=8,
    port_no=1516,
    duration=60,
    rocm_path='',
):
    log.info(
        '%s: msg_size=%s qp=%s start (%ss, %d nodes)',
        bw_test,
        msg_size,
        qp_count,
        duration,
        len(bck_nic_dict),
    )
    app_port = port_no
    result_dict = {}
    i = 0
    cmd_dict = {}
    phdl.exec('sudo rm -rf /tmp/ib_cmds_file.txt', print_console=False)
    phdl.exec('sudo rm -rf /tmp/ib_perf*', print_console=False)
    phdl.exec('touch /tmp/ib_cmds_file.txt', print_console=False)
    if rocm_path:
        log.debug('Setting LD_LIBRARY_PATH to %s/lib for perftest binaries', rocm_path)
        phdl.exec(
            f'echo "export LD_LIBRARY_PATH={rocm_path}/lib:$LD_LIBRARY_PATH" >> /tmp/ib_cmds_file.txt',
            print_console=False,
        )
    server_addr = None
    for node in bck_nic_dict.keys():
        result_dict[node] = {}
        cmd_dict[node] = []
        # even nodes make it server and odd as clients
        if i % 2 == 0:
            # server
            server_addr = node
            cmd = 'echo "sleep 1" >> /tmp/ib_cmds_file.txt'
            cmd_dict[node].append(cmd)
            inst_count = 0
            port_no = app_port
            for gpu_no in range(0, 8):
                card_no = 'card' + str(gpu_no)
                rdma_dev = gpu_nic_dict[node][card_no]['rdma_dev']
                cmd = f'numactl --physcpubind={gpu_numa_dict[node][card_no]["local_cpulist"]} --localalloc {app_path}/{bw_test} -d {rdma_dev} --use_rocm={gpu_no} -x {gid_index} --report_gbits -b -F -D {duration} -p {port_no} -s {msg_size} -q {qp_count} > /tmp/ib_perf_{inst_count}_logs &  2>&1'
                cmd_dict[node].append(f'echo "{cmd}" >> /tmp/ib_cmds_file.txt')
                inst_count = inst_count + 1
                port_no = port_no + 1
        else:
            # client
            cmd = 'echo "sleep 5" >> /tmp/ib_cmds_file.txt'
            cmd_dict[node].append(cmd)
            inst_count = 0
            port_no = app_port
            for gpu_no in range(0, 8):
                card_no = 'card' + str(gpu_no)
                rdma_dev = gpu_nic_dict[node][card_no]['rdma_dev']
                cmd = f'numactl --physcpubind={gpu_numa_dict[node][card_no]["local_cpulist"]} --localalloc {app_path}/{bw_test} -d {rdma_dev} --use_rocm={gpu_no} -x {gid_index} --report_gbits -b -F -D {duration} -p {port_no} -s {msg_size} -q {qp_count} {server_addr} > /tmp/ib_perf_{inst_count}_logs &  2>&1'
                cmd_dict[node].append(f'echo "{cmd}" >> /tmp/ib_cmds_file.txt')
                inst_count = inst_count + 1
                port_no = port_no + 1
        i = i + 1

    # Get number of commands in cmd_dict - should be same for all nodes
    node_list = list(cmd_dict.keys())
    first_node_cmd_list = cmd_dict[node_list[0]]

    for j in range(0, len(first_node_cmd_list)):
        cmd_list = []
        for node in cmd_dict.keys():
            cmd_list.append(cmd_dict[node][j])
        # print(cmd_list)
        phdl.exec_cmd_list(cmd_list, print_console=False)

    # Clean up any stale instances from earlier runs (ignore if none running)
    log.debug('Killing stale %s processes before starting test', bw_test)
    phdl.exec(f'killall {bw_test} 2>/dev/null || true', print_console=False)
    time.sleep(2)
    phdl.exec('source /tmp/ib_cmds_file.txt', print_console=False)

    time.sleep(20)

    for instance_no in range(0, inst_count):
        try:
            bw_pps_dict = get_ib_bw_pps(phdl, msg_size, f'cat /tmp/ib_perf_{instance_no}_logs', instance_no=instance_no)
            for node in bw_pps_dict.keys():
                result_dict[node][instance_no] = {}
                result_dict[node][instance_no]['pps'] = bw_pps_dict[node]['pps']
                result_dict[node][instance_no]['bw'] = bw_pps_dict[node]['bw']
        except Exception:
            log.error(
                'Failed to collect BW/PPS for msg_size=%s qp_count=%s instance=%d',
                msg_size,
                qp_count,
                instance_no,
                exc_info=True,
            )

    _log_bw_run_done(bw_test, msg_size, qp_count, result_dict)
    return result_dict


def run_ib_perf_lat_test(
    phdl, lat_test, gpu_numa_dict, gpu_nic_dict, bck_nic_dict, app_path, msg_size, gid_index, port_no=1516, rocm_path=''
):
    log.info('%s: msg_size=%s start (%d nodes)', lat_test, msg_size, len(bck_nic_dict))
    app_port = port_no
    result_dict = {}
    i = 0
    cmd_dict = {}
    phdl.exec('sudo rm -rf /tmp/ib_cmds_file.txt', print_console=False)
    phdl.exec('sudo rm -rf /tmp/ib_perf*', print_console=False)
    phdl.exec('touch /tmp/ib_cmds_file.txt', print_console=False)
    if rocm_path:
        log.debug('Setting LD_LIBRARY_PATH to %s/lib for perftest binaries', rocm_path)
        phdl.exec(
            f'echo "export LD_LIBRARY_PATH={rocm_path}/lib:$LD_LIBRARY_PATH" >> /tmp/ib_cmds_file.txt',
            print_console=False,
        )
    server_addr = None
    for node in bck_nic_dict.keys():
        result_dict[node] = {}
        cmd_dict[node] = []
        # even nodes make it server and odd as clients
        if i % 2 == 0:
            # server
            server_addr = node
            cmd = 'echo "sleep 1" >> /tmp/ib_cmds_file.txt'
            cmd_dict[node].append(cmd)
            inst_count = 0
            port_no = app_port
            for gpu_no in range(0, 8):
                card_no = 'card' + str(gpu_no)
                rdma_dev = gpu_nic_dict[node][card_no]['rdma_dev']
                cmd = f'numactl --physcpubind={gpu_numa_dict[node][card_no]["local_cpulist"]} --localalloc {app_path}/{lat_test} -d {rdma_dev} --use_rocm={gpu_no} -x {gid_index} --report_gbits -F -p {port_no} -s {msg_size} > /tmp/ib_perf_{inst_count}_logs &  2>&1'
                cmd_dict[node].append(f'echo "{cmd}" >> /tmp/ib_cmds_file.txt')
                inst_count = inst_count + 1
                port_no = port_no + 1
        else:
            # client
            cmd = 'echo "sleep 5" >> /tmp/ib_cmds_file.txt'
            cmd_dict[node].append(cmd)
            inst_count = 0
            port_no = app_port
            for gpu_no in range(0, 8):
                card_no = 'card' + str(gpu_no)
                rdma_dev = gpu_nic_dict[node][card_no]['rdma_dev']
                cmd = f'numactl --physcpubind={gpu_numa_dict[node][card_no]["local_cpulist"]} --localalloc {app_path}/{lat_test} -d {rdma_dev} --use_rocm={gpu_no} -x {gid_index} --report_gbits -F -p {port_no} -s {msg_size} {server_addr} > /tmp/ib_perf_{inst_count}_logs &  2>&1'
                cmd_dict[node].append(f'echo "{cmd}" >> /tmp/ib_cmds_file.txt')
                inst_count = inst_count + 1
                port_no = port_no + 1
        i = i + 1

    # Get number of commands in cmd_dict - should be same for all nodes
    node_list = list(cmd_dict.keys())
    first_node_cmd_list = cmd_dict[node_list[0]]

    for j in range(0, len(first_node_cmd_list)):
        cmd_list = []
        for node in cmd_dict.keys():
            cmd_list.append(cmd_dict[node][j])
        # print(cmd_list)
        phdl.exec_cmd_list(cmd_list, print_console=False)

    # Clean up any stale instances from earlier runs (ignore if none running)
    log.debug('Killing stale %s processes before starting test', lat_test)
    phdl.exec(f'killall {lat_test} 2>/dev/null || true', print_console=False)
    time.sleep(2)
    phdl.exec('source /tmp/ib_cmds_file.txt', print_console=False)

    # wait for the duration of test
    time.sleep(20)

    for instance_no in range(0, inst_count):
        try:
            lat_dict = get_ib_lat_numb(phdl, msg_size, f'cat /tmp/ib_perf_{instance_no}_logs', instance_no=instance_no)
            for node in bck_nic_dict.keys():
                result_dict[node][instance_no] = {}
                result_dict[node][instance_no]['t_min'] = lat_dict[node]['t_min']
                result_dict[node][instance_no]['t_max'] = lat_dict[node]['t_max']
                result_dict[node][instance_no]['t_typical'] = lat_dict[node]['t_typical']
                result_dict[node][instance_no]['t_avg'] = lat_dict[node]['t_avg']
                result_dict[node][instance_no]['t_stdev'] = lat_dict[node]['t_stdev']
                result_dict[node][instance_no]['t_99_pct'] = lat_dict[node]['t_99_pct']
                result_dict[node][instance_no]['t_99_9_pct'] = lat_dict[node]['t_99_9_pct']
        except Exception:
            log.error(
                'Failed to collect latency for msg_size=%s instance=%d',
                msg_size,
                instance_no,
                exc_info=True,
            )

    _log_lat_run_done(lat_test, msg_size, result_dict)
    return result_dict


def split_list_into_n_chunks(original_list, n):
    """
    Splits a list into n approximately equal chunks.
    """
    if not original_list:
        return [[] for _ in range(n)]

    avg_chunk_size = len(original_list) // n
    remainder = len(original_list) % n
    result = []
    current_index = 0

    for i in range(n):
        chunk_size = avg_chunk_size + (1 if i < remainder else 0)
        result.append(original_list[current_index : current_index + chunk_size])
        current_index += chunk_size
    return result


def _to_float(x):
    """Convert ints/floats/strings to float. Accepts '1,234.5' and spaces."""
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip().replace(",", "").replace("_", "")
        # allow trailing percent via uncomment lines below if you want:
        # if s.endswith("%"):
        #     return float(s[:-1]) / 100.0
        return float(s)  # will raise ValueError if not numeric
    raise TypeError(f"Unsupported type {type(x)} for value {x!r}")


def average_of_lists(lists):
    """
    Element-wise average of a list of equal-length lists.
    Normalizes all values to float first.
    """
    if not lists:
        return []
    # normalize
    norm = [[_to_float(v) for v in row] for row in lists]
    # length check
    n = len(norm[0])
    if any(len(row) != n for row in norm):
        raise ValueError("All inner lists must be the same length")
    # average column-wise
    return [sum(col) / len(norm) for col in zip(*norm)]


def round_vals(list_a):
    rounded_vals = [round(float(x), 2) for x in list_a]
    return rounded_vals


# Sample res_dict
# {'ib_write_bw': {2: {'8': {'x.x.x.x': {0: {'pps': '6.229094', 'bw': '0.099666'}, 1: {'pps': '6.391922', 'bw': '0.102271'}, .... 7: { 'pps': '6.79', 'bw': '0.10227' }}
# Top level - application like ib_write_bw, ib_read_bw
# 2 - msg_size
# 8 - Number of QPs
# x.x.x.x - Node IP
# 0 - NIC Instance (will have 0-7)
# pps - packets per sec
# BW - Bandwidth in Gbps


def generate_ibperf_bw_chart(res_dict, excel_file='ib_bw_pps_perf.xlsx'):
    workbook = xlsxwriter.Workbook(excel_file)

    app_list = list(res_dict.keys())

    merge_format = workbook.add_format(
        {
            "bold": 1,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "bg_color": "yellow",
        }
    )
    node_merge_format = workbook.add_format(
        {
            "bold": 1,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
        }
    )

    node_list = []
    msg_size_list = []
    qp_count_list = []
    gpu_no_list = [0, 1, 2, 3, 4, 5, 6, 7]

    for app_name in app_list:
        for msg_size in res_dict[app_name].keys():
            if msg_size not in msg_size_list:
                msg_size_list.append(msg_size)

            for qp_count in res_dict[app_name][msg_size].keys():
                if qp_count not in qp_count_list:
                    qp_count_list.append(qp_count)

                for node in res_dict[app_name][msg_size][qp_count].keys():
                    if node not in node_list:
                        node_list.append(node)

    for app_name in app_list:
        for qp_count in qp_count_list:
            sheet_name = app_name + "_qp_" + str(qp_count)
            worksheet = workbook.add_worksheet(sheet_name)
            bold = workbook.add_format({"bold": 1})

            heading = "Test {} - BW, MPPS Numbers for {} QPs".format(app_name, qp_count)
            worksheet.merge_range("A1:T1", heading, merge_format)

            # Init data lists
            d_msg_size_list = []
            d_bw_gpu_0_list = []
            d_bw_gpu_1_list = []
            d_bw_gpu_2_list = []
            d_bw_gpu_3_list = []
            d_bw_gpu_4_list = []
            d_bw_gpu_5_list = []
            d_bw_gpu_6_list = []
            d_bw_gpu_7_list = []
            d_pps_gpu_0_list = []
            d_pps_gpu_1_list = []
            d_pps_gpu_2_list = []
            d_pps_gpu_3_list = []
            d_pps_gpu_4_list = []
            d_pps_gpu_5_list = []
            d_pps_gpu_6_list = []
            d_pps_gpu_7_list = []
            d_pps_total_list = []
            d_bw_total_list = []

            for node in node_list:
                d_msg_size_list.extend(msg_size_list)
                for msg_size in msg_size_list:
                    tot_pps = 0.0
                    tot_bw = 0.0
                    for gpu_no in gpu_no_list:
                        if gpu_no == 0:
                            d_bw_gpu_0_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                            d_pps_gpu_0_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_pps = tot_pps + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_bw = tot_bw + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                        elif gpu_no == 1:
                            d_bw_gpu_1_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                            d_pps_gpu_1_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_pps = tot_pps + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_bw = tot_bw + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                        elif gpu_no == 2:
                            d_bw_gpu_2_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                            d_pps_gpu_2_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_pps = tot_pps + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_bw = tot_bw + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                        elif gpu_no == 3:
                            d_bw_gpu_3_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                            d_pps_gpu_3_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_pps = tot_pps + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_bw = tot_bw + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                        elif gpu_no == 4:
                            d_bw_gpu_4_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                            d_pps_gpu_4_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_pps = tot_pps + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_bw = tot_bw + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                        elif gpu_no == 5:
                            d_bw_gpu_5_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                            d_pps_gpu_5_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_pps = tot_pps + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_bw = tot_bw + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                        elif gpu_no == 6:
                            d_bw_gpu_6_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                            d_pps_gpu_6_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_pps = tot_pps + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_bw = tot_bw + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                        elif gpu_no == 7:
                            d_bw_gpu_7_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                            d_pps_gpu_7_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_pps = tot_pps + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_bw = tot_bw + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                    d_bw_total_list.append(tot_bw)
                    d_pps_total_list.append(tot_pps)

            split_bw_list = split_list_into_n_chunks(d_bw_total_list, len(node_list))
            split_pps_list = split_list_into_n_chunks(d_pps_total_list, len(node_list))
            split_bw_gpu0_list = split_list_into_n_chunks(d_bw_gpu_0_list, len(node_list))
            split_bw_gpu1_list = split_list_into_n_chunks(d_bw_gpu_1_list, len(node_list))
            split_bw_gpu2_list = split_list_into_n_chunks(d_bw_gpu_2_list, len(node_list))
            split_bw_gpu3_list = split_list_into_n_chunks(d_bw_gpu_3_list, len(node_list))
            split_bw_gpu4_list = split_list_into_n_chunks(d_bw_gpu_4_list, len(node_list))
            split_bw_gpu5_list = split_list_into_n_chunks(d_bw_gpu_5_list, len(node_list))
            split_bw_gpu6_list = split_list_into_n_chunks(d_bw_gpu_6_list, len(node_list))
            split_bw_gpu7_list = split_list_into_n_chunks(d_bw_gpu_7_list, len(node_list))
            split_pps_gpu0_list = split_list_into_n_chunks(d_pps_gpu_0_list, len(node_list))
            split_pps_gpu1_list = split_list_into_n_chunks(d_pps_gpu_1_list, len(node_list))
            split_pps_gpu2_list = split_list_into_n_chunks(d_pps_gpu_2_list, len(node_list))
            split_pps_gpu3_list = split_list_into_n_chunks(d_pps_gpu_3_list, len(node_list))
            split_pps_gpu4_list = split_list_into_n_chunks(d_pps_gpu_4_list, len(node_list))
            split_pps_gpu5_list = split_list_into_n_chunks(d_pps_gpu_5_list, len(node_list))
            split_pps_gpu6_list = split_list_into_n_chunks(d_pps_gpu_6_list, len(node_list))
            split_pps_gpu7_list = split_list_into_n_chunks(d_pps_gpu_7_list, len(node_list))

            avg_bw_data = average_of_lists(split_bw_list)
            avg_pps_data = average_of_lists(split_pps_list)
            avg_bw_gpu0_data = average_of_lists(split_bw_gpu0_list)
            avg_bw_gpu1_data = average_of_lists(split_bw_gpu1_list)
            avg_bw_gpu2_data = average_of_lists(split_bw_gpu2_list)
            avg_bw_gpu3_data = average_of_lists(split_bw_gpu3_list)
            avg_bw_gpu4_data = average_of_lists(split_bw_gpu4_list)
            avg_bw_gpu5_data = average_of_lists(split_bw_gpu5_list)
            avg_bw_gpu6_data = average_of_lists(split_bw_gpu6_list)
            avg_bw_gpu7_data = average_of_lists(split_bw_gpu7_list)

            avg_pps_gpu0_data = average_of_lists(split_pps_gpu0_list)
            avg_pps_gpu1_data = average_of_lists(split_pps_gpu1_list)
            avg_pps_gpu2_data = average_of_lists(split_pps_gpu2_list)
            avg_pps_gpu3_data = average_of_lists(split_pps_gpu3_list)
            avg_pps_gpu4_data = average_of_lists(split_pps_gpu4_list)
            avg_pps_gpu5_data = average_of_lists(split_pps_gpu5_list)
            avg_pps_gpu6_data = average_of_lists(split_pps_gpu6_list)
            avg_pps_gpu7_data = average_of_lists(split_pps_gpu7_list)

            # Merge cols for Node IP
            worksheet.merge_range("A2:A3", "Node IP", bold)
            worksheet.merge_range("B2:B3", "Msg Size", bold)
            # Merge 2 cols for every GPU and Total
            worksheet.merge_range("C2:D2", "Total", bold)
            worksheet.merge_range("E2:F2", "GPU0", bold)
            worksheet.merge_range("G2:H2", "GPU1", bold)
            worksheet.merge_range("I2:J2", "GPU2", bold)
            worksheet.merge_range("K2:L2", "GPU3", bold)
            worksheet.merge_range("M2:N2", "GPU4", bold)
            worksheet.merge_range("O2:P2", "GPU5", bold)
            worksheet.merge_range("Q2:R2", "GPU6", bold)
            worksheet.merge_range("S2:T2", "GPU7", bold)

            headings = []
            for i in range(0, 9):
                headings.append("BW")
                headings.append("PPS")
            # Write headings for BW, PPS for total and every GPU
            worksheet.write_row("C3", headings, bold)

            cur_row = 4
            msg_list_len = len(msg_size_list)
            for node_ip in node_list:
                worksheet.merge_range(f"A{cur_row}:A{cur_row + msg_list_len - 1}", node_ip, node_merge_format)
                cur_row = cur_row + msg_list_len

            # For the cluster Avg
            worksheet.merge_range(f"A{cur_row}:A{cur_row + msg_list_len - 1}", "Cluster Avg")

            worksheet.write_column("B4", round_vals(d_msg_size_list))
            worksheet.write_column("C4", round_vals(d_bw_total_list))
            worksheet.write_column("D4", round_vals(d_pps_total_list))
            worksheet.write_column("E4", round_vals(d_bw_gpu_0_list))
            worksheet.write_column("F4", round_vals(d_pps_gpu_0_list))
            worksheet.write_column("G4", round_vals(d_bw_gpu_1_list))
            worksheet.write_column("H4", round_vals(d_pps_gpu_1_list))
            worksheet.write_column("I4", round_vals(d_bw_gpu_2_list))
            worksheet.write_column("J4", round_vals(d_pps_gpu_2_list))
            worksheet.write_column("K4", round_vals(d_bw_gpu_3_list))
            worksheet.write_column("L4", round_vals(d_pps_gpu_3_list))
            worksheet.write_column("M4", round_vals(d_bw_gpu_4_list))
            worksheet.write_column("N4", round_vals(d_pps_gpu_4_list))
            worksheet.write_column("O4", round_vals(d_bw_gpu_5_list))
            worksheet.write_column("P4", round_vals(d_pps_gpu_5_list))
            worksheet.write_column("Q4", round_vals(d_bw_gpu_6_list))
            worksheet.write_column("R4", round_vals(d_pps_gpu_6_list))
            worksheet.write_column("S4", round_vals(d_bw_gpu_7_list))
            worksheet.write_column("T4", round_vals(d_pps_gpu_7_list))

            x_chart_index = len(d_msg_size_list) + 4

            worksheet.write_column(f"B{x_chart_index}", msg_size_list)
            worksheet.write_column(f"C{x_chart_index}", round_vals(avg_bw_data))
            worksheet.write_column(f"D{x_chart_index}", round_vals(avg_pps_data))
            worksheet.write_column(f"E{x_chart_index}", round_vals(avg_bw_gpu0_data))
            worksheet.write_column(f"F{x_chart_index}", round_vals(avg_pps_gpu0_data))
            worksheet.write_column(f"G{x_chart_index}", round_vals(avg_bw_gpu1_data))
            worksheet.write_column(f"H{x_chart_index}", round_vals(avg_pps_gpu1_data))
            worksheet.write_column(f"I{x_chart_index}", round_vals(avg_bw_gpu2_data))
            worksheet.write_column(f"J{x_chart_index}", round_vals(avg_pps_gpu2_data))
            worksheet.write_column(f"K{x_chart_index}", round_vals(avg_bw_gpu3_data))
            worksheet.write_column(f"L{x_chart_index}", round_vals(avg_pps_gpu3_data))
            worksheet.write_column(f"M{x_chart_index}", round_vals(avg_bw_gpu4_data))
            worksheet.write_column(f"N{x_chart_index}", round_vals(avg_pps_gpu4_data))
            worksheet.write_column(f"O{x_chart_index}", round_vals(avg_bw_gpu5_data))
            worksheet.write_column(f"P{x_chart_index}", round_vals(avg_pps_gpu5_data))
            worksheet.write_column(f"Q{x_chart_index}", round_vals(avg_bw_gpu6_data))
            worksheet.write_column(f"R{x_chart_index}", round_vals(avg_pps_gpu6_data))
            worksheet.write_column(f"S{x_chart_index}", round_vals(avg_bw_gpu7_data))
            worksheet.write_column(f"T{x_chart_index}", round_vals(avg_pps_gpu7_data))

            x_chart_index = len(d_msg_size_list) + len(msg_size_list) + 7
            y_chart_index = 2
            chart = workbook.add_chart({"type": "column"})

            row_end = 4 + len(d_msg_size_list)

            chart.add_series(
                {
                    "name": "={}!$B$2".format(sheet_name),
                    "categories": "={}!$B${}:$B${}".format(sheet_name, row_end, row_end + len(msg_size_list)),
                    "values": "={}!$C${}:$C${}".format(sheet_name, row_end, row_end + len(msg_size_list)),
                    "data_labels": {
                        "value": True,
                        "font": {
                            "rotation": -45,
                        },
                    },
                }
            )

            chart.set_size({'width': 1200, 'height': 720})
            chart.set_title({"name": "Bandwidth in Gbps"})
            chart.set_x_axis({"name": "Msg Size"})
            chart.set_y_axis({"name": "Bandwidth in Gbps"})
            chart.set_style(11)
            worksheet.insert_chart(x_chart_index, y_chart_index, chart)

    workbook.close()
    log.info('BW performance chart saved: %s', excel_file)


def generate_ibperf_lat_chart(res_dict, excel_file='ib_lat_perf.xlsx'):
    workbook = xlsxwriter.Workbook(excel_file)

    app_list = list(res_dict.keys())

    merge_format = workbook.add_format(
        {
            "bold": 1,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "bg_color": "yellow",
        }
    )
    node_merge_format = workbook.add_format(
        {
            "bold": 1,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
        }
    )

    node_list = []
    msg_size_list = []
    gpu_no_list = [0, 1, 2, 3, 4, 5, 6, 7]

    for app_name in app_list:
        for msg_size in res_dict[app_name].keys():
            if msg_size not in msg_size_list:
                msg_size_list.append(msg_size)

                for node in res_dict[app_name][msg_size].keys():
                    if node not in node_list:
                        node_list.append(node)

    for app_name in app_list:
        sheet_name = app_name + "_lat"
        worksheet = workbook.add_worksheet(sheet_name)
        bold = workbook.add_format({"bold": 1})

        heading = "Test {} - latency results".format(app_name)
        worksheet.merge_range("A1:Z1", heading, merge_format)

        # Init data lists
        d_msg_size_list = []
        d_tmin_gpu_0_list = []
        d_tmin_gpu_1_list = []
        d_tmin_gpu_2_list = []
        d_tmin_gpu_3_list = []
        d_tmin_gpu_4_list = []
        d_tmin_gpu_5_list = []
        d_tmin_gpu_6_list = []
        d_tmin_gpu_7_list = []
        d_tmax_gpu_0_list = []
        d_tmax_gpu_1_list = []
        d_tmax_gpu_2_list = []
        d_tmax_gpu_3_list = []
        d_tmax_gpu_4_list = []
        d_tmax_gpu_5_list = []
        d_tmax_gpu_6_list = []
        d_tmax_gpu_7_list = []
        d_tavg_gpu_0_list = []
        d_tavg_gpu_1_list = []
        d_tavg_gpu_2_list = []
        d_tavg_gpu_3_list = []
        d_tavg_gpu_4_list = []
        d_tavg_gpu_5_list = []
        d_tavg_gpu_6_list = []
        d_tavg_gpu_7_list = []
        d_tstdev_gpu_0_list = []
        d_tstdev_gpu_1_list = []
        d_tstdev_gpu_2_list = []
        d_tstdev_gpu_3_list = []
        d_tstdev_gpu_4_list = []
        d_tstdev_gpu_5_list = []
        d_tstdev_gpu_6_list = []
        d_tstdev_gpu_7_list = []
        d_t99pct_gpu_0_list = []
        d_t99pct_gpu_1_list = []
        d_t99pct_gpu_2_list = []
        d_t99pct_gpu_3_list = []
        d_t99pct_gpu_4_list = []
        d_t99pct_gpu_5_list = []
        d_t99pct_gpu_6_list = []
        d_t99pct_gpu_7_list = []

        d_avg_tmin_list = []
        d_avg_tmax_list = []
        d_avg_tavg_list = []
        d_avg_tstdev_list = []
        d_avg_t99pct_list = []

        for node in node_list:
            d_msg_size_list.extend(msg_size_list)
            for msg_size in msg_size_list:
                tot_tmin = 0.0
                tot_tmax = 0.0
                tot_tavg = 0.0
                tot_tstdev = 0.0
                tot_t99pct = 0.0
                for gpu_no in gpu_no_list:
                    if gpu_no == 0:
                        d_tmin_gpu_0_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_min'])
                        d_tmax_gpu_0_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_max'])
                        d_tavg_gpu_0_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_avg'])
                        d_tstdev_gpu_0_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_stdev'])
                        d_t99pct_gpu_0_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_99_pct'])
                    elif gpu_no == 1:
                        d_tmin_gpu_1_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_min'])
                        d_tmax_gpu_1_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_max'])
                        d_tavg_gpu_1_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_avg'])
                        d_tstdev_gpu_1_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_stdev'])
                        d_t99pct_gpu_1_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_99_pct'])
                    elif gpu_no == 2:
                        d_tmin_gpu_2_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_min'])
                        d_tmax_gpu_2_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_max'])
                        d_tavg_gpu_2_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_avg'])
                        d_tstdev_gpu_2_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_stdev'])
                        d_t99pct_gpu_2_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_99_pct'])
                    elif gpu_no == 3:
                        d_tmin_gpu_3_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_min'])
                        d_tmax_gpu_3_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_max'])
                        d_tavg_gpu_3_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_avg'])
                        d_tstdev_gpu_3_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_stdev'])
                        d_t99pct_gpu_3_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_99_pct'])
                    elif gpu_no == 4:
                        d_tmin_gpu_4_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_min'])
                        d_tmax_gpu_4_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_max'])
                        d_tavg_gpu_4_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_avg'])
                        d_tstdev_gpu_4_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_stdev'])
                        d_t99pct_gpu_4_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_99_pct'])
                    elif gpu_no == 5:
                        d_tmin_gpu_5_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_min'])
                        d_tmax_gpu_5_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_max'])
                        d_tavg_gpu_5_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_avg'])
                        d_tstdev_gpu_5_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_stdev'])
                        d_t99pct_gpu_5_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_99_pct'])
                    elif gpu_no == 6:
                        d_tmin_gpu_6_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_min'])
                        d_tmax_gpu_6_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_max'])
                        d_tavg_gpu_6_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_avg'])
                        d_tstdev_gpu_6_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_stdev'])
                        d_t99pct_gpu_6_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_99_pct'])
                    elif gpu_no == 7:
                        d_tmin_gpu_7_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_min'])
                        d_tmax_gpu_7_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_max'])
                        d_tavg_gpu_7_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_avg'])
                        d_tstdev_gpu_7_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_stdev'])
                        d_t99pct_gpu_7_list.append(res_dict[app_name][msg_size][node][gpu_no]['t_99_pct'])
                    tot_tmin = tot_tmin + float(res_dict[app_name][msg_size][node][gpu_no]['t_min'])
                    tot_tmax = tot_tmax + float(res_dict[app_name][msg_size][node][gpu_no]['t_max'])
                    tot_tavg = tot_tavg + float(res_dict[app_name][msg_size][node][gpu_no]['t_avg'])
                    tot_tstdev = tot_tstdev + float(res_dict[app_name][msg_size][node][gpu_no]['t_stdev'])
                    tot_t99pct = tot_t99pct + float(res_dict[app_name][msg_size][node][gpu_no]['t_99_pct'])
                d_avg_tmin_list.append(tot_tmin / 8)
                d_avg_tmax_list.append(tot_tmax / 8)
                d_avg_tavg_list.append(tot_tavg / 8)
                d_avg_tstdev_list.append(tot_tstdev / 8)
                d_avg_t99pct_list.append(tot_t99pct / 8)

        split_tmin_list = split_list_into_n_chunks(d_avg_tmin_list, len(node_list))
        split_tmax_list = split_list_into_n_chunks(d_avg_tmax_list, len(node_list))
        split_tavg_list = split_list_into_n_chunks(d_avg_tavg_list, len(node_list))
        split_tstdev_list = split_list_into_n_chunks(d_avg_tstdev_list, len(node_list))
        split_t99pct_list = split_list_into_n_chunks(d_avg_t99pct_list, len(node_list))

        split_tmin_gpu0_list = split_list_into_n_chunks(d_tmin_gpu_0_list, len(node_list))
        split_tmin_gpu1_list = split_list_into_n_chunks(d_tmin_gpu_1_list, len(node_list))
        split_tmin_gpu2_list = split_list_into_n_chunks(d_tmin_gpu_2_list, len(node_list))
        split_tmin_gpu3_list = split_list_into_n_chunks(d_tmin_gpu_3_list, len(node_list))
        split_tmin_gpu4_list = split_list_into_n_chunks(d_tmin_gpu_4_list, len(node_list))
        split_tmin_gpu5_list = split_list_into_n_chunks(d_tmin_gpu_5_list, len(node_list))
        split_tmin_gpu6_list = split_list_into_n_chunks(d_tmin_gpu_6_list, len(node_list))
        split_tmin_gpu7_list = split_list_into_n_chunks(d_tmin_gpu_7_list, len(node_list))

        split_tmax_gpu0_list = split_list_into_n_chunks(d_tmax_gpu_0_list, len(node_list))
        split_tmax_gpu1_list = split_list_into_n_chunks(d_tmax_gpu_1_list, len(node_list))
        split_tmax_gpu2_list = split_list_into_n_chunks(d_tmax_gpu_2_list, len(node_list))
        split_tmax_gpu3_list = split_list_into_n_chunks(d_tmax_gpu_3_list, len(node_list))
        split_tmax_gpu4_list = split_list_into_n_chunks(d_tmax_gpu_4_list, len(node_list))
        split_tmax_gpu5_list = split_list_into_n_chunks(d_tmax_gpu_5_list, len(node_list))
        split_tmax_gpu6_list = split_list_into_n_chunks(d_tmax_gpu_6_list, len(node_list))
        split_tmax_gpu7_list = split_list_into_n_chunks(d_tmax_gpu_7_list, len(node_list))

        split_tavg_gpu0_list = split_list_into_n_chunks(d_tavg_gpu_0_list, len(node_list))
        split_tavg_gpu1_list = split_list_into_n_chunks(d_tavg_gpu_1_list, len(node_list))
        split_tavg_gpu2_list = split_list_into_n_chunks(d_tavg_gpu_2_list, len(node_list))
        split_tavg_gpu3_list = split_list_into_n_chunks(d_tavg_gpu_3_list, len(node_list))
        split_tavg_gpu4_list = split_list_into_n_chunks(d_tavg_gpu_4_list, len(node_list))
        split_tavg_gpu5_list = split_list_into_n_chunks(d_tavg_gpu_5_list, len(node_list))
        split_tavg_gpu6_list = split_list_into_n_chunks(d_tavg_gpu_6_list, len(node_list))
        split_tavg_gpu7_list = split_list_into_n_chunks(d_tavg_gpu_7_list, len(node_list))

        split_tstdev_gpu0_list = split_list_into_n_chunks(d_tstdev_gpu_0_list, len(node_list))
        split_tstdev_gpu1_list = split_list_into_n_chunks(d_tstdev_gpu_1_list, len(node_list))
        split_tstdev_gpu2_list = split_list_into_n_chunks(d_tstdev_gpu_2_list, len(node_list))
        split_tstdev_gpu3_list = split_list_into_n_chunks(d_tstdev_gpu_3_list, len(node_list))
        split_tstdev_gpu4_list = split_list_into_n_chunks(d_tstdev_gpu_4_list, len(node_list))
        split_tstdev_gpu5_list = split_list_into_n_chunks(d_tstdev_gpu_5_list, len(node_list))
        split_tstdev_gpu6_list = split_list_into_n_chunks(d_tstdev_gpu_6_list, len(node_list))
        split_tstdev_gpu7_list = split_list_into_n_chunks(d_tstdev_gpu_7_list, len(node_list))

        split_t99pct_gpu0_list = split_list_into_n_chunks(d_t99pct_gpu_0_list, len(node_list))
        split_t99pct_gpu1_list = split_list_into_n_chunks(d_t99pct_gpu_1_list, len(node_list))
        split_t99pct_gpu2_list = split_list_into_n_chunks(d_t99pct_gpu_2_list, len(node_list))
        split_t99pct_gpu3_list = split_list_into_n_chunks(d_t99pct_gpu_3_list, len(node_list))
        split_t99pct_gpu4_list = split_list_into_n_chunks(d_t99pct_gpu_4_list, len(node_list))
        split_t99pct_gpu5_list = split_list_into_n_chunks(d_t99pct_gpu_5_list, len(node_list))
        split_t99pct_gpu6_list = split_list_into_n_chunks(d_t99pct_gpu_6_list, len(node_list))
        split_t99pct_gpu7_list = split_list_into_n_chunks(d_t99pct_gpu_7_list, len(node_list))

        avg_tmin_data = average_of_lists(split_tmin_list)
        avg_tmax_data = average_of_lists(split_tmax_list)
        avg_tavg_data = average_of_lists(split_tavg_list)
        avg_tstdev_data = average_of_lists(split_tstdev_list)
        avg_t99pct_data = average_of_lists(split_t99pct_list)

        avg_tmin_gpu0_data = average_of_lists(split_tmin_gpu0_list)
        avg_tmin_gpu1_data = average_of_lists(split_tmin_gpu1_list)
        avg_tmin_gpu2_data = average_of_lists(split_tmin_gpu2_list)
        avg_tmin_gpu3_data = average_of_lists(split_tmin_gpu3_list)
        avg_tmin_gpu4_data = average_of_lists(split_tmin_gpu4_list)
        avg_tmin_gpu5_data = average_of_lists(split_tmin_gpu5_list)
        avg_tmin_gpu6_data = average_of_lists(split_tmin_gpu6_list)
        avg_tmin_gpu7_data = average_of_lists(split_tmin_gpu7_list)

        avg_tmax_gpu0_data = average_of_lists(split_tmax_gpu0_list)
        avg_tmax_gpu1_data = average_of_lists(split_tmax_gpu1_list)
        avg_tmax_gpu2_data = average_of_lists(split_tmax_gpu2_list)
        avg_tmax_gpu3_data = average_of_lists(split_tmax_gpu3_list)
        avg_tmax_gpu4_data = average_of_lists(split_tmax_gpu4_list)
        avg_tmax_gpu5_data = average_of_lists(split_tmax_gpu5_list)
        avg_tmax_gpu6_data = average_of_lists(split_tmax_gpu6_list)
        avg_tmax_gpu7_data = average_of_lists(split_tmax_gpu7_list)

        avg_tavg_gpu0_data = average_of_lists(split_tavg_gpu0_list)
        avg_tavg_gpu1_data = average_of_lists(split_tavg_gpu1_list)
        avg_tavg_gpu2_data = average_of_lists(split_tavg_gpu2_list)
        avg_tavg_gpu3_data = average_of_lists(split_tavg_gpu3_list)
        avg_tavg_gpu4_data = average_of_lists(split_tavg_gpu4_list)
        avg_tavg_gpu5_data = average_of_lists(split_tavg_gpu5_list)
        avg_tavg_gpu6_data = average_of_lists(split_tavg_gpu6_list)
        avg_tavg_gpu7_data = average_of_lists(split_tavg_gpu7_list)

        avg_tstdev_gpu0_data = average_of_lists(split_tstdev_gpu0_list)
        avg_tstdev_gpu1_data = average_of_lists(split_tstdev_gpu1_list)
        avg_tstdev_gpu2_data = average_of_lists(split_tstdev_gpu2_list)
        avg_tstdev_gpu3_data = average_of_lists(split_tstdev_gpu3_list)
        avg_tstdev_gpu4_data = average_of_lists(split_tstdev_gpu4_list)
        avg_tstdev_gpu5_data = average_of_lists(split_tstdev_gpu5_list)
        avg_tstdev_gpu6_data = average_of_lists(split_tstdev_gpu6_list)
        avg_tstdev_gpu7_data = average_of_lists(split_tstdev_gpu7_list)

        avg_t99pct_gpu0_data = average_of_lists(split_t99pct_gpu0_list)
        avg_t99pct_gpu1_data = average_of_lists(split_t99pct_gpu1_list)
        avg_t99pct_gpu2_data = average_of_lists(split_t99pct_gpu2_list)
        avg_t99pct_gpu3_data = average_of_lists(split_t99pct_gpu3_list)
        avg_t99pct_gpu4_data = average_of_lists(split_t99pct_gpu4_list)
        avg_t99pct_gpu5_data = average_of_lists(split_t99pct_gpu5_list)
        avg_t99pct_gpu6_data = average_of_lists(split_t99pct_gpu6_list)
        avg_t99pct_gpu7_data = average_of_lists(split_t99pct_gpu7_list)

        # data = [ node_list, d_msg_size_list, d_bw_gpu_0_list, d_bw_gpu_1_list, d_bw_gpu_2_list, d_bw_gpu_3_list, \
        #         d_bw_gpu_4_list, d_bw_gpu_5_list, d_bw_gpu_6_list ]
        # Merge cols for Node IP
        worksheet.merge_range("A2:A3", "Node IP", bold)
        worksheet.merge_range("B2:B3", "Msg Size", bold)
        # Merge 2 cols for every GPU and Total
        worksheet.merge_range("C2:G2", "Node Avg ", bold)
        worksheet.merge_range("H2:L2", "GPU0", bold)
        worksheet.merge_range("M2:Q2", "GPU1", bold)
        worksheet.merge_range("R2:V2", "GPU2", bold)
        worksheet.merge_range("W2:AA2", "GPU3", bold)
        worksheet.merge_range("AB2:AF2", "GPU4", bold)
        worksheet.merge_range("AG2:AK2", "GPU5", bold)
        worksheet.merge_range("AL2:AP2", "GPU6", bold)
        worksheet.merge_range("AQ2:AU2", "GPU7", bold)

        headings = []
        for i in range(0, 9):
            headings.append("Tmin")
            headings.append("Tmax")
            headings.append("TAvg")
            headings.append("TStdev")
            headings.append("T99pct")
        # Write headings for BW, PPS for total and every GPU
        worksheet.write_row("C3", headings, bold)

        cur_row = 4
        msg_list_len = len(msg_size_list)
        for node_ip in node_list:
            worksheet.merge_range(f"A{cur_row}:A{cur_row + msg_list_len - 1}", node_ip, node_merge_format)
            cur_row = cur_row + msg_list_len

        # For the cluster Avg
        worksheet.merge_range(f"A{cur_row}:A{cur_row + msg_list_len - 1}", "Cluster Avg")

        worksheet.write_column("B4", round_vals(d_msg_size_list))
        worksheet.write_column("C4", round_vals(d_avg_tmin_list))
        worksheet.write_column("D4", round_vals(d_avg_tmax_list))
        worksheet.write_column("E4", round_vals(d_avg_tavg_list))
        worksheet.write_column("F4", round_vals(d_avg_tstdev_list))
        worksheet.write_column("G4", round_vals(d_avg_t99pct_list))

        worksheet.write_column("H4", round_vals(d_tmin_gpu_0_list))
        worksheet.write_column("I4", round_vals(d_tmax_gpu_0_list))
        worksheet.write_column("J4", round_vals(d_tavg_gpu_0_list))
        worksheet.write_column("K4", round_vals(d_tstdev_gpu_0_list))
        worksheet.write_column("L4", round_vals(d_t99pct_gpu_0_list))

        worksheet.write_column("M4", round_vals(d_tmin_gpu_1_list))
        worksheet.write_column("N4", round_vals(d_tmax_gpu_1_list))
        worksheet.write_column("O4", round_vals(d_tavg_gpu_1_list))
        worksheet.write_column("P4", round_vals(d_tstdev_gpu_1_list))
        worksheet.write_column("Q4", round_vals(d_t99pct_gpu_1_list))

        worksheet.write_column("R4", round_vals(d_tmin_gpu_2_list))
        worksheet.write_column("S4", round_vals(d_tmax_gpu_2_list))
        worksheet.write_column("T4", round_vals(d_tavg_gpu_2_list))
        worksheet.write_column("U4", round_vals(d_tstdev_gpu_2_list))
        worksheet.write_column("V4", round_vals(d_t99pct_gpu_2_list))

        worksheet.write_column("W4", round_vals(d_tmin_gpu_3_list))
        worksheet.write_column("X4", round_vals(d_tmax_gpu_3_list))
        worksheet.write_column("Y4", round_vals(d_tavg_gpu_3_list))
        worksheet.write_column("Z4", round_vals(d_tstdev_gpu_3_list))
        worksheet.write_column("AA4", round_vals(d_t99pct_gpu_3_list))

        worksheet.write_column("AB4", round_vals(d_tmin_gpu_4_list))
        worksheet.write_column("AC4", round_vals(d_tmax_gpu_4_list))
        worksheet.write_column("AD4", round_vals(d_tavg_gpu_4_list))
        worksheet.write_column("AE4", round_vals(d_tstdev_gpu_4_list))
        worksheet.write_column("AF4", round_vals(d_t99pct_gpu_4_list))

        worksheet.write_column("AG4", round_vals(d_tmin_gpu_5_list))
        worksheet.write_column("AH4", round_vals(d_tmax_gpu_5_list))
        worksheet.write_column("AI4", round_vals(d_tavg_gpu_5_list))
        worksheet.write_column("AJ4", round_vals(d_tstdev_gpu_5_list))
        worksheet.write_column("AK4", round_vals(d_t99pct_gpu_5_list))

        worksheet.write_column("AL4", round_vals(d_tmin_gpu_6_list))
        worksheet.write_column("AM4", round_vals(d_tmax_gpu_6_list))
        worksheet.write_column("AN4", round_vals(d_tavg_gpu_6_list))
        worksheet.write_column("AO4", round_vals(d_tstdev_gpu_6_list))
        worksheet.write_column("AP4", round_vals(d_t99pct_gpu_6_list))

        worksheet.write_column("AQ4", round_vals(d_tmin_gpu_7_list))
        worksheet.write_column("AR4", round_vals(d_tmax_gpu_7_list))
        worksheet.write_column("AS4", round_vals(d_tavg_gpu_7_list))
        worksheet.write_column("AT4", round_vals(d_tstdev_gpu_7_list))
        worksheet.write_column("AU4", round_vals(d_t99pct_gpu_7_list))

        x_chart_index = len(d_msg_size_list) + 4

        worksheet.write_column(f"B{x_chart_index}", msg_size_list)
        worksheet.write_column(f"C{x_chart_index}", round_vals(avg_tmin_data))
        worksheet.write_column(f"D{x_chart_index}", round_vals(avg_tmax_data))
        worksheet.write_column(f"E{x_chart_index}", round_vals(avg_tavg_data))
        worksheet.write_column(f"F{x_chart_index}", round_vals(avg_tstdev_data))
        worksheet.write_column(f"G{x_chart_index}", round_vals(avg_t99pct_data))

        worksheet.write_column(f"H{x_chart_index}", round_vals(avg_tmin_gpu0_data))
        worksheet.write_column(f"I{x_chart_index}", round_vals(avg_tmax_gpu0_data))
        worksheet.write_column(f"J{x_chart_index}", round_vals(avg_tavg_gpu0_data))
        worksheet.write_column(f"K{x_chart_index}", round_vals(avg_tstdev_gpu0_data))
        worksheet.write_column(f"L{x_chart_index}", round_vals(avg_t99pct_gpu0_data))
        worksheet.write_column(f"M{x_chart_index}", round_vals(avg_tmin_gpu1_data))
        worksheet.write_column(f"N{x_chart_index}", round_vals(avg_tmax_gpu1_data))
        worksheet.write_column(f"O{x_chart_index}", round_vals(avg_tavg_gpu1_data))
        worksheet.write_column(f"P{x_chart_index}", round_vals(avg_tstdev_gpu1_data))
        worksheet.write_column(f"Q{x_chart_index}", round_vals(avg_t99pct_gpu1_data))
        worksheet.write_column(f"R{x_chart_index}", round_vals(avg_tmin_gpu2_data))
        worksheet.write_column(f"S{x_chart_index}", round_vals(avg_tmax_gpu2_data))
        worksheet.write_column(f"T{x_chart_index}", round_vals(avg_tavg_gpu2_data))
        worksheet.write_column(f"U{x_chart_index}", round_vals(avg_tstdev_gpu2_data))
        worksheet.write_column(f"V{x_chart_index}", round_vals(avg_t99pct_gpu2_data))
        worksheet.write_column(f"W{x_chart_index}", round_vals(avg_tmin_gpu3_data))
        worksheet.write_column(f"X{x_chart_index}", round_vals(avg_tmax_gpu3_data))
        worksheet.write_column(f"Y{x_chart_index}", round_vals(avg_tavg_gpu3_data))
        worksheet.write_column(f"Z{x_chart_index}", round_vals(avg_tstdev_gpu3_data))
        worksheet.write_column(f"AA{x_chart_index}", round_vals(avg_t99pct_gpu3_data))
        worksheet.write_column(f"AB{x_chart_index}", round_vals(avg_tmin_gpu4_data))
        worksheet.write_column(f"AC{x_chart_index}", round_vals(avg_tmax_gpu4_data))
        worksheet.write_column(f"AD{x_chart_index}", round_vals(avg_tavg_gpu4_data))
        worksheet.write_column(f"AE{x_chart_index}", round_vals(avg_tstdev_gpu4_data))
        worksheet.write_column(f"AF{x_chart_index}", round_vals(avg_t99pct_gpu4_data))
        worksheet.write_column(f"AG{x_chart_index}", round_vals(avg_tmin_gpu5_data))
        worksheet.write_column(f"AH{x_chart_index}", round_vals(avg_tmax_gpu5_data))
        worksheet.write_column(f"AI{x_chart_index}", round_vals(avg_tavg_gpu5_data))
        worksheet.write_column(f"AJ{x_chart_index}", round_vals(avg_tstdev_gpu5_data))
        worksheet.write_column(f"AK{x_chart_index}", round_vals(avg_t99pct_gpu5_data))
        worksheet.write_column(f"AL{x_chart_index}", round_vals(avg_tmin_gpu6_data))
        worksheet.write_column(f"AM{x_chart_index}", round_vals(avg_tmax_gpu6_data))
        worksheet.write_column(f"AN{x_chart_index}", round_vals(avg_tavg_gpu6_data))
        worksheet.write_column(f"AO{x_chart_index}", round_vals(avg_tstdev_gpu6_data))
        worksheet.write_column(f"AP{x_chart_index}", round_vals(avg_t99pct_gpu6_data))
        worksheet.write_column(f"AQ{x_chart_index}", round_vals(avg_tmin_gpu7_data))
        worksheet.write_column(f"AR{x_chart_index}", round_vals(avg_tmax_gpu7_data))
        worksheet.write_column(f"AS{x_chart_index}", round_vals(avg_tavg_gpu7_data))
        worksheet.write_column(f"AT{x_chart_index}", round_vals(avg_tstdev_gpu7_data))
        worksheet.write_column(f"AU{x_chart_index}", round_vals(avg_t99pct_gpu7_data))

        x_chart_index = len(d_msg_size_list) + len(msg_size_list) + 7
        y_chart_index = 2
        chart = workbook.add_chart({"type": "column"})

        row_end = 4 + len(d_msg_size_list)

        chart.add_series(
            {
                "name": "={}!$B$2".format(sheet_name),
                "categories": "={}!$B${}:$B${}".format(sheet_name, row_end, row_end + len(msg_size_list)),
                "values": "={}!$E${}:$E${}".format(sheet_name, row_end, row_end + len(msg_size_list)),
                "data_labels": {
                    "value": True,
                    "font": {
                        "rotation": -45,
                    },
                },
            }
        )

        chart.set_size({'width': 1200, 'height': 720})
        chart.set_title({"name": "Latency in us"})
        chart.set_x_axis({"name": "Msg Size"})
        chart.set_y_axis({"name": "Latency in us"})
        chart.set_style(11)
        worksheet.insert_chart(x_chart_index, y_chart_index, chart)

    workbook.close()
    log.info('Latency performance chart saved: %s', excel_file)
