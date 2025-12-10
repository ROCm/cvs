'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest

import re
import sys
import os
import sys
import time
import json
import logging


from cvs.lib.parallel_ssh_lib import *
from cvs.lib.utils_lib import *

from cvs.lib import globals

log = globals.log


# Importing additional cmd line args to script ..
@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    return pytestconfig.getoption("config_file")


# Importing the cluster and cofig files to script to access node, switch, test config params
@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)

    # Resolve path placeholders like {user-id} in cluster config
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)

    log.info(cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict):
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['transferbench']

    # Resolve path placeholders like {user-id}, {home-mount-dir}, etc.
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)

    log.info(config_dict)
    return config_dict


def parse_tb_a2a_bw(out_dict, exp_dict):
    for node in out_dict.keys():
        print(exp_dict)
        rtotal_list = re.findall(
            'RTotal\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+',
            out_dict[node],
        )
        for gpu_bw in list(rtotal_list[0]):
            if float(gpu_bw) < float(exp_dict['gpu_to_gpu_a2a_rtotal']):
                fail_test(
                    f"Actual GPU a2a bandwidth {gpu_bw} in transferbench a2a test lower than expected {exp_dict['gpu_to_gpu_a2a_rtotal']} on node {node}"
                )


def parse_tb_p2p_bw(out_dict, exp_dict):
    for node in out_dict.keys():
        match = re.search(
            'Averages\s+\(During\s+UniDir\):\s+[0-9\.]+\s+[0-9\.]+\s+[0-9\.]+\s+([0-9\.]+)', out_dict[node], re.I
        )
        avg_unidir = float(match.group(1))
        if float(avg_unidir) < float(exp_dict['avg_gpu_to_gpu_p2p_unidir_bw']):
            fail_test(
                f"Actual Avg UniDir GPU to GPU bandwidth {avg_unidir} is less than expected {exp_dict['avg_gpu_to_gpu_p2p_unidir_bw']} on node {node}"
            )
        match = re.search(
            'Averages\s+\(During\s+BiDir\):\s+[0-9\.]+\s+[0-9\.]+\s+[0-9\.]+\s+([0-9\.]+)', out_dict[node], re.I
        )
        avg_bidir = float(match.group(1))
        if float(avg_bidir) < float(exp_dict['avg_gpu_to_gpu_p2p_bidir_bw']):
            fail_test(
                f"Actual Avg BiDir GPU to GPU bandwidth {avg_bidir} is less than expected {exp_dict['avg_gpu_to_gpu_p2p_bidir_bw']} on node {node}"
            )


def parse_tb_scaling_bw(out_dict, exp_dict):
    for node in out_dict.keys():
        print(f"^^^^^ {out_dict[node]} ^^^^^^")
        match = re.search('Best\s+[0-9\.]+\(\s[0-9]+\)\s+[0-9\.]+\(\s[0-9]+\)\s+([0-9\.]+)', out_dict[node])
        gpu0_bw = float(match.group(1))
        if float(gpu0_bw) < float(exp_dict['best_gpu0_bw']):
            fail_test(
                f"Actual Best BW from GPU0 in scaling test is lower than expected {exp_dict['best_gpu0_bw']} on node {node}"
            )


def parse_tb_schmoo_bw(out_dict, exp_dict):
    for node in out_dict.keys():
        match = re.search(
            '\s+32\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)', out_dict[node]
        )
        local_read = float(match.group(1))
        local_write = float(match.group(2))
        local_copy = float(match.group(3))
        remote_read = float(match.group(4))
        remote_write = float(match.group(5))
        remote_copy = float(match.group(6))
        if float(local_read) < float(exp_dict['32_cu_local_read']):
            fail_test(
                f"Actual local read for 32 CU {local_read} is less than expected {exp_dict['32_cu_local_read']} for node {node}"
            )
        if float(local_write) < float(exp_dict['32_cu_local_write']):
            fail_test(
                f"Actual local write for 32 CU {local_write} is less than expected {exp_dict['32_cu_local_write']} for node {node}"
            )
        if float(local_copy) < float(exp_dict['32_cu_local_copy']):
            fail_test(
                f"Actual local copy for 32 CU {local_copy} is less than expected {exp_dict['32_cu_local_copy']} for node {node}"
            )
        if float(remote_read) < float(exp_dict['32_cu_rem_read']):
            fail_test(
                f"Actual remote read for 32 CU {remote_read} is less than expected {exp_dict['32_cu_rem_read']} for node {node}"
            )
        if float(remote_write) < float(exp_dict['32_cu_rem_write']):
            fail_test(
                f"Actual remote write for 32 CU {remote_write} is less than expected {exp_dict['32_cu_rem_write']} for node {node}"
            )
        if float(remote_copy) < float(exp_dict['32_cu_rem_copy']):
            fail_test(
                f"Actual remote copy for 32 CU {remote_copy} is less than expected {exp_dict['32_cu_rem_copy']} for node {node}"
            )


def parse_tb_example_test_results(out_dict, exp_dict):
    for node in out_dict.keys():
        # Test 1 results parse
        match = re.search('(Test\s1:\n.*\n.*\n.*\n)', out_dict[node])
        test1_out = match.group(1)
        match = re.search('Transfer\s+[0-9]+\s+\|\s+([0-9\.]+)\sGB\/s', test1_out, re.I)
        test1_res = match.group(1)
        if float(test1_res) < float(exp_dict['test1']):
            fail_test(
                f"Transfer Bench example test1 failed actual value {test1_res} is less than expected {exp_dict['test1']}"
            )

        # Test 2 results parse
        match = re.search('(Test\s2:\n.*\n.*\n.*\n)', out_dict[node])
        test2_out = match.group(1)
        match = re.search('Transfer\s+[0-9]+\s+\|\s+([0-9\.]+)\sGB\/s', test2_out, re.I)
        test2_res = match.group(1)
        if float(test2_res) < float(exp_dict['test2']):
            fail_test(
                f"Transfer Bench example test2 failed actual value {test2_res} is less than expected {exp_dict['test2']}"
            )

        # Test 3 results parse
        match = re.search('(Test\s3:\n.*\n.*\n.*\n.*\n.*\n)', out_dict[node])
        test3_out = match.group(1)

        match = re.search('Transfer\s+[0-9]+\s+\|\s+([0-9\.]+)\sGB\/s[\s0-9\.\|a-z]+\s+G0 \-\>', test3_out, re.I)
        test3_res_0_1 = match.group(1)
        if float(test3_res_0_1) < float(exp_dict['test3_0_to_1']):
            fail_test(
                f"Transfer Bench example test3 failed actual value {test3_res_0_1} is less than expected {exp_dict['test3_0_to_1']}"
            )

        match = re.search('Transfer\s+[0-9]+\s+\|\s+([0-9\.]+)\sGB\/s[\s0-9\.\|a-z]+\s+G1 \-\>', test3_out, re.I)
        test3_res_1_0 = match.group(1)
        if float(test3_res_1_0) < float(exp_dict['test3_1_to_0']):
            fail_test(
                f"Transfer Bench example test3 failed actual value {test3_res_1_0} is less than expected {exp_dict['test3_1_to_0']}"
            )

        # Test 4 results parse
        match = re.search('(Test\s4:\n.*\n.*\n.*\n)', out_dict[node])
        test4_out = match.group(1)
        match = re.search('Transfer\s+[0-9]+\s+\|\s+([0-9\.]+)\sGB\/s', test4_out, re.I)
        test4_res = match.group(1)
        if float(test4_res) < float(exp_dict['test4']):
            fail_test(
                f"Transfer Bench example test4 failed actual value {test4_res} is less than expected {exp_dict['test4']}"
            )

        # Test 5 CPU only .. skip
        # Test 6 results parse
        match = re.search('(Test\s6:\n.*\n.*\n.*\n)', out_dict[node])
        test6_out = match.group(1)
        match = re.search('Transfer\s+[0-9]+\s+\|\s+([0-9\.]+)\sGB\/s', test6_out, re.I)
        test6_res = match.group(1)
        if float(test6_res) < float(exp_dict['test6']):
            fail_test(
                f"Transfer Bench example test6 failed actual value {test6_res} is less than expected {exp_dict['test6']}"
            )


@pytest.fixture(scope="module")
def phdl(cluster_dict):
    print(cluster_dict)
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
    return phdl


def test_transfer_bench_example_tests_1_6_t(phdl, config_dict):
    globals.error_list = []
    log.info('Testcase Run TransferBench example tests 1-6')
    path = config_dict['path']
    print(config_dict)
    example_path = config_dict['example_tests_path']
    out_dict = phdl.exec(f'sudo {path}/TransferBench {example_path}/example.cfg', timeout=(60 * 5))
    print_test_output(log, out_dict)
    scan_test_results(out_dict)
    parse_tb_example_test_results(out_dict, config_dict['results']['example_results'])
    update_test_result()


def test_transfer_bench_a2a(
    phdl,
    config_dict,
):
    globals.error_list = []
    log.info('Testcase Run Transferbench a2a')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/TransferBench a2a', timeout=(60 * 5))
    print_test_output(log, out_dict)
    scan_test_results(out_dict)
    parse_tb_a2a_bw(out_dict, config_dict['results'])
    scan_test_results(out_dict)
    update_test_result()


def test_transfer_bench_p2p(
    phdl,
    config_dict,
):
    globals.error_list = []
    log.info('Testcase Run Transferbench p2p')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/TransferBench p2p', timeout=(60 * 5))
    print_test_output(log, out_dict)
    parse_tb_p2p_bw(out_dict, config_dict['results'])
    scan_test_results(out_dict)
    update_test_result()


def test_transfer_bench_healthcheck(
    phdl,
    config_dict,
):
    globals.error_list = []
    log.info('Testcase Run TransferBench healthcheck')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/TransferBench healthcheck', timeout=(60 * 2))
    print_test_output(log, out_dict)
    scan_test_results(out_dict)
    update_test_result()


def test_transfer_bench_a2asweep(
    phdl,
    config_dict,
):
    globals.error_list = []
    log.info('Testcase Run TransferBench a2asweep')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/TransferBench a2asweep', timeout=(60 * 10))
    print_test_output(log, out_dict)
    scan_test_results(out_dict)
    update_test_result()


def test_transfer_bench_scaling(
    phdl,
    config_dict,
):
    globals.error_list = []
    log.info('Testcase Run TransferBench scaling')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/TransferBench scaling', timeout=(60 * 10))
    print_test_output(log, out_dict)
    parse_tb_scaling_bw(out_dict, config_dict['results'])
    scan_test_results(out_dict)
    update_test_result()


def test_transfer_bench_schmoo(
    phdl,
    config_dict,
):
    globals.error_list = []
    log.info('Testcase Run TransferBench schmoo')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/TransferBench schmoo', timeout=(60 * 5))
    print_test_output(log, out_dict)
    scan_test_results(out_dict)
    parse_tb_schmoo_bw(out_dict, config_dict['results'])
    update_test_result()
