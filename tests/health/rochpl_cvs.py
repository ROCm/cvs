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

sys.path.insert( 0, './lib' )
from parallel_ssh_lib import *
from utils_lib import *

log = logging.getLogger()


# Importing additional cmd line args to script ..
@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    return pytestconfig.getoption("config_file")


# Importing the cluster and cofig files to script to access node, switch, test config params
@pytest.fixture(scope="module")
def  cluster_dict(cluster_file):
     with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)
     log.info(cluster_dict)
     return cluster_dict

@pytest.fixture(scope="module")
def  config_dict(config_file):
     with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
     config_dict = config_dict_t['transferbench']
     log.info(config_dict)
     return config_dict



def parse_tb_a2a_bw( out_dict, exp_dict ):

    for node in out_dict.keys():
        rtotal_list = re.findall( 'RTotal\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+', out_dict[node] )
        for gpu_bw in list(rtotal_list[0]):
            if float(gpu_bw) < float(exp_dict['gpu_to_gpu_a2a_rtotal']):
                fail_test(f"Actual GPU a2a bandwidth {gpu_bw} in transferbench a2a test lower than expected {exp_dict['gpu_to_gpu_a2a_rtotal']} on node {node}")




# Create connection to DUT, MTPs, Switches and export for later use ..
@pytest.fixture(scope="module")
def  phdl(cluster_dict):
     nhdl_dict = {}
     print(cluster_dict)
     node_list = list(cluster_dict['node_dict'].keys())
     phdl = Pssh( log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'] )
     return phdl


###### REDO EVERYTHING

def test_rochpl_a2a(phdl, config_dict, ):
    log.info('Testcase Run rochpl')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/TransferBench a2a', timeout=(60*5))
    print_test_output( log, out_dict )
    scan_test_results( out_dict )
    parse_tb_a2a_bw( out_dict, config_dict['results'] )
    scan_test_results( out_dict )


def test_transfer_bench_p2p(phdl, config_dict, ):
    log.info('Testcase Run Transferbench p2p')
    path = config_dict['path']
    model_dict = phdl.exec(f'sudo {path}/rocm-smi --showproduct')
    out_dict = phdl.exec(f'sudo {path}/TransferBench p2p', timeout=(60*5))
    print_test_output( log, out_dict )
    parse_tb_p2p_bw( out_dict, config_dict['results'] )
    scan_test_results( out_dict )


def test_transfer_bench_healthcheck(phdl, config_dict ):
    log.info('Testcase Run TransferBench healthcheck')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/TransferBench healthcheck', timeout=(60*2))
    print_test_output( log, out_dict )
    scan_test_results( out_dict )

def test_transfer_bench_a2asweep(phdl, config_dict ):
    log.info('Testcase Run TransferBench a2asweep')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/TransferBench a2asweep', timeout=(60*10))
    print_test_output( log, out_dict )
    scan_test_results( out_dict )


def test_transfer_bench_scaling(phdl, config_dict ):
    log.info('Testcase Run TransferBench scaling')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/TransferBench scaling', timeout=(60*10))
    print_test_output( log, out_dict )
    parse_tb_scaling_bw( out_dict, config_dict['results'] )
    scan_test_results( out_dict )

def test_transfer_bench_schmoo(phdl, config_dict ):
    log.info('Testcase Run TransferBench schmoo')
    path = config_dict['path']
    model_dict = phdl.exec(f'sudo {path}/rocm-smi --showproduct')
    out_dict = phdl.exec(f'sudo {path}/TransferBench schmoo', timeout=(60*5))
    print_test_output( log, out_dict )
    scan_test_results( out_dict )
    parse_tb_schmoo_bw( out_dict, config_dict['results'] )


#def test_transfer_bench_sweep(phdl, config_dict, ):
#    log.info('Testcase Run TransferBench sweep')
#    path = config_dict['transferbench']['path']
#    out_dict = phdl.exec(f'sudo {path}/TransferBench sweep', timeout=(60*60*3))
#    print_test_output( out_dict )
