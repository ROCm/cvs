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
import itertools


sys.path.insert( 0, './lib' )
import ibperf_lib

from parallel_ssh_lib import *
from utils_lib import *
from verify_lib import *

import globals

log = globals.log

ib_bw_dict = {}
ib_lat_dict = {}

rccl_res_dict = {}


# Importing additional cmd line args to script ..
@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    """
    Return the path to the cluster configuration JSON file passed via pytest CLI.

    Expects:
      - pytest to be invoked with: --cluster_file <path>

    Args:
      pytestconfig: Built-in pytest config object used to access CLI options.

    Returns:
      str: Filesystem path to the cluster configuration file.
    """
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    """
    Return the path to the test configuration JSON file passed via pytest CLI.

    Expects:
      - pytest to be invoked with: --config_file <path>

    Args:
      pytestconfig: Built-in pytest config object used to access CLI options.

    Returns:
      str: Filesystem path to the test configuration file.
    """
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def  cluster_dict(cluster_file):
     """
    Load and expose full cluster configuration for the test module.

    Behavior:
      - Opens the JSON at cluster_file and parses it into a Python dict.
      - Logs the parsed dictionary for visibility and debugging.
      - Returns the entire cluster configuration (node list, credentials, etc.).

    Args:
      cluster_file (str): Path to the cluster configuration JSON.

    Returns:
      dict: Parsed cluster configuration. Expected keys include:
            - 'node_dict': Map of node name -> node metadata
            - 'username': SSH username
            - 'priv_key_file': Path to SSH private key
    """
     with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)
     log.info(cluster_dict)
     return cluster_dict




@pytest.fixture(scope="module")
def config_dict(config_file):
    """
    Load and return the RCCL-specific configuration dictionary for the test module.

    Args:
      config_file (str): Path to a JSON config file provided by another fixture.

    Notes:
      - Expects the JSON file to contain a top-level key "ibperf".
      - Uses module scope so the config is parsed once per test module.
     """
    with open(config_file) as json_file:
       config_dict_t = json.load(json_file)
    config_dict = config_dict_t['ibperf']
    log.info(config_dict)
    return config_dict




@pytest.fixture(scope="module")
def phdl(cluster_dict):
    """
    Build and return a parallel SSH handle (Pssh) for all cluster nodes.

    Args:
      cluster_dict (dict): Cluster metadata fixture containing:
        - node_dict: dict of node_name -> node_details
        - username: SSH username
        - priv_key_file: path to SSH private key

    Returns:
      Pssh: Handle configured for all nodes (for broadcast/parallel operations).

    Notes:
      - Prints the cluster_dict for quick debugging; consider replacing with log.debug.
      - Module-scoped so a single shared handle is used across all tests in the module.
      - nhdl_dict is currently unused; it can be removed unless used elsewhere.
      - Assumes Pssh(log, node_list, user=..., pkey=...) is available in scope.
    """
    nhdl_dict = {}
    print(cluster_dict)
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh( log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'] )
    return phdl


@pytest.fixture(scope="module")
def shdl(cluster_dict):
    """
    Build and return a parallel SSH handle (Pssh) for the head node only.

    Args:
      cluster_dict (dict): Cluster metadata fixture (see phdl docstring).

    Returns:
      Pssh: Handle configured for the first node (head node) in node_dict.

    Notes:
      - Useful when commands should be executed only from a designated head node.
      - Module scope ensures a single connection context for the duration of the module.
      - nhdl_dict is currently unused; it can be removed unless used elsewhere.
    """
    nhdl_dict = {}
    node_list = list(cluster_dict['node_dict'].keys())
    head_node = node_list[0]
    shdl = Pssh( log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'] )
    return shdl




@pytest.fixture(scope="module")
def vpc_node_list(cluster_dict):
    """
    Collect and return a list of VPC IPs for all nodes in the cluster.

    Args:
      cluster_dict (dict): Cluster metadata fixture containing node_dict with vpc_ip per node.

    Returns:
      list[str]: List of VPC IP addresses in the cluster, ordered by node_dict iteration.

    Notes:
      - Iteration order depends on the insertion order of node_dict.
      - Consider validating that each node entry contains a 'vpc_ip' key.
    """
    vpc_node_list = []
    for node in list(cluster_dict['node_dict'].keys()):
        vpc_node_list.append(cluster_dict['node_dict'][node]['vpc_ip']) 
    return vpc_node_list





def test_install_ib_perf(phdl, shdl, config_dict ):
    # We install on the first node using shdl handle
    # install standard rdma packages
    globals.error_list = []

    if re.search( 'true', config_dict['install_perf_package'], re.I ):
        shdl.exec( f'mkdir -p {config_dict["install_dir"]}')
        phdl.exec( 'sudo apt update -y', timeout=200 )
        phdl.exec( 'sudo apt install -y git build-essential autoconf automake libtool pkg-config', timeout=200 )
        phdl.exec( 'sudo apt install -y libibverbs-dev librdmacm-dev ibverbs-providers rdma-core', timeout=200 )
        phdl.exec( 'sudo apt install -y libibumad-dev' )
        phdl.exec( 'sudo apt install -y libpci-dev' )
        phdl.exec( 'sudo apt install -y numactl' )
        shdl.exec( f'cd {config_dict["install_dir"]}; git clone https://github.com/linux-rdma/perftest' )
        shdl.exec( f'cd {config_dict["install_dir"]}/perftest; ./autogen.sh', timeout=100 )
        shdl.exec( f'cd {config_dict["install_dir"]}/perftest; ./configure --prefix={config_dict['install_dir']}/perftest --with-rocm={config_dict["rocm_dir"]} --enable-rocm', timeout=200 )
        shdl.exec( f'cd {config_dict["install_dir"]}/perftest; make', timeout=100 )
        shdl.exec( f'cd {config_dict["install_dir"]}/perftest; make install', timeout=100 )

        # Verify if the installation went fine ..
        out_dict = phdl.exec( f'{config_dict["install_dir"]}/perftest/ib_write_bw -h | grep -i rocm --color=never' )
        for node in out_dict.keys():
            if not re.search( 'GPUDirect RDMA', out_dict[node], re.I ):
                fail_test('IB Perf package installation on node {node} failed, ib_write_bw not showing expected use_rocm output')
    update_test_result()



# Start of test cases.

@pytest.mark.parametrize( "bw_test", [ "ib_write_bw", ] )


def test_ib_bw_perf( phdl, bw_test, config_dict ):

    # Get IB_backend_nics for each node
    # Get the NIC to GPU mapping dict
    # Generate the command list for all nodes
    # Run the commands
    # Get the bandwidth numbers as a dict for each node and NIC/GPU for every msg size

    globals.error_list = []
    ib_bw_dict[bw_test] = {}

    gpu_nic_dict = linux_utils.get_gpu_nic_mapping_dict(phdl)
    gpu_numa_dict = linux_utils.get_gpu_numa_dict( phdl )

    print(gpu_nic_dict)
    bck_nic_dict_lshw = linux_utils.get_backend_nic_dict( phdl )
    rdma_nic_dict = linux_utils.get_active_rdma_nic_dict( phdl )

    bck_nic_dict = {}
    for node in rdma_nic_dict.keys():
        bck_nic_dict[node] = {}
        for rdma_dev in rdma_nic_dict[node].keys():
            print(bck_nic_dict_lshw[node])
            if rdma_nic_dict[node][rdma_dev]['eth_device'] in bck_nic_dict_lshw[node]:
                bck_nic_dict[node][rdma_dev] = rdma_nic_dict[node][rdma_dev]


    for msg_size in config_dict['msg_size_list']:
        ib_bw_dict[bw_test][msg_size] = {}
        for qp_count in config_dict['qp_count_list']:
            # Log a message to Dmesg to create a timestamp record
            start_time = phdl.exec('date +"%a %b %e %H:%M"')
            phdl.exec( f'Starting Test {bw_test} for {msg_size} and QP count {qp_count} | sudo tee /dev/kmsg' )
            ib_bw_dict[bw_test][msg_size][qp_count] = ibperf_lib.run_ib_perf_test( phdl, bw_test, gpu_numa_dict, \
                    gpu_nic_dict, bck_nic_dict, f'{config_dict["install_dir"]}/perftest/bin', \
                    msg_size, config_dict['gid_index'], qp_count, int(config_dict['port_no']), \
                    int(config_dict['duration']) )
            end_time = phdl.exec('date +"%a %b %e %H:%M"')
            verify_dmesg_for_errors( phdl, start_time, end_time, till_end_flag=True )
            if re.search( 'True', config_dict['verify_bw'], re.I ):
                ibperf_lib.verify_expected_bw( bw_test, msg_size, qp_count, ib_bw_dict[bw_test][msg_size][qp_count], \
                        config_dict['expected_results'])

 
    print(ib_bw_dict)
    update_test_result()

