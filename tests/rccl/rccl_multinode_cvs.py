import pytest

import re
import sys
import os
import sys
import time
import json
import logging

sys.path.insert( 0, './lib' )
import rccl_lib
from parallel_ssh_lib import *
from utils_lib import *
from verify_lib import *

import globals

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
def  cluster_dict(cluster_file):
     with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)
     log.info(cluster_dict)
     return cluster_dict

@pytest.fixture(scope="module")
def  config_dict(config_file):
     with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
     config_dict = config_dict_t['rccl']
     log.info(config_dict)
     return config_dict




@pytest.fixture(scope="module")
def  phdl(cluster_dict):
     nhdl_dict = {}
     print(cluster_dict)
     node_list = list(cluster_dict['node_dict'].keys())
     phdl = Pssh( log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'] )
     return phdl


@pytest.fixture(scope="module")
def  shdl(cluster_dict):
     nhdl_dict = {}
     node_list = list(cluster_dict['node_dict'].keys())
     head_node = node_list[0]
     shdl = Pssh( log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'] )
     return shdl




@pytest.fixture(scope="module")
def  vpc_node_list(cluster_dict):
     vpc_node_list = []
     for node in list(cluster_dict['node_dict'].keys()):
         vpc_node_list.append(cluster_dict['node_dict'][node]['vpc_ip']) 
     return vpc_node_list



def pytest_generate_tests(metafunc):
    config_file = metafunc.config.getoption("config_file")
    if not config_file:
        return
    with open(config_file) as fp:
        config = json.load(fp)

    # Build test parameters
    if 'rccl_collective' in config['rccl']:
        rccl_collective_list = config['rccl']['rccl_collective']
    else:
        rccl_collective_list = [ "all_reduce_perf", "all_gather_perf", \
                                 "scatter_perf", "gather_perf", \
                                 "reduce_scatter_perf", "sendrecv_perf", \
                                 "alltoall_perf", "alltoallv_perf", \
                                 "reduce_scatter_perf", "broadcast_perf"]

    if 'rccl_algo' in config['rccl']:
        rccl_algo_list = config['rccl']['rccl_algo']
    else:
        rccl_algo_list = ["ring", "tree" ]

    if 'rccl_protocol' in config['rccl']:
        rccl_protocol_list = config['rccl']['rccl_protocol']
    else:
        rccl_protocol_list = ["simple", "LL128", "LL" ]

    if 'qp_scale' in config['rccl']:
        qp_scale_list = config['rccl']['qp_scale']
    else:
        qp_scale_list = [ "1", "2", ]


    # Invoke parametrize
    if "rccl_collective" in metafunc.fixturenames:
        metafunc.parametrize( "rccl_collective", rccl_collective_list )

    if "rccl_algo" in metafunc.fixturenames:
        metafunc.parametrize( "rccl_algo", rccl_algo_list )


    if "rccl_protocol" in metafunc.fixturenames:
        metafunc.parametrize( "rccl_protocol", rccl_protocol_list )

    if "qp_scale" in metafunc.fixturenames:
        metafunc.parametrize( "qp_scale", qp_scale_list )




# Start of test cases.

def test_collect_hostinfo( phdl ):
    globals.error_list = []
    phdl.exec('cat /opt/rocm/.info/version')
    phdl.exec('hipconfig')
    phdl.exec('rocm_agent_enumerator')
    update_test_result()



def test_collect_networkinfo( phdl ):
    globals.error_list = []
    phdl.exec('rdma link')
    phdl.exec('ibv_devinfo')
    update_test_result()



def test_rccl_perf(phdl, shdl, cluster_dict, config_dict, rccl_collective, rccl_algo, \
       rccl_protocol, qp_scale ):

    start_time = phdl.exec('date')
    globals.error_list = []
    node_list = list(cluster_dict['node_dict'].keys())
    vpc_node_list = []
    for node in list(cluster_dict['node_dict'].keys()):
        vpc_node_list.append(cluster_dict['node_dict'][node]['vpc_ip']) 


    #Get cluster snapshot ..
    if re.search( 'True', config_dict['cluster_snapshot_debug'], re.I ):
        cluster_dict_before = create_cluster_metrics_snapshot( phdl )


    if not re.search( 'None', config_dict['env_source_script'], re.I ):
        phdl.exec(f'bash {config_dict['env_source_script']}')

    result_dict = rccl_lib.rccl_cluster_test( phdl, shdl, \
       test_name               = rccl_collective, \
       cluster_node_list       = node_list, \
       vpc_node_list           = vpc_node_list, \
       user_name               = cluster_dict['username'], \
       ib_hca_list             = config_dict['ib_hca_list'], \
       net_dev_list            = config_dict['net_dev_list'], \
       oob_port                = config_dict['oob_port'], \
       no_of_global_ranks      = config_dict['no_of_global_ranks'], \
       rocm_path_var           = config_dict['rocm_path_var'], \
       mpi_dir                 = config_dict['mpi_dir'], \
       mpi_path_var            = config_dict['mpi_path_var'], \
       rccl_dir                = config_dict['rccl_dir'], \
       rccl_path_var           = config_dict['rccl_path_var'], \
       rccl_tests_dir          = config_dict['rccl_tests_dir'], \
       nccl_algo               = rccl_algo, \
       nccl_proto              = rccl_protocol, \
       gid_index               = config_dict['gid_index'], \
       qp_count                = qp_scale, \
       start_msg_size          = config_dict['start_msg_size'], \
       end_msg_size            = config_dict['end_msg_size'], \
       step_function           = config_dict['step_function'], \
       threads_per_gpu         = config_dict['threads_per_gpu'], \
       warmup_iterations       = config_dict['warmup_iterations'], \
       no_of_iterations        = config_dict['no_of_iterations'], \
       check_iteration_count   = config_dict['check_iteration_count'], \
       debug_level             = config_dict['debug_level'], \
       rccl_result_file        = config_dict['rccl_result_file'], \
       no_of_local_ranks       = config_dict['no_of_local_ranks'], \
       ib_rx_queue_len         = config_dict['ib_rx_queue_len'], \
       ucx_tls                 = config_dict['ucx_tls'], \
       hcoll_enable_mcast_all  = config_dict['hcoll_enable_mcast_all'], \
       nccl_cumem_enable       = config_dict['nccl_cumem_enable'], \
       nccl_ib_timeout         = config_dict['nccl_ib_timeout'], \
       nccl_ib_sl              = config_dict['nccl_ib_sl'], \
       nccl_ib_tc              = config_dict['nccl_ib_tc'], \
       nccl_ib_split_data_on_qps  = config_dict['nccl_ib_split_data_on_qps'], \
       nccl_pxn_disable        = config_dict['nccl_pxn_disable'], \
       nccl_net_plugin         = config_dict['nccl_net_plugin'], \
       user_key_file           = cluster_dict['priv_key_file'], \
       verify_bus_bw           = config_dict['verify_bus_bw'], \
       exp_results_dict        = config_dict['results']
    )


    # Scan dmesg between start and end times cluster wide ..
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )

    # Get new cluster snapshot and compare ..
    if re.search( 'True', config_dict['cluster_snapshot_debug'], re.I ):
        cluster_dict_after = create_cluster_metrics_snapshot( phdl )
        compare_cluster_metrics_snapshots( cluster_dict_before, cluster_dict_after )

    # Update test results based on any failures ..
    update_test_result()


