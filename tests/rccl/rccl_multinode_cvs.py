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




# Create connection to DUT, MTPs, Switches and export for later use ..
@pytest.fixture(scope="module")
def  phdl(cluster_dict):
     nhdl_dict = {}
     print(cluster_dict)
     node_list = list(cluster_dict['node_dict'].keys())
     phdl = Pssh( log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'] )
     return phdl


@pytest.fixture(scope="module")
def  vpc_node_list(cluster_dict):
     vpc_node_list = []
     for node in list(cluster_dict['node_dict'].keys()):
         vpc_node_list.append(cluster_dict['node_dict'][node]['vpc_ip']) 
     return vpc_node_list






# Parametrize all collectives
@pytest.mark.parametrize( "rccl_collective", ["all_reduce_perf", "all_gather_perf", \
                                            "scatter_perf", "gather_perf", \
                                            "reduce_scatter_perf", "sendrecv_perf", \
                                            "alltoall_perf", "alltoallv_perf", \
                                            "broadcast_perf"])

#@pytest.mark.parametrize( "rccl_collective", ["all_reduce_perf"] )

# Parametrize the algorithms ..
@pytest.mark.parametrize( "rccl_algo", ["ring", "tree"] )

# Parametrize the protocols ..
@pytest.mark.parametrize( "rccl_protocol", ["simple", "LL128"] )

# Parametrize the QP scale
@pytest.mark.parametrize( "qp_scale", ["1", "8", "16", "32", "64", "128"] )
#@pytest.mark.parametrize( "qp_scale", ["1"]  )


def test_rccl_perf(phdl, cluster_dict, config_dict, rccl_collective, rccl_algo, \
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


    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name=rccl_collective, \
       cluster_node_list       = node_list, \
       vpc_node_list           = vpc_node_list, \
       user_name               = cluster_dict['username'], \
       ib_hca_list             = config_dict['ib_hca_list'], \
       net_dev_list            = config_dict['net_dev_list'], \
       oob_port                = config_dict['oob_port'], \
       no_of_global_ranks      = config_dict['no_of_global_ranks'], \
       rocm_path               = config_dict['rocm_path'], \
       ucx_path                = config_dict['ucx_path'], \
       mpi_path                = config_dict['mpi_path'], \
       rccl_path               = config_dict['rccl_path'], \
       rccl_tests_path         = config_dict['rccl_tests_path'], \
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
       nccl_ib_timeout         = config_dict['nccl_ib_timeout'], \
       debug_level             = config_dict['debug_level'], \
       rccl_result_file        = config_dict['rccl_result_file'], \
       no_of_local_ranks       = config_dict['no_of_local_ranks'], \
       user_key_file           = cluster_dict['priv_key_file'], \
       verify_bus_bw           = config_dict['verify_bus_bw'], \
       verify_avg_bus_bw       = config_dict['verify_avg_bus_bw'], \
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



