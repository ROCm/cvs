import pytest

import re
import sys
import os
import sys
import time
import json
import logging

sys.path.insert( 0, '../../lib' )
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






#ALGO_PROTO combination
def test_cluster_all_reduce_ring_simple(phdl, cluster_dict, vpc_node_list, config_dict, ):

    start_time = phdl.exec('date')
    globals.error_list = []
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='all_reduce_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='ring', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )

    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()
              


def test_cluster_all_gather_ring_simple(phdl, cluster_dict, vpc_node_list, config_dict, ):
    start_time = phdl.exec('date')

    globals.error_list = []
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='all_gather_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='ring', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()




def test_cluster_gather_ring_simple(phdl, cluster_dict, vpc_node_list, config_dict, ):

    globals.error_list = []
    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='gather_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='ring', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()




def test_cluster_scatter_ring_simple(phdl, cluster_dict, vpc_node_list, config_dict, ):

    globals.error_list = []
    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='scatter_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='ring', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()



def test_cluster_reduce_ring_simple(phdl, cluster_dict, vpc_node_list, config_dict, ):

    globals.error_list = []
    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='reduce_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='ring', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()



def test_cluster_alltoall_ring_simple(phdl, cluster_dict, vpc_node_list, config_dict, ):

    globals.error_list = []
    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='alltoall_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='ring', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()




def test_cluster_alltoallv_ring_simple(phdl, cluster_dict, vpc_node_list, config_dict, ):

    globals.error_list = []
    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='alltoallv_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='ring', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()



def test_cluster_sendrecv_ring_simple(phdl, cluster_dict, vpc_node_list, config_dict, ):

    globals.error_list = []
    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='sendrecv_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='ring', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()



def test_cluster_broadcast_ring_simple(phdl, cluster_dict, vpc_node_list, config_dict, ):

    globals.error_list = []
    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='broadcast_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='ring', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()





#========#




def test_cluster_all_reduce_tree_simple(phdl, cluster_dict, vpc_node_list, config_dict, ):

    globals.error_list = []
    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='all_reduce_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='tree', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']

       )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()




def test_cluster_gather_tree_simple(phdl, cluster_dict, vpc_node_list, config_dict, ):

    globals.error_list = []
    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='gather_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='tree', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()




def test_cluster_scatter_tree_simple(phdl, cluster_dict, vpc_node_list, config_dict, ):
    start_time = phdl.exec('date')
    globals.error_list = []

    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='scatter_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='tree', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()




def test_cluster_alltoall_tree_simple(phdl, cluster_dict, vpc_node_list, config_dict, ):

    globals.error_list = []
    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='alltoall_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='tree', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()




def test_cluster_alltoallv_tree_simple(phdl, cluster_dict, vpc_node_list, config_dict, ):

    globals.error_list = []
    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='alltoallv_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='tree', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()



def test_cluster_sendrecv_tree_simple(phdl, cluster_dict, vpc_node_list, config_dict, ):

    globals.error_list = []
    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='sendrecv_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='tree', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()



def test_cluster_broadcast_tree_simple(phdl, cluster_dict, vpc_node_list, config_dict, ):
    globals.error_list = []

    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='broadcast_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='tree', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \

    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()




#========#




def test_cluster_all_reduce_ring_ll128(phdl, cluster_dict, vpc_node_list, config_dict, ):

    globals.error_list = []
    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='all_reduce_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='ring', nccl_proto='LL128', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']

       )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()




def test_cluster_all_gather_ring_ll128(phdl, cluster_dict, vpc_node_list, config_dict, ):

    globals.error_list = []
    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='all_gather_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='ring', nccl_proto='LL128', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()




def test_cluster_gather_ring_ll128(phdl, cluster_dict, vpc_node_list, config_dict, ):

    globals.error_list = []
    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='gather_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='ring', nccl_proto='LL128', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()




def test_cluster_scatter_ring_ll128(phdl, cluster_dict, vpc_node_list, config_dict, ):

    globals.error_list = []
    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='scatter_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='ring', nccl_proto='LL128', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()



def test_cluster_reduce_ring_ll128(phdl, cluster_dict, vpc_node_list, config_dict, ):

    globals.error_list = []
    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='reduce_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='ring', nccl_proto='LL128', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()



def test_cluster_alltoall_ring_ll128(phdl, cluster_dict, vpc_node_list, config_dict, ):

    globals.error_list = []
    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='alltoall_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='ring', nccl_proto='LL128', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()




def test_cluster_alltoallv_ring_ll128(phdl, cluster_dict, vpc_node_list, config_dict, ):

    globals.error_list = []
    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='alltoallv_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='ring', nccl_proto='LL128', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()



def test_cluster_sendrecv_ring_ll128(phdl, cluster_dict, vpc_node_list, config_dict, ):

    globals.error_list = []
    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='sendrecv_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='ring', nccl_proto='LL128', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()



def test_cluster_broadcast_ring_ll128(phdl, cluster_dict, vpc_node_list, config_dict, ):

    globals.error_list = []
    start_time = phdl.exec('date')
    node_list = list(cluster_dict['node_dict'].keys())
    result_dict = rccl_lib.rccl_cluster_test( phdl, test_name='broadcast_perf', \
       node_list=node_list, \
       vpc_node_list=vpc_node_list, \
       user_name=cluster_dict['username'], \
       ib_hca_list=config_dict['ib_hca_list'], \
       net_dev_list=config_dict['net_dev_list'], \
       oob_port=config_dict['oob_port'], \
       no_of_global_ranks=config_dict['no_of_global_ranks'], \
       rocm_path=config_dict['rocm_path'], \
       ucx_path=config_dict['ucx_path'], \
       mpi_path=config_dict['mpi_path'], \
       rccl_path=config_dict['rccl_path'], \
       rccl_tests_path=config_dict['rccl_tests_path'], \
       nccl_algo='ring', nccl_proto='LL128', \
       gid_index=config_dict['gid_index'], \
       qp_count=config_dict['qp_count'], \
       start_msg_size=config_dict['start_msg_size'], \
       end_msg_size=config_dict['end_msg_size'], \
       step_function=config_dict['step_function'], \
       threads_per_gpu=config_dict['threads_per_gpu'], \
       warmup_iterations=config_dict['warmup_iterations'], \
       no_of_iterations=config_dict['no_of_iterations'], \
       check_iteration_count=config_dict['check_iteration_count'], \
       nccl_ib_timeout=config_dict['nccl_ib_timeout'], \
       debug_level=config_dict['debug_level'], \
       rccl_result_file=config_dict['rccl_result_file'], \
       no_of_local_ranks=config_dict['no_of_local_ranks'], \
       user_key_file=cluster_dict['priv_key_file'], \
       verify_bus_bw=config_dict['verify_bus_bw'], \
       verify_avg_bus_bw=config_dict['verify_avg_bus_bw'], \
       exp_results_dict=config_dict['results']
    )
    end_time = phdl.exec('date')
    verify_dmesg_for_errors( phdl, start_time, end_time )
    update_test_result()

