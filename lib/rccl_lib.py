import re
import sys
import os

import globals

log = globals.log

from utils_lib import *
from verify_lib import *



rccl_err_dict = {
   'orte': 'ORTE does not know how to route|ORTE was unable to reliably start',
   'nccl': 'NCCL ERROR|Test failure',
   'fs_err': 'No such file or directory'
}


def scan_rccl_logs( output ):
    error_list = []
    warn_list = []
    for line in output.split("\n"):
        for err_key in rccl_err_dict.keys():
            if re.search( f'{rccl_err_dict[err_key]}', line ):
                error_list.append(line)
                fail_test(f'ERROR - {line}')
        if re.search('NCCL WARN', line ):
            warn_list.append(line)
    if len(warn_list) > 0:
        print('Following warnings were observed in the RCCL test')
        print('#============#')
        print(warn_list)
        print('#============#')
    if not re.search('#\sAvg bus bandwidth', output ):
        fail_test('RCCL test did not complete successfully, no bandwidth numbers printed - pls check')
  


def check_avg_bus_bw( output, exp_res_dict ):
    if re.search('#\sAvg bus bandwidth\s+:\s+[0-9\.]+', output, re.I ):
        match = re.search('#\sAvg bus bandwidth\s+:\s+([0-9\.]+)', output, re.I )
        actual_bw = float(match.group(1))
        if actual_bw < float(exp_res_dict['avg_bus_bw']):
            fail_test(f"Actual Avg Bus BW {actual_bw} is less than the expected Avg BW {exp_res_dict['avg_bus_bw']}") 




def check_bus_bw( test_name, output, exp_res_dict ):
    actual_bw_dict = {}
    msg_size_list = list(exp_res_dict['bus_bw'].keys())
    print(test_name)
    act_res_dict = json.loads(output.replace( '\n', '').replace( '\r', ''))
    if re.search( 'alltoall|all_to_all', test_name, re.I ):
        for act_dict in act_res_dict:
            if act_dict['inPlace'] == 0:
                for msg_size in msg_size_list:
                    if str(msg_size) == str(act_dict['size']):
                        if float(act_dict['busBw']) < float(exp_res_dict['bus_bw'][msg_size]):
                            fail_test(f"The actual out-of-place bus BW {act_dict['busBw']} for msg size {act_dict['size']} is lower than expected bus BW {exp_res_dict['bus_bw'][msg_size]}")
    else:
        for act_dict in act_res_dict:
            if act_dict['inPlace'] == 1:
                for msg_size in msg_size_list:
                    if str(msg_size) == str(act_dict['size']):
                        if float(act_dict['busBw']) < float(exp_res_dict['bus_bw'][msg_size]):
                            fail_test(f"The actual out-of-place bus BW {act_dict['busBw']} for msg size {act_dict['size']} is lower than expected bus BW {exp_res_dict['bus_bw'][msg_size]}")

 


def rccl_cluster_test( phdl, shdl, test_name, cluster_node_list, vpc_node_list, user_name, ib_hca_list, \
        net_dev_list, oob_port, no_of_global_ranks, rocm_path_var, mpi_dir, mpi_path_var, \
        rccl_dir, rccl_path_var, rccl_tests_dir, nccl_algo='ring', \
        nccl_proto='simple', gid_index=1, qp_count=1, start_msg_size=1024, end_msg_size='16g', \
        step_function=2, threads_per_gpu=1, warmup_iterations=10, no_of_iterations=1, \
        check_iteration_count=1, debug_level='INFO', \
        rccl_result_file='/tmp/rccl_result_output.json', no_of_local_ranks=8, \
        ib_rx_queue_len=8192, ucx_tls='tcp', hcoll_enable_mcast_all=0, \
        nccl_cumem_enable=0, nccl_ib_timeout=30, nccl_ib_sl=0, \
        nccl_ib_tc=41, nccl_ib_split_data_on_qps=0, nccl_pxn_disable=0, \
        nccl_net_plugin=None, user_password=None, \
        min_channels=64, max_channels=64, \
        user_key_file=None, verify_bus_bw=False, \
        exp_results_dict=None ):

    print(f'Starting RCCL Test ..........................................{test_name}')
    ROCM_PATH=rocm_path_var

    #MPI_PATH=f'{mpi_path}/install/bin'
    MPI_PATH=f'{mpi_path_var}'
    MPI_INSTALL_DIR=f'{mpi_dir}'
    RCCL_INSTALL_DIR=f'{rccl_dir}'
    RCCL_PATH=f'{rccl_path_var}'
    RCCL_TESTS_INSTALL_DIR=f'{rccl_tests_dir}'

    PATH=f'{MPI_PATH}/bin:{ROCM_PATH}/bin:$PATH'
    LD_LIBRARY_PATH=f'{RCCL_PATH}:{MPI_PATH}/lib:{ROCM_PATH}/lib:$LD_LIBRARY_PATH'

    print(f'%% VPC Node IPs {vpc_node_list}')

    head_node = cluster_node_list[0]
    host_params=''
    proc_per_node = int(int(no_of_global_ranks)/len(cluster_node_list))
    for node in vpc_node_list:
        host_params = f'{host_params}{node}:{proc_per_node},'

    host_params = host_params.rstrip(',')
    print(f'RCCL Hosts -H value {host_params}')

        
    cmd = f'''{MPI_INSTALL_DIR}/mpirun --np {no_of_global_ranks} \
        --allow-run-as-root \
        -H {host_params} \
        -x NCCL_DEBUG={debug_level} \
        --bind-to numa \
        -x NCCL_IB_GID_INDEX={gid_index} \
        -x UCX_UNIFIED_MODE=y \
        -x NCCL_IB_PCI_RELAXED_ORDERING=1 \
        -x PATH={PATH} \
        -x LD_LIBRARY_PATH={LD_LIBRARY_PATH} \
        -x NCCL_IB_HCA={ib_hca_list} \
        --mca btl ^vader,openib \
        --mca btl_tcp_if_include {oob_port}\
        -x UCX_NET_DEVICES={net_dev_list} \
        -x NCCL_ALGO={nccl_algo} \
        -x NCCL_MIN_NCHANNELS={min_channels} \
        -x NCCL_MAX_NCHANNELS={max_channels} \
        -x NCCL_IB_QPS_PER_CONNECTION={qp_count} \
        -x IB_RX_QUEUE_LEN={ib_rx_queue_len} \
        -x UCX_TLS={ucx_tls} \
        -x HCOLL_ENABLE_MCAST_ALL={hcoll_enable_mcast_all} \
        -x NCCL_CUMEM_ENABLE={nccl_cumem_enable} \
        -x NCCL_IB_TIMEOUT={nccl_ib_timeout} \
        -x NCCL_IB_SL={nccl_ib_sl} \
        -x NCCL_IB_TC={nccl_ib_tc} \
        -x NCCL_IB_SPLIT_DATA_ON_QPS={nccl_ib_split_data_on_qps} \
        -x NCCL_PXN_DISABLE={nccl_pxn_disable} \
        -x NCCL_NET_PLUGIN={nccl_net_plugin} \
        {RCCL_TESTS_INSTALL_DIR}/{test_name} -b {start_msg_size} -e {end_msg_size} -f {step_function} \
        -g {threads_per_gpu} -c {check_iteration_count} -w {warmup_iterations} \
        -Z json -x {rccl_result_file}
        '''

    print('%%%%%%%%%%%%%%%%')
    print(cmd)
    print('%%%%%%%%%%%%%%%%')
    try:
        out_dict = shdl.exec(cmd, timeout=500)
        output = out_dict[head_node]
        #print(output)
        scan_rccl_logs(output)
    except Exception as e:
        log.error(f'Hit Exceptions with rccl cmd {cmd} - exception {e}')
        fail_test(f'Hit Exceptions with rccl cmd {cmd} - exception {e}')

    result_dict_out = shdl.exec(f'cat {rccl_result_file}')
    result_out = result_dict_out[head_node]
    smi_out_dict = shdl.exec('rocm-smi -a | head -30')
    smi_out = smi_out_dict[head_node]
    model=get_model_from_rocm_smi_output(smi_out)
    if re.search( 'True', verify_bus_bw, re.I ):
        check_bus_bw( test_name, result_out, exp_results_dict[test_name] )

    return result_out
