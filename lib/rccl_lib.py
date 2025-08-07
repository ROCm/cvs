import re
import sys
import os
import netmiko
from netmiko import ConnectHandler
from netmiko import redispatch

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




def check_bus_bw( output, exp_res_dict ):
    actual_bw_dict = {}
    msg_size_list = list(exp_res_dict['out_bus_bw'].keys())
    print(msg_size_list)
    print(exp_res_dict)
    for line in output.split("\n"):
        line_dict = json.loads(line)
        print('^^^^^^')
        print(line_dict)
        if line_dict['inPlace'] == 0:
            msg_size = str(line_dict['size'])
            if msg_size in msg_size_list:
               
                if float(line_dict['busBw']) < float(exp_res_dict['out_bus_bw'][msg_size]):
                    fail_test(f"The actual out-of-place bus BW {line_dict['busBw']} for msg size {line_dict['size']} is lower than expected bus BW {exp_res_dict['out_bus_bw'][msg_size]}")

 


def rccl_cluster_test( phdl, shdl, test_name, cluster_node_list, vpc_node_list, user_name, ib_hca_list, \
        net_dev_list, oob_port, no_of_global_ranks, rocm_path, ucx_path, mpi_path, \
        rccl_path, rccl_tests_path, nccl_algo='ring', \
        nccl_proto='simple', gid_index=1, qp_count=1, start_msg_size=1024, end_msg_size='16g', \
        step_function=2, threads_per_gpu=1, warmup_iterations=10, no_of_iterations=1, \
        check_iteration_count=1, nccl_ib_timeout=5, debug_level='INFO', \
        rccl_result_file='/tmp/rccl_result_output.json', no_of_local_ranks=8, user_password=None, \
        min_channels=64, max_channels=64, \
        user_key_file=None, verify_avg_bus_bw=False, verify_bus_bw=False, \
        exp_results_dict=None ):

    print(test_name)
    ROCM_PATH=rocm_path
    MPI_PATH=f'{mpi_path}/install/bin'
    UCX_INSTALL_DIR=f'{ucx_path}/install'
    MPI_INSTALL_DIR=f'{mpi_path}/install'
    RCCL_INSTALL_DIR=f'{rccl_path}/build/release'
    RCCL_TESTS_INSTALL_DIR=f'{rccl_tests_path}/build'

    PATH=f'{MPI_INSTALL_DIR}/bin:{ROCM_PATH}/bin:$PATH'
    LD_LIBRARY_PATH=f'{RCCL_INSTALL_DIR}/lib:{MPI_INSTALL_DIR}/lib:$LD_LIBRARY_PATH'

    print('%%%% vpc_node_list %%%%')
    print(vpc_node_list)
    print(vpc_node_list[0])

    head_node = cluster_node_list[0]
    host_params=''
    proc_per_node = int(int(no_of_global_ranks)/len(cluster_node_list))
    for node in vpc_node_list:
        host_params = f'{host_params}{node}:{proc_per_node},'

    host_params = host_params.rstrip(',')
    print(host_params)

    #if user_password is None and user_key_file is None:
    #    print('ERROR !! Both password and key file cannot be none, need one to login to the host')
    #    return
    #elif user_key_file is None:
    #    hdl = ConnectHandler( ip=head_node, device_type='linux', username=user_name, \
    #          password = user_password )
    #elif user_password is None:
    #    hdl = ConnectHandler( ip=head_node, device_type='linux', username=user_name, \
    #          use_keys=True, key_file=user_key_file )
        
    cmd = f'''{MPI_PATH}/mpirun --np {no_of_global_ranks} \
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
        -x NCCL_IB_TIMEOUT={nccl_ib_timeout} \
        -x NCCL_IB_QPS_PER_CONNECTION={qp_count} \
        {RCCL_TESTS_INSTALL_DIR}/{test_name} -b {start_msg_size} -e {end_msg_size} -f {step_function} \
        -g {threads_per_gpu} -N {no_of_iterations} -c {check_iteration_count} -w {warmup_iterations} \
        -Z json -x {rccl_result_file}'''

    print('%%%%%%%%%%%%%%%%')
    print(cmd)
    print('%%%%%%%%%%%%%%%%')
    #output = hdl.send_command(cmd, read_timeout=300)
    try:
        out_dict = shdl.exec(cmd, timeout=500)
        output = out_dict[head_node]
        print(output)
        scan_rccl_logs(output)
    except Exception as e:
        log.error(f'Hit Exceptions with rccl cmd {cmd} - exception {e}')
        fail_test(f'Hit Exceptions with rccl cmd {cmd} - exception {e}')

    #result_out = hdl.send_command(f'cat {rccl_result_file}') 
    result_dict_out = shdl.exec(f'cat {rccl_result_file}')
    result_out = result_dict_out[head_node]
    smi_out_dict = shdl.exec('rocm-smi -a | head -30')
    smi_out = smi_out_dict[head_node]
    model=get_model_from_rocm_smi_output(smi_out)
    if re.search( 'True', verify_avg_bus_bw, re.I ):
        check_avg_bus_bw( output, exp_results_dict[model][test_name] )

    if re.search( 'True', verify_bus_bw, re.I ):
        check_bus_bw( result_out, exp_results_dict[model][test_name] )

    return result_out
    
