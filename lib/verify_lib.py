import os
import re
import sys

from utils_lib import *
from rocm_plib import *
import linux_utils


err_patterns_dict = {

    'gpu_reset': 'GPU reset begin|GPU hang|cp might be in an unrecoverable state|fence wait loop timeout expired',
    'crash': 'crashed|Traceback|cut here|Bug:|Call Trace|end trace|amdgpu: Fatal error',
    'test_fail': 'Test failure',
    'fault': 'no-retry page fault|Illegal register access|PROTECTION_FAULT_STATUS',
    'driver': 'Queue preemption failed for queue|Failed to evict process queues',
    'hardware': 'hardware error|hardware fail|ras error|uncorrectable|correctable err',
    'down': 'NIC Link is Down|link is down'

}


err_stats_pattern = 'err|drop|discard|overflow|fcs|nak|uncorrect|loss'
warn_stats_pattern = 'retry|timeout|exceeded|ooo|retransmit'
threshold_stats_pattern = 'cnp|ecn'
threshold_counter_val=1000



def verify_gpu_pcie_bus_width( phdl, expected_cards=8, gpu_pcie_speed=32, gpu_pcie_width=16):
    out_dict = get_gpu_pcie_bus_dict( phdl )
    cmd_list = []
    node_0 = list(out_dict.keys())[0]
    card_list = list(out_dict[node_0].keys())
    for node in out_dict.keys():
        card_list = out_dict[node].keys()
        if len(card_list)!= expected_cards:
            fail_test('ERROR !! Number of cards not matching expected no {expected_cards} on node {node}')
    # Let us take the last card_list for further checks ..
    for card_no in card_list:
        cmd_list = []
        for node in out_dict.keys(): 
            bus_no = out_dict[node][card_no]['PCI Bus']
            cmd_list.append(f'sudo lspci -vvv -s {bus_no} | grep "LnkSta:" --color=never')
        pci_dict = phdl.exec_cmd_list( cmd_list )
        for p_node in pci_dict.keys():
            if not re.search( f'Speed {gpu_pcie_speed}GT', pci_dict[p_node] ):
                fail_test(f'ERROR !! PCIe speed not matching for bus {bus_no} on node {p_node}, expected {gpu_pcie_speed}GT/s but got {pci_dict[p_node]}')
            if not re.search( f'Width x{gpu_pcie_width}', pci_dict[p_node] ):
                fail_test(f'ERROR !! PCIe width not matching for bus {bus_no} on node {p_node}, expected {gpu_pcie_width} but got {pci_dict[p_node]}')
            if re.search( 'downgrade', pci_dict[p_node] ):
                fail_test(f'ERROR !! PCIe in downgraded state for bus {bus_no} on node {p_node}')




def verify_gpu_pcie_errors( phdl ):
    metrics_dict = get_gpu_metrics_dict( phdl )
    for node in metrics_dict.keys():
        d_dict = metrics_dict[node]
        for card in d_dict.keys():
            if int(d_dict[card]['pcie_l0_to_recov_count_acc (Count)']) > 100:
                fail_test(f"ERROR !! Node {node} card {card} having higher L0 to recovery counter - \
                    {d_dict[card]['pcie_l0_to_recov_count_acc (Count)']}") 
            if int(d_dict[card]['pcie_nak_sent_count_acc (Count)']) > 100:
                fail_test(f"ERROR !! Node {node} card {card} having PCIe NAK Sent counter - \
                    {d_dict[card]['pcie_nak_sent_count_acc (Count)']}") 
            if int(d_dict[card]['pcie_nak_rcvd_count_acc (Count)']) > 100:
                fail_test(f"ERROR !! Node {node} card {card} having PCIe NAK Recv counter - \
                    {d_dict[card]['pcie_nak_rcvd_count_acc (Count)']}") 


def verify_dmesg_for_errors(phdl, start_time_dict, end_time_dict ):
    print('scan dmesg')
    node0 = list(start_time_dict.keys())[0]
    start_time = start_time_dict[node0]
    end_time = end_time_dict[node0]
    match = re.search( '([a-zA-Z]+\s+[a-zA-Z]+\s+[0-9]+\s+[0-9]+\:[0-9]+\:[0-9]+)\s', start_time)
    start_pattern = match.group(1)
    match = re.search( '([a-zA-Z]+\s+[a-zA-Z]+\s+[0-9]+\s+[0-9]+\:[0-9]+\:[0-9]+)\s', end_time)
    end_pattern = match.group(1)
    output_dict = phdl.exec(f"sudo dmesg -T | awk '/{start_pattern}.*/,/{end_pattern}.*/' | grep -v ALLOWED --color=never")
    #print(output_dict) 
    for node in output_dict.keys():
        for line in output_dict[node].split("\n"):
            for err_key in err_patterns_dict.keys():
                if re.search( f'{err_patterns_dict[err_key]}', line, re.I ):
                    fail_test(f'ERROR - Failue pattern ** {line} ** seen in Dmesg')




def verify_nic_link_flap( phdl ):
    nic_stats_dict = linux_utils.get_nic_ethtool_stats_dict( phdl )
    for node in nic_stats_dict:
        for intf in nic_stats_dict[node].keys():
            for counter in nic_stats_dict[node][intf].keys():
                if re.search( 'reset|down|flap', counter, re.I ):
                    if int(nic_stats_dict[node][intf][counter]) > 0:
                        fail_test(f'ERROR !! {node} {intf} {counter} {nic_stats_dict[node][intf][counter]}')

    nic_dmesg_dict = phdl.exec( 'sudo dmesg -T | grep -i link --color=never')
    for node in nic_dmesg_dict.keys():
        if re.search( 'NIC Link is down', nic_dmesg_dict[node], re.I ):
            fail_test(f'ERROR !! NIC Link down dmesg logs seen on node {node}')




def verify_host_lspci( phdl, pcie_speed=32, pcie_width=16 ):
    out_dict = phdl.exec('sudo amd-smi list | grep BDF --color=never')
    bdf_dict = {}
    for node in out_dict.keys():
        bdf_list_out =  out_dict[node]
        bdf_list = re.findall( 'BDF:\s+([0-9a-f\:\.]+)', out_dict[node], re.I )
        bdf_dict[node] = bdf_list
    for i in range(0,len(bdf_list)):
        cmd_list = []
        for node in out_dict.keys():
                   cmd_list.append(f'sudo lspci -vvv -s {bdf_list[i]} | grep Sta: --color=never')
        lspci_dict = phdl.exec_cmd_list(cmd_list)
        for lnode in lspci_dict.keys():
            if not re.search( f'LnkSta:\s+Speed\s+{pcie_speed}GT', lspci_dict[lnode], re.I ):
                fail_test(f'ERROR !! PCIe Link speed not matching with expected output on node {lnode} - expected {pcie_speed}')
            if not re.search( f'Width\s+x{pcie_width}', lspci_dict[lnode], re.I ):
                fail_test(f'ERROR !! PCIe Link width not matching with expected output on node {lnode} - expected {pcie_width}')
            if not re.search( 'CorrErr+|FatalErr+|RxErr+|BadTLP+|BadDLLP+|DLP+|SDES+|ExOF+|TLP+|MalfTLP+', lspci_dict[lnode], re.I ):
                fail_test(f'ERROR !! PCIe corretable or uncorrectable error indications on Host side on node {lnode}')






def full_dmesg_scan(phdl,):
    print('scan dmesg')
    output_dict = phdl.exec(f"sudo dmesg -T | grep -v initialized | grep -v ALLOWED | grep -v denied --color=never")
    for node in output_dict.keys():
        for line in output_dict[node].split("\n"):
            for err_key in err_patterns_dict.keys():
                if re.search( f'{err_patterns_dict[err_key]}', line, re.I ):
                    fail_test(f'ERROR - Failure pattern *** {line} *** seen in Dmesg on node {node}')




def verify_driver_errors( phdl ):
    print('')
    out_dict = phdl.exec("sudo dmesg -T | grep -i amdgpu  | egrep -i 'fail|error|reset|hang|traceback' --color=never")
    for node in out_dict.keys():
        if re.search( 'fail|error', out_dict[node], re.I ):
            fail_test(f'ERROR !! Dmesg has amdgpu driver errors on node {node}')



def create_cluster_metrics_snapshot( phdl, ):
    s_dict = {}
    s_dict['eth_stats'] = linux_utils.get_nic_ethtool_stats_dict( phdl )
    s_dict['rdma_stats'] = linux_utils.get_rdma_stats_dict( phdl )
    s_dict['gpu_ras_stats'] = get_amd_smi_ras_metrics_dict( phdl )
    s_dict['gpu_pcie_stats'] = get_amd_smi_pcie_metrics_dict( phdl )
    #s_dict['gpu_stats'] = get_gpu_metrics_dict( phdl )
    return s_dict





def get_metrics_snapshot_diff_dict( s_dict_before, s_dict_after ):
    diff_dict = {}
    # Initialize diff_dict
    for key_nam in s_dict_before.keys():
        diff_dict[key_nam] = {}
        for node in s_dict_before[key_nam].keys():
            diff_dict[key_nam][node] = {}
            for dev_nam in s_dict_before[key_nam][node].keys():
                diff_dict[key_nam][node][dev_nam] = {}

    for key_nam in s_dict_before.keys():
        for node in s_dict_before[key_nam].keys():
            for dev_nam in s_dict_before[key_nam][node].keys():
                for stat_nam in s_dict_before[key_nam][node][dev_nam].keys():
                    if not isinstance( s_dict_before[key_nam][node][dev_nam][stat_nam], list ):
                        if isinstance( s_dict_before[key_nam][node][dev_nam][stat_nam], str ):
                            if not re.search( '[a-z\.\_\-]+', s_dict_before[key_nam][node][dev_nam][stat_nam], re.I ):
                                diff_dict[key_nam][node][dev_nam][stat_nam] =  \
                                       int(s_dict_after[key_nam][node][dev_nam][stat_nam]) - \
                                       int(s_dict_before[key_nam][node][dev_nam][stat_nam])
                        elif isinstance( s_dict_before[key_nam][node][dev_nam][stat_nam], int ): 
                            diff_dict[key_nam][node][dev_nam][stat_nam] =  \
                                s_dict_after[key_nam][node][dev_nam][stat_nam] - \
                                s_dict_before[key_nam][node][dev_nam][stat_nam]
    return diff_dict
                      
            


# top level key_nam is type like eth_stats, rdma_stats
def compare_cluster_metrics_snapshots( s_dict_before, s_dict_after ):
    print('Compare 2 cluster snapshots')
    diff_dict = get_metrics_snapshot_diff_dict( s_dict_before, s_dict_after )
    for key_nam in diff_dict.keys():
        for node in diff_dict[key_nam].keys():
            for dev_nam in diff_dict[key_nam][node].keys():
                for stat_nam in diff_dict[key_nam][node][dev_nam].keys():
                    if re.search( f'{warn_stats_pattern}', stat_nam, re.I ):
                        if int(diff_dict[key_nam][node][dev_nam][stat_nam]) > 0:
                            log.warn(f'WARN !! cluster snapshot showing some warning counters going up - {key_nam} {node} {dev_nam} {stat_nam} has incremented by {diff_dict[key_nam][node][dev_nam][stat_nam]} Before = {s_dict_before[key_nam][node][dev_nam][stat_nam]} After = {s_dict_after[key_nam][node][dev_nam][stat_nam]}')
                            print(f'WARN !! cluster snapshot showing some warning counters going up - {key_nam} {node} {dev_nam} {stat_nam} has incremented by {diff_dict[key_nam][node][dev_nam][stat_nam]} Before = {s_dict_before[key_nam][node][dev_nam][stat_nam]} After = {s_dict_after[key_nam][node][dev_nam][stat_nam]}')
                    elif re.search( f'{err_stats_pattern}', stat_nam, re.I ):
                        if int(diff_dict[key_nam][node][dev_nam][stat_nam]) > 0:
                            log.error(f'ERROR !! cluster snapshot showing some error counters going up - {key_nam} {node} {dev_nam} {stat_nam} has incremented by {diff_dict[key_nam][node][dev_nam][stat_nam]} Before = {s_dict_before[key_nam][node][dev_nam][stat_nam]} After = {s_dict_after[key_nam][node][dev_nam][stat_nam]}')
                            print(f'ERROR !! cluster snapshot showing some error counters going up - {key_nam} {node} {dev_nam} {stat_nam} has incremented by {diff_dict[key_nam][node][dev_nam][stat_nam]} Before = {s_dict_before[key_nam][node][dev_nam][stat_nam]} After = {s_dict_after[key_nam][node][dev_nam][stat_nam]}')
                    elif re.search( f'{threshold_stats_pattern}', stat_nam, re.I ):
                        if int(diff_dict[key_nam][node][dev_nam][stat_nam]) > threshold_counter_val:
                            log.error(f'WARN !! cluster snapshot showing some threshold warn counters going up - {key_nam} {node} {dev_nam} {stat_nam} has incremented by {diff_dict[key_nam][node][dev_nam][stat_nam]} Before = {s_dict_before[key_nam][node][dev_nam][stat_nam]} After = {s_dict_after[key_nam][node][dev_nam][stat_nam]}')
                            print(f'WARN !! cluster snapshot showing some threshold warn counters going up - {key_nam} {node} {dev_nam} {stat_nam} has incremented by {diff_dict[key_nam][node][dev_nam][stat_nam]} Before = {s_dict_before[key_nam][node][dev_nam][stat_nam]} After = {s_dict_after[key_nam][node][dev_nam][stat_nam]}')



