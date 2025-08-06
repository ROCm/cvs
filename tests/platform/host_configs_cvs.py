import pytest

import re
import sys
import os
import sys
import time
import json
import logging
import json

sys.path.insert( 0, './lib' )
from parallel_ssh_lib import *
from utils_lib import *
from verify_lib import *
from rocm_plib import *

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
     config_dict = config_dict_t['host']
     log.info(config_dict)
     return config_dict



# Create connection to DUT, Switches and export for later use ..
@pytest.fixture(scope="module")
def  phdl(cluster_dict):
     nhdl_dict = {}
     print(cluster_dict)
     node_list = list(cluster_dict['node_dict'].keys())
     phdl = Pssh( log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'] )
     return phdl



def test_check_os_release(phdl, config_dict, ):
    globals.error_list = []
    log.info('Testcase check OS Version')
    os_version = config_dict['os_version']
    out_dict = phdl.exec('cat /etc/os-release')
    for node in out_dict.keys():
        if not re.search( f'{os_version}', out_dict[node], re.I ):
            match = re.search( 'VERSION=\"([0-9\.\-\_A-Z]+)\s+)', out_dict[node], re.I )
            actual_ver = match.group(1)
            fail_test(f'Installed OS Version {actual_ver} not matching expected version {os_version} on node {node}')
    update_test_result()
     


def test_check_kernel_version( phdl, config_dict ):
    globals.error_list = []
    log.info('Testcase check Kernel Version')
    kernel_version = config_dict['kernel_version']
    out_dict = phdl.exec('uname -a')
    for node in out_dict.keys():
        if not re.search( f'{kernel_version}', out_dict[node], re.I ):
            match = re.search( '([0-9\.\-\_]+generic)', out_dict[node], re.I )
            actual_ver =  match.group(1)
            fail_test(f'Installed Kernel Version {actual_ver} not matching expected version {kernel_version} on node {node}')
    update_test_result()
     


def test_check_bios_version( phdl, config_dict ):
    globals.error_list = []
    log.info('Testcase check BIOS Version')
    bios_version = config_dict['bios_version']
    out_dict = phdl.exec('sudo dmidecode -s bios-version')
    for node in out_dict.keys():
        if not re.search( f'{bios_version}', out_dict[node], re.I ):
            match = re.search( '([a-z0-9\_\.\-]+)', out_dict[node], re.I )
            act_bios_ver = match.group(1)
            fail_test(f'Installed BIOS Version {act_bios_ver} not matching expected version {bios_version} on node {node}')
    update_test_result()
     



def test_check_rocm_version( phdl, config_dict ):
    globals.error_list = []
    log.info('Testcase check rocm version')
    rocm_version = config_dict['rocm_version']
    out_dict = phdl.exec('amd-smi version')
    for node in out_dict.keys():
        if not re.search( f'{rocm_version}', out_dict[node], re.I ):
            match = re.search('ROCm version:\s+([0-9\.]+)', out_dict[node], re.I )
            actual_rocm_version = match.group(1)
            fail_test(f'Installed rocm version {actual_rocm_version} not matching expected version {rocm_version} on node {node}')
    update_test_result()
     


def test_check_gpu_fw_version( phdl, config_dict ):
    globals.error_list = []
    log.info('Testcase check GPU Firmware versions')
    fw_dict = config_dict['fw_dict']
    out_dict = get_amd_smi_fw_dict(phdl)
    for node in out_dict.keys():
        for gpu_dict in out_dict[node]:
            gpu_no = gpu_dict['gpu']
            for fw_list_dict in gpu_dict['fw_list']:
                fw_key = fw_list_dict['fw_id']
                print(f'&&&&& {fw_list_dict} &&&')
                print(f'&&&&& {fw_dict} &&&')
                if fw_list_dict['fw_version'] != fw_dict[fw_key]:
                    fail_test(f"For Firmware {fw_key} actual FW version {fw_list_dict['fw_version']} for gpu {gpu_no} on node {node} is not matching expected FW version {fw_dict[fw_key]}")
    update_test_result()




def test_check_pci_realloc( phdl, config_dict ):
    globals.error_list = []
    log.info('Testcase check pci realloc')
    pci_realloc = config_dict['pci_realloc']
    out_dict = phdl.exec('cat /proc/cmdline')
    for node in out_dict.keys():
        if not re.search( f'pci=realloc={pci_realloc}', out_dict[node], re.I ):
            fail_test(f'PCI realloc flag not set to {pci_realloc} on node {node}')
    update_test_result()
     


def test_check_iommu_pt( phdl, config_dict ):
    globals.error_list = []
    log.info('Testcase check IOMMU PT')
    out_dict = phdl.exec('cat /proc/cmdline')
    for node in out_dict.keys():
        if not re.search( f'iommu=pt', out_dict[node], re.I ):
            fail_test(f'IOMMU not set to pt on node {node}')
    update_test_result()
     


def test_check_numa_balancing( phdl, config_dict ):
    globals.error_list = []
    log.info('Testcase check NUMA balancing')
    out_dict = phdl.exec('sudo sysctl kernel.numa_balancing')
    for node in out_dict.keys():
        if not re.search( f'=0|= 0', out_dict[node], re.I ):
            fail_test(f'NUMA balancing not disabled on node {node}')
    update_test_result()
     



def test_check_online_memory( phdl, config_dict ):
    globals.error_list = []
    log.info('Testcase check online memory')
    online_mem = config_dict['online_memory']
    out_dict = phdl.exec('lsmem')
    for node in out_dict.keys():
        if not re.search( f'Total online memory:\s+{online_mem}', out_dict[node], re.I ):
            match =  re.search('Total online memory:\s+([0-9\.A-Za-z]+)', out_dict[node] )
            actual_mem = match.group(1)
            fail_test(f'Total online memory {actual_mem} not matching expected online mem {online_mem} on node {node}')
    update_test_result()
     


def test_check_pci_accelerators( phdl, config_dict ):
    globals.error_list = []
    log.info('Testcase check online GPUs in pcie')
    gpu_count = config_dict['gpu_count']
    out_dict = phdl.exec('lspci | grep "accelerators" --color=never')
    for node in out_dict.keys():
        match_list = re.findall( 'accelerators:\s+Advanced', out_dict[node], re.I )
        actual_gpu_count = len(match_list)
        if int(gpu_count) != actual_gpu_count:
            fail_test(f'Expected GPU count in PCI {gpu_count} not matching actual GPU count {actual_gpu_count} on node {node}')
    update_test_result()



def test_check_pci_speed_width( phdl, config_dict ):
    globals.error_list = []
    log.info('Testcase check online GPUs in pcie')
    gpu_pcie_speed = config_dict['gpu_pcie_speed']
    gpu_pcie_width = config_dict['gpu_pcie_width']
    out_dict = get_gpu_pcie_bus_dict( phdl )
    cmd_list = []
    node_0 = list(out_dict.keys())[0]
    card_list = list(out_dict[node_0].keys())

    print('&&&&& {card_list} &&&&')
    # We are making an assumption that it is a homogenous cluster
    # and all nodes have same PCI Bus number
    for card_no in card_list:
        cmd_list = []
        for node in out_dict.keys():
            bus_no = out_dict[node][card_no]['PCI Bus']
            cmd_list.append(f'sudo lspci -vvv -s {bus_no} | grep "LnkSta:" --color=never')
        pci_dict = phdl.exec_cmd_list( cmd_list )
        for p_node in pci_dict.keys():
            if not re.search( f'Speed {gpu_pcie_speed}GT', pci_dict[p_node] ):
                fail_test(f'PCIe speed not matching for bus {bus_no} on node {p_node}, expected {gpu_pcie_speed}GT/s but got {pci_dict[p_node]}')
            if not re.search( f'Width x{gpu_pcie_width}', pci_dict[p_node] ):
                fail_test(f'PCIe width not matching for bus {bus_no} on node {p_node}, expected {gpu_pcie_width} but got {pci_dict[p_node]}')
            if re.search( 'downgrade', pci_dict[p_node] ):
                fail_test(f'PCIe in downgraded state for bus {bus_no} on node {p_node}')
    update_test_result()
                    
                 
   

def test_check_pci_acs( phdl, config_dict ):
    globals.error_list = []
    out_dict = phdl.exec('sudo lspci -vv | grep ACSCtl | grep SrcValid+ --color=never')
    for node in out_dict.keys():
        if re.search( 'ACSCtl:', out_dict[node], re.I ):
            fail_test('PCIe ACS not disabled on node {node}')
    update_test_result()



def test_check_dmesg_driver_errors( phdl, config_dict):
    globals.error_list = []
    out_dict = phdl.exec("sudo dmesg -T | grep -i amdgpu  | egrep -i 'fail|error' --color=never")
    for node in out_dict.keys():
        if re.search( 'fail|error', out_dict[node], re.I ):
            fail_test(f'Dmesg has amdgpu driver errors on node {node}')
    update_test_result()
    out_dict = phdl.exec("sudo dmesg -T | grep -i amdgpu  | egrep -i 'reset|hang|traceback' --color=never")
    for node in out_dict.keys():
        if re.search( 'reset|hang', out_dict[node], re.I ):
            fail_test(f'Dmesg has amdgpu reset/hang errors on node {node}')
    update_test_result()
   

