import re
import os
import sys
import parallel_ssh_lib

from utils_lib import *


def get_rocm_smi_dict( phdl ):
    rocm_smi_dict = convert_phdl_json_to_dict( phdl.exec('sudo rocm-smi -a --json') )
    return rocm_smi_dict


def get_gpu_partition_dict( phdl ):
    amd_part_dict = convert_phdl_json_to_dict( phdl.exec('sudo amd-smi partition --json') )
    return amd_part_dict


def get_gpu_process_dict( phdl ):
    amd_proc_dict = convert_phdl_json_to_dict( phdl.exec('sudo amd-smi process --json') )
    return amd_proc_dict


def get_amd_smi_metric_dict( phdl ):
    amd_metric_dict = convert_phdl_json_to_dict( phdl.exec('sudo amd-smi metric --json') )
    return amd_metric_dict


def get_amd_smi_fw_dict( phdl ):
    firmware_dict = convert_phdl_json_to_dict( phdl.exec('sudo amd-smi firmware --json') )
    return firmware_dict


def get_amd_smi_ras_metrics_dict( phdl ):
    ras_dict = {}
    ras_dict_t = convert_phdl_json_to_dict( phdl.exec( 'sudo amd-smi metric --ecc --json' ))
    print(ras_dict_t)
    for node in ras_dict_t.keys():
        ras_dict[node] = {}
        print('^^^^^')
        print(ras_dict_t[node])
        if isinstance( ras_dict_t[node], dict ):
            if 'gpu_data' in ras_dict_t[node].keys():
                for gpu_dict in list(ras_dict_t[node]['gpu_data']):
                    ras_dict[node][gpu_dict['gpu']] = gpu_dict['ecc']
        elif isinstance( ras_dict_t[node], list ):
            for gpu_dict in ras_dict_t[node]:
                ras_dict[node][gpu_dict['gpu']] = gpu_dict['ecc']
            
    return ras_dict


def get_amd_smi_pcie_metrics_dict( phdl ):
    pcie_dict = {}
    pcie_dict_t = convert_phdl_json_to_dict( phdl.exec( 'sudo amd-smi metric --pcie --json' ))
    for node in pcie_dict_t.keys():
        pcie_dict[node] = {}
        if isinstance( pcie_dict_t[node], dict ):
            if 'gpu_data' in pcie_dict_t[node].keys():
                for gpu_dict in list(pcie_dict_t[node]['gpu_data']):
                    pcie_dict[node][gpu_dict['gpu']] = gpu_dict['pcie']
        elif isinstance( pcie_dict_t[node], list ):
            for gpu_dict in pcie_dict_t[node]:
                pcie_dict[node][gpu_dict['gpu']] = gpu_dict['pcie']
    return pcie_dict


def get_gpu_mem_use_dict( phdl ):
    d_dict = convert_phdl_json_to_dict( phdl.exec('sudo rocm-smi --showmemuse --json'))
    return d_dict


def get_gpu_use_dict( phdl ):
    d_dict = convert_phdl_json_to_dict( phdl.exec('sudo rocm-smi --showuse --json'))
    return d_dict

def get_gpu_metrics_dict( phdl ):
    d_dict = convert_phdl_json_to_dict( phdl.exec('sudo rocm-smi --showmetric --json'))
    return d_dict


def get_gpu_fw_dict( phdl ):
    d_dict = convert_phdl_json_to_dict( phdl.exec('sudo rocm-smi --showfwinfo --json'))
    return d_dict

def get_gpu_pcie_bus_dict( phdl ):
    d_dict = convert_phdl_json_to_dict( phdl.exec('sudo rocm-smi --showbus --json'))
    return d_dict

def get_gpu_model_dict( phdl ):
    d_dict = convert_phdl_json_to_dict( phdl.exec('sudo rocm-smi --showproductname --json'))
    return d_dict

def get_gpu_temp_dict( phdl ):
    d_dict = convert_phdl_json_to_dict( phdl.exec('sudo rocm-smi --showtemp --json'))
    return d_dict




