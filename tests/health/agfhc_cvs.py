import pytest

import re
import sys
import os
import sys
import time
import json
import logging

sys.path.insert( 0, '../../lib' )
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
     config_dict = config_dict_t['agfhc']
     log.info(config_dict)
     return config_dict




def scan_agfc_results(out_dict):

    for host in out_dict.keys():
        if not re.search( 'return code AGFHC_SUCCESS', out_dict[host], re.I ):
            fail_test(f'Test failed on node {host} - AGFHC_SUCCESS code NOT seen in test result')

        if re.search( 'FAIL|ERROR|ABORT', out_dict[host], re.I ):
            fail_test(f'Test failed on node {host} - FAIL or ERROR or ABORT patterns seen')



# Create connection to DUT, MTPs, Switches and export for later use ..
@pytest.fixture(scope="module")
def  phdl(cluster_dict):
     nhdl_dict = {}
     print(cluster_dict)
     node_list = list(cluster_dict['node_dict'].keys())
     phdl = Pssh( log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'] )
     return phdl



@pytest.mark.dependency(name="init")
def test_install_agfhc(phdl, config_dict, ):
    globals.error_list = []
    log.info('Testcase install agfhc')
    path = config_dict['path']
    package_path = config_dict['package_path']
    print(package_path)
    out_dict = phdl.exec(f'cd {package_path};sudo ./install', timeout=90)
    for node in out_dict.keys():
        print(out_dict[node])
        if re.search( 'Error|No such file', out_dict[node], re.I ):
            fail_test(f'Installation of AGFHC failed on node {node}')
    out_dict = phdl.exec(f'ls -l {path}')
    for node in out_dict.keys():
        print(out_dict[node])
        if re.search( 'No such file', out_dict[node], re.I ):
            fail_test(f'Installation of AGFHC failed on node {node}')
    update_test_result()
     


@pytest.mark.dependency(depends=["init"])
def test_agfhc_hbm(phdl, config_dict, ):
    globals.error_list = []
    log.info('Testcase Run HBM Test')
    path = config_dict['path']
    duration = convert_hms_to_secs(config_dict['hbm_test_duration'])
    (hours,mins,secs) = config_dict['hbm_test_duration'].split(":")
    out_dict = phdl.exec(f'sudo {path}/agfhc -t hbm:d={hours}h{mins}m{secs}s', timeout=duration+120)
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


# 2 hrs
@pytest.mark.dependency(depends=["init"])
def test_agfhc_hbm1_lvl5(phdl, config_dict, ):
    globals.error_list = []
    log.info('Testcase Run HBM1 Test - hbm_lvl5')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -s hbm1 -r hbm_lvl5', timeout=(60*300))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


# 2 hrs
@pytest.mark.dependency(depends=["init"])
def test_agfhc_hbm2_lvl5(phdl, config_dict, ):
    globals.error_list = []
    log.info('Testcase Run HBM2 Test - hbm_lvl5')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -s hbm2 -r hbm_lvl5', timeout=(60*300))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()



# 30 min
@pytest.mark.dependency(depends=["init"])
def test_agfhc_hbm3_lvl3(phdl, config_dict, ):
    globals.error_list = []
    log.info('Testcase Run HBM3 Test - hbm_lvl3')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -s hbm3 -r hbm_lvl3', timeout=(60*100))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()




@pytest.mark.dependency(depends=["init"])
def test_agfhc_dma_lvl1(phdl, config_dict, ):
    globals.error_list = []
    log.info('Testcase Run all_lvl1')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -r all_lvl1', timeout=(60*30))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


@pytest.mark.dependency(depends=["init"])
def test_agfhc_dma_lvl1(phdl, config_dict, ):
    globals.error_list = []
    log.info('Testcase Run DMA lvl1')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -r dma_lvl1', timeout=(60*30))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


@pytest.mark.dependency(depends=["init"])
def test_agfhc_gfx_lvl1(phdl, config_dict, ):
    globals.error_list = []
    log.info('Testcase Run GFX lvl1')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -r gfx_lvl1', timeout=(60*60))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


@pytest.mark.dependency(depends=["init"])
def test_agfhc_pcie_lvl1(phdl, config_dict, ):
    globals.error_list = []
    log.info('Testcase Run PCIe lvl1')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -r pcie_lvl1', timeout=(60*60))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


@pytest.mark.dependency(depends=["init"])
def test_agfhc_pcie_lvl3(phdl, config_dict, ):
    globals.error_list = []
    log.info('Testcase Run PCIe lvl3')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -r pcie_lvl3', timeout=(60*60))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()




@pytest.mark.dependency(depends=["init"])
def test_agfhc_xgmi_lvl1(phdl, config_dict, ):
    globals.error_list = []
    log.info('Testcase Run XGMI lvl1')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -r xgmi_lvl1', timeout=(60*90))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


@pytest.mark.dependency(depends=["init"])
def test_agfhc_all_perf(phdl, config_dict, ):
    globals.error_list = []
    log.info('Testcase Run all_perf')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -r all_perf', timeout=(60*90))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()



@pytest.mark.dependency(depends=["init"])
def test_agfhc_all_lvl5(phdl, config_dict, ):
    log.info('Testcase all lvl5')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc --recipe-info all_lvl5', timeout=(60*260))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


#180m
#@pytest.mark.dependency(depends=["init"])
#def test_agfhc_rochpl(phdl, config_dict, ):
#    log.info('Testcase rochpl for 180 min')
#    path = config_dict['path']
#    out_dict = phdl.exec(f'sudo {path}/agfhc rochpl:d=180m', timeout=(60*650))
#    scan_agfc_results(out_dict)
#    print_test_output(log, out_dict)
#    update_test_result()
