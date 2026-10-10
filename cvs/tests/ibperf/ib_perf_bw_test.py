'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest

import logging
import re


from cvs.lib import ibperf_lib

from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *

from cvs.lib import globals

log = logging.getLogger(__name__)

ib_bw_dict = {}
ib_lat_dict = {}

rccl_res_dict = {}


def can_read_dmesg(orch):
    if all(get_passwordless_sudo_status(orch.all).values()):
        return True
    log.warning('SKIPPING the dmesg error check: no passwordless sudo to read dmesg on one or more nodes')
    return False


@pytest.mark.parametrize("bw_test", ["ib_write_bw", "ib_read_bw", "ib_send_bw"])
def test_ib_bw_perf(orch, bw_test, config_dict):
    globals.error_list = []
    ib_bw_dict[bw_test] = {}
    scan_dmesg = can_read_dmesg(orch)

    gpu_nic_dict = linux_utils.get_gpu_nic_mapping_dict(orch.all)
    gpu_numa_dict = linux_utils.get_gpu_numa_dict(orch.all)

    bck_nic_dict_lshw = linux_utils.get_backend_nic_dict(orch.all)
    rdma_nic_dict = linux_utils.get_active_rdma_nic_dict(orch.all)

    bck_nic_dict = {}
    for node in rdma_nic_dict.keys():
        bck_nic_dict[node] = {}
        for rdma_dev in rdma_nic_dict[node].keys():
            if rdma_nic_dict[node][rdma_dev]['eth_device'] in bck_nic_dict_lshw[node]:
                bck_nic_dict[node][rdma_dev] = rdma_nic_dict[node][rdma_dev]

    rocm_path = ibperf_lib.detect_rocm_path(orch.all, config_dict.get('rocm_dir', ''))
    for msg_size in config_dict['msg_size_list']:
        ib_bw_dict[bw_test][msg_size] = {}
        for qp_count in config_dict['qp_count_list']:
            start_time = orch.all.exec('date +"%a %b %e %H:%M"', print_console=False)
            orch.all.exec(
                f'echo "Starting Test {bw_test} for {msg_size} and QP count {qp_count}" | sudo -n tee /dev/kmsg',
                print_console=False,
            )
            ib_bw_dict[bw_test][msg_size][qp_count] = ibperf_lib.run_ib_perf_bw_test(
                orch.head,
                orch.all,
                bw_test,
                gpu_numa_dict,
                gpu_nic_dict,
                bck_nic_dict,
                f'{config_dict["install_dir"]}/perftest/bin',
                msg_size,
                config_dict['gid_index'],
                qp_count,
                int(config_dict['port_no']),
                int(config_dict['duration']),
                rocm_path=rocm_path,
                use_rocm_dmabuf=bool(re.search('True', config_dict.get('use_rocm_dmabuf', 'False'), re.I)),
            )
            end_time = orch.all.exec('date +"%a %b %e %H:%M"', print_console=False)
            if scan_dmesg:
                verify_dmesg_for_errors(orch.all, start_time, end_time, till_end_flag=True)
            if re.search('True', config_dict['verify_bw'], re.I):
                ibperf_lib.verify_expected_bw(
                    bw_test,
                    msg_size,
                    qp_count,
                    ib_bw_dict[bw_test][msg_size][qp_count],
                    config_dict['expected_results'],
                )

    log.debug('ib_bw_dict: %s', ib_bw_dict)
    update_test_result()


@pytest.mark.parametrize("lat_test", ["ib_write_lat", "ib_send_lat"])
def test_ib_lat_perf(orch, lat_test, config_dict):
    globals.error_list = []
    ib_lat_dict[lat_test] = {}
    scan_dmesg = can_read_dmesg(orch)

    gpu_nic_dict = linux_utils.get_gpu_nic_mapping_dict(orch.all)
    gpu_numa_dict = linux_utils.get_gpu_numa_dict(orch.all)

    bck_nic_dict_lshw = linux_utils.get_backend_nic_dict(orch.all)
    rdma_nic_dict = linux_utils.get_active_rdma_nic_dict(orch.all)

    bck_nic_dict = {}
    for node in rdma_nic_dict.keys():
        bck_nic_dict[node] = {}
        for rdma_dev in rdma_nic_dict[node].keys():
            if rdma_nic_dict[node][rdma_dev]['eth_device'] in bck_nic_dict_lshw[node]:
                bck_nic_dict[node][rdma_dev] = rdma_nic_dict[node][rdma_dev]

    rocm_path = ibperf_lib.detect_rocm_path(orch.all, config_dict.get('rocm_dir', ''))
    for msg_size in config_dict['msg_size_list']:
        ib_lat_dict[lat_test][msg_size] = {}
        start_time = orch.all.exec('date +"%a %b %e %H:%M"', print_console=False)
        orch.all.exec(
            f'echo "Starting Test {lat_test} for {msg_size}" | sudo -n tee /dev/kmsg',
            print_console=False,
        )
        ib_lat_dict[lat_test][msg_size] = ibperf_lib.run_ib_perf_lat_test(
            orch.head,
            orch.all,
            lat_test,
            gpu_numa_dict,
            gpu_nic_dict,
            bck_nic_dict,
            f'{config_dict["install_dir"]}/perftest/bin',
            msg_size,
            config_dict['gid_index'],
            int(config_dict['port_no']),
            rocm_path=rocm_path,
            use_rocm_dmabuf=bool(re.search('True', config_dict.get('use_rocm_dmabuf', 'False'), re.I)),
        )
        end_time = orch.all.exec('date +"%a %b %e %H:%M"', print_console=False)
        if scan_dmesg:
            verify_dmesg_for_errors(orch.all, start_time, end_time, till_end_flag=True)
        if re.search('True', config_dict['verify_bw'], re.I):
            ibperf_lib.verify_expected_lat(
                lat_test, msg_size, ib_lat_dict[lat_test][msg_size], config_dict['expected_results']
            )

    log.debug('ib_lat_dict: %s', ib_lat_dict)
    update_test_result()


def test_build_ib_bw_perf_chart():
    globals.error_list = []
    ibperf_lib.generate_ibperf_bw_chart(ib_bw_dict, excel_file='ib_bw_pps_perf.xlsx')
    update_test_result()


def test_build_ib_lat_perf_chart():
    globals.error_list = []
    ibperf_lib.generate_ibperf_lat_chart(ib_lat_dict, excel_file='ib_lat_perf.xlsx')
    update_test_result()
