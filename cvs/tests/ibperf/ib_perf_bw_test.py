'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import logging
import re
import json
import os


from cvs.lib import ibperf_lib
from cvs.lib.rdma_gid_lib import DEFAULT_GID_TYPE

from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *

from cvs.lib import globals

log = logging.getLogger(__name__)

ib_bw_dict = {}
ib_lat_dict = {}
ib_dmabuf_dict = {}

rccl_res_dict = {}


def pytest_generate_tests(metafunc):
    """Parametrize bandwidth and latency test names from the ibperf config."""
    wants_bw = 'bw_test' in metafunc.fixturenames
    wants_lat = 'lat_test' in metafunc.fixturenames
    if not (wants_bw or wants_lat):
        return

    ibperf_config = {}
    config_file = metafunc.config.getoption('config_file')
    if config_file and os.path.exists(config_file):
        with open(config_file) as fp:
            ibperf_config = json.load(fp).get('ibperf', {})

    if wants_bw:
        metafunc.parametrize('bw_test', ibperf_lib.configured_bw_tests(ibperf_config))
    if wants_lat:
        metafunc.parametrize('lat_test', ibperf_lib.configured_lat_tests(ibperf_config))


def _fail_unmapped_gpus(orch):
    """Map GPUs to RDMA NICs on orch.all, failing each card left without one before perftest starts."""
    gpu_nic_dict = linux_utils.get_gpu_nic_mapping_dict(orch.all, use_sudo=False)
    for node, card_dict in gpu_nic_dict.items():
        for card, nic_info in card_dict.items():
            if not nic_info.get('rdma_dev'):
                fail_test(
                    f'No active backend RDMA NIC in the PCI domain of {card} '
                    f'(GPU {nic_info.get("gpu_bdf")}) on node {node}; not running perftest over a NIC outside it'
                )
    return gpu_nic_dict


def can_read_dmesg(orch):
    if all(get_passwordless_sudo_status(orch.all).values()):
        return True
    log.warning('SKIPPING the dmesg error check: no passwordless sudo to read dmesg on one or more nodes')
    return False


def test_ib_bw_perf(orch, bw_test, config_dict, request):
    globals.error_list = []
    ib_bw_dict[bw_test] = {}
    scan_dmesg = can_read_dmesg(orch)
    rocm_path = ibperf_lib.detect_rocm_path(orch.all, config_dict.get('rocm_dir', ''))
    app_path = f'{config_dict["install_dir"]}/perftest/bin'
    use_dmabuf = ibperf_lib.check_perftest_dmabuf(
        orch.all,
        f'{app_path}/{bw_test}',
        rocm_path=rocm_path,
        require_dmabuf=ibperf_lib.is_dmabuf_required(config_dict),
    )
    ib_dmabuf_dict[bw_test] = use_dmabuf
    request.node.user_properties.append(("dmabuf", "on" if use_dmabuf else "off"))
    update_test_result()

    gpu_nic_dict = _fail_unmapped_gpus(orch)
    update_test_result()
    gpu_numa_dict = linux_utils.get_gpu_numa_dict(orch.all, use_sudo=False)
    hip_dev_dict = ibperf_lib.get_hip_device_dict(orch.all, gpu_nic_dict)

    bck_nic_dict_lshw = linux_utils.get_backend_nic_dict(orch.all, use_sudo=False)
    rdma_nic_dict = linux_utils.get_active_rdma_nic_dict(orch.all)

    bck_nic_dict = {}
    for node in rdma_nic_dict.keys():
        bck_nic_dict[node] = {}
        for rdma_dev in rdma_nic_dict[node].keys():
            if rdma_nic_dict[node][rdma_dev]['eth_device'] in bck_nic_dict_lshw[node]:
                bck_nic_dict[node][rdma_dev] = rdma_nic_dict[node][rdma_dev]

    gid_index = ibperf_lib.resolve_gid_index(
        orch.all,
        ibperf_lib.get_ibperf_rdma_devices(gpu_nic_dict, bck_nic_dict),
        config_dict.get('gid_index'),
        config_dict.get('gid_type', DEFAULT_GID_TYPE),
    )
    update_test_result()
    for msg_size in config_dict['msg_size_list']:
        ib_bw_dict[bw_test][msg_size] = {}
        for qp_count in config_dict['qp_count_list']:
            start_time = orch.all.exec('date +"%a %b %e %H:%M"', print_console=False)
            if scan_dmesg:
                orch.all.exec(
                    f'echo "Starting Test {bw_test} for {msg_size} and QP count {qp_count}" | sudo -n tee /dev/kmsg',
                    print_console=False,
                )
            ib_bw_dict[bw_test][msg_size][qp_count] = ibperf_lib.run_ib_perf_bw_test(
                orch.all,
                bw_test,
                gpu_numa_dict,
                gpu_nic_dict,
                bck_nic_dict,
                app_path,
                msg_size,
                gid_index,
                qp_count,
                int(config_dict['port_no']),
                int(config_dict['duration']),
                rocm_path=rocm_path,
                use_dmabuf=use_dmabuf,
                hip_dev_dict=hip_dev_dict,
            )
            end_time = orch.all.exec('date +"%a %b %e %H:%M"', print_console=False)
            if scan_dmesg:
                verify_dmesg_for_errors(orch.all, start_time, end_time, till_end_flag=True)
            if re.search('True', config_dict.get('verify_bw', ''), re.I):
                ibperf_lib.verify_expected_bw(
                    bw_test,
                    msg_size,
                    qp_count,
                    ib_bw_dict[bw_test][msg_size][qp_count],
                    config_dict.get('expected_results', {}),
                )

    log.debug('ib_bw_dict: %s', ib_bw_dict)
    update_test_result()


def test_ib_lat_perf(orch, lat_test, config_dict, request):
    globals.error_list = []
    ib_lat_dict[lat_test] = {}
    scan_dmesg = can_read_dmesg(orch)
    rocm_path = ibperf_lib.detect_rocm_path(orch.all, config_dict.get('rocm_dir', ''))
    app_path = f'{config_dict["install_dir"]}/perftest/bin'
    use_dmabuf = ibperf_lib.check_perftest_dmabuf(
        orch.all,
        f'{app_path}/{lat_test}',
        rocm_path=rocm_path,
        require_dmabuf=ibperf_lib.is_dmabuf_required(config_dict),
    )
    ib_dmabuf_dict[lat_test] = use_dmabuf
    request.node.user_properties.append(("dmabuf", "on" if use_dmabuf else "off"))
    update_test_result()

    gpu_nic_dict = _fail_unmapped_gpus(orch)
    update_test_result()
    gpu_numa_dict = linux_utils.get_gpu_numa_dict(orch.all, use_sudo=False)
    hip_dev_dict = ibperf_lib.get_hip_device_dict(orch.all, gpu_nic_dict)

    bck_nic_dict_lshw = linux_utils.get_backend_nic_dict(orch.all, use_sudo=False)
    rdma_nic_dict = linux_utils.get_active_rdma_nic_dict(orch.all)

    bck_nic_dict = {}
    for node in rdma_nic_dict.keys():
        bck_nic_dict[node] = {}
        for rdma_dev in rdma_nic_dict[node].keys():
            if rdma_nic_dict[node][rdma_dev]['eth_device'] in bck_nic_dict_lshw[node]:
                bck_nic_dict[node][rdma_dev] = rdma_nic_dict[node][rdma_dev]

    gid_index = ibperf_lib.resolve_gid_index(
        orch.all,
        ibperf_lib.get_ibperf_rdma_devices(gpu_nic_dict, bck_nic_dict),
        config_dict.get('gid_index'),
        config_dict.get('gid_type', DEFAULT_GID_TYPE),
    )
    update_test_result()
    for msg_size in config_dict['msg_size_list']:
        ib_lat_dict[lat_test][msg_size] = {}
        start_time = orch.all.exec('date +"%a %b %e %H:%M"', print_console=False)
        if scan_dmesg:
            orch.all.exec(
                f'echo "Starting Test {lat_test} for {msg_size}" | sudo -n tee /dev/kmsg',
                print_console=False,
            )
        ib_lat_dict[lat_test][msg_size] = ibperf_lib.run_ib_perf_lat_test(
            orch.all,
            lat_test,
            gpu_numa_dict,
            gpu_nic_dict,
            bck_nic_dict,
            app_path,
            msg_size,
            gid_index,
            int(config_dict['port_no']),
            rocm_path=rocm_path,
            use_dmabuf=use_dmabuf,
            hip_dev_dict=hip_dev_dict,
        )
        end_time = orch.all.exec('date +"%a %b %e %H:%M"', print_console=False)
        if scan_dmesg:
            verify_dmesg_for_errors(orch.all, start_time, end_time, till_end_flag=True)
        if re.search('True', config_dict.get('verify_bw', ''), re.I):
            ibperf_lib.verify_expected_lat(
                lat_test, msg_size, ib_lat_dict[lat_test][msg_size], config_dict.get('expected_results', {})
            )

    log.debug('ib_lat_dict: %s', ib_lat_dict)
    update_test_result()


def test_build_ib_bw_perf_chart():
    globals.error_list = []
    ibperf_lib.generate_ibperf_bw_chart(ib_bw_dict, excel_file='ib_bw_pps_perf.xlsx', dmabuf_dict=ib_dmabuf_dict)
    update_test_result()


def test_build_ib_lat_perf_chart():
    globals.error_list = []
    try:
        ibperf_lib.generate_ibperf_lat_chart(ib_lat_dict, excel_file='ib_lat_perf.xlsx', dmabuf_dict=ib_dmabuf_dict)
    except ValueError as exc:
        fail_test(str(exc))
    update_test_result()
