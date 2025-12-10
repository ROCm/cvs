'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest

import re
import time
import json

from netmiko import ConnectHandler


from cvs.lib.parallel_ssh_lib import *
from cvs.lib.utils_lib import *
from cvs.lib.linux_utils import (
    detect_distro,
    install_package,
    update_package_cache,
    map_packages
)

from cvs.lib import globals

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
def cluster_dict(cluster_file):
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)

    # Resolve path placeholders like {user-id} in cluster config
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)

    log.info(cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict):
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['rocblas']

    # Resolve path placeholders like {user-id}, {home-mount-dir}, etc.
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)

    log.info(config_dict)
    return config_dict


def parse_rocblas_fp32(
    out_dict,
    exp_dict,
):
    for node in out_dict.keys():
        match = re.search(
            r'N,T,4000,4000,4000,[0-9\.]+,[0-9\.]+,[0-9\.]+,[0-9\.]+,[0-9\.]+,\s+[0-9\.]+,\s+[0-9\.]+,\s+([0-9\.]+),\s+[0-9\.]+',
            out_dict[node],
        )
        fp32_gflops = float(match.group(1))
        if float(fp32_gflops) < float(exp_dict['fp32_gflops']):
            fail_test(
                f"Node {node} Actual GFLOPs for rocblas with FP32 {fp32_gflops} is lower than the expected GFLOPs {exp_dict['fp32_gflops']}"
            )


def parse_rocblas_bf16(out_dict, exp_dict):
    for node in out_dict.keys():
        match = re.search(
            r'N,T,1024,2048,512,[0-9\.]+,[0-9\.]+,[0-9\.]+,[0-9\.]+,[0-9\.]+,[0-9\.]+,[0-9\.]+,[0-9\.]+,[0-9\.]+,[0-9\.]+,[0-9\.]+,\s+[0-9\.]+,\s+[0-9\.]+,\s+([0-9\.]+),\s+[0-9\.]+',
            out_dict[node],
        )
        bf16_gflops = float(match.group(1))
        if float(bf16_gflops) < float(exp_dict['bf16_gflops']):
            fail_test(
                f"Node {node} Actual GFLOPs for rocblas with BF16 {bf16_gflops} is lower than the expected GFLOPs {exp_dict['bf16_gflops']}"
            )


def parse_rocblas_int8(out_dict, exp_dict):
    for node in out_dict.keys():
        match = re.search(
            r'N,T,1024,2048,512,[0-9\.]+,[0-9\.]+,[0-9\.]+,[0-9\.]+,[0-9\.]+,[0-9\.]+,[0-9\.]+,[0-9\.]+,[0-9\.]+,[0-9\.]+,[0-9\.]+,\s+[0-9\.]+,\s+[0-9\.]+,\s+([0-9\.]+),\s+[0-9\.]+',
            out_dict[node],
        )
        int8_gflops = float(match.group(1))
        if float(int8_gflops) < float(exp_dict['int8_gflops']):
            fail_test(
                f"Node {node} Actual GFLOPs for rocblas with INT8 {int8_gflops} is lower than the expected GFLOPs {exp_dict['int8_gflops']}"
            )


# Create connection to DUT, Switches and export for later use ..
@pytest.fixture(scope="module")
def phdl(cluster_dict):
    print(cluster_dict)
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
    return phdl


# Connect to first node to install packages in NFS mounted common directories
@pytest.fixture(scope="module")
def hdl(cluster_dict):
    node_list = list(cluster_dict['node_dict'].keys())
    hdl = ConnectHandler(
        ip=node_list[0],
        device_type='linux',
        username=cluster_dict['username'],
        use_keys=True,
        key_file=cluster_dict['priv_key_file'],
    )
    out = hdl.send_command('pwd')
    log.info(out)
    return hdl


@pytest.mark.dependency(name="init")
def test_rocblas_install(
    hdl,
    phdl,
    config_dict,
):
    globals.error_list = []
    log.info('Testcase install rocblas')
    log.info(f'{config_dict}')
    path = config_dict['path']
    package_path = config_dict['package_path']
    rocm_version = config_dict['rocm_version']
    phdl.exec('sudo rm -rf /home/venksrin/rocBLAS')
    time.sleep(5)
    git_url = config_dict['git_url']
    out_dict = phdl.exec('sudo apt update -y', timeout=200)
    out_dict = phdl.exec('sudo apt install -y libgtest-dev', timeout=200)
    out_dict = phdl.exec('sudo apt install -y cmake', timeout=200)
    out_dict = phdl.exec('sudo apt install -y gfortran', timeout=200)
    time.sleep(3)
    log.info(out_dict)
    log.info(f'Inputs - {package_path}, {path}, {git_url}')
    print('%%%%%%%%%%%%%%%%%')
    print('%%%%%%%%%%%%%%%%%')
    # hdl.write_channel('git init')
    # hdl.write_channel(f'cd {package_path};git clone {git_url};cd\r\r' )
    print("cmd = git init")
    out_dict = phdl.exec('git init')
    time.sleep(5)
    print(f'cmd = cd {package_path};git clone {git_url};cd')
    out_dict = phdl.exec(f'cd {package_path};git clone {git_url};cd', timeout=100)
    time.sleep(10)

    # hdl.write_channel(f'cd {package_path}/rocBLAS;git checkout rocm-{rocm_version};cd\r\r')
    print(f'cmd = cd {package_path}/rocBLAS;git checkout rocm-{rocm_version}')
    out_dict = phdl.exec(f'cd {package_path}/rocBLAS;git checkout rocm-{rocm_version}', timeout=60)
    time.sleep(10)
    # time.sleep(30)
    # hdl.write_channel(f'cd {package_path}/rocBLAS;./install.sh --clients-only --library-path /opt/rocm-{rocm_version}\r\r')
    print(f'cmd = cd {package_path}/rocBLAS;./install.sh --clients-only --library-path /opt/rocm')
    out_dict = phdl.exec(f'cd {package_path}/rocBLAS;./install.sh --clients-only --library-path /opt/rocm', timeout=700)
    out_dict = phdl.exec(f'ls -l {path}')
    for node in out_dict.keys():
        if not re.search('rocblas-bench', out_dict[node], re.I):
            fail_test(f'rocblas installation failed, rocblas-bench not found, aborting !! {node}')
    update_test_result()


@pytest.mark.dependency(depends=["init"])
def test_rocblas_fp32_benchmark(
    phdl,
    config_dict,
):
    globals.error_list = []
    log.info('Testcase Run rocblas FP32 benchmark')
    path = config_dict['path']
    out_dict = phdl.exec(
        f'sudo {path}/rocblas-bench -f gemm -r s -m 4000 -n 4000 -k 4000 --lda 4000 --ldb 4000 --ldc 4000 --transposeA N --transposeB T',
        timeout=(60 * 5),
    )
    print_test_output(log, out_dict)
    parse_rocblas_fp32(out_dict, config_dict['results'])
    scan_test_results(out_dict)
    update_test_result()


@pytest.mark.dependency(depends=["init"])
def test_rocblas_bf16_benchmark(
    phdl,
    config_dict,
):
    globals.error_list = []
    log.info('Testcase Run rocblas BF16 benchmark')
    path = config_dict['path']
    out_dict = phdl.exec(
        f'sudo {path}/rocblas-bench -f gemm_strided_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type h --lda 1024 --stride_a 4096 --b_type h --ldb 2048 --stride_b 4096 --c_type s --ldc 1024 --stride_c 2097152 --d_type s --ldd 1024 --stride_d 2097152 --compute_type s --alpha 1.1 --beta 1 --batch_count 5',
        timeout=(60 * 5),
    )
    print_test_output(log, out_dict)
    parse_rocblas_bf16(out_dict, config_dict['results'])
    scan_test_results(out_dict)
    update_test_result()


@pytest.mark.dependency(depends=["init"])
def test_rocblas_int8_benchmark(
    phdl,
    config_dict,
):
    globals.error_list = []
    log.info('Testcase Run rocblas INT8 benchmark')
    path = config_dict['path']
    out_dict = phdl.exec(
        f'sudo {path}/rocblas-bench -f gemm_strided_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type i8_r --lda 1024 --stride_a 4096 --b_type i8_r --ldb 2048 --stride_b 4096 --c_type i32_r --ldc 1024 --stride_c 2097152 --d_type i32_r --ldd 1024 --stride_d 2097152 --compute_type i32_r --alpha 1.1 --beta 1 --batch_count 5',
        timeout=(60 * 5),
    )
    print_test_output(log, out_dict)
    parse_rocblas_int8(out_dict, config_dict['results'])
    scan_test_results(out_dict)
    update_test_result()
