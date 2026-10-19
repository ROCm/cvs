'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import logging
import re


from cvs.core.scheduler import is_managed_compute
from cvs.lib import ibperf_lib
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *

from cvs.lib import globals

log = logging.getLogger(__name__)

ib_bw_dict = {}
ib_lat_dict = {}

rccl_res_dict = {}

RUNTIME_DEPS_CHECK = (
    'command -v numactl >/dev/null || echo MISSING:numactl; '
    'ldconfig -p | grep -q libibverbs.so.1 || echo MISSING:libibverbs; '
    'ldconfig -p | grep -q librdmacm.so.1 || echo MISSING:librdmacm'
)


def test_install_ib_perf(orch, config_dict):
    globals.error_list = []

    if re.search('true', config_dict['install_perf_package'], re.I):
        install_dir = config_dict['install_dir']
        log.info('Installing perftest to %s', install_dir)
        orch.head.exec(f'mkdir -p {install_dir}', print_console=False)
        install_system_packages = re.search('true', config_dict.get('install_system_packages', 'False'), re.I)
        if install_system_packages and is_managed_compute():
            log.warning('Managed Spur/Slurm step: skipping apt install of perftest dependencies')
            install_system_packages = False
        if install_system_packages:
            log.warning('install_system_packages is true: running apt on every node')
            orch.all.exec('sudo apt update -y', timeout=200, print_console=False)
            orch.all.exec(
                'sudo apt install -y git build-essential autoconf automake libtool pkg-config',
                timeout=200,
                print_console=False,
            )
            orch.all.exec(
                'sudo apt install -y libibverbs-dev librdmacm-dev ibverbs-providers rdma-core',
                timeout=200,
                print_console=False,
            )
            orch.all.exec('sudo apt install -y libibumad-dev', print_console=False)
            orch.all.exec('sudo apt install -y libpci-dev', print_console=False)
            orch.all.exec('sudo apt install -y numactl', print_console=False)
        else:
            for node, out in orch.all.exec(RUNTIME_DEPS_CHECK, print_console=False).items():
                missing = re.findall(r'MISSING:(\S+)', out)
                if missing:
                    fail_test(
                        f'perftest runtime dependencies missing on node {node}: {", ".join(missing)}; '
                        'install them or set install_system_packages to True'
                    )
            update_test_result()
        orch.head.exec(f'cd {install_dir}; git clone https://github.com/linux-rdma/perftest', print_console=False)
        orch.head.exec(f'cd {install_dir}/perftest; ./autogen.sh', timeout=100, print_console=False)
        rocm_path = ibperf_lib.detect_rocm_path(orch.head, config_dict.get('rocm_dir', '<changeme>'))
        ibperf_lib.configure_perftest(orch.head, install_dir, rocm_path)
        orch.head.exec(f'cd {install_dir}/perftest; make', timeout=100, print_console=False)
        orch.head.exec(f'cd {install_dir}/perftest; make install', timeout=100, print_console=False)

        out_dict = orch.all.exec(
            f'{install_dir}/perftest/ib_write_bw -h | grep -i rocm --color=never',
            print_console=False,
        )
        verified_nodes = 0
        for node in out_dict.keys():
            if not re.search('GPUDirect RDMA', out_dict[node], re.I):
                fail_test(
                    f'IB Perf package installation on node {node} failed, ib_write_bw not showing expected use_rocm output'
                )
            else:
                verified_nodes += 1
        if verified_nodes:
            log.info('Perftest installation verified on %d node(s)', verified_nodes)
        ibperf_lib.check_perftest_dmabuf(
            orch.all, f'{install_dir}/perftest/ib_write_bw', rocm_path=rocm_path, require_dmabuf=False
        )
    else:
        log.info('Skipping perftest installation (install_perf_package is not true)')
    update_test_result()
