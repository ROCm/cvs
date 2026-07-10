'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Shared test steps for all Megatron training suites (single-node and distributed).
Import via `from ._shared import *` in each test file.
'''

from cvs.lib.utils_lib import *
from cvs.lib import docker_lib
from cvs.lib import globals

log = globals.log

__all__ = [
    'test_cleanup_stale_containers',
    'test_launch_megatron_containers',
]


def test_cleanup_stale_containers(phdl, training_dict):
    container_name = training_dict['container_name']
    docker_lib.kill_docker_container(phdl, container_name)
    docker_lib.delete_all_containers_and_volumes(phdl)


def test_launch_megatron_containers(phdl, training_dict, lifecycle):
    log.info('Testcase launch Megatron containers')
    globals.error_list = []
    container_name = training_dict['container_name']
    docker_lib.launch_docker_container(
        phdl,
        container_name,
        training_dict['container_image'],
        training_dict['container_config']['device_list'],
        training_dict['container_config']['volume_dict'],
        shm_size='128G',
        timeout=60 * 20,
    )
    if globals.error_list:
        lifecycle.failed = True
    update_test_result()
