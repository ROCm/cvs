'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import re
import json
import time

from cvs.lib import globals
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *

log = globals.log


def get_running_docker_containers(phdl):
    cont_dict = {}
    out_dict = phdl.exec('docker ps --format="{{json .}}"')
    for node in out_dict.keys():
        cont_dict[node] = {}
        for line in out_dict[node].split("\n"):
            line_dict = json.loads(line)
            cont_dict[node][line_dict['Names']] = line_dict
    return cont_dict


def check_if_docker_client_running(phdl):
    out_dict = phdl.exec('docker ps')
    for node in out_dict.keys():
        if not re.search('CONTAINER', out_dict[node], re.I):
            fail_test(f'Docker Not running on node {node} .. pls check')
            return False
    return True


def killall_docker_containers(phdl):
    phdl.exec('docker kill $(docker ps -q)')


def kill_docker_container(phdl, container_name):
    phdl.exec(f'docker kill {container_name}')


def delete_all_containers_and_volumes(phdl):
    # out_dict = phdl.exec('docker rm -vf $(docker ps -aq)')
    print('Deleting all containers and volumes')
    # out_dict = phdl.exec('docker system prune -a --volumes --force', timeout=60*10)
    phdl.exec('docker system prune --force', timeout=60 * 10)


def delete_all_images(phdl):
    phdl.exec('docker rmi -f $(docker images -aq)')


def old_install_docker_on_ubuntu(phdl):
    phdl.exec('sudo apt-get -y update')
    phdl.exec('sudo apt-get -y install ca-certificates curl')
    phdl.exec('sudo install -m 0755 -d /etc/apt/keyrings')
    phdl.exec(
        'sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
         -o /etc/apt/keyrings/docker.asc'
    )
    phdl.exec('sudo chmod a+r /etc/apt/keyrings/docker.asc')

    # Add the repository to Apt sources:
    cmd = '''echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
        https://download.docker.com/linux/ubuntu $(. /etc/os-release && \
        echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null'''
    phdl.exec(cmd, timeout=100)
    phdl.exec('sudo apt-get -y update')
    time.sleep(3)
    out_dict = phdl.exec(
        'sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
        docker-buildx-plugin docker-compose-plugin',
        timeout=500,
    )
    for node in out_dict.keys():
        if re.search('error|failed|missing', out_dict[node], re.I):
            fail_test(f'Failed to install docker packages on node {node}, please check')


def install_docker_on_ubuntu(phdl):
    phdl.exec('sudo rm /etc/apt/keyrings/docker.gpg')
    phdl.exec('sudo rm /etc/apt/sources.list.d/docker.list')
    phdl.exec('sudo apt-get -y update')
    phdl.exec('sudo apt install -y apt-transport-https ca-certificates curl software-properties-common')
    phdl.exec('curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -')
    phdl.exec('sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"')
    phdl.exec('apt-cache policy docker-ce')
    phdl.exec('sudo apt install -y docker-ce')
    time.sleep(3)
    phdl.exec('sudo systemctl start docker')
    time.sleep(3)
    phdl.exec('sudo systemctl status docker')


def launch_docker_container(
    phdl,
    container_name,
    image,
    device_list=[],
    volume_dict={},
    env_dict={},
    network='host',
    shm_size='64G',
    timeout=60 * 10,
):
    cmd = f'docker run -d --network {network} --ipc {network} \
            --cap-add=IPC_LOCK --security-opt seccomp=unconfined --privileged '
    for device in device_list:
        cmd = cmd + f' --device {device} '
    for src_vol in volume_dict.keys():
        cmd = cmd + f' -v {src_vol}:{volume_dict[src_vol]}'
    for env_var in env_dict.keys():
        cmd = cmd + f' -e {env_var}={env_dict[env_var]}'
    cmd = cmd + f' --shm-size {shm_size} --name {container_name} {image} '
    cmd = cmd + 'tail -f /dev/null'

    print(f'cmd = {cmd}')
    out_dict = phdl.exec(cmd, timeout=timeout)
    time.sleep(15)

    # out_dict = phdl.exec( f'docker start {container_name}', timeout=timeout)
    # time.sleep(3)

    # out_dict = phdl.exec( f'docker exec -it {container_name} /bin/bash', timeout=timeout)
    # time.sleep(3)
    for node in out_dict.keys():
        if re.search('error|fail', out_dict[node], re.I):
            fail_test('Failed to launch containers, please check logs')
            return

    out_dict = phdl.exec('docker ps')
    for node in out_dict.keys():
        if not re.search(f'{container_name}', out_dict[node]):
            time.sleep(60)
            out_dict_n = phdl.exec('docker ps')
            for node in out_dict_n.keys():
                if not re.search(f'{container_name}', out_dict_n[node], re.I):
                    out_dict = phdl.exec(cmd, timeout=timeout)
                    time.sleep(3)
                    fail_test(f'Failed to launch container {container_name} on node {node}')
