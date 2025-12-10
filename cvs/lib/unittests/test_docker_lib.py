# cvs/lib/unittests/test_docker_lib.py
import unittest
from unittest.mock import MagicMock
import cvs.lib.docker_lib as docker_lib


class TestDockerLib(unittest.TestCase):
    def setUp(self):
        self.mock_phdl = MagicMock()

    def test_killall_docker_containers(self):
        docker_lib.killall_docker_containers(self.mock_phdl)
        self.mock_phdl.exec.assert_called_once_with('docker kill $(docker ps -q)')

    def test_kill_docker_container(self):
        container_name = 'test_container'
        docker_lib.kill_docker_container(self.mock_phdl, container_name)
        self.mock_phdl.exec.assert_called_once_with(f'docker kill {container_name}')

    def test_delete_all_containers_and_volumes(self):
        docker_lib.delete_all_containers_and_volumes(self.mock_phdl)
        self.mock_phdl.exec.assert_called_once_with('docker system prune --force', timeout=60 * 10)

    def test_delete_all_images(self):
        docker_lib.delete_all_images(self.mock_phdl)
        self.mock_phdl.exec.assert_called_once_with('docker rmi -f $(docker images -aq)')


if __name__ == '__main__':
    unittest.main()
