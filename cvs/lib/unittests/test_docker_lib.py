# cvs/lib/unittests/test_docker_lib.py
import os
import unittest
from unittest.mock import MagicMock, patch

import cvs.lib.docker_lib as docker_lib
from cvs.unittests.support import FAKE_CRED


class TestDockerLib(unittest.TestCase):
    def setUp(self):
        self.mock_phdl = MagicMock()

    @patch("cvs.lib.docker_lib.time.sleep")
    @patch("cvs.lib.docker_lib.log")
    def test_launch_docker_container_logs_masked_command(self, mock_log, _sleep):
        self.mock_phdl.exec.side_effect = [{"n1": "ok"}, {"n1": "c1"}]
        docker_lib.launch_docker_container(
            self.mock_phdl, "c1", "img", env_dict={"HF_TOKEN": FAKE_CRED}, ulimit_nofile="1:1"
        )
        self.assertIn(f"-e HF_TOKEN={FAKE_CRED}", self.mock_phdl.exec.call_args_list[0][0][0])
        logged = [c.args[0] for c in mock_log.info.call_args_list if str(c.args[0]).startswith("cmd = ")]
        self.assertIn("-e HF_TOKEN=<redacted>", logged[0])
        self.assertNotIn(FAKE_CRED, logged[0])

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

    @patch.dict(os.environ, {"CVS_SKIP_DOCKER_SYSTEM_PRUNE": "1"}, clear=False)
    def test_delete_all_containers_skips_prune_when_env_set(self):
        docker_lib.delete_all_containers_and_volumes(self.mock_phdl)
        self.mock_phdl.exec.assert_not_called()

    def test_delete_all_images(self):
        docker_lib.delete_all_images(self.mock_phdl)
        self.mock_phdl.exec.assert_called_once_with('docker rmi -f $(docker images -aq)')

    def test_nodes_missing_docker_image_none_missing(self):
        self.mock_phdl.exec.return_value = {"n1": "IMG_OK\n", "n2": "IMG_OK\n"}
        self.assertEqual(docker_lib.nodes_missing_docker_image(self.mock_phdl, "repo/img:tag"), [])
        self.mock_phdl.exec.assert_called_once()
        cmd = self.mock_phdl.exec.call_args[0][0]
        self.assertIn("docker image inspect", cmd)
        self.assertIn("repo/img:tag", cmd)

    def test_nodes_missing_docker_image_some_missing(self):
        self.mock_phdl.exec.return_value = {"n1": "IMG_OK\n", "n2": "IMG_MISSING\n"}
        self.assertEqual(docker_lib.nodes_missing_docker_image(self.mock_phdl, "x:y"), ["n2"])


if __name__ == '__main__':
    unittest.main()
