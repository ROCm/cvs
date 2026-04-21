"""Unit tests for cvs/lib/multinode.py (CVS docker-mode P13)."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from cvs.lib import multinode


class TestEphemeralKey(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._patcher = patch.object(
            multinode, "EPHEMERAL_KEY_DIR", self._tmp
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_generate_writes_priv_and_pub(self):
        priv, pub = multinode.generate_ephemeral_key("cvs-runner")
        self.assertTrue(os.path.exists(priv))
        self.assertTrue(os.path.exists(priv + ".pub"))
        self.assertTrue(pub.startswith("ssh-rsa "))
        self.assertIn("cvs-runner-ephemeral-cvs-runner", pub)

    def test_generate_overwrites_stale_key(self):
        priv1, _ = multinode.generate_ephemeral_key("cvs-runner")
        with open(priv1, "rb") as f:
            blob1 = f.read()
        priv2, _ = multinode.generate_ephemeral_key("cvs-runner")
        with open(priv2, "rb") as f:
            blob2 = f.read()
        self.assertNotEqual(blob1, blob2)

    def test_teardown_removes_key_files(self):
        priv, _ = multinode.generate_ephemeral_key("cvs-runner")
        multinode.teardown_multinode_ssh("cvs-runner")
        self.assertFalse(os.path.exists(priv))
        self.assertFalse(os.path.exists(priv + ".pub"))


class TestPushHelpers(unittest.TestCase):
    def test_push_authorized_key_uses_docker_exec(self):
        phdl = MagicMock()
        multinode.push_authorized_key(phdl, "cvs-runner", "ssh-rsa AAAA test@x")
        cmd = phdl.exec.call_args.args[0]
        self.assertIn("docker exec -i cvs-runner", cmd)
        self.assertIn("authorized_keys", cmd)
        # Key is base64-encoded, so the raw key text should NOT appear literally:
        self.assertNotIn("ssh-rsa AAAA test@x", cmd)
        self.assertIn("base64 -d", cmd)

    def test_push_ssh_config_includes_target_path_and_base64(self):
        phdl = MagicMock()
        multinode.push_ssh_config(phdl, "cvs-runner", ["node-01", "node-02"])
        cmd = phdl.exec.call_args.args[0]
        self.assertIn("/root/.ssh/config", cmd)
        self.assertIn("base64 -d", cmd)
        self.assertIn("docker exec -i cvs-runner", cmd)

    def test_push_private_key_base64_encodes(self):
        phdl = MagicMock()
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("-----BEGIN OPENSSH PRIVATE KEY-----\nAAAA\n-----END OPENSSH PRIVATE KEY-----\n")
            path = f.name
        try:
            multinode.push_private_key(phdl, "cvs-runner", path)
            cmd = phdl.exec.call_args.args[0]
            self.assertIn("base64 -d > /root/.ssh/id_rsa", cmd)
            self.assertIn("chmod 600", cmd)
        finally:
            os.unlink(path)


class TestVerifyContainerSshd(unittest.TestCase):
    def test_sshd_listening_per_node(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            "node-01": "  16: 00000000:08AE 00000000:0000 0A 00000000\n12345\n",
            "node-02": "",
        }
        result = multinode.verify_container_sshd(phdl, "cvs-runner")
        self.assertTrue(result["node-01"])
        self.assertFalse(result["node-02"])

    def test_pgrep_fallback_alone_is_enough(self):
        phdl = MagicMock()
        phdl.exec.return_value = {"node-01": "12345\n"}
        self.assertTrue(multinode.verify_container_sshd(phdl, "cvs-runner")["node-01"])


class TestVerifyContainerToContainerSsh(unittest.TestCase):
    def test_success_when_hostname_returned(self):
        phdl = MagicMock()
        phdl.exec.return_value = {"node-01": "node-02\n"}
        self.assertTrue(
            multinode.verify_container_to_container_ssh(
                phdl, "cvs-runner", "node-01", "node-02"
            )
        )

    def test_failure_when_connection_refused(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            "node-01": "ssh: connect to host node-02 port 2222: Connection refused\n"
        }
        self.assertFalse(
            multinode.verify_container_to_container_ssh(
                phdl, "cvs-runner", "node-01", "node-02"
            )
        )

    def test_failure_when_empty(self):
        phdl = MagicMock()
        phdl.exec.return_value = {"node-01": ""}
        self.assertFalse(
            multinode.verify_container_to_container_ssh(
                phdl, "cvs-runner", "node-01", "node-02"
            )
        )


if __name__ == "__main__":
    unittest.main()
