"""Unit tests for cvs/lib/driver_recovery.py (CVS docker-mode P8)."""

import unittest
from unittest.mock import MagicMock, patch

from cvs.lib import driver_recovery


def _phdl_with_responses(*responses):
    """Build a Pssh mock whose .exec() returns each response in turn."""
    m = MagicMock()
    m.exec.side_effect = responses
    return m


class TestIsDriverLive(unittest.TestCase):
    def test_single_node_live(self):
        phdl = MagicMock()
        phdl.exec.return_value = {"node-01": "live\n"}
        self.assertEqual(driver_recovery.is_driver_live(phdl), {"node-01": True})

    def test_single_node_dead(self):
        phdl = MagicMock()
        phdl.exec.return_value = {"node-01": "NOT_LIVE"}
        self.assertEqual(driver_recovery.is_driver_live(phdl), {"node-01": False})

    def test_dead_when_kfd_missing(self):
        # The shell `test -e /dev/kfd && cat ... || echo NOT_LIVE` would
        # short-circuit to NOT_LIVE if /dev/kfd is missing.
        phdl = MagicMock()
        phdl.exec.return_value = {"node-01": "NOT_LIVE"}
        self.assertFalse(driver_recovery.is_driver_live(phdl)["node-01"])

    def test_dead_when_initstate_not_live(self):
        phdl = MagicMock()
        phdl.exec.return_value = {"node-01": "starting\n"}
        self.assertFalse(driver_recovery.is_driver_live(phdl)["node-01"])

    def test_multi_node_mixed(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            "node-01": "live\n",
            "node-02": "NOT_LIVE",
            "node-03": "live\n",
        }
        self.assertEqual(
            driver_recovery.is_driver_live(phdl),
            {"node-01": True, "node-02": False, "node-03": True},
        )


class TestVerifyOrRecoverDriver(unittest.TestCase):
    @patch.object(driver_recovery.time, "sleep", lambda *_: None)
    def test_all_live_is_noop(self):
        phdl = _phdl_with_responses({"node-01": "live\n", "node-02": "live\n"})
        result = driver_recovery.verify_or_recover_driver(phdl)
        # Only one exec call (the initial check); no modprobe sent.
        self.assertEqual(phdl.exec.call_count, 1)
        for node in ("node-01", "node-02"):
            self.assertTrue(result[node]["before"])
            self.assertFalse(result[node]["attempted"])
            self.assertTrue(result[node]["after"])

    @patch.object(driver_recovery.time, "sleep", lambda *_: None)
    def test_recovery_succeeds(self):
        phdl = _phdl_with_responses(
            {"node-01": "NOT_LIVE"},  # initial check
            {"node-01": "(modprobe output)"},  # modprobe call
            {"node-01": "live\n"},  # post-modprobe re-check
        )
        result = driver_recovery.verify_or_recover_driver(phdl)
        self.assertEqual(phdl.exec.call_count, 3)
        self.assertFalse(result["node-01"]["before"])
        self.assertTrue(result["node-01"]["attempted"])
        self.assertTrue(result["node-01"]["after"])
        # Verify modprobe command shape
        modprobe_call = phdl.exec.call_args_list[1]
        self.assertIn("modprobe amdgpu", modprobe_call[0][0])

    @patch.object(driver_recovery.time, "sleep", lambda *_: None)
    def test_recovery_fails_raises(self):
        phdl = _phdl_with_responses(
            {"node-01": "NOT_LIVE"},
            {"node-01": "modprobe: failed"},
            {"node-01": "NOT_LIVE"},  # still dead after modprobe
        )
        with self.assertRaises(RuntimeError) as ctx:
            driver_recovery.verify_or_recover_driver(phdl)
        msg = str(ctx.exception)
        self.assertIn("still not live", msg)
        self.assertIn("node-01", msg)

    @patch.object(driver_recovery.time, "sleep", lambda *_: None)
    def test_partial_recovery_one_dead_others_live(self):
        phdl = _phdl_with_responses(
            {"node-01": "live\n", "node-02": "NOT_LIVE"},  # initial
            {"node-01": "(noop)", "node-02": "(modprobe ok)"},  # modprobe to all
            {"node-01": "live\n", "node-02": "live\n"},  # post-recheck
        )
        result = driver_recovery.verify_or_recover_driver(phdl)
        self.assertTrue(result["node-01"]["before"])
        self.assertFalse(result["node-01"]["attempted"])
        self.assertFalse(result["node-02"]["before"])
        self.assertTrue(result["node-02"]["attempted"])
        self.assertTrue(result["node-02"]["after"])


class TestRestartContainer(unittest.TestCase):
    @patch.object(driver_recovery.time, "sleep", lambda *_: None)
    def test_restart_success(self):
        phdl = _phdl_with_responses(
            {"node-01": "cvs-runner\n"},  # docker restart output
            {"node-01": "cvs-runner"},    # docker ps output (still running)
        )
        out = driver_recovery.restart_container(phdl, "cvs-runner")
        self.assertEqual(out, {"node-01": True})
        # Verify the restart command shape
        restart_call = phdl.exec.call_args_list[0]
        self.assertIn("docker restart cvs-runner", restart_call[0][0])

    @patch.object(driver_recovery.time, "sleep", lambda *_: None)
    def test_restart_failure_returns_false(self):
        phdl = _phdl_with_responses(
            {"node-01": "Error: No such container"},
            {"node-01": ""},  # not in docker ps output
        )
        out = driver_recovery.restart_container(phdl, "cvs-runner")
        self.assertEqual(out, {"node-01": False})


if __name__ == "__main__":
    unittest.main()
