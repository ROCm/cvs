"""Unit tests for cvs/lib/host_sanitize.py (CVS docker-mode P11)."""

import json
import unittest
from unittest.mock import MagicMock

from cvs.lib import host_sanitize


def _phdl(side_effects):
    m = MagicMock()
    m.exec.side_effect = side_effects
    m.hosts = ["node-01"]
    return m


class TestCapturePreState(unittest.TestCase):
    def test_basic_capture(self):
        phdl = _phdl(
            [
                {"node-01": "performance\n"},
                {"node-01": "auto\n"},
            ]
        )
        snap = host_sanitize.capture_pre_state(phdl)
        self.assertEqual(snap["node-01"]["cpu_governor"], "performance")
        self.assertEqual(snap["node-01"]["rocm_perflevel"], "auto")

    def test_unknown_when_command_fails(self):
        phdl = _phdl(
            [
                {"node-01": "unknown\n"},
                {"node-01": "unknown\n"},
            ]
        )
        snap = host_sanitize.capture_pre_state(phdl)
        self.assertEqual(snap["node-01"]["cpu_governor"], "unknown")
        self.assertEqual(snap["node-01"]["rocm_perflevel"], "unknown")


class TestApplySanitize(unittest.TestCase):
    def test_apply_runs_one_remote_command(self):
        phdl = _phdl([{"node-01": ""}])
        host_sanitize.apply_sanitize(phdl)
        # Should have issued exactly ONE exec
        self.assertEqual(phdl.exec.call_count, 1)
        cmd = phdl.exec.call_args.args[0]
        self.assertIn("scaling_governor", cmd)
        self.assertIn("performance", cmd)
        self.assertIn("drop_caches", cmd)
        self.assertIn("rocm-smi --setperflevel high", cmd)


class TestRestoreHost(unittest.TestCase):
    def test_restore_with_explicit_snapshot(self):
        snap = {"node-01": {"cpu_governor": "ondemand", "rocm_perflevel": "low"}}
        phdl = _phdl([{"node-01": ""}])
        out = host_sanitize.restore_host(phdl, snapshot=snap)
        self.assertEqual(out["node-01"]["cpu_governor"], "ondemand")
        self.assertEqual(out["node-01"]["rocm_perflevel"], "low")
        cmd = phdl.exec.call_args.args[0]
        self.assertIn("ondemand", cmd)
        self.assertIn("low", cmd)

    def test_restore_with_unknown_falls_back_to_safe_defaults(self):
        snap = {"node-01": {"cpu_governor": "unknown", "rocm_perflevel": ""}}
        phdl = _phdl([{"node-01": ""}])
        out = host_sanitize.restore_host(phdl, snapshot=snap)
        self.assertEqual(out["node-01"]["cpu_governor"], host_sanitize.SAFE_DEFAULT_GOVERNOR)
        self.assertEqual(out["node-01"]["cpu_governor"], "powersave")
        self.assertEqual(out["node-01"]["rocm_perflevel"], host_sanitize.SAFE_DEFAULT_PERFLEVEL)


class TestSnapshotIO(unittest.TestCase):
    def test_read_snapshot_parses_remote_file(self):
        snap = {"node-01": {"cpu_governor": "performance", "rocm_perflevel": "high"}}
        phdl = _phdl([{"node-01": json.dumps(snap)}])
        parsed = host_sanitize.read_snapshot(phdl)
        self.assertEqual(parsed["node-01"]["cpu_governor"], "performance")
        self.assertEqual(parsed["node-01"]["rocm_perflevel"], "high")

    def test_read_snapshot_missing_file_returns_empty_per_node(self):
        phdl = _phdl([{"node-01": "{}"}])
        parsed = host_sanitize.read_snapshot(phdl)
        self.assertEqual(parsed["node-01"], {})


class TestDiffState(unittest.TestCase):
    def test_no_diff(self):
        snap = {"a": {"x": 1, "y": 2}}
        self.assertEqual(host_sanitize.diff_state(snap, snap), {})

    def test_diff_field_changed(self):
        before = {"a": {"x": 1, "y": 2}}
        after = {"a": {"x": 9, "y": 2}}
        diff = host_sanitize.diff_state(before, after)
        self.assertEqual(diff, {"a": {"x": (1, 9)}})


if __name__ == "__main__":
    unittest.main()
