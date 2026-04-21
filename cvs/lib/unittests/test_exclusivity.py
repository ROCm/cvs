"""Unit tests for cvs/lib/exclusivity.py (CVS docker-mode P10)."""

import unittest
from unittest.mock import MagicMock

from cvs.lib import exclusivity
from cvs.lib.runtime_config import RuntimeConfig


def _phdl(per_call_returns):
    """Build a mock Pssh whose .exec() returns each item in turn."""
    m = MagicMock()
    m.exec.side_effect = per_call_returns
    return m


def _cfg(allowed=None, container_name="cvs-runner", excl="warn"):
    return RuntimeConfig(
        mode="docker",
        image="x",
        container_name=container_name,
        allowed_containers=list(allowed or []),
        exclusivity=excl,
    )


class TestStrayContainers(unittest.TestCase):
    def test_clean_node_no_containers(self):
        phdl = _phdl([{"node-01": ""}])
        result = exclusivity.check_stray_containers(phdl, _cfg())
        self.assertEqual(result, {"node-01": []})

    def test_only_cvs_runner_is_clean(self):
        phdl = _phdl([{"node-01": "cvs-runner\n"}])
        result = exclusivity.check_stray_containers(phdl, _cfg())
        self.assertEqual(result["node-01"], [])

    def test_default_allowlist_node_exporter(self):
        # node-exporter is in the built-in allowlist (Conductor infra container)
        phdl = _phdl([{"node-01": "node-exporter.service\ncvs-runner\n"}])
        result = exclusivity.check_stray_containers(phdl, _cfg())
        self.assertEqual(result["node-01"], [])

    def test_stray_container_detected(self):
        phdl = _phdl([{"node-01": "cvs-runner\nfoo\nbar\n"}])
        result = exclusivity.check_stray_containers(phdl, _cfg())
        self.assertEqual(sorted(result["node-01"]), ["bar", "foo"])

    def test_user_allowlist(self):
        phdl = _phdl([{"node-01": "cvs-runner\nmy-monitor\n"}])
        result = exclusivity.check_stray_containers(
            phdl, _cfg(allowed=["my-monitor"])
        )
        self.assertEqual(result["node-01"], [])

    def test_substring_match(self):
        # "node-exporter" matches "node-exporter.service"
        phdl = _phdl([{"node-01": "node-exporter.service\n"}])
        result = exclusivity.check_stray_containers(phdl, _cfg())
        self.assertEqual(result["node-01"], [])


class TestKfdHolders(unittest.TestCase):
    def test_no_holders(self):
        phdl = _phdl([{"node-01": ""}])
        self.assertEqual(exclusivity.check_kfd_holders(phdl)["node-01"], [])

    def test_holders_returned(self):
        phdl = _phdl([{"node-01": "1234\n5678\n"}])
        self.assertEqual(
            exclusivity.check_kfd_holders(phdl)["node-01"], ["1234", "5678"]
        )


class TestReservedPorts(unittest.TestCase):
    def test_no_listener(self):
        phdl = _phdl([{"node-01": ""}])
        self.assertEqual(exclusivity.check_reserved_ports(phdl)["node-01"], [])

    def test_2222_listener(self):
        phdl = _phdl(
            [{"node-01": "LISTEN 0 128 0.0.0.0:2222 0.0.0.0:* users:((sshd,pid=1,fd=3))"}]
        )
        self.assertEqual(exclusivity.check_reserved_ports(phdl)["node-01"], [2222])

    def test_only_substring_match_does_not_false_positive(self):
        # ":12222" should NOT match ":2222" thanks to \b boundary
        phdl = _phdl([{"node-01": "LISTEN 0 128 0.0.0.0:12222 0.0.0.0:*"}])
        self.assertEqual(exclusivity.check_reserved_ports(phdl)["node-01"], [])


class TestSummaryHelpers(unittest.TestCase):
    def test_violation_count_zero(self):
        summary = {
            "stray_containers": {"a": []},
            "kfd_holders": {"a": []},
            "reserved_ports": {"a": []},
        }
        self.assertEqual(exclusivity.violation_count(summary), 0)

    def test_violation_count_mixed(self):
        summary = {
            "stray_containers": {"a": ["x", "y"], "b": ["z"]},
            "kfd_holders": {"a": ["1"]},
            "reserved_ports": {"a": [2222]},
        }
        # 3 + 1 + 1 = 5
        self.assertEqual(exclusivity.violation_count(summary), 5)

    def test_render_violations_empty(self):
        summary = {"stray_containers": {}, "kfd_holders": {}, "reserved_ports": {}}
        self.assertEqual(exclusivity.render_violations(summary), "(no violations)")

    def test_render_violations_format(self):
        summary = {
            "stray_containers": {"node-01": ["foo"]},
            "kfd_holders": {"node-01": ["999"]},
            "reserved_ports": {"node-01": [2222]},
        }
        rendered = exclusivity.render_violations(summary)
        self.assertIn("node-01: stray container 'foo'", rendered)
        self.assertIn("node-01: PID 999 holds /dev/kfd", rendered)
        self.assertIn("node-01: port 2222 already listening", rendered)


if __name__ == "__main__":
    unittest.main()
