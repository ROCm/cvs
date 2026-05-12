'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/core/orchestrators/baremetal.py: BaremetalOrchestrator construction
# and command-dispatch surface (exec, exec_on_head, cleanup) used by the migrated
# rvs_cvs.py orch fixture. Mocks Pssh so tests run with no SSH.

import unittest
from unittest.mock import MagicMock, patch

from cvs.core.orchestrators.factory import OrchestratorConfig
from cvs.core.orchestrators.baremetal import BaremetalOrchestrator


def _make_orch_config():
    """Minimal OrchestratorConfig that satisfies BaremetalOrchestrator.__init__
    without touching disk or SSH."""
    return OrchestratorConfig(
        orchestrator="baremetal",
        node_dict={"10.0.0.1": {}, "10.0.0.2": {}},
        username="testuser",
        priv_key_file="/dev/null",
        password=None,
        head_node_dict={"mgmt_ip": "10.0.0.1"},
        container={},
    )


class TestBaremetalOrchestrator(unittest.TestCase):
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_init_constructs_pssh_handles(self, mock_pssh):
        BaremetalOrchestrator(MagicMock(), _make_orch_config())
        # __init__ creates two Pssh handles: self.head and self.all.
        self.assertEqual(mock_pssh.call_count, 2)

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_init_sets_orchestrator_type(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        self.assertEqual(orch.orchestrator_type, "baremetal")

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_init_picks_first_node_as_head(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        # _make_orch_config inserts 10.0.0.1 first.
        self.assertEqual(orch.head_node, "10.0.0.1")

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_exec_delegates_to_all_when_targeting_full_set(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch.all = MagicMock()
        orch.all.exec.return_value = {"10.0.0.1": "ok", "10.0.0.2": "ok"}
        result = orch.exec("ls", timeout=5)
        orch.all.exec.assert_called_once_with("ls", timeout=5)
        self.assertEqual(result, {"10.0.0.1": "ok", "10.0.0.2": "ok"})

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_exec_on_head_delegates_to_head_handle(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch.head = MagicMock()
        orch.head.exec.return_value = {"10.0.0.1": "ok"}
        result = orch.exec_on_head("hostname", timeout=10)
        orch.head.exec.assert_called_once_with("hostname", timeout=10)
        self.assertEqual(result, {"10.0.0.1": "ok"})

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_cleanup_returns_true(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        self.assertTrue(orch.cleanup(orch.hosts))


if __name__ == "__main__":
    unittest.main()
