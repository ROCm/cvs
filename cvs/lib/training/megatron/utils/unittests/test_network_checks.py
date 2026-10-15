'''Unit tests for Megatron network preflight helpers.'''

import unittest
from unittest.mock import patch

from cvs.lib.training.megatron.utils.network_checks import backend_nic_link_speed_violations


class FakeHandle:
    def __init__(self, outputs_by_batch):
        self.outputs_by_batch = list(outputs_by_batch)
        self.commands = []

    def exec_cmd_list(self, cmd_list):
        self.commands.append(cmd_list)
        return self.outputs_by_batch.pop(0)


class FakeOrch:
    def __init__(self, outputs_by_batch):
        self.all = FakeHandle(outputs_by_batch)


def _backend_nics():
    return {
        "node0": {
            "rdma0": {"eth_device": "eno0"},
            "rdma1": {"eth_device": "eno1"},
        },
        "node1": {
            "rdma0": {"eth_device": "ens10"},
            "rdma1": {"eth_device": "ens11"},
        },
    }


class TestNetworkChecks(unittest.TestCase):
    @patch("cvs.lib.training.megatron.utils.network_checks.linux_utils.get_backend_rdma_nic_dict")
    def test_passes_when_all_backend_nics_report_expected_speed(self, mock_backend):
        mock_backend.return_value = _backend_nics()
        orch = FakeOrch(
            [
                {"node0": "400000\n", "node1": "400000\n"},
                {"node0": "400000\n", "node1": "400000\n"},
            ]
        )

        self.assertEqual(backend_nic_link_speed_violations(orch), [])

    @patch("cvs.lib.training.megatron.utils.network_checks.linux_utils.get_backend_rdma_nic_dict")
    def test_flags_backend_nic_below_expected_speed(self, mock_backend):
        mock_backend.return_value = _backend_nics()
        orch = FakeOrch(
            [
                {"node0": "400000\n", "node1": "200000\n"},
                {"node0": "400000\n", "node1": "400000\n"},
            ]
        )

        violations = backend_nic_link_speed_violations(orch)

        self.assertEqual(len(violations), 1)
        self.assertIn("node1: backend NIC ens10 (rdma0) link speed 200000 < expected 400000", violations[0])

    @patch("cvs.lib.training.megatron.utils.network_checks.linux_utils.get_backend_rdma_nic_dict")
    def test_flags_missing_speed(self, mock_backend):
        mock_backend.return_value = {"node0": {"rdma0": {"eth_device": "eno0"}}}
        orch = FakeOrch([{"node0": "__CVS_NIC_SPEED_MISSING__\n"}])

        violations = backend_nic_link_speed_violations(orch)

        self.assertEqual(len(violations), 1)
        self.assertIn("node0: backend NIC eno0 (rdma0) has no readable link speed", violations[0])

    @patch("cvs.lib.training.megatron.utils.network_checks.linux_utils.get_backend_rdma_nic_dict")
    def test_flags_non_numeric_speed(self, mock_backend):
        mock_backend.return_value = {"node0": {"rdma0": {"eth_device": "eno0"}}}
        orch = FakeOrch([{"node0": "unknown\n"}])

        violations = backend_nic_link_speed_violations(orch)

        self.assertEqual(len(violations), 1)
        self.assertIn("node0: backend NIC eno0 (rdma0) reported non-numeric link speed", violations[0])

    @patch("cvs.lib.training.megatron.utils.network_checks.linux_utils.get_backend_rdma_nic_dict")
    def test_flags_no_backend_nics(self, mock_backend):
        mock_backend.return_value = {}
        orch = FakeOrch([])

        violations = backend_nic_link_speed_violations(orch)

        self.assertEqual(violations, ["no backend RDMA NICs discovered; expected link speed 400000"])


if __name__ == "__main__":
    unittest.main()
