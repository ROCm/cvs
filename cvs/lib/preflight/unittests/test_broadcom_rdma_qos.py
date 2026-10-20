"""Unit tests for the Broadcom Thor NIC RDMA QoS preflight check."""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.broadcom_rdma_qos import BroadcomRdmaQosCheck

_EXPECTED_DSCP_TABLE = [
    {'priority': 7, 'sel': 5, 'dscp': 48},
    {'priority': 3, 'sel': 5, 'dscp': 26},
    {'priority': 3, 'sel': 3, 'udp_dccp': 4791},
]


def _make_output(
    pci="0000:06:00.0",
    prio_map="0:0 1:0 2:0 3:1 4:0 5:0 6:0 7:2",
    tc_bw="50% 50% 0%",
    tsa_map="0:ets 1:ets 2:strict",
    pfc_enabled="3",
    support_rdma="True",
    perf_profile="RoCE",
    pcie_ro="Enabled",
    sriov=None,
):
    """Build realistic command output matching the check's awk-parsed format."""
    lines = [
        "VENDOR:BROADCOM",
        f"DEV_BEGIN:{pci}",
        f"PRIO_MAP:{prio_map}",
        f"TC_BW:{tc_bw}",
        f"TSA_MAP:{tsa_map}",
        f"PFC_ENABLED:{pfc_enabled}",
        "APP_DSCP:7:5:48",
        "APP_DSCP:3:5:26",
        "APP_UDP:3:3:4791",
        f"SUPPORT_RDMA:{support_rdma}",
        f"PERF_PROFILE:{perf_profile}",
        f"PCIE_RO:{pcie_ro}",
    ]
    if sriov is not None:
        lines.append(f"SRIOV:{sriov}")
    lines.append(f"DEV_END:{pci}")
    return "\n".join(lines) + "\n"


def _default_checker(orch):
    return BroadcomRdmaQosCheck(
        orch,
        expected_prio_map='0:0 1:0 2:0 3:1 4:0 5:0 6:0 7:2',
        expected_tc_bandwidth='50% 50% 0%',
        expected_tsa_map='0:ets 1:ets 2:strict',
        expected_pfc_enabled='3',
        expected_dscp_app_table=_EXPECTED_DSCP_TABLE,
        expected_support_rdma=True,
        expected_performance_profile='RoCE',
        expected_pcie_relaxed_ordering='Enabled',
        expected_enable_sriov=None,
    )


class TestBroadcomRdmaQosCheckDetection(unittest.TestCase):
    def test_non_broadcom_node_skipped(self):
        orch = MagicMock()
        orch.exec.return_value = {'node1': 'VENDOR:NOT_BROADCOM'}
        results = BroadcomRdmaQosCheck(orch).run()
        self.assertEqual(results['node1']['status'], 'SKIPPED')
        self.assertEqual(results['node1']['vendor'], 'NOT_BROADCOM')

    def test_niccli_missing_warns(self):
        orch = MagicMock()
        orch.exec.return_value = {'node1': 'VENDOR:BROADCOM\nNICCLI:MISSING'}
        results = BroadcomRdmaQosCheck(orch).run()
        self.assertEqual(results['node1']['status'], 'WARNING')
        self.assertTrue(any('niccli not found' in e for e in results['node1']['errors']))

    def test_no_devices_warns(self):
        orch = MagicMock()
        orch.exec.return_value = {'node1': 'VENDOR:BROADCOM\nNICCLI:NO_DEVICES'}
        results = BroadcomRdmaQosCheck(orch).run()
        self.assertEqual(results['node1']['status'], 'WARNING')


class TestBroadcomRdmaQosCheckValidation(unittest.TestCase):
    def _run(self, **output_kwargs):
        orch = MagicMock()
        orch.exec.return_value = {'node1': _make_output(**output_kwargs)}
        return _default_checker(orch).run()

    def test_all_matching_pass(self):
        results = self._run()
        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['errors'], [])

    def test_prio_map_mismatch_fails(self):
        results = self._run(prio_map='0:0 1:0 2:0 3:0 4:0 5:0 6:0 7:0')
        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertTrue(any('PRIO_MAP' in e for e in results['node1']['errors']))

    def test_tc_bandwidth_mismatch_fails(self):
        results = self._run(tc_bw='33% 33% 34%')
        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertTrue(any('TC Bandwidth' in e for e in results['node1']['errors']))

    def test_performance_profile_mismatch_fails(self):
        results = self._run(perf_profile='default')
        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertTrue(any('performance_profile' in e for e in results['node1']['errors']))

    def test_pcie_relaxed_ordering_mismatch_fails(self):
        results = self._run(pcie_ro='Auto')
        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertTrue(any('pcie_relaxed_ordering' in e for e in results['node1']['errors']))

    def test_sriov_mismatch_fails(self):
        orch = MagicMock()
        orch.exec.return_value = {'node1': _make_output(sriov='Disabled')}
        checker = BroadcomRdmaQosCheck(
            orch, expected_dscp_app_table=_EXPECTED_DSCP_TABLE, expected_enable_sriov='Enabled'
        )
        results = checker.run()
        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertTrue(any('enable_sriov' in e for e in results['node1']['errors']))

    def test_sriov_skipped_when_none(self):
        orch = MagicMock()
        orch.exec.return_value = {'node1': _make_output()}
        checker = BroadcomRdmaQosCheck(orch, expected_dscp_app_table=_EXPECTED_DSCP_TABLE, expected_enable_sriov=None)
        self.assertEqual(checker.run()['node1']['status'], 'PASS')


class TestBroadcomRdmaQosCheckSudo(unittest.TestCase):
    def test_sudo_used_by_default(self):
        orch = MagicMock()
        orch.exec.return_value = {'node1': _make_output()}
        _default_checker(orch).run()
        self.assertIn('sudo -n niccli', orch.exec.call_args[0][0])

    def test_sudo_disabled(self):
        orch = MagicMock()
        orch.exec.return_value = {'node1': _make_output()}
        BroadcomRdmaQosCheck(orch, expected_dscp_app_table=_EXPECTED_DSCP_TABLE, use_sudo=False).run()
        self.assertNotIn('sudo -n niccli', orch.exec.call_args[0][0])


if __name__ == '__main__':
    unittest.main()
