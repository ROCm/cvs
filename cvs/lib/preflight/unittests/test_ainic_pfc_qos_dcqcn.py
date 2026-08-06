"""Unit tests for the AINIC PFC/QoS/DCQCN control-plane preflight checks."""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.ainic_pfc_qos_dcqcn import (
    DcqcnValidationCheck,
    PfcValidationCheck,
    QosValidationCheck,
    _parse_result_line,
)


class TestParseResultLine(unittest.TestCase):
    def test_parses_well_formed_line(self):
        output = "RESULT=PASS|CHECK=PFC|CARDS=8|PASSED=8|FAILED=0"
        fields = _parse_result_line(output)
        self.assertEqual(fields, {'RESULT': 'PASS', 'CHECK': 'PFC', 'CARDS': '8', 'PASSED': '8', 'FAILED': '0'})

    def test_returns_empty_dict_for_malformed_output(self):
        self.assertEqual(_parse_result_line("no result markers here"), {})
        self.assertEqual(_parse_result_line(""), {})
        self.assertEqual(_parse_result_line(None), {})


class TestPfcValidationCheck(unittest.TestCase):
    def test_all_cards_pass(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': "RESULT=PASS|CHECK=PFC|CARDS=8|PASSED=8|FAILED=0",
        }
        checker = PfcValidationCheck(phdl, expected_card_count=8)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['cards'], 8)
        self.assertEqual(results['node1']['errors'], [])

    def test_card_count_mismatch_fails(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': "RESULT=FAIL|CHECK=PFC|CARDS=4|PASSED=4|FAILED=0",
        }
        checker = PfcValidationCheck(phdl, expected_card_count=8)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('Expected 8 AINIC card(s), found 4', results['node1']['errors'][0])

    def test_pause_type_mismatch_fails(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': "RESULT=FAIL|CHECK=PFC|CARDS=8|PASSED=7|FAILED=1|FAILED_CARDS=3:PAUSE_TYPE=None",
        }
        checker = PfcValidationCheck(phdl, expected_card_count=8, expected_pause_type='PFC')
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('PFC mismatch on card(s): 3:PAUSE_TYPE=None', results['node1']['errors'][0])

    def test_no_cards_discovered_fails(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': "RESULT=ERROR|CHECK=PFC|REASON=no_card_ids_found",
        }
        checker = PfcValidationCheck(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('no_card_ids_found', results['node1']['errors'][0])

    def test_malformed_empty_output_fails(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': ''}
        checker = PfcValidationCheck(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('malformed or empty nicctl output', results['node1']['errors'][0])


class TestQosValidationCheck(unittest.TestCase):
    def test_all_cards_pass(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': "RESULT=PASS|CHECK=QOS|CARDS=8|PASSED=8|FAILED=0",
        }
        checker = QosValidationCheck(phdl, expected_card_count=8)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['errors'], [])

    def test_dscp_mismatch_fails(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': "RESULT=FAIL|CHECK=QOS|CARDS=8|PASSED=7|FAILED=1|DETAILS=0:[dscp24_priority=5]",
        }
        checker = QosValidationCheck(phdl, expected_card_count=8)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('QoS mismatch on card(s): 0:[dscp24_priority=5]', results['node1']['errors'][0])

    def test_no_cards_discovered_fails(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': "RESULT=ERROR|CHECK=QOS|REASON=no_card_ids_found",
        }
        checker = QosValidationCheck(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')

    def test_malformed_empty_output_fails(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': None}
        checker = QosValidationCheck(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('malformed or empty nicctl output', results['node1']['errors'][0])


class TestDcqcnValidationCheck(unittest.TestCase):
    def _golden_field_blob(self, checker, overrides=None):
        overrides = overrides or {}
        labels = list(checker.golden.keys())
        parts = []
        for i, label in enumerate(labels):
            value = overrides.get(label, checker.golden[label])
            parts.append(f"F{i}={value}")
        return "|".join(parts)

    def test_all_devices_pass(self):
        phdl = MagicMock()
        checker = DcqcnValidationCheck(phdl, expected_device_count=1)
        blob = self._golden_field_blob(checker)
        phdl.exec.return_value = {
            'node1': f"DEV_FIELDS:mlx5_0:{blob}\nRESULT=RAW|CHECK=DCQCN|DEVICES=1",
        }
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'PASS')
        self.assertEqual(results['node1']['devices'], 1)
        self.assertEqual(results['node1']['passed'], 1)
        self.assertEqual(results['node1']['errors'], [])

    def test_device_count_mismatch_fails(self):
        phdl = MagicMock()
        checker = DcqcnValidationCheck(phdl, expected_device_count=2)
        blob = self._golden_field_blob(checker)
        phdl.exec.return_value = {
            'node1': f"DEV_FIELDS:mlx5_0:{blob}\nRESULT=RAW|CHECK=DCQCN|DEVICES=1",
        }
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('Expected 2 AINIC device(s), found 1', results['node1']['errors'][0])

    def test_parameter_mismatch_fails(self):
        phdl = MagicMock()
        checker = DcqcnValidationCheck(phdl, expected_device_count=1, ai_rate="160")
        blob = self._golden_field_blob(checker, overrides={'Rate increase in AI phase': '999'})
        phdl.exec.return_value = {
            'node1': f"DEV_FIELDS:mlx5_0:{blob}\nRESULT=RAW|CHECK=DCQCN|DEVICES=1",
        }
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('DCQCN mismatch on device(s)', results['node1']['errors'][-1])
        self.assertIn('Rate increase in AI phase=999 (expected 160)', results['node1']['errors'][-1])

    def test_no_devices_found_fails(self):
        phdl = MagicMock()
        phdl.exec.return_value = {
            'node1': "RESULT=ERROR|CHECK=DCQCN|REASON=no_devices_found",
        }
        checker = DcqcnValidationCheck(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('no_devices_found', results['node1']['errors'][0])

    def test_malformed_empty_output_fails(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': ''}
        checker = DcqcnValidationCheck(phdl)
        results = checker.run()

        self.assertEqual(results['node1']['status'], 'FAIL')
        self.assertIn('malformed or empty output', results['node1']['errors'][0])


if __name__ == '__main__':
    unittest.main()
