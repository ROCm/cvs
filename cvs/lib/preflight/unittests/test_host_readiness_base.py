"""Unit tests for the shared host readiness plumbing."""

import unittest
from unittest.mock import MagicMock

from cvs.lib.preflight.host_readiness_base import (
    HostReadinessCheck,
    findings,
    first_marker,
    is_unreachable,
    node_result,
    parse_markers,
    to_number,
)


class _EchoCheck(HostReadinessCheck):
    def build_command(self):
        return 'echo VALUE=$(cat /x)'

    def evaluate_node(self, node, output):
        value = first_marker(parse_markers(output), 'VALUE')
        if value == 'bad':
            return findings(mismatches=[f'{node} value is bad'])
        if value == '':
            return findings(errors=[f'{node} unreadable'])
        return findings(details={'value': value})


class TestHelpers(unittest.TestCase):
    def test_parse_markers_keeps_order_and_ignores_other_lines(self):
        markers = parse_markers('noise\nA=1\nlower=x\nA=2\r\nB_2=x=y\n')
        self.assertEqual(markers, {'A': ['1', '2'], 'B_2': ['x=y']})

    def test_first_marker_strips_and_defaults(self):
        self.assertEqual(first_marker({'A': [' v ']}, 'A'), 'v')
        self.assertEqual(first_marker({}, 'A', 'd'), 'd')

    def test_to_number(self):
        self.assertEqual(to_number('42'), 42)
        self.assertEqual(to_number(' 1.5 '), 1.5)
        self.assertIsNone(to_number('N/A'))
        self.assertIsNone(to_number(True))
        self.assertIsNone(to_number(None))

    def test_node_result_fail_mode_turns_mismatches_into_errors(self):
        result = node_result(findings(['m'], ['e'], ['w']), 'fail')
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['errors'], ['m', 'e'])
        self.assertEqual(result['warnings'], ['w'])

    def test_node_result_warn_mode_keeps_probe_errors_fatal(self):
        warned = node_result(findings(['m'], [], ['w']), 'warn')
        self.assertEqual(warned['status'], 'PASS')
        self.assertEqual(warned['warnings'], ['m', 'w'])
        failed = node_result(findings(['m'], ['e']), 'warn')
        self.assertEqual(failed['status'], 'FAIL')
        self.assertEqual(failed['errors'], ['e'])

    def test_is_unreachable(self):
        self.assertTrue(is_unreachable('ABORT: Host Unreachable Error'))
        self.assertFalse(is_unreachable('VALUE=1'))


class TestHostReadinessCheck(unittest.TestCase):
    def test_invalid_on_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            _EchoCheck(MagicMock(), on_mismatch='ignore')

    def test_run_executes_one_probe_and_evaluates_every_node(self):
        orch = MagicMock()
        orch.all.exec.return_value = {
            'n1': 'VALUE=ok',
            'n2': 'VALUE=bad',
            'n3': '',
            'n4': 'ABORT: Host Unreachable Error',
        }
        results = _EchoCheck(orch).run()
        orch.all.exec.assert_called_once_with('echo VALUE=$(cat /x)', timeout=120)
        self.assertEqual(results['n1']['status'], 'PASS')
        self.assertEqual(results['n1']['details'], {'value': 'ok'})
        self.assertEqual(results['n2']['errors'], ['n2 value is bad'])
        self.assertEqual(results['n3']['errors'], ['n3 unreadable'])
        self.assertEqual(results['n4']['status'], 'FAIL')
        self.assertIn('unreachable', results['n4']['errors'][0])

    def test_warn_mode_passes_mismatched_nodes(self):
        orch = MagicMock()
        orch.all.exec.return_value = {'n1': 'VALUE=bad'}
        results = _EchoCheck(orch, on_mismatch='warn').run()
        self.assertEqual(results['n1']['status'], 'PASS')
        self.assertEqual(results['n1']['warnings'], ['n1 value is bad'])


if __name__ == '__main__':
    unittest.main()
