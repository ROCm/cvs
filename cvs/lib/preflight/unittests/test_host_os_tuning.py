"""Unit tests for the OS tuning (C-state and governor) readiness check."""

import unittest
from unittest.mock import MagicMock

from cvs.lib.preflight.host_os_tuning import OsTuningCheck, parse_governors, parse_idle_states
from cvs.lib.preflight.host_readiness_base import parse_markers


def _probe(states=((0, 64, 0), (1, 64, 0), (2, 64, 64)), governors=(('performance', 64),), cmdline='ro'):
    lines = [f'CMDLINE={cmdline}', 'CPUIDLE_DRIVER=acpi_idle']
    names = {0: 'POLL', 1: 'C1', 2: 'C2', 3: 'C3'}
    for index, cpus, disabled in states:
        lines.append(f'IDLE_STATE={index} {cpus} {disabled}')
        lines.append(f'IDLE_NAME={index} {names[index]}')
    for name, count in governors:
        lines.append(f'GOVERNOR={name} {count}')
    return '\n'.join(lines)


class TestParsing(unittest.TestCase):
    def test_parse_idle_states_joins_names_and_skips_garbage(self):
        markers = parse_markers('IDLE_STATE=2 8 3\nIDLE_NAME=2 C2\nIDLE_STATE=x 1 1\nIDLE_STATE=3 8 8\n')
        self.assertEqual(
            parse_idle_states(markers),
            {2: {'name': 'C2', 'cpus': 8, 'disabled': 3}, 3: {'name': 'state3', 'cpus': 8, 'disabled': 8}},
        )

    def test_parse_governors(self):
        self.assertEqual(parse_governors(parse_markers('GOVERNOR=schedutil 4\nGOVERNOR=bad\n')), {'schedutil': 4})


class TestOsTuningCheck(unittest.TestCase):
    def test_deep_states_disabled_passes(self):
        result = OsTuningCheck(MagicMock()).evaluate('n1', _probe())
        self.assertEqual(result['status'], 'PASS', result)
        self.assertEqual(result['details']['idle_states']['state2']['disabled'], 64)

    def test_partially_enabled_deep_state_fails_with_fix_hint(self):
        result = OsTuningCheck(MagicMock()).evaluate('n1', _probe(states=((1, 64, 0), (2, 64, 60), (3, 64, 64))))
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(len(result['errors']), 1)
        self.assertIn('C2 (state2) is enabled on 4/64 CPUs', result['errors'][0])
        self.assertIn('cpupower idle-set -d 2', result['errors'][0])

    def test_threshold_controls_which_states_matter(self):
        result = OsTuningCheck(MagicMock(), disable_idle_states_from=3).evaluate(
            'n1', _probe(states=((2, 64, 0), (3, 64, 64)))
        )
        self.assertEqual(result['status'], 'PASS')

    def test_no_cpuidle_passes_with_note(self):
        result = OsTuningCheck(MagicMock()).evaluate('n1', 'CMDLINE=ro\nCPUIDLE_DRIVER=none')
        self.assertEqual(result['status'], 'PASS')
        self.assertIn('cpuidle is not active', result['details']['notes'][0])

    def test_null_threshold_skips_idle_states(self):
        result = OsTuningCheck(MagicMock(), disable_idle_states_from=None).evaluate('n1', _probe(states=((2, 64, 0),)))
        self.assertEqual(result['status'], 'PASS')

    def test_max_cstate_params_use_cmdline_matching(self):
        check = OsTuningCheck(MagicMock(), max_cstate_params=['processor.max_cstate=1'])
        self.assertEqual(check.evaluate('n1', _probe(cmdline='processor.max_cstate=1'))['status'], 'PASS')
        failed = check.evaluate('n1', _probe(cmdline='processor.max_cstate=5'))
        self.assertIn('processor.max_cstate=1 is not set (found: processor.max_cstate=5)', failed['errors'][0])

    def test_governor_mismatch_lists_offending_governors(self):
        check = OsTuningCheck(MagicMock(), cpu_governor='performance')
        result = check.evaluate('n1', _probe(governors=(('performance', 60), ('powersave', 4))))
        self.assertEqual(
            result['errors'], ['CPU frequency governor is not performance everywhere: powersave on 4 CPUs']
        )
        unreadable = check.evaluate('n1', _probe(governors=()))
        self.assertIn('not readable', unreadable['errors'][0])

    def test_warn_mode(self):
        result = OsTuningCheck(MagicMock(), on_mismatch='warn').evaluate('n1', _probe(states=((2, 64, 0),)))
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(len(result['warnings']), 1)


if __name__ == '__main__':
    unittest.main()
