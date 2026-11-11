"""Unit tests for the persistent environment readiness check."""

import unittest
from unittest.mock import MagicMock

from cvs.lib.preflight.host_env import (
    PersistentEnvCheck,
    parse_assignment_value,
    parse_session,
    session_command,
    validate_variable_names,
)
from cvs.lib.preflight.host_readiness_base import parse_markers

REQUIRED = {'HIP_FORCE_DEV_KERNARG': '1', 'HSA_OVERRIDE_CPU_AFFINITY_DEBUG': '0'}
GOOD = (
    'ENVLINE=/etc/environment:3:HIP_FORCE_DEV_KERNARG=1\n'
    'ENVLINE=/etc/environment:4:HSA_OVERRIDE_CPU_AFFINITY_DEBUG="0"\n'
    'SESSION=HIP_FORCE_DEV_KERNARG=1\n'
    'SESSION=HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0\n'
)


def _check(**kwargs):
    return PersistentEnvCheck(MagicMock(), REQUIRED, ['/etc/environment'], **kwargs)


class TestHelpers(unittest.TestCase):
    def test_parse_assignment_value(self):
        self.assertEqual(parse_assignment_value('1'), ('1', False))
        self.assertEqual(parse_assignment_value('"1"  # note'), ('1', False))
        self.assertEqual(parse_assignment_value("'$X'"), ('$X', False))
        self.assertEqual(parse_assignment_value('1 # comment'), ('1', False))
        self.assertEqual(parse_assignment_value('"${X:-1}"'), ('${X:-1}', True))
        self.assertEqual(parse_assignment_value('`cat f`'), ('`cat f`', True))

    def test_validate_variable_names(self):
        validate_variable_names(['A_1', '_B'])
        with self.assertRaises(ValueError):
            validate_variable_names(['1A'])
        with self.assertRaises(ValueError):
            validate_variable_names(['A;rm -rf /'])

    def test_session_round_trip(self):
        command = session_command(['A'])
        self.assertIn('printenv A', command)
        markers = parse_markers('SESSION=A=x=y\nSESSION_UNSET=B\n')
        self.assertEqual(parse_session(markers), {'A': 'x=y', 'B': None})


class TestBuildCommand(unittest.TestCase):
    def test_command_quotes_sources_and_greps_required_names(self):
        command = PersistentEnvCheck(MagicMock(), REQUIRED, ['/etc/environment', '/etc/my env']).build_command()
        self.assertIn("'/etc/my env'", command)
        self.assertIn('HIP_FORCE_DEV_KERNARG|HSA_OVERRIDE_CPU_AFFINITY_DEBUG', command)
        self.assertIn('ENV_SOURCE_MISSING', command)
        self.assertIn('printenv HIP_FORCE_DEV_KERNARG', command)

    def test_session_probe_omitted_when_disabled(self):
        self.assertNotIn('printenv', _check(check_session=False).build_command())


class TestEvaluate(unittest.TestCase):
    def test_good_node_passes(self):
        result = _check().evaluate('n1', GOOD)
        self.assertEqual(result['status'], 'PASS', result)
        self.assertEqual(result['details']['session']['HIP_FORCE_DEV_KERNARG'], '1')

    def test_missing_variable_fails_with_fix(self):
        result = _check(check_session=False).evaluate('n1', GOOD.splitlines()[0])
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('HSA_OVERRIDE_CPU_AFFINITY_DEBUG is not set persistently', result['errors'][0])

    def test_wrong_last_value_and_conflicts(self):
        output = (
            'ENVLINE=/etc/environment:1:HIP_FORCE_DEV_KERNARG=1\n'
            'ENVLINE=/etc/environment:9:export HIP_FORCE_DEV_KERNARG=0\n'
            'ENVLINE=/etc/environment:4:HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0\n'
        )
        result = _check(check_session=False).evaluate('n1', output)
        self.assertEqual(len(result['errors']), 2)
        self.assertIn('conflicting persistent definitions', result['errors'][0])
        self.assertIn('HIP_FORCE_DEV_KERNARG=0 at /etc/environment:9', result['errors'][1])

    def test_dynamic_value_warns(self):
        output = GOOD.replace('HIP_FORCE_DEV_KERNARG=1\n', 'HIP_FORCE_DEV_KERNARG=${HIP:-1}\n', 1)
        result = _check(check_session=False).evaluate('n1', output)
        self.assertEqual(result['status'], 'PASS')
        self.assertIn('cannot be checked statically', result['warnings'][0])

    def test_session_override_fails_and_invisible_session_warns(self):
        overridden = GOOD.replace('SESSION=HIP_FORCE_DEV_KERNARG=1', 'SESSION=HIP_FORCE_DEV_KERNARG=0')
        result = _check().evaluate('n1', overridden)
        self.assertIn('in a remote command session overrides', result['errors'][0])
        hidden = GOOD.replace('SESSION=HIP_FORCE_DEV_KERNARG=1', 'SESSION_UNSET=HIP_FORCE_DEV_KERNARG')
        warned = _check().evaluate('n1', hidden)
        self.assertEqual(warned['status'], 'PASS')
        self.assertIn('pam_env', warned['warnings'][0])

    def test_missing_source_warns(self):
        result = _check().evaluate('n1', GOOD + 'ENV_SOURCE_MISSING=/etc/profile.d\n')
        self.assertIn('Environment source /etc/profile.d does not exist', result['warnings'])

    def test_container_session_is_checked(self):
        orch = MagicMock()
        orch.exec.return_value = {
            'n1': 'SESSION=HIP_FORCE_DEV_KERNARG=1\nSESSION_UNSET=HSA_OVERRIDE_CPU_AFFINITY_DEBUG\n'
        }
        orch.all.exec.return_value = {'n1': GOOD}
        results = PersistentEnvCheck(orch, REQUIRED, ['/etc/environment'], check_container=True).run()
        orch.exec.assert_called_once()
        self.assertEqual(results['n1']['status'], 'FAIL')
        self.assertIn('HSA_OVERRIDE_CPU_AFFINITY_DEBUG is unset inside the CVS container', results['n1']['errors'][0])


if __name__ == '__main__':
    unittest.main()
