"""Unit tests for the kernel command-line readiness check."""

import unittest
from unittest.mock import MagicMock

from cvs.lib.preflight.host_kernel_params import (
    KernelParamsCheck,
    blacklisted_modules,
    conflicting_scalars,
    parse_cmdline,
    requirement_met,
)

REQUIRED = ['pci=realloc=off', 'pci=bfsort', 'iommu=pt', 'numa_balancing=disable', 'modprobe.blacklist=amdgpu']
GOOD_CMDLINE = (
    'BOOT_IMAGE=/vmlinuz root=UUID=1 ro pci=realloc=off,bfsort iommu=pt numa_balancing=disable '
    'modprobe.blacklist=amdgpu'
)


def _probe(cmdline=GOOD_CMDLINE, vendor='AuthenticAMD', numa='0', initstate='live'):
    return f'CMDLINE={cmdline}\nCPU_VENDOR={vendor}\nNUMA_BALANCING={numa}\nAMDGPU_INITSTATE={initstate}\n'


class TestParsing(unittest.TestCase):
    def test_parse_cmdline_keeps_bare_flags_and_quoted_values(self):
        pairs = parse_cmdline('ro quiet console="ttyS0,115200 n8" a=b=c')
        self.assertEqual(pairs, [('ro', None), ('quiet', None), ('console', 'ttyS0,115200 n8'), ('a', 'b=c')])

    def test_parse_cmdline_unbalanced_quote_falls_back_to_split(self):
        self.assertEqual(parse_cmdline('a="b c'), [('a', '"b'), ('c', None)])

    def test_list_params_match_by_option_across_occurrences(self):
        pairs = parse_cmdline('pci=bfsort pci=realloc=off')
        self.assertTrue(requirement_met(pairs, 'pci=bfsort'))
        self.assertTrue(requirement_met(pairs, 'pci=realloc=off'))
        self.assertFalse(requirement_met(pairs, 'pci=realloc=on'))

    def test_later_list_option_wins(self):
        pairs = parse_cmdline('pci=realloc=off pci=realloc=on')
        self.assertFalse(requirement_met(pairs, 'pci=realloc=off'))

    def test_substring_cannot_satisfy_requirement(self):
        pairs = parse_cmdline('amd_iommu=pt')
        self.assertFalse(requirement_met(pairs, 'iommu=pt'))

    def test_scalar_last_value_wins_and_bare_key_means_present(self):
        pairs = parse_cmdline('iommu=pt iommu=on nosmt')
        self.assertFalse(requirement_met(pairs, 'iommu=pt'))
        self.assertTrue(requirement_met(pairs, 'iommu'))
        self.assertTrue(requirement_met(pairs, 'nosmt'))
        self.assertEqual(conflicting_scalars(pairs), {'iommu': ['pt', 'on']})

    def test_blacklisted_modules_reads_both_parameters(self):
        pairs = parse_cmdline('modprobe.blacklist=nouveau,amdgpu module_blacklist=ast')
        self.assertEqual(blacklisted_modules(pairs), {'nouveau', 'amdgpu', 'ast'})


class TestKernelParamsCheck(unittest.TestCase):
    def _check(self, **kwargs):
        kwargs.setdefault('required_intel', ['intel_iommu=on'])
        return KernelParamsCheck(MagicMock(), REQUIRED, **kwargs)

    def test_good_amd_host_passes(self):
        result = self._check().evaluate('n1', _probe())
        self.assertEqual(result['status'], 'PASS', result)
        self.assertEqual(result['warnings'], [])
        self.assertEqual(result['details']['cpu_vendor'], 'AuthenticAMD')

    def test_intel_host_needs_intel_iommu(self):
        result = self._check().evaluate('n1', _probe(vendor='GenuineIntel'))
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('intel_iommu=on is not set', result['errors'][0])
        ok = self._check().evaluate('n1', _probe(GOOD_CMDLINE + ' intel_iommu=on,sm_on', vendor='GenuineIntel'))
        self.assertEqual(ok['status'], 'PASS', ok)

    def test_missing_parameter_reports_what_was_found(self):
        result = self._check().evaluate('n1', _probe(GOOD_CMDLINE.replace('iommu=pt', 'amd_iommu=pt')))
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['errors'], ['Kernel parameter iommu=pt is not set (found: absent)'])

    def test_unknown_vendor_warns(self):
        result = self._check().evaluate('n1', _probe(vendor=''))
        self.assertEqual(result['status'], 'PASS')
        self.assertIn('Unknown CPU vendor', result['warnings'][0])

    def test_forbidden_parameter_fails(self):
        result = self._check(forbidden=['iommu=pt']).evaluate('n1', _probe())
        self.assertIn('Forbidden kernel parameter iommu=pt is set', result['errors'])

    def test_runtime_numa_balancing_must_be_zero(self):
        result = self._check().evaluate('n1', _probe(numa='1'))
        self.assertIn('kernel.numa_balancing is 1 at runtime; it must be 0', result['errors'])

    def test_blacklisted_amdgpu_must_be_loaded_even_in_warn_mode(self):
        result = self._check(on_mismatch='warn').evaluate('n1', _probe(initstate=''))
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('amdgpu module is not loaded', result['errors'][0])

    def test_unreadable_cmdline_is_an_error(self):
        result = self._check(on_mismatch='warn').evaluate('n1', 'CPU_VENDOR=AuthenticAMD')
        self.assertEqual(result['errors'], ['Could not read /proc/cmdline on n1'])

    def test_warn_mode_downgrades_missing_parameters(self):
        result = self._check(on_mismatch='warn').evaluate('n1', _probe('ro'))
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(len(result['warnings']), len(REQUIRED))


if __name__ == '__main__':
    unittest.main()
