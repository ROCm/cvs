"""Unit tests for the RDHC-derived platform readiness extras."""

import unittest
from unittest.mock import MagicMock

from cvs.lib.preflight.host_platform_extras import PlatformExtrasCheck, effective_limits, parse_limit_entries
from cvs.lib.preflight.host_readiness_base import parse_markers

GIB = 1024 * 1024 * 1024
LIMITS = {'memlock': 'unlimited', 'nofile_min': 1048576}
MODULES = ['ib_core', 'ib_uverbs']

GOOD = '\n'.join(
    [
        f'BAR=card1 {192 * GIB} {192 * GIB}',
        'ATOMIC=0000:05:00.0 AtomicOpsCtl: ReqEn+',
        'CWSR=1',
        'LIMITS_USER=alice',
        'LIMITS_GROUPS=alice video',
        'LIMIT=/etc/security/limits.conf:50:* - memlock unlimited',
        'LIMIT=/etc/security/limits.d/99-rocm.conf:1:@video soft nofile 1048576',
        'LIMIT=/etc/security/limits.d/99-rocm.conf:2:@video hard nofile 1048576',
        'MODULE=ib_core',
        'MODULE=ib_uverbs',
        'IBDEV=rocep28s0 bnxt_re',
    ]
)


def _check(**kwargs):
    kwargs.setdefault('limits', LIMITS)
    kwargs.setdefault('rdma_modules', MODULES)
    return PlatformExtrasCheck(MagicMock(), **kwargs)


class TestLimits(unittest.TestCase):
    def _effective(self, lines, user='alice', groups=('alice', 'video')):
        markers = parse_markers('\n'.join(f'LIMIT={line}' for line in lines))
        return effective_limits(parse_limit_entries(markers), user, set(groups))

    def test_dash_sets_both_kinds(self):
        chosen = self._effective(['/f:1:* - memlock unlimited'])
        self.assertEqual(set(chosen), {('memlock', 'soft'), ('memlock', 'hard')})

    def test_user_beats_group_beats_wildcard_regardless_of_order(self):
        chosen = self._effective(['/f:1:alice soft nofile 10', '/f:2:@video soft nofile 20', '/f:3:* soft nofile 30'])
        self.assertEqual(chosen[('nofile', 'soft')]['value'], '10')
        chosen = self._effective(['/f:1:@video soft nofile 20', '/f:2:* soft nofile 30'])
        self.assertEqual(chosen[('nofile', 'soft')]['value'], '20')

    def test_later_entry_of_same_rank_wins(self):
        chosen = self._effective(['/f:1:* soft nofile 30', '/g:1:* soft nofile 40'])
        self.assertEqual(chosen[('nofile', 'soft')]['path'], '/g')

    def test_root_ignores_wildcard_and_groups(self):
        chosen = self._effective(['/f:1:* - memlock unlimited', '/f:2:@root - memlock unlimited'], 'root', ['root'])
        self.assertEqual(chosen, {})

    def test_comments_and_short_lines_are_skipped(self):
        markers = parse_markers('LIMIT=/f:1:* soft nofile 5 # tuned\nLIMIT=/f:2:* soft\nLIMIT=bad')
        self.assertEqual([e['value'] for e in parse_limit_entries(markers)], ['5'])


class TestBuildCommand(unittest.TestCase):
    def test_disabled_sections_are_not_probed(self):
        command = PlatformExtrasCheck(MagicMock(), large_bar=False, atomic_ops=False).build_command()
        self.assertNotIn('mem_info_vram_total', command)
        self.assertNotIn('lspci', command)
        self.assertNotIn('limits.conf', command)
        self.assertNotIn('ldd', command)
        self.assertIn('cwsr_enable', command)

    def test_enabled_sections_quote_inputs(self):
        command = _check(
            rocm_lib_deps={'max_depth': 2, 'timeout_sec': 60}, rocm_path='/opt/rocm 7', rdma_modules=['ib_core']
        ).build_command()
        self.assertIn("'/opt/rocm 7/lib'", command)
        self.assertIn('-maxdepth 2', command)
        self.assertIn('timeout 60', command)
        self.assertIn('sudo -n lspci -vvv', command)
        self.assertIn('/etc/security/limits.d/*.conf', command)
        self.assertIn('for _cvs_m in ib_core;', command)

    def test_ldd_scan_extends_timeout(self):
        self.assertEqual(_check(rocm_lib_deps={'timeout_sec': 600}).probe_timeout, 720)


class TestEvaluate(unittest.TestCase):
    def test_good_node_passes(self):
        result = _check(cwsr_enable=1).evaluate('n1', GOOD)
        self.assertEqual(result['status'], 'PASS', result)
        self.assertEqual(result['details']['rdma']['devices'], {'rocep28s0': 'bnxt_re'})
        self.assertEqual(result['details']['limits']['values']['hard_memlock'], 'unlimited')

    def test_small_bar_fails_with_bios_hint(self):
        output = GOOD.replace(f'BAR=card1 {192 * GIB} {192 * GIB}', f'BAR=card1 {192 * GIB} {256 * 1024 * 1024}')
        result = _check().evaluate('n1', output)
        self.assertIn('card1: Large BAR is not enabled (CPU-visible VRAM 256 MiB of 196608 MiB)', result['errors'][0])

    def test_atomic_ops_disabled_and_denied(self):
        output = GOOD.replace('ReqEn+', 'ReqEn-') + '\nCVS_CMD_DENIED lspci 0000:06:00.0'
        result = _check(on_mismatch='warn').evaluate('n1', output)
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn("'sudo -n lspci -vvv' was denied for 1 GPU function(s)", result['errors'][0])
        self.assertIn('PCIe AtomicOps requests are not enabled', result['warnings'][0])

    def test_cwsr_mismatch(self):
        result = _check(cwsr_enable=0).evaluate('n1', GOOD)
        self.assertIn('amdgpu cwsr_enable is 1 (expected 0)', result['errors'])

    def test_limits_missing_and_too_low(self):
        output = GOOD.replace('@video hard nofile 1048576', '@video hard nofile 1024').replace(
            'LIMIT=/etc/security/limits.conf:50:* - memlock unlimited',
            'LIMIT=/etc/security/limits.conf:50:* soft memlock 64',
        )
        result = _check().evaluate('n1', output)
        joined = '\n'.join(result['errors'])
        self.assertIn('soft memlock is 64 at /etc/security/limits.conf:50 (expected unlimited)', joined)
        self.assertIn('No hard memlock limit applies to user alice', joined)
        self.assertIn('hard nofile is 1024 at /etc/security/limits.d/99-rocm.conf:2', joined)

    def test_rdma_modules_and_devices(self):
        output = GOOD.replace('MODULE=ib_uverbs', '').replace('IBDEV=rocep28s0 bnxt_re', '')
        result = _check().evaluate('n1', output)
        self.assertIn('RDMA kernel module(s) not loaded: ib_uverbs', result['errors'])
        self.assertIn('No RDMA devices are registered in /sys/class/infiniband', result['errors'])

    def test_ldd_missing_dependencies_and_missing_dir(self):
        output = GOOD + '\nLDD_CHECKED=40\nLDD_MISSING=/opt/rocm/lib/libfoo.so libbar.so.1'
        result = _check(rocm_lib_deps={'max_depth': 1}).evaluate('n1', output)
        self.assertIn('1 unresolved ROCm library dependencies: /opt/rocm/lib/libfoo.so libbar.so.1', result['errors'])
        missing = _check(rocm_lib_deps={'max_depth': 1}).evaluate('n1', GOOD + '\nROCM_LIB_MISSING=/opt/rocm/lib')
        self.assertIn('ROCm library directory /opt/rocm/lib does not exist', missing['errors'])

    def test_no_gpu_sizes_warns(self):
        result = _check().evaluate('n1', GOOD.replace(f'BAR=card1 {192 * GIB} {192 * GIB}', ''))
        self.assertIn('No amdgpu VRAM sizes in sysfs; Large BAR not checked', result['warnings'])


if __name__ == '__main__':
    unittest.main()
