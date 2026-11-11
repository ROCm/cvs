"""Unit tests for the GPU idle-state readiness check."""

import json
import unittest
from unittest.mock import MagicMock

from cvs.lib.preflight.gpu_idle_state import (
    AMD_SMI_MISSING,
    PROCESS_SEP,
    SAMPLE_SEP,
    GpuIdleStateCheck,
    _load_json,
    gpu_entries,
    parse_processes,
)

TEMPS = {'hotspot': 85, 'mem': 85, 'edge': 85}


def _gpu(gpu=0, gfx=0, umc=0, used=200, total=196592, hotspot=40, mem='N/A'):
    return {
        'gpu': gpu,
        'usage': {'gfx_activity': {'value': gfx, 'unit': '%'}, 'umc_activity': {'value': umc, 'unit': '%'}},
        'mem_usage': {'used_vram': {'value': used, 'unit': 'MB'}, 'total_vram': {'value': total, 'unit': 'MB'}},
        'temperature': {'edge': 'N/A', 'hotspot': {'value': hotspot, 'unit': 'C'}, 'mem': mem},
    }


def _probe(samples, processes='No running processes found'):
    parts = []
    for sample in samples:
        parts.append(SAMPLE_SEP)
        parts.append(json.dumps({'gpu_data': sample}))
    parts.append(PROCESS_SEP)
    parts.append(processes if isinstance(processes, str) else json.dumps(processes))
    return '\n'.join(parts)


def _check(**kwargs):
    kwargs.setdefault('temp_max_c', TEMPS)
    kwargs.setdefault('allowed_procs', 'gpuagent,amd-smi')
    return GpuIdleStateCheck(MagicMock(), **kwargs)


class TestParsing(unittest.TestCase):
    def test_gpu_entries_accepts_wrapper_list_and_single_dict(self):
        self.assertEqual(len(gpu_entries({'gpu_data': [_gpu(0), _gpu(1)]})), 2)
        self.assertEqual(len(gpu_entries([_gpu(0)])), 1)
        self.assertEqual(len(gpu_entries(_gpu(0))), 1)
        self.assertEqual(gpu_entries('N/A'), [])

    def test_load_json_skips_bracketed_warning_text(self):
        self.assertEqual(_load_json('[WARN] driver busy\n[{"gpu": 0}]\ntrailer'), [{'gpu': 0}])
        self.assertIsNone(_load_json('no json here'))

    def test_parse_processes(self):
        self.assertEqual(parse_processes('No running processes found'), ([], True))
        self.assertEqual(parse_processes(''), ([], False))
        payload = [
            {'gpu': 1, 'process_list': [{'process_info': {'name': 'python3', 'pid': 42}}]},
            {'gpu': 2, 'process_list': [{'process_info': 'No running processes found'}]},
        ]
        self.assertEqual(parse_processes(json.dumps(payload)), ([{'gpu': 1, 'name': 'python3', 'pid': 42}], True))


class TestBuildCommand(unittest.TestCase):
    def test_command_samples_targeted_metrics_only(self):
        command = _check(samples=4, interval_sec=3).build_command()
        self.assertIn('metric --usage --mem-usage --temperature --json', command)
        self.assertNotIn('xgmi', command)
        self.assertIn('-lt 4', command)
        self.assertIn('sleep 3', command)
        self.assertIn('/opt/rocm/bin/amd-smi', command)
        self.assertIn('process --json', command)

    def test_timeout_covers_sampling(self):
        self.assertEqual(_check(samples=10, interval_sec=5).probe_timeout, 170)


class TestEvaluate(unittest.TestCase):
    def test_idle_gpus_pass(self):
        result = _check().evaluate('n1', _probe([[_gpu(0), _gpu(1)]] * 3))
        self.assertEqual(result['status'], 'PASS', result)
        self.assertEqual(result['details']['samples'], 3)
        self.assertEqual(result['details']['gpus']['0']['temperature'], {'hotspot': 40})

    def test_transient_spike_uses_lowest_sample(self):
        result = _check().evaluate('n1', _probe([[_gpu(gfx=90)], [_gpu(gfx=1)], [_gpu(gfx=50)]]))
        self.assertEqual(result['status'], 'PASS', result)

    def test_busy_hot_and_full_gpu_fail(self):
        busy = _gpu(gpu=3, gfx=80, umc=40, used=150000, hotspot=91)
        result = _check().evaluate('n1', _probe([[busy]] * 2))
        self.assertEqual(result['status'], 'FAIL')
        joined = '\n'.join(result['errors'])
        self.assertIn('GPU 3: GFX activity 80% is above the idle limit of 5%', joined)
        self.assertIn('GPU 3: memory activity 40%', joined)
        self.assertIn('GPU 3: VRAM', joined)
        self.assertIn('GPU 3: hotspot temperature 91 C is above 85 C', joined)

    def test_temperature_uses_highest_sample(self):
        result = _check().evaluate('n1', _probe([[_gpu(hotspot=60)], [_gpu(hotspot=86)]]))
        self.assertIn('hotspot temperature 86 C', result['errors'][0])

    def test_foreign_process_fails_and_allowlisted_process_passes(self):
        payload = [
            {'gpu': 0, 'process_list': [{'process_info': {'name': 'gpuagent', 'pid': 7}}]},
            {'gpu': 1, 'process_list': [{'process_info': {'name': 'python3', 'pid': 42}}]},
        ]
        result = _check().evaluate('n1', _probe([[_gpu(0)]], payload))
        self.assertEqual(result['errors'], ['GPU 1: process python3 (pid 42) is using the GPU'])
        relaxed = _check(fail_on_gpu_processes=False).evaluate('n1', _probe([[_gpu(0)]], payload))
        self.assertEqual(relaxed['status'], 'PASS')

    def test_missing_amd_smi_and_unparseable_output_are_errors(self):
        self.assertEqual(
            _check(on_mismatch='warn').evaluate('n1', AMD_SMI_MISSING)['errors'], ['amd-smi was not found on n1']
        )
        garbage = f'{SAMPLE_SEP}\nERROR: amdgpu driver not loaded\n{PROCESS_SEP}\n'
        result = _check(on_mismatch='warn').evaluate('n1', garbage)
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('Could not parse amd-smi metric output', result['errors'][0])

    def test_no_gpus_is_an_error(self):
        result = _check().evaluate('n1', _probe([[]]))
        self.assertEqual(result['errors'], ['amd-smi reported no GPUs on n1'])

    def test_partial_samples_and_unreadable_process_list_warn(self):
        output = f'{SAMPLE_SEP}\n{json.dumps([_gpu(0)])}\n{SAMPLE_SEP}\ngarbage\n{PROCESS_SEP}\n'
        result = _check().evaluate('n1', output)
        self.assertEqual(result['status'], 'PASS')
        self.assertIn('1 of 2 amd-smi samples were unreadable', result['warnings'])
        self.assertIn("Could not read the GPU process list from 'amd-smi process'", result['warnings'])

    def test_null_limits_skip_checks(self):
        check = _check(
            gfx_util_max_pct=None, mem_util_max_pct=None, vram_used_max_pct=None, temp_max_c={'hotspot': None}
        )
        busy = _gpu(gfx=99, umc=99, used=190000, hotspot=99)
        self.assertEqual(check.evaluate('n1', _probe([[busy]]))['status'], 'PASS')


if __name__ == '__main__':
    unittest.main()
