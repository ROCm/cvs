import json
import unittest
from unittest.mock import MagicMock, patch

import cvs.lib.rocm_plib as rocm_plib


def _zero_blocks():
    return {block: {field: 0 for field in rocm_plib.ECC_COUNTER_FIELDS} for block in rocm_plib.RCCL_ECC_BLOCKS}


def _blocks_with(**overrides):
    blocks = _zero_blocks()
    for block, fields in overrides.items():
        blocks[block].update(fields)
    return blocks


def _logged_text(mock_log):
    chunks = []
    for call in mock_log.call_args_list:
        args = call[0]
        if len(args) == 1:
            chunks.append(str(args[0]))
        else:
            chunks.append(args[0] % args[1:])
    return '\n'.join(chunks)


def _metric_gpu(gpu_id, ecc_blocks, extra_ecc=None):
    gpu = {'gpu': gpu_id, 'ecc_blocks': ecc_blocks}
    if extra_ecc is not None:
        gpu['ecc'] = extra_ecc
    return gpu


class TestIterAmdSmiGpuDicts(unittest.TestCase):
    def test_list_payload(self):
        gpus = [{'gpu': 0}, {'gpu': 1}]
        self.assertEqual(rocm_plib._iter_amd_smi_gpu_dicts(gpus), gpus)

    def test_gpu_data_wrapper(self):
        gpus = [{'gpu': 0}]
        self.assertEqual(rocm_plib._iter_amd_smi_gpu_dicts({'gpu_data': gpus}), gpus)

    def test_empty_or_unknown(self):
        self.assertEqual(rocm_plib._iter_amd_smi_gpu_dicts({}), [])
        self.assertEqual(rocm_plib._iter_amd_smi_gpu_dicts(None), [])


class TestGetAmdSmiEccBlocksDict(unittest.TestCase):
    def test_parses_list_shaped_metric_json(self):
        blocks = _blocks_with(UMC={'correctable_count': 1, 'uncorrectable_count': 2})
        payload = json.dumps([_metric_gpu(0, blocks)])
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': payload}

        result = rocm_plib.get_amd_smi_ecc_blocks_dict(phdl)
        self.assertEqual(result['node1'][0]['UMC']['correctable_count'], 1)
        self.assertEqual(result['node1'][0]['UMC']['uncorrectable_count'], 2)
        self.assertEqual(result['node1'][0]['UMC']['deferred_count'], 0)
        self.assertEqual(result['node1'][0]['GFX']['correctable_count'], 0)
        phdl.exec.assert_called_once()
        self.assertIn('metric -g all', phdl.exec.call_args[0][0])
        self.assertFalse(phdl.exec.call_args.kwargs.get('print_console', True))

    def test_parses_gpu_data_wrapper(self):
        blocks = _zero_blocks()
        payload = json.dumps({'gpu_data': [_metric_gpu(1, blocks)]})
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': payload}

        result = rocm_plib.get_amd_smi_ecc_blocks_dict(phdl)
        self.assertIn(1, result['node1'])

    def test_ignores_extra_blocks_and_numeric_strings(self):
        raw = _zero_blocks()
        raw['UMC'] = {'correctable_count': '4', 'uncorrectable_count': '0', 'deferred_count': '2'}
        raw['FOO'] = {'correctable_count': 9, 'uncorrectable_count': 9}
        payload = json.dumps([_metric_gpu(0, raw)])
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': payload}

        result = rocm_plib.get_amd_smi_ecc_blocks_dict(phdl)
        self.assertEqual(result['node1'][0]['UMC']['correctable_count'], 4)
        self.assertEqual(result['node1'][0]['UMC']['deferred_count'], 2)
        self.assertNotIn('FOO', result['node1'][0])

    def test_ecc_totals_without_ecc_blocks_are_zero(self):
        gpu = {
            'gpu': 0,
            'ecc': {
                'total_correctable_count': 11,
                'total_uncorrectable_count': 7,
            },
        }
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': json.dumps([gpu])}

        with patch.object(rocm_plib.log, 'debug') as mock_debug:
            result = rocm_plib.get_amd_smi_ecc_blocks_dict(phdl)
        self.assertEqual(result['node1'][0], _zero_blocks())
        self.assertFalse(any('total_correctable_count=11' in str(c) for c in mock_debug.call_args_list))
        missing_blocks = [c.args[3] for c in mock_debug.call_args_list if c.args and 'not present' in c.args[0]]
        self.assertEqual(missing_blocks, list(rocm_plib.RCCL_ECC_BLOCKS))

    def test_empty_amd_smi_skips_without_fail_test(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': '[]'}
        with patch('cvs.lib.rocm_plib.fail_test') as mock_fail:
            result = rocm_plib.get_amd_smi_ecc_blocks_dict(phdl)
        self.assertEqual(result, {'node1': {}})
        mock_fail.assert_not_called()

    def test_invalid_json_warns_without_fail_test(self):
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': 'not-json'}
        with patch.object(rocm_plib.log, 'warning') as mock_warn, patch('cvs.lib.rocm_plib.fail_test') as mock_fail:
            result = rocm_plib.get_amd_smi_ecc_blocks_dict(phdl)
        self.assertEqual(result, {'node1': {}})
        mock_fail.assert_not_called()
        mock_warn.assert_called()


class TestCompareEccBlocksSnapshots(unittest.TestCase):
    def test_no_delta_logs_table_not_warning(self):
        snap = {'node1': {0: _zero_blocks()}}
        with patch.object(rocm_plib.log, 'info') as mock_info, patch.object(rocm_plib.log, 'warning') as mock_warn:
            increases = rocm_plib.compare_ecc_blocks_snapshots(snap, snap, collective='all_reduce_perf')
        self.assertEqual(increases, [])
        mock_warn.assert_not_called()
        text = _logged_text(mock_info)
        self.assertTrue(text.startswith('======== ECC_BLOCKS report ========\nECC_BLOCKS  collective=all_reduce_perf'))
        self.assertIn('ECC_BLOCKS  collective=all_reduce_perf  node=node1  result=CLEAN', text)
        self.assertIn('gpu=0  result=CLEAN', text)
        self.assertIn('CE before', text)
        self.assertIn('CE Delta', text)
        self.assertIn('UE Delta', text)
        self.assertIn('DE Delta', text)
        self.assertIn('UMC', text)
        self.assertIn('XGMI_WAFL', text)
        self.assertNotIn('ECC_BLOCKS before:', text)
        self.assertNotIn('ECC_BLOCKS after:', text)
        self.assertNotIn('ECC_BLOCKS delta:', text)

    def test_correctable_increase_table_and_warn(self):
        before = {'node1': {0: _zero_blocks()}}
        after = {'node1': {0: _blocks_with(UMC={'correctable_count': 3})}}
        with patch.object(rocm_plib.log, 'info') as mock_info, patch.object(rocm_plib.log, 'warning') as mock_warn:
            increases = rocm_plib.compare_ecc_blocks_snapshots(before, after)
        self.assertEqual(len(increases), 1)
        self.assertEqual(increases[0]['field'], 'correctable_count')
        text = _logged_text(mock_info)
        self.assertIn('result=INCREASED', text)
        self.assertIn('+3', text)
        self.assertIn('increased', mock_warn.call_args[0][0])

    def test_uncorrectable_increase_warns(self):
        before = {'node1': {0: _zero_blocks()}}
        after = {'node1': {0: _blocks_with(SDMA={'uncorrectable_count': 1})}}
        increases = rocm_plib.compare_ecc_blocks_snapshots(before, after)
        self.assertEqual(increases[0]['block'], 'SDMA')
        self.assertEqual(increases[0]['field'], 'uncorrectable_count')

    def test_deferred_increase_warns(self):
        before = {'node1': {0: _zero_blocks()}}
        after = {'node1': {0: _blocks_with(GFX={'deferred_count': 2})}}
        with patch.object(rocm_plib.log, 'info') as mock_info:
            increases = rocm_plib.compare_ecc_blocks_snapshots(before, after)
        self.assertEqual(increases[0]['field'], 'deferred_count')
        self.assertEqual(increases[0]['delta'], 2)
        self.assertIn('+2', _logged_text(mock_info))

    def test_missing_block_after_treated_as_zero(self):
        before = {'node1': {0: _blocks_with(GFX={'correctable_count': 2})}}
        after = {'node1': {0: {}}}
        increases = rocm_plib.compare_ecc_blocks_snapshots(before, after)
        self.assertEqual(increases, [])

    def test_decrease_warns_but_is_not_an_increase(self):
        before = {'node1': {0: _blocks_with(HDP={'correctable_count': 4})}}
        after = {'node1': {0: _zero_blocks()}}
        with patch.object(rocm_plib.log, 'warning') as mock_warn, patch.object(rocm_plib.log, 'info') as mock_info:
            increases = rocm_plib.compare_ecc_blocks_snapshots(before, after)
        self.assertEqual(increases, [])
        self.assertIn('decreased', mock_warn.call_args[0][0])
        text = _logged_text(mock_info)
        self.assertIn('result=DECREASED', text)
        self.assertIn('-4', text)

    def test_mixed_increase_and_decrease(self):
        before = {'node1': {0: _blocks_with(UMC={'correctable_count': 1}, HDP={'correctable_count': 4})}}
        after = {'node1': {0: _blocks_with(UMC={'correctable_count': 5})}}
        with patch.object(rocm_plib.log, 'info') as mock_info:
            increases = rocm_plib.compare_ecc_blocks_snapshots(before, after)
        self.assertEqual(len(increases), 1)
        text = _logged_text(mock_info)
        self.assertIn('result=MIXED', text)
        self.assertIn('+4', text)
        self.assertIn('-4', text)

    def test_snapshot_log_is_debug_not_info(self):
        snap = {'node1': {0: _zero_blocks()}}
        with patch.object(rocm_plib.log, 'debug') as mock_debug, patch.object(rocm_plib.log, 'info') as mock_info:
            rocm_plib.log_ecc_blocks_snapshot('before', snap)
        mock_info.assert_not_called()
        line = _logged_text(mock_debug)
        self.assertIn('ECC_BLOCKS before:', line)
        for block in rocm_plib.RCCL_ECC_BLOCKS:
            self.assertIn(f'{block}.correctable_count=', line)
            self.assertIn(f'{block}.deferred_count=', line)


class TestEccDeltaCheckEnabled(unittest.TestCase):
    def test_default_false_when_key_missing(self):
        self.assertFalse(rocm_plib.ecc_delta_check_enabled({}))
        self.assertTrue(rocm_plib.ecc_delta_check_enabled({'cvs_params': {'verify_ecc_delta': 'True'}}))
        self.assertFalse(rocm_plib.ecc_delta_check_enabled({'cvs_params': {'verify_ecc_delta': 'False'}}))


class TestResolveEccBlocks(unittest.TestCase):
    def test_missing_key_returns_all_blocks(self):
        self.assertEqual(rocm_plib.resolve_ecc_blocks({}), rocm_plib.RCCL_ECC_BLOCKS)
        self.assertEqual(
            rocm_plib.resolve_ecc_blocks({'cvs_params': {'verify_ecc_blocks': []}}),
            rocm_plib.RCCL_ECC_BLOCKS,
        )

    def test_subset_from_list(self):
        blocks = rocm_plib.resolve_ecc_blocks({'cvs_params': {'verify_ecc_blocks': ['UMC', 'SDMA']}})
        self.assertEqual(blocks, ('UMC', 'SDMA'))

    def test_normalizes_case(self):
        blocks = rocm_plib.resolve_ecc_blocks({'cvs_params': {'verify_ecc_blocks': ['umc', 'xgmi_wafl']}})
        self.assertEqual(blocks, ('UMC', 'XGMI_WAFL'))

    def test_comma_separated_string(self):
        blocks = rocm_plib.resolve_ecc_blocks({'cvs_params': {'verify_ecc_blocks': 'UMC, GFX'}})
        self.assertEqual(blocks, ('UMC', 'GFX'))

    def test_unknown_entries_warn_and_fallback_to_all(self):
        with patch.object(rocm_plib.log, 'warning') as mock_warn:
            blocks = rocm_plib.resolve_ecc_blocks({'cvs_params': {'verify_ecc_blocks': ['FOO']}})
        self.assertEqual(blocks, rocm_plib.RCCL_ECC_BLOCKS)
        messages = [call[0][0] for call in mock_warn.call_args_list]
        self.assertIn('ECC_BLOCKS: unknown verify_ecc_blocks entries %s; ignoring', messages)
        self.assertIn('ECC_BLOCKS: no valid verify_ecc_blocks entries; using all blocks', messages)

    def test_mixed_valid_and_unknown_keeps_valid_only(self):
        with patch.object(rocm_plib.log, 'warning') as mock_warn:
            blocks = rocm_plib.resolve_ecc_blocks({'cvs_params': {'verify_ecc_blocks': ['UMC', 'FOO']}})
        self.assertEqual(blocks, ('UMC',))
        mock_warn.assert_called_once()

    def test_invalid_type_warns_and_returns_all(self):
        with patch.object(rocm_plib.log, 'warning') as mock_warn:
            blocks = rocm_plib.resolve_ecc_blocks({'cvs_params': {'verify_ecc_blocks': 42}})
        self.assertEqual(blocks, rocm_plib.RCCL_ECC_BLOCKS)
        self.assertIn('must be a list', mock_warn.call_args[0][0])


class TestFilteredEccBlocks(unittest.TestCase):
    def test_get_amd_smi_projects_subset_only(self):
        blocks = _blocks_with(UMC={'correctable_count': 1}, GFX={'correctable_count': 9})
        payload = json.dumps([_metric_gpu(0, blocks)])
        phdl = MagicMock()
        phdl.exec.return_value = {'node1': payload}

        result = rocm_plib.get_amd_smi_ecc_blocks_dict(phdl, blocks=('UMC',))
        self.assertEqual(set(result['node1'][0].keys()), {'UMC'})
        self.assertEqual(result['node1'][0]['UMC']['correctable_count'], 1)

    def test_compare_ignores_non_selected_blocks(self):
        before = {'node1': {0: _zero_blocks()}}
        after = {'node1': {0: _blocks_with(GFX={'correctable_count': 5})}}
        with patch.object(rocm_plib.log, 'info') as mock_info, patch.object(rocm_plib.log, 'warning') as mock_warn:
            increases = rocm_plib.compare_ecc_blocks_snapshots(before, after, blocks=('UMC',))
        self.assertEqual(increases, [])
        mock_warn.assert_not_called()
        text = _logged_text(mock_info)
        self.assertIn('UMC', text)
        self.assertNotIn('GFX', text)

    def test_compare_reports_selected_block_increase(self):
        before = {'node1': {0: _zero_blocks()}}
        after = {'node1': {0: _blocks_with(UMC={'correctable_count': 2})}}
        increases = rocm_plib.compare_ecc_blocks_snapshots(before, after, blocks=('UMC',))
        self.assertEqual(len(increases), 1)
        self.assertEqual(increases[0]['block'], 'UMC')


if __name__ == '__main__':
    unittest.main()
