"""Unit tests for the IFoE TransferBench smoketest preflight check (AIMVT-181).

All fixtures are entirely synthetic — generic node names, generic vpod/ppod
IDs, and synthesised TransferBench output that mimics the documented
candidate-branch smoketest format. No real cluster serial numbers, MAC
addresses, or hostnames are embedded.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.transferbench_smoke import (
    EXIT_SENTINEL,
    SmoketestParser,
    TransferBenchSmokeCheck,
    evaluate_smoketest,
    extract_node_pod_membership,
    parse_amd_smi_fabric_text,
    reconcile_cluster_vpod,
)


# ---------------------------------------------------------------------------
# amd-smi fabric topology fixtures (synthetic schema, generic IDs)
# ---------------------------------------------------------------------------

FABRIC_TOPOLOGY_VPOD0_4GPU = [
    {
        'gpu': 0,
        'bdf': '0000:01:00.0',
        'fabric': {'ppod_id': 0, 'vpod_id': 0, 'ppod_size': 4, 'vpod_size': 4},
    },
    {
        'gpu': 1,
        'bdf': '0001:01:00.0',
        'fabric': {'ppod_id': 0, 'vpod_id': 0, 'ppod_size': 4, 'vpod_size': 4},
    },
    {
        'gpu': 2,
        'bdf': '0002:01:00.0',
        'fabric': {'ppod_id': 0, 'vpod_id': 0, 'ppod_size': 4, 'vpod_size': 4},
    },
    {
        'gpu': 3,
        'bdf': '0003:01:00.0',
        'fabric': {'ppod_id': 0, 'vpod_id': 0, 'ppod_size': 4, 'vpod_size': 4},
    },
]

FABRIC_TOPOLOGY_VPOD1_4GPU = [
    {
        'gpu': g,
        'bdf': f'{g:04d}:01:00.0',
        'fabric': {'ppod_id': 1, 'vpod_id': 1, 'ppod_size': 4, 'vpod_size': 4},
    }
    for g in range(4)
]

FABRIC_TOPOLOGY_MIXED_VPODS_ON_ONE_NODE = [
    {'gpu': 0, 'fabric': {'ppod_id': 0, 'vpod_id': 0, 'vpod_size': 4}},
    {'gpu': 1, 'fabric': {'ppod_id': 0, 'vpod_id': 0, 'vpod_size': 4}},
    {'gpu': 2, 'fabric': {'ppod_id': 0, 'vpod_id': 1, 'vpod_size': 4}},
    {'gpu': 3, 'fabric': {'ppod_id': 0, 'vpod_id': 1, 'vpod_size': 4}},
]

FABRIC_TOPOLOGY_GPU_DATA_WRAPPER = {
    'gpu_data': FABRIC_TOPOLOGY_VPOD0_4GPU,
}

FABRIC_TOPOLOGY_KEYED = {
    'gpu0': FABRIC_TOPOLOGY_VPOD0_4GPU[0],
    'gpu1': FABRIC_TOPOLOGY_VPOD0_4GPU[1],
    'gpu2': FABRIC_TOPOLOGY_VPOD0_4GPU[2],
    'gpu3': FABRIC_TOPOLOGY_VPOD0_4GPU[3],
}

FABRIC_TOPOLOGY_NO_POD = [{'gpu': 0, 'bdf': '0000:01:00.0'}]

FABRIC_TEXT_VPOD0 = """\
GPU 0
  PPOD_ID  : 0
  VPOD_ID  : 0
  PPOD_SIZE: 4
  VPOD_SIZE: 4

GPU 1
  PPOD_ID  : 0
  VPOD_ID  : 0
  PPOD_SIZE: 4
  VPOD_SIZE: 4
"""

FABRIC_TEXT_VPOD2 = """\
GPU 0
  PPOD_ID  : 2
  VPOD_ID  : 2
  PPOD_SIZE: 4
  VPOD_SIZE: 4
"""


# ---------------------------------------------------------------------------
# TransferBench smoketest output fixtures (synthetic shape that exercises
# every code path the parser is asked to handle).
# ---------------------------------------------------------------------------

SMOKETEST_PASS_OUTPUT = """\
TransferBench (candidate) starting smoketest preset
Detected 4 local GPUs in vpod_id=0, ppod_id=0

Running 16 tests with sizes [1K, 16M], 2 iterations, validate=1
Test  1: H2D BDMA           [PASS]   bw= 25.0 GB/s
Test  2: D2H BDMA           [PASS]   bw= 25.1 GB/s
Test  3: D2D SDMA           [PASS]   bw= 64.0 GB/s
Test  4: D2D GFX            [PASS]   bw= 70.2 GB/s
Test  5: Broadcast SDMA     [PASS]   bw= 60.0 GB/s
Test  6: Broadcast GFX      [PASS]   bw= 65.5 GB/s
Test  7: Gather SDMA        [PASS]   bw= 60.0 GB/s
Test  8: Gather GFX         [PASS]   bw= 65.5 GB/s
Test  9: A2A SDMA           [PASS]   bw=200.0 GB/s
Test 10: A2A GFX            [PASS]   bw=220.0 GB/s
Test 11: H2D-validate       [PASS]
Test 12: D2H-validate       [PASS]
Test 13: D2D-validate       [PASS]
Test 14: Broadcast-validate [PASS]
Test 15: Gather-validate    [PASS]
Test 16: A2A-validate       [PASS]

Smoketest summary: 16/16 PASS, 0 FAIL, 0 SKIP
"""

SMOKETEST_FAIL_OUTPUT = """\
TransferBench (candidate) starting smoketest preset
Detected 4 local GPUs in vpod_id=0, ppod_id=0

Test  1: H2D BDMA           [PASS]   bw= 25.0 GB/s
Test  2: D2H BDMA           [PASS]   bw= 25.1 GB/s
Test  3: D2D SDMA           [FAIL]   error=validation mismatch
Test  4: D2D GFX            [FAIL]   error=hipMemcpyDtoD returned -1
Test  5: Broadcast SDMA     [PASS]   bw= 60.0 GB/s
Test  6: Broadcast GFX      [PASS]   bw= 65.5 GB/s
Test  7: Gather SDMA        [PASS]   bw= 60.0 GB/s
Test  8: Gather GFX         [PASS]   bw= 65.5 GB/s
Test  9: A2A SDMA           [PASS]   bw=200.0 GB/s
Test 10: A2A GFX            [PASS]   bw=220.0 GB/s
Test 11: H2D-validate       [PASS]
Test 12: D2H-validate       [PASS]
Test 13: D2D-validate       [SKIP]   reason=preceding D2D test failed
Test 14: Broadcast-validate [PASS]
Test 15: Gather-validate    [PASS]
Test 16: A2A-validate       [PASS]

Smoketest summary: 13/16 PASS, 2 FAIL, 1 SKIP
"""

SMOKETEST_SKIP_HEAVY_OUTPUT = """\
TransferBench (candidate) starting smoketest preset
Detected 1 local GPU in vpod_id=0, ppod_id=0

Test  1: H2D BDMA           [PASS]   bw= 25.0 GB/s
Test  2: D2H BDMA           [PASS]   bw= 25.1 GB/s
Test  3: D2D SDMA           [SKIP]   reason=requires >= 2 local GPUs
Test  4: D2D GFX            [SKIP]   reason=requires >= 2 local GPUs
Test  5: Broadcast SDMA     [SKIP]   reason=requires >= 2 local GPUs
Test  6: Broadcast GFX      [SKIP]   reason=requires >= 2 local GPUs
Test  7: Gather SDMA        [SKIP]   reason=requires >= 2 local GPUs
Test  8: Gather GFX         [SKIP]   reason=requires >= 2 local GPUs
Test  9: A2A SDMA           [SKIP]   reason=requires >= 2 local GPUs
Test 10: A2A GFX            [SKIP]   reason=requires >= 2 local GPUs
Test 11: H2D-validate       [PASS]
Test 12: D2H-validate       [PASS]
Test 13: D2D-validate       [SKIP]   reason=requires >= 2 local GPUs
Test 14: Broadcast-validate [SKIP]   reason=requires >= 2 local GPUs
Test 15: Gather-validate    [SKIP]   reason=requires >= 2 local GPUs
Test 16: A2A-validate       [SKIP]   reason=requires >= 2 local GPUs

Smoketest summary: 4/16 PASS, 0 FAIL, 12 SKIP
"""

SMOKETEST_FATAL_PRECONDITION_OUTPUT = """\
TransferBench (candidate) starting smoketest preset
FATAL: ranks span multiple vPods (rank 0 vpod_id=0, rank 1 vpod_id=1)
ERR_FATAL: pod-membership precondition failed
"""

SMOKETEST_MARKER_TABLE_OUTPUT = """\
TransferBench (candidate) starting smoketest preset
   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
**.**F***.******
"""


def _with_sentinel(output: str, exit_code: int) -> str:
    return output.rstrip('\n') + f'\n{EXIT_SENTINEL}={exit_code}\n'


# ---------------------------------------------------------------------------
# Pod-membership parser tests
# ---------------------------------------------------------------------------


class TestExtractNodePodMembership(unittest.TestCase):
    def test_flat_list_payload(self):
        m = extract_node_pod_membership(FABRIC_TOPOLOGY_VPOD0_4GPU)
        self.assertEqual(m['gpus'], 4)
        self.assertEqual(m['vpod_ids'], [0])
        self.assertEqual(m['ppod_ids'], [0])
        self.assertEqual(m['vpod_size'], 4)
        self.assertEqual(m['ppod_size'], 4)
        self.assertEqual(m['errors'], [])

    def test_gpu_data_wrapper_payload(self):
        m = extract_node_pod_membership(FABRIC_TOPOLOGY_GPU_DATA_WRAPPER)
        self.assertEqual(m['gpus'], 4)
        self.assertEqual(m['vpod_ids'], [0])

    def test_keyed_dict_payload(self):
        m = extract_node_pod_membership(FABRIC_TOPOLOGY_KEYED)
        self.assertEqual(m['gpus'], 4)
        self.assertEqual(m['vpod_ids'], [0])

    def test_mixed_vpods_within_node_reported_as_error(self):
        m = extract_node_pod_membership(FABRIC_TOPOLOGY_MIXED_VPODS_ON_ONE_NODE)
        self.assertEqual(sorted(m['vpod_ids']), [0, 1])
        self.assertTrue(m['errors'])
        self.assertIn('Multiple vPod IDs', m['errors'][0])

    def test_payload_with_no_pod_fields(self):
        m = extract_node_pod_membership(FABRIC_TOPOLOGY_NO_POD)
        self.assertEqual(m['gpus'], 0)
        self.assertTrue(m['errors'])

    def test_plaintext_fallback(self):
        m = extract_node_pod_membership(FABRIC_TEXT_VPOD0)
        self.assertEqual(m['gpus'], 2)
        self.assertEqual(m['vpod_ids'], [0])
        self.assertEqual(m['ppod_ids'], [0])

    def test_plaintext_garbage_inputs(self):
        m = extract_node_pod_membership('bash: amd-smi: command not found\n')
        self.assertEqual(m['gpus'], 0)
        self.assertTrue(m['errors'])

    def test_parse_amd_smi_fabric_text_extracts_multiple_blocks(self):
        blocks = parse_amd_smi_fabric_text(FABRIC_TEXT_VPOD0)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]['vpod_id'], 0)
        self.assertEqual(blocks[0]['ppod_id'], 0)


class TestReconcileClusterVpod(unittest.TestCase):
    def test_uniform_vpod_passes(self):
        per_node = {
            'nodeA': extract_node_pod_membership(FABRIC_TOPOLOGY_VPOD0_4GPU),
            'nodeB': extract_node_pod_membership(FABRIC_TOPOLOGY_VPOD0_4GPU),
        }
        rec = reconcile_cluster_vpod(per_node)
        self.assertEqual(rec['status'], 'PASS')
        self.assertEqual(rec['vpod_id'], 0)
        self.assertEqual(rec['ppod_id'], 0)
        self.assertEqual(rec['errors'], [])

    def test_split_vpod_fails(self):
        per_node = {
            'nodeA': extract_node_pod_membership(FABRIC_TOPOLOGY_VPOD0_4GPU),
            'nodeB': extract_node_pod_membership(FABRIC_TOPOLOGY_VPOD1_4GPU),
        }
        rec = reconcile_cluster_vpod(per_node)
        self.assertEqual(rec['status'], 'FAIL')
        self.assertIsNone(rec['vpod_id'])
        self.assertTrue(any('multiple vPods' in e for e in rec['errors']))

    def test_missing_vpod_on_some_nodes_fails(self):
        per_node = {
            'nodeA': extract_node_pod_membership(FABRIC_TOPOLOGY_VPOD0_4GPU),
            'nodeB': extract_node_pod_membership(FABRIC_TOPOLOGY_NO_POD),
        }
        rec = reconcile_cluster_vpod(per_node)
        self.assertEqual(rec['status'], 'FAIL')
        self.assertTrue(rec['errors'])

    def test_empty_input_fails(self):
        rec = reconcile_cluster_vpod({})
        self.assertEqual(rec['status'], 'FAIL')

    def test_multi_local_vpod_fails(self):
        per_node = {
            'nodeA': extract_node_pod_membership(FABRIC_TOPOLOGY_MIXED_VPODS_ON_ONE_NODE),
            'nodeB': extract_node_pod_membership(FABRIC_TOPOLOGY_VPOD0_4GPU),
        }
        rec = reconcile_cluster_vpod(per_node)
        self.assertEqual(rec['status'], 'FAIL')


# ---------------------------------------------------------------------------
# Smoketest parser tests
# ---------------------------------------------------------------------------


class TestSmoketestParser(unittest.TestCase):
    def test_passing_output(self):
        parsed = SmoketestParser.parse(_with_sentinel(SMOKETEST_PASS_OUTPUT, 0))
        self.assertEqual(parsed['exit_code'], 0)
        self.assertEqual(parsed['num_pass'], 16)
        self.assertEqual(parsed['num_fail'], 0)
        self.assertEqual(parsed['num_skip'], 0)
        self.assertEqual(parsed['num_tests'], 16)
        self.assertEqual(parsed['summary_total'], 16)
        self.assertEqual(parsed['summary_pass'], 16)
        self.assertEqual(parsed['errors'], [])

    def test_failing_output(self):
        parsed = SmoketestParser.parse(_with_sentinel(SMOKETEST_FAIL_OUTPUT, 1))
        self.assertEqual(parsed['exit_code'], 1)
        self.assertEqual(parsed['num_pass'], 13)
        self.assertEqual(parsed['num_fail'], 2)
        self.assertEqual(parsed['num_skip'], 1)
        self.assertEqual(parsed['summary_total'], 16)

    def test_skip_heavy_output(self):
        parsed = SmoketestParser.parse(_with_sentinel(SMOKETEST_SKIP_HEAVY_OUTPUT, 0))
        self.assertEqual(parsed['exit_code'], 0)
        self.assertEqual(parsed['num_pass'], 4)
        self.assertEqual(parsed['num_fail'], 0)
        self.assertEqual(parsed['num_skip'], 12)
        self.assertEqual(parsed['num_tests'], 16)
        self.assertTrue(parsed['warnings'])

    def test_fatal_precondition_output(self):
        parsed = SmoketestParser.parse(_with_sentinel(SMOKETEST_FATAL_PRECONDITION_OUTPUT, 2))
        self.assertEqual(parsed['exit_code'], 2)
        self.assertEqual(parsed['num_tests'], 0)
        self.assertTrue(any('ERR_FATAL' in e for e in parsed['errors']))

    def test_marker_table_fallback(self):
        parsed = SmoketestParser.parse(_with_sentinel(SMOKETEST_MARKER_TABLE_OUTPUT, 1))
        self.assertEqual(parsed['exit_code'], 1)
        self.assertEqual(parsed['num_fail'], 1)
        self.assertGreaterEqual(parsed['num_pass'], 10)
        self.assertGreaterEqual(parsed['num_skip'], 2)

    def test_empty_output(self):
        parsed = SmoketestParser.parse('')
        self.assertIsNone(parsed['exit_code'])
        self.assertTrue(parsed['parse_errors'])

    def test_garbage_output_no_sentinel(self):
        parsed = SmoketestParser.parse('bash: TransferBench: command not found\n')
        self.assertIsNone(parsed['exit_code'])
        self.assertEqual(parsed['num_tests'], 0)


# ---------------------------------------------------------------------------
# Verdict logic tests
# ---------------------------------------------------------------------------


class TestEvaluateSmoketest(unittest.TestCase):
    def test_pass_verdict(self):
        parsed = SmoketestParser.parse(_with_sentinel(SMOKETEST_PASS_OUTPUT, 0))
        verdict, errors = evaluate_smoketest(parsed, max_skip_pct=25.0)
        self.assertEqual(verdict, 'PASS')
        self.assertEqual(errors, [])

    def test_fail_on_exit_code(self):
        parsed = SmoketestParser.parse(_with_sentinel(SMOKETEST_FAIL_OUTPUT, 1))
        verdict, errors = evaluate_smoketest(parsed)
        self.assertEqual(verdict, 'FAIL')
        self.assertTrue(any('non-zero exit code' in e for e in errors))

    def test_fail_on_precondition_exit_2(self):
        parsed = SmoketestParser.parse(
            _with_sentinel(SMOKETEST_FATAL_PRECONDITION_OUTPUT, 2)
        )
        verdict, errors = evaluate_smoketest(parsed)
        self.assertEqual(verdict, 'FAIL')
        self.assertTrue(any('ERR_FATAL precondition' in e for e in errors))

    def test_fail_when_sentinel_missing(self):
        parsed = SmoketestParser.parse(SMOKETEST_PASS_OUTPUT)
        verdict, errors = evaluate_smoketest(parsed)
        self.assertEqual(verdict, 'FAIL')
        self.assertTrue(any('exit code not captured' in e for e in errors))

    def test_warning_on_excessive_skips(self):
        parsed = SmoketestParser.parse(_with_sentinel(SMOKETEST_SKIP_HEAVY_OUTPUT, 0))
        verdict, errors = evaluate_smoketest(parsed, max_skip_pct=25.0)
        self.assertEqual(verdict, 'WARNING')
        self.assertTrue(any('skipped' in e.lower() for e in errors))

    def test_pass_when_skips_within_budget(self):
        parsed = SmoketestParser.parse(_with_sentinel(SMOKETEST_SKIP_HEAVY_OUTPUT, 0))
        verdict, _ = evaluate_smoketest(parsed, max_skip_pct=80.0)
        self.assertEqual(verdict, 'PASS')

    def test_fail_when_exit_zero_but_fail_markers_present(self):
        bogus = SMOKETEST_FAIL_OUTPUT  # has FAIL cells
        parsed = SmoketestParser.parse(_with_sentinel(bogus, 0))
        verdict, errors = evaluate_smoketest(parsed)
        self.assertEqual(verdict, 'FAIL')


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


class TestTransferBenchSmokeCheck(unittest.TestCase):
    def _make_phdl(self, reachable_hosts, exec_responses=None, cmd_list_responses=None):
        phdl = MagicMock()
        phdl.reachable_hosts = list(reachable_hosts)
        if exec_responses is not None:
            phdl.exec = MagicMock(side_effect=list(exec_responses))
        else:
            phdl.exec = MagicMock(return_value={})
        if cmd_list_responses is not None:
            phdl.exec_cmd_list = MagicMock(side_effect=list(cmd_list_responses))
        else:
            phdl.exec_cmd_list = MagicMock(return_value={})
        return phdl

    def _fabric_payload(self, payload):
        return json.dumps(payload)

    def test_build_command_per_node_defaults(self):
        check = TransferBenchSmokeCheck(MagicMock())
        cmd = check.build_command(rank=0, num_ranks=1, master_addr='127.0.0.1')
        self.assertIn('TransferBench', cmd)
        self.assertIn('smoketest', cmd)
        self.assertIn('NUM_ITERATIONS=2', cmd)
        self.assertIn('ALWAYS_VALIDATE=1', cmd)
        self.assertIn('RUN_PARALLEL=1', cmd)
        self.assertIn('FORCE_SINGLE_POD=1', cmd)
        self.assertIn(EXIT_SENTINEL, cmd)
        self.assertNotIn('TB_NUM_RANKS=', cmd)
        self.assertNotIn('TB_RANK=', cmd)

    def test_build_command_multi_rank_includes_socket_env(self):
        check = TransferBenchSmokeCheck(MagicMock(), rank_mode='multi_rank')
        cmd = check.build_command(rank=2, num_ranks=4, master_addr='10.0.0.1')
        self.assertIn('TB_NUM_RANKS=4', cmd)
        self.assertIn('TB_RANK=2', cmd)
        self.assertIn('TB_MASTER_ADDR=10.0.0.1', cmd)
        self.assertIn('TB_MASTER_PORT=31337', cmd)

    def test_build_command_respects_sudo_and_rocm_path(self):
        check = TransferBenchSmokeCheck(
            MagicMock(),
            use_sudo=True,
            rocm_path='/opt/rocm',
        )
        cmd = check.build_command(rank=0, num_ranks=1, master_addr='127.0.0.1')
        self.assertTrue(cmd.startswith('sudo '))
        self.assertIn('/opt/rocm/bin', cmd)
        self.assertIn('/opt/rocm/lib', cmd)

    def test_run_pass_per_node(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA', 'nodeB'],
            exec_responses=[
                {
                    'nodeA': self._fabric_payload(FABRIC_TOPOLOGY_VPOD0_4GPU),
                    'nodeB': self._fabric_payload(FABRIC_TOPOLOGY_VPOD0_4GPU),
                },
                {
                    'nodeA': _with_sentinel(SMOKETEST_PASS_OUTPUT, 0),
                    'nodeB': _with_sentinel(SMOKETEST_PASS_OUTPUT, 0),
                },
            ],
        )
        check = TransferBenchSmokeCheck(phdl, rank_mode='per_node')
        results = check.run()
        self.assertEqual(results['status'], 'PASS')
        self.assertEqual(results['pod_membership']['status'], 'PASS')
        self.assertEqual(results['pod_membership']['vpod_id'], 0)
        self.assertEqual(results['totals']['nodes_pass'], 2)
        for node in ('nodeA', 'nodeB'):
            self.assertEqual(results['nodes'][node]['status'], 'PASS')
            self.assertEqual(results['nodes'][node]['exit_code'], 0)

    def test_run_fails_when_vpods_diverge(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA', 'nodeB'],
            exec_responses=[
                {
                    'nodeA': self._fabric_payload(FABRIC_TOPOLOGY_VPOD0_4GPU),
                    'nodeB': self._fabric_payload(FABRIC_TOPOLOGY_VPOD1_4GPU),
                },
            ],
        )
        check = TransferBenchSmokeCheck(phdl, rank_mode='multi_rank')
        results = check.run()
        self.assertEqual(results['status'], 'FAIL')
        self.assertEqual(results['pod_membership']['status'], 'FAIL')
        for node in ('nodeA', 'nodeB'):
            self.assertEqual(results['nodes'][node]['status'], 'SKIPPED')
        phdl.exec_cmd_list.assert_not_called()

    def test_run_fails_when_one_node_smoketest_fails(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA', 'nodeB'],
            exec_responses=[
                {
                    'nodeA': self._fabric_payload(FABRIC_TOPOLOGY_VPOD0_4GPU),
                    'nodeB': self._fabric_payload(FABRIC_TOPOLOGY_VPOD0_4GPU),
                },
                {
                    'nodeA': _with_sentinel(SMOKETEST_PASS_OUTPUT, 0),
                    'nodeB': _with_sentinel(SMOKETEST_FAIL_OUTPUT, 1),
                },
            ],
        )
        check = TransferBenchSmokeCheck(phdl, rank_mode='per_node')
        results = check.run()
        self.assertEqual(results['status'], 'FAIL')
        self.assertEqual(results['nodes']['nodeA']['status'], 'PASS')
        self.assertEqual(results['nodes']['nodeB']['status'], 'FAIL')
        self.assertEqual(results['totals']['nodes_pass'], 1)
        self.assertEqual(results['totals']['nodes_fail'], 1)
        self.assertEqual(results['totals']['tests_fail'], 2)

    def test_run_warning_when_skips_exceed_budget(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA'],
            exec_responses=[
                {'nodeA': self._fabric_payload(FABRIC_TOPOLOGY_VPOD0_4GPU)},
                {'nodeA': _with_sentinel(SMOKETEST_SKIP_HEAVY_OUTPUT, 0)},
            ],
        )
        check = TransferBenchSmokeCheck(phdl, rank_mode='per_node', max_skip_pct=25.0)
        results = check.run()
        self.assertEqual(results['status'], 'WARNING')
        self.assertEqual(results['nodes']['nodeA']['status'], 'WARNING')
        self.assertEqual(results['totals']['nodes_warning'], 1)

    def test_run_multi_rank_dispatch_uses_cmd_list(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeB', 'nodeA'],
            exec_responses=[
                {
                    'nodeA': self._fabric_payload(FABRIC_TOPOLOGY_VPOD0_4GPU),
                    'nodeB': self._fabric_payload(FABRIC_TOPOLOGY_VPOD0_4GPU),
                },
            ],
            cmd_list_responses=[
                {
                    'nodeA': _with_sentinel(SMOKETEST_PASS_OUTPUT, 0),
                    'nodeB': _with_sentinel(SMOKETEST_PASS_OUTPUT, 0),
                }
            ],
        )
        check = TransferBenchSmokeCheck(phdl, rank_mode='multi_rank')
        results = check.run()
        self.assertEqual(results['status'], 'PASS')
        self.assertEqual(results['rank_mode'], 'multi_rank')
        phdl.exec_cmd_list.assert_called_once()
        args, kwargs = phdl.exec_cmd_list.call_args
        cmd_list_arg = args[0]
        self.assertEqual(len(cmd_list_arg), 2)
        joined = '\n'.join(cmd_list_arg)
        self.assertIn('TB_NUM_RANKS=2', joined)
        self.assertIn('TB_RANK=0', joined)
        self.assertIn('TB_RANK=1', joined)
        self.assertIn('TB_MASTER_ADDR=nodeA', joined)

    def test_multi_rank_degrades_to_per_node_with_one_host(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA'],
            exec_responses=[
                {'nodeA': self._fabric_payload(FABRIC_TOPOLOGY_VPOD0_4GPU)},
                {'nodeA': _with_sentinel(SMOKETEST_PASS_OUTPUT, 0)},
            ],
        )
        check = TransferBenchSmokeCheck(phdl, rank_mode='multi_rank')
        results = check.run()
        self.assertEqual(results['rank_mode'], 'per_node')
        self.assertEqual(results['status'], 'PASS')
        phdl.exec_cmd_list.assert_not_called()

    def test_skip_pod_check_bypasses_precondition(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA'],
            exec_responses=[
                {'nodeA': _with_sentinel(SMOKETEST_PASS_OUTPUT, 0)},
            ],
        )
        check = TransferBenchSmokeCheck(phdl, rank_mode='per_node', skip_pod_check=True)
        results = check.run()
        self.assertEqual(results['pod_membership']['status'], 'SKIPPED')
        self.assertEqual(results['status'], 'PASS')
        self.assertEqual(phdl.exec.call_count, 1)

    def test_run_fail_on_precondition_exit_2(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA'],
            exec_responses=[
                {'nodeA': self._fabric_payload(FABRIC_TOPOLOGY_VPOD0_4GPU)},
                {'nodeA': _with_sentinel(SMOKETEST_FATAL_PRECONDITION_OUTPUT, 2)},
            ],
        )
        check = TransferBenchSmokeCheck(phdl, rank_mode='per_node')
        results = check.run()
        self.assertEqual(results['status'], 'FAIL')
        node_result = results['nodes']['nodeA']
        self.assertEqual(node_result['status'], 'FAIL')
        self.assertEqual(node_result['exit_code'], 2)
        self.assertTrue(any('ERR_FATAL' in e for e in node_result['errors']))

    def test_run_handles_plaintext_fabric_output(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA'],
            exec_responses=[
                {'nodeA': FABRIC_TEXT_VPOD0},
                {'nodeA': _with_sentinel(SMOKETEST_PASS_OUTPUT, 0)},
            ],
        )
        check = TransferBenchSmokeCheck(phdl, rank_mode='per_node')
        results = check.run()
        self.assertEqual(results['status'], 'PASS')
        self.assertEqual(results['pod_membership']['vpod_id'], 0)

    def test_run_fails_when_no_reachable_hosts(self):
        phdl = self._make_phdl(reachable_hosts=[])
        check = TransferBenchSmokeCheck(phdl)
        results = check.run()
        self.assertEqual(results['status'], 'FAIL')
        self.assertTrue(any('No reachable hosts' in e for e in results['errors']))


if __name__ == '__main__':
    unittest.main()
