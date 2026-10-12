'''Unit tests for the preflight Run Deck status-matrix capture.'''

import unittest

import pytest

from cvs.lib.preflight.rundeck import CHECK_IDS, build_preflight_deck, make_meta
from cvs.lib.report.profile import load_json_profile
from cvs.tests.preflight import preflight_checks


def _smoke_payload():
    return {
        'status': 'PASS',
        'tier1': {
            'per_gpu': [
                {'gpu': 0, 'status': 'PASS', 'details': {'gemm_tflops': 757.44, 'hbm_gbs': 4410.0}},
                {'gpu': 1, 'status': 'PASS', 'details': {'gemm_tflops': 700.0, 'hbm_gbs': 4300.0}},
            ],
            'gpu_processes': {'ok': True},
            'nics': {'expected_count': 8, 'issues': []},
            'host_limits': {'fail_reasons': []},
            'gpu_low_level': {'ok': True},
            'xgmi': {'ok': True},
            'tooling': {'ok': True},
            'gpu_visibility': {'fail_reasons': [], 'torch_visible': 8},
        },
        'tier2': {'rccl': {'status': 'PASS', 'gbs': 277.3}},
    }


class TestPreflightDeck(unittest.TestCase):
    def test_meta_uses_cluster_and_detected_rocm(self):
        meta = make_meta({'cluster_name': 'lab'}, 'preflight_checks', '7.2.1')
        self.assertEqual(meta['cluster'], 'lab')
        self.assertEqual(meta['version'], '7.2.1')
        self.assertEqual(meta['version_label'], 'ROCm version')
        deck = build_preflight_deck(
            {
                'rocm_versions': {
                    'n1': {'status': 'PASS', 'detected_version': '7.2.1', 'expected_version': '7.2.1', 'errors': []},
                    'n2': {
                        'status': 'FAIL',
                        'detected_version': '6.2.0',
                        'expected_version': '7.2.1',
                        'errors': ['mismatch'],
                    },
                }
            },
            {'cluster_name': 'lab', 'node_dict': {'n1': {}, 'n2': {}}},
        )
        self.assertEqual(deck['_meta']['version'], '7.2.1')
        self.assertEqual(deck['groups']['rocm_versions']['nodes']['n1']['status'], 'pass')
        self.assertEqual(deck['groups']['rocm_versions']['nodes']['n2']['status'], 'fail')
        self.assertIn('6.2.0', deck['groups']['rocm_versions']['nodes']['n2']['items_summary'])

    def test_skipped_rdma_is_na_on_every_node(self):
        deck = build_preflight_deck(
            {
                'rdma_connectivity': {
                    'status': 'SKIPPED',
                    'skipped': True,
                    'message': 'RDMA interface, GID, and connectivity validation skipped by configuration',
                }
            },
            {'node_dict': {'n1': {}, 'n2': {}}},
        )
        nodes = deck['groups']['rdma_connectivity']['nodes']
        self.assertEqual(set(nodes), {'n1', 'n2'})
        self.assertTrue(all(record['status'] == 'na' for record in nodes.values()))
        self.assertIn('skipped by configuration', nodes['n1']['items_summary'])

    def test_reachability_marks_unreachable_nodes(self):
        deck = build_preflight_deck(
            {
                'node_reachability': {
                    'status': 'WARNING',
                    'unreachable_nodes': ['n2'],
                    'nodes': {'n1': {'status': 'PASS'}, 'n2': {'status': 'FAIL'}},
                }
            },
            {'node_dict': {'n1': {}, 'n2': {}}},
        )
        nodes = deck['groups']['node_reachability']['nodes']
        self.assertEqual(nodes['n1']['status'], 'pass')
        self.assertEqual(nodes['n2']['status'], 'fail')

    def test_node_smoke_scores_reported_checks_and_keeps_gemm_numbers(self):
        payload = _smoke_payload()
        deck = build_preflight_deck(
            {
                'node_smoke_tier1': {
                    'status': 'PASS',
                    'tier2_perf': True,
                    'gpus_per_node': 2,
                    'tier2_thresholds': {'gemm_tflops_min': 600, 'hbm_gbs_min': 2000, 'rccl_gbs_min': 100},
                    'node_results': {'n1': {'status': 'PASS', 'fail_reasons': [], 'node_payload': payload}},
                }
            },
            {'node_dict': {'n1': {}}},
        )
        tier1 = deck['groups']['node_smoke_tier1']['nodes']['n1']
        self.assertEqual(tier1['status'], 'pass')
        by_name = {item['name']: item for item in tier1['items']}
        self.assertEqual(by_name['GPU 0']['status'], 'pass')
        self.assertEqual(by_name['RDMA NICs']['status'], 'na')
        self.assertEqual(by_name['Host limits']['status'], 'na')
        self.assertEqual(by_name['GPU visibility']['status'], 'na')
        self.assertEqual(by_name['GPU processes']['status'], 'pass')

        tier2 = deck['groups']['node_smoke_tier2']['nodes']['n1']
        self.assertEqual(tier2['status'], 'pass')
        tier2_items = {item['name']: item for item in tier2['items']}
        self.assertEqual(tier2_items['Local RCCL all-reduce']['status'], 'pass')
        self.assertEqual(tier2_items['GPU 0 Large GEMM TFLOPS']['status'], 'na')
        self.assertIn('757.44', tier2_items['GPU 0 Large GEMM TFLOPS']['message'])
        gemm = next(metric for metric in tier2['metrics'] if metric['name'] == 'large_gemm')
        self.assertEqual(gemm['value'], 700.0)
        self.assertEqual(gemm['status'], 'pass')
        self.assertEqual(gemm['threshold'], 600)
        self.assertTrue(tier2['series'])

    def test_tier2_group_omitted_when_perf_disabled(self):
        deck = build_preflight_deck(
            {
                'node_smoke_tier1': {
                    'status': 'PASS',
                    'tier2_perf': False,
                    'gpus_per_node': 2,
                    'node_results': {
                        'n1': {'status': 'PASS', 'node_payload': {'tier1': {'per_gpu': [{'gpu': 0, 'status': 'PASS'}]}}}
                    },
                }
            }
        )
        self.assertIn('node_smoke_tier1', deck['groups'])
        self.assertNotIn('node_smoke_tier2', deck['groups'])

    def test_profile_labels_match_check_ids(self):
        profile = load_json_profile('preflight_checks')
        self.assertEqual(profile['lifecycle']['session_labels'], list(CHECK_IDS))
        self.assertEqual(profile['sources']['results'], 'preflight_res_dict')

    def test_publish_fills_the_slot_before_a_skip(self):
        previous_slot = dict(preflight_checks._rundeck_slot)
        previous_results = dict(preflight_checks.preflight_results)
        target = {}
        try:
            preflight_checks.preflight_results.clear()
            preflight_checks.preflight_results['gid_consistency'] = {
                'status': 'SKIPPED',
                'skipped': True,
                'message': 'RDMA GID validation skipped because RDMA connectivity mode is skip',
            }
            preflight_checks._rundeck_slot['target'] = target
            preflight_checks._rundeck_slot['cluster'] = {'cluster_name': 'lab', 'node_dict': {'n1': {}}}
            with self.assertRaises(pytest.skip.Exception):
                preflight_checks.preflight_update_test_result(preflight_checks.preflight_results['gid_consistency'])
            self.assertEqual(target['groups']['gid_consistency']['nodes']['n1']['status'], 'na')
            self.assertEqual(target['_meta']['cluster'], 'lab')
        finally:
            preflight_checks._rundeck_slot.clear()
            preflight_checks._rundeck_slot.update(previous_slot)
            preflight_checks.preflight_results.clear()
            preflight_checks.preflight_results.update(previous_results)


if __name__ == '__main__':
    unittest.main()
