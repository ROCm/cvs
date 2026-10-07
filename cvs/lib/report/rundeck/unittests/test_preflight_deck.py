'''Payload/render test for the preflight status_matrix deck.'''

import unittest

from cvs.lib.preflight.rundeck import build_preflight_deck
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.config_adapter import resolve_report_config
from cvs.lib.report.rundeck.payload import SummaryMetaApplier, build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html


def _deck():
    return build_preflight_deck(
        {
            'rocm_versions': {
                'n1': {
                    'status': 'FAIL',
                    'detected_version': '6.2.0',
                    'expected_version': '7.2.1',
                    'errors': ['ROCm version mismatch on n1'],
                }
            },
            'rdma_connectivity': {
                'status': 'SKIPPED',
                'skipped': True,
                'message': 'RDMA connectivity skipped by configuration',
            },
        },
        {'cluster_name': 'lab', 'node_dict': {'n1': {}}},
    )


class TestPreflightDeck(unittest.TestCase):
    def setUp(self):
        self.profile = load_json_profile('preflight_checks')
        deck = _deck()
        store = {'cvs_results_dict': deck, 'inf_res_dict': deck}
        self.payload = SummaryMetaApplier(resolve_report_config(self.profile)).apply(
            build_rundeck_payload(profile=self.profile, store=store, provenance={}, cvs_version='dev')
        )
        self.html = render_rundeck_html(self.payload)

    def test_profile_cards_overview_and_viewer(self):
        types = [card['type'] for card in self.profile['cards']]
        self.assertEqual(
            types,
            ['run_card', 'lifecycle_timeline', 'status_overview', 'metric_charts', 'status_matrix'],
        )
        metrics = next(card for card in self.profile['cards'] if card['type'] == 'metric_charts')
        self.assertEqual(metrics['title'], 'Perf sanity')
        self.assertEqual(metrics['when_empty'], 'hide')
        self.assertTrue(self.profile['interactive_viewer'])
        self.assertEqual(self.profile['sources']['results'], 'preflight_res_dict')
        self.assertEqual(self.profile['sources']['lifecycle'], 'lifecycle')
        self.assertEqual(self.profile['dataset_builder'], 'status_matrix')
        self.assertEqual(self.profile['report_basename'], 'preflight_run_deck')
        self.assertIn('Pass rate', self.html)
        self.assertIn('Columns are preflight checks', self.html)
        self.assertIn('preflight_run_deck_viewer.html', self.html)
        self.assertNotIn('Perf sanity', self.html)
        self.assertNotIn('item breakdown', self.html)

    def test_lifecycle_draws_recorded_checks(self):
        deck = _deck()
        store = {
            'cvs_results_dict': deck,
            'inf_res_dict': deck,
            'lifecycle_report': {'health': [('node_reachability', 0.5, 's'), ('node_smoke_tier1', 12.0, 's')]},
        }
        payload = build_rundeck_payload(profile=self.profile, store=store, provenance={}, cvs_version='dev')
        html = render_rundeck_html(payload)
        self.assertIn('Lifecycle timeline', html)
        self.assertIn('12.0s', html)
        self.assertIn('node smoke tier1', html)

    def test_overall_status_is_fail(self):
        self.assertEqual(self.payload['overall_status'], 'fail')

    def test_run_card_names_rocm_version(self):
        self.assertIn('ROCm version', self.html)
        self.assertIn('6.2.0', self.html)
        self.assertIn('lab', self.html)

    def test_status_matrix_shows_failed_check_and_skipped_rdma(self):
        self.assertIn('status-matrix', self.html)
        self.assertIn('rocm_versions', self.html)
        self.assertIn('rdma_connectivity', self.html)
        self.assertIn('n1', self.html)
        self.assertIn('ROCm version mismatch on n1', self.html)

    def test_no_inference_gate_vocabulary(self):
        self.assertNotIn('Gate matrix', self.html)
        self.assertNotIn('hover for tier status', self.html)


if __name__ == '__main__':
    unittest.main()
