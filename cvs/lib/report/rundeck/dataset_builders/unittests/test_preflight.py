'''Unit tests for the preflight Run Deck dataset builder.'''

import unittest

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.dataset_builders.preflight import build_preflight_datasets
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html


def _sample_results():
    return {
        'node_reachability': {
            'total_nodes': 2,
            'reachable_nodes': 1,
            'unreachable_nodes': ['node-b'],
            'status': 'WARNING',
        },
        'node_health': {
            'status': 'PASS',
            'node_results': {
                'node-a': {'status': 'PASS'},
                'node-b': {'status': 'PASS'},
            },
        },
        'gid_consistency': {
            'node-a': {'status': 'PASS'},
            'node-b': {'status': 'FAIL'},
        },
        'rdma_connectivity': {
            'status': 'BLOCKED',
            'blocked': True,
            'node_status': {},
        },
        'summary': {
            'overall_status': 'FAIL',
            'checks': {
                'ssh_reachability': {
                    'status': 'WARNING',
                    'unreachable_nodes': ['node-b'],
                    'summary': '1/2 nodes reachable',
                },
                'node_health': {
                    'status': 'PASS',
                    'summary': '2/2 nodes passed node-health admission',
                },
                'gid_consistency': {
                    'status': 'FAIL',
                    'failed_nodes': ['node-b'],
                    'summary': '1/2 interfaces have valid GID',
                },
                'node_smoke_tier1': {
                    'status': 'SKIPPED',
                    'summary': 'Node Smoke Tier 1 was not enabled',
                },
                'rdma_connectivity': {
                    'status': 'BLOCKED',
                    'summary': 'RDMA connectivity blocked by node health',
                },
            },
        },
    }


class TestPreflightDatasetBuilder(unittest.TestCase):
    def test_builds_headline_matrix_and_failures_first_tables(self):
        datasets = build_preflight_datasets(
            {
                'results': _sample_results(),
                'variant': {'cluster_nodes': ['node-b', 'node-a']},
            },
            {},
        )

        self.assertEqual(datasets['overall_status'], 'fail')
        self.assertEqual(datasets['headline']['rows'], [[1, 2, 1, 1, 5]])
        self.assertEqual(datasets['matrix']['headers'], ['Check', 'Overall', 'node-a', 'node-b'])
        matrix_rows = {row[0]: row[1:] for row in datasets['matrix']['rows']}
        self.assertEqual(matrix_rows['Ssh Reachability'], ['WARNING', 'PASS', 'FAIL'])
        self.assertEqual(matrix_rows['Gid Consistency'], ['FAIL', 'PASS', 'FAIL'])
        self.assertEqual(matrix_rows['Node Smoke Tier 1'], ['SKIPPED', 'SKIPPED', 'SKIPPED'])
        self.assertEqual(
            [row[0] for row in datasets['failures']['rows']],
            ['FAIL', 'BLOCKED', 'WARNING'],
        )
        self.assertEqual(datasets['failures']['rows'][0][2], 'node-b')
        warning = next(row for row in datasets['failures']['rows'] if row[0] == 'WARNING')
        self.assertEqual(warning[2], 'node-b')

    def test_profile_renders_operator_summary_and_detailed_report_link(self):
        profile = load_json_profile('preflight_checks')
        payload = build_rundeck_payload(
            profile=profile,
            store={
                'cvs_results_dict': _sample_results(),
                'variant_config': {
                    'cluster_nodes': ['node-a', 'node-b'],
                    'cluster_size': 2,
                    'enabled_checks': ['SSH Reachability', 'Node Health'],
                    'detailed_report_path': 'preflight_report.html',
                },
            },
        )

        document = render_rundeck_html(payload)
        self.assertIn('Preflight Checks Run Deck', document)
        self.assertIn('Check totals', document)
        self.assertIn('Failures first', document)
        self.assertIn('Check × node matrix', document)
        self.assertIn('href="preflight_report.html"', document)
        self.assertNotIn('Sweep analytics', document)

    def test_package_import_registers_preflight_builder(self):
        import cvs.lib.report.rundeck.dataset_builders  # noqa: F401
        from cvs.lib.report.rundeck.dataset_builders.registry import get_dataset_builder

        self.assertIsNotNone(get_dataset_builder('preflight'))

    def test_clean_run_hides_failures_and_missing_summary_is_explicit(self):
        clean = {
            'summary': {
                'overall_status': 'PASS',
                'checks': {'node_health': {'status': 'PASS', 'summary': 'ok'}},
            }
        }
        datasets = build_preflight_datasets({'results': clean, 'variant': {'cluster_nodes': ['n0']}}, {})
        self.assertEqual(datasets['failures'], {})
        document = render_rundeck_html(
            build_rundeck_payload(
                profile=load_json_profile('preflight_checks'),
                store={'cvs_results_dict': clean, 'variant_config': {'cluster_nodes': ['n0'], 'cluster_size': 1}},
            )
        )
        self.assertNotIn('Failures first', document)

        empty = build_preflight_datasets({'results': {}, 'variant': {}}, {})
        self.assertEqual(empty['overall_status'], 'na')
        self.assertIn('No preflight summary recorded', empty['headline']['rows'][0][0])


if __name__ == '__main__':
    unittest.main()
