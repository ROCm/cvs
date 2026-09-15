'''Unit tests for the preflight Run Deck run card.'''

import unittest

from cvs.lib.report.profiles.hooks.preflight_run_card import preflight_run_card_display


class TestPreflightRunCard(unittest.TestCase):
    def test_displays_cluster_checks_and_detailed_report(self):
        rows = preflight_run_card_display(
            {
                'cluster_size': 4,
                'enabled_checks': ['SSH Reachability', 'Node Health'],
                'detailed_report_path': 'logs/preflight_report.html',
            },
            {},
        )

        self.assertEqual(rows[0], ('Cluster size', '4', False))
        self.assertEqual(rows[1], ('Checks enabled', 'SSH Reachability, Node Health', False))
        self.assertEqual(rows[2], ('Detailed preflight report', 'logs/preflight_report.html', True))

    def test_handles_missing_context(self):
        self.assertEqual(
            preflight_run_card_display(None, {}),
            [
                ('Cluster size', '0', False),
                ('Checks enabled', 'None', False),
            ],
        )


if __name__ == '__main__':
    unittest.main()
