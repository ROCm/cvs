'''Unit tests for shared pytest sub-test report detection.'''

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cvs.lib.report import subtest_reports


class TestSubtestReports(unittest.TestCase):
    def test_is_subtest_report_matches_either_implementation(self):
        class FakeA:
            pass

        class FakeB:
            pass

        with (
            patch.object(subtest_reports, '_BuiltinSubtestReport', FakeA),
            patch.object(subtest_reports, '_PluginSubtestReport', FakeB),
        ):
            self.assertTrue(subtest_reports.is_subtest_report(FakeA()))
            self.assertTrue(subtest_reports.is_subtest_report(FakeB()))
            self.assertFalse(subtest_reports.is_subtest_report(SimpleNamespace()))
        with (
            patch.object(subtest_reports, '_BuiltinSubtestReport', None),
            patch.object(subtest_reports, '_PluginSubtestReport', None),
        ):
            self.assertFalse(subtest_reports.is_subtest_report(FakeA()))

    def test_called_from_subtest_context_detects_subtest_exit(self):
        source = 'class Context:\n    def __exit__(self):\n        return called_from_subtest_context()\n'
        for path, expected in (
            ('/x/_pytest/subtests.py', True),
            ('/x/pytest_subtests/plugin.py', True),
            ('/x/other.py', False),
        ):
            with self.subTest(path=path):
                namespace = {'called_from_subtest_context': subtest_reports.called_from_subtest_context}
                exec(compile(source, path, 'exec'), namespace)
                self.assertEqual(namespace['Context']().__exit__(), expected)

    def test_not_in_subtest_context_by_default(self):
        self.assertFalse(subtest_reports.called_from_subtest_context())
