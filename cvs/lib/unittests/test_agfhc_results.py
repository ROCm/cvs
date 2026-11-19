'''Unit tests for AGFHC results.json classification and read details.'''

import unittest

from cvs.lib.agfhc_results import classify_results_json, describe_read, results_text


class TestClassifyResultsJson(unittest.TestCase):
    def test_zero_passes(self):
        self.assertEqual(classify_results_json('{\n  "total_failed": 0,\n}\n'), ('pass', None))

    def test_nonzero_reports_count(self):
        kind, detail = classify_results_json('"total_failed": 3,')
        self.assertEqual(kind, 'nonzero')
        self.assertIn('"total_failed": 3', detail)

    def test_missing_file_is_unread(self):
        kind, detail = classify_results_json('cat: /var/tmp/x/results.json: No such file or directory\n')
        self.assertEqual(kind, 'unread')
        self.assertIn('No such file or directory', detail)

    def test_ssh_error_is_unread(self):
        output = '("Error connecting to host \'%s:%s\' - %s - retry %s/%s", \'node-ssh-error\', 22, \'Connection refused\', 3, 3)\n'
        kind, detail = classify_results_json(output)
        self.assertEqual(kind, 'unread')
        self.assertIn('Connection refused', detail)

    def test_empty_is_unread_no_output(self):
        for output in ('', '  \n'):
            with self.subTest(output=output):
                self.assertEqual(classify_results_json(output), ('unread', 'no output'))

    def test_zero_gate_unchanged(self):
        kind, _detail = classify_results_json('"total_failed":0}')
        self.assertEqual(kind, 'nonzero')


class TestDescribeRead(unittest.TestCase):
    def test_collapses_whitespace(self):
        self.assertEqual(describe_read('a\n  b\tc'), 'a b c')

    def test_truncates_long_output(self):
        detail = describe_read('x' * 500)
        self.assertEqual(len(detail), 203)
        self.assertTrue(detail.endswith('...'))

    def test_dict_output(self):
        self.assertEqual(describe_read({'output': 'boom'}), 'boom')

    def test_none_output(self):
        self.assertEqual(describe_read(None), 'no output')


class TestResultsText(unittest.TestCase):
    def test_preserves_strings_and_normalizes_other_outputs(self):
        self.assertEqual(results_text('raw\n'), 'raw\n')
        self.assertEqual(results_text({'output': 'boom'}), 'boom')
        self.assertEqual(results_text({'output': None}), '')
        self.assertEqual(results_text(None), '')
        self.assertEqual(results_text(3), '3')


if __name__ == '__main__':
    unittest.main()
