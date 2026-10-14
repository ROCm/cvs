'''Real pytest-html and JUnit coverage for RCCL case reports.'''

import html
import json
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
import xml.etree.ElementTree as element_tree
from pathlib import Path


def _html_tests(html_text):
    match = re.search(r'data-jsonblob="([^"]*)"', html_text)
    return json.loads(html.unescape(match.group(1)))['tests']


def _cases(entry):
    cases = []
    for extra in entry.get('extras') or []:
        content = str(extra.get('content') or '')
        cases.extend(
            (outcome, html.unescape(label))
            for outcome, label in re.findall(
                r"<td class='col-result'>(\w+)</td><td class='col-testId'>([^<]+)</td>", content
            )
        )
    return cases


SOURCE = textwrap.dedent(
    '''
    import pytest

    from cvs.lib import globals
    from cvs.lib.utils_lib import fail_test, update_test_result
    from cvs.tests.rccl._case_report import RcclCaseReporter

    FAILURE = ('The actual in-place bus BW 350.0 for msg size 4294967296 is lower than expected bus BW 9999.0 '
               '(threshold with 5% tolerance: 9499.05)')
    VERDICTS = [
        {'check': 'bus_bw', 'dtype': 'float', 'size': 1073741824, 'actual': 300.0, 'threshold': 270.0,
         'unit': 'GB/s', 'status': 'pass', 'message': ''},
        {'check': 'bus_bw', 'dtype': 'float', 'size': 4294967296, 'actual': 350.0, 'threshold': 9499.05,
         'unit': 'GB/s', 'status': 'fail', 'message': FAILURE},
        {'check': 'bw_dip', 'dtype': 'float', 'size': 4294967296, 'actual': 350.0, 'threshold': 285.0,
         'unit': 'GB/s', 'status': 'pass', 'message': ''},
    ]

    @pytest.fixture
    def setup_output():
        print('RCCL-SETUP-MARKER')

    @pytest.mark.parametrize('rccl_collective', ['all_reduce_perf'])
    def test_rccl_perf(rccl_collective, setup_output, request, subtests):
        globals.error_list = []
        print('RCCL-PARENT-MARKER')
        fail_test(FAILURE)
        RcclCaseReporter(request, subtests).report_verdicts(VERDICTS, rccl_collective)
        update_test_result()

    def test_rccl_pairwise(request, subtests):
        globals.error_list = []
        reporter = RcclCaseReporter(request, subtests)
        reporter.report_phase('0', 'n0', 'Phase0 sanity n0', True, '', best_bw=150.0)
        reporter.report_phase('1', 'n1', 'Phase1 n0 <-> n1', True, '', best_bw=300.0)
        update_test_result()
    '''
)

PERF_CASES = [
    ('Passed', 'all_reduce_perf bus_bw float size=1073741824: 300.00 GB/s (threshold >= 270.00 GB/s)'),
    ('Failed', 'all_reduce_perf bus_bw float size=4294967296: 350.00 GB/s (threshold >= 9499.05 GB/s)'),
    ('Passed', 'all_reduce_perf bw_dip float size=4294967296: 350.00 GB/s (threshold >= 285.00 GB/s)'),
]
PAIRWISE_CASES = [
    ('Passed', 'phase 0 n0 (Phase0 sanity n0): 150.00 GB/s'),
    ('Passed', 'phase 1 n1 (Phase1 n0 <-> n1): 300.00 GB/s'),
]


class TestRcclReportingIntegration(unittest.TestCase):
    def _run(self, root, with_root_plugin=False):
        test_path = root / 'test_rccl_reporting.py'
        html_path = root / 'report.html'
        xml_path = root / 'report.xml'
        test_path.write_text(SOURCE, encoding='utf-8')
        args = [
            sys.executable,
            '-m',
            'pytest',
            str(test_path),
            '-p',
            'cvs.tests.rccl.conftest',
            f'--html={html_path}',
            '--self-contained-html',
            f'--junitxml={xml_path}',
            '-v',
        ]
        if with_root_plugin:
            cluster_path = root / 'cluster.json'
            config_path = root / 'config.json'
            cluster_path.write_text('{}', encoding='utf-8')
            config_path.write_text('{}', encoding='utf-8')
            args += ['-p', 'cvs.conftest', f'--cluster_file={cluster_path}', f'--config_file={config_path}']
        completed = subprocess.run(args, check=False, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 1, f'stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}')
        tests = _html_tests(html_path.read_text(encoding='utf-8'))
        perf = [
            (nodeid, entries)
            for nodeid, entries in tests.items()
            if nodeid.endswith('::test_rccl_perf[all_reduce_perf]')
        ]
        pairwise = [(nodeid, entries) for nodeid, entries in tests.items() if nodeid.endswith('::test_rccl_pairwise')]
        self.assertEqual(len(perf), 1)
        self.assertEqual(len(pairwise), 1)
        self.assertEqual(len(perf[0][1]), 1)
        self.assertEqual(len(pairwise[0][1]), 1)
        return (
            completed,
            html_path.read_text(encoding='utf-8'),
            element_tree.parse(xml_path).getroot(),
            perf[0][1][0],
            pairwise[0][1][0],
        )

    def test_rccl_case_rows_html_junit_and_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed, html_text, xml_root, perf, pairwise = self._run(Path(tmp))
        self.assertIn('Failed', perf['resultsTableRow'][0])
        self.assertIn('cvs-benchmark-collapsible', perf['resultsTableRow'][0])
        self.assertEqual(_cases(perf), PERF_CASES)
        self.assertIn('Passed', pairwise['resultsTableRow'][0])
        self.assertEqual(_cases(pairwise), PAIRWISE_CASES)
        self.assertIn('cvs-subtests-count">5 subtests,', html_text)
        self.assertIn('> 1 Failed,</span>', html_text)
        self.assertIn('> 4 Passed,</span>', html_text)
        perf_case = next(
            case for case in xml_root.findall('.//testcase') if case.get('name') == 'test_rccl_perf[all_reduce_perf]'
        )
        self.assertGreaterEqual(len(perf_case.findall('failure')), 2)
        for failure in perf_case.findall('failure'):
            self.assertIn('4294967296', failure.get('message', '') + (failure.text or ''))
        pairwise_case = next(
            case for case in xml_root.findall('.//testcase') if case.get('name') == 'test_rccl_pairwise'
        )
        self.assertEqual(pairwise_case.findall('failure'), [])
        self.assertIn('SUBFAIL', completed.stdout)
        self.assertIn('size=4294967296', completed.stdout)

    def test_parent_full_log_survives_subtests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, perf, _ = self._run(root, with_root_plugin=True)
            full_logs = [extra for extra in perf.get('extras') or [] if extra.get('name') == 'Full Log']
            self.assertEqual(len(full_logs), 1)
            self.assertIn('RCCL-PARENT-MARKER', (root / full_logs[0]['content']).read_text(encoding='utf-8'))
            self.assertEqual(len(list(root.glob('test_rccl_reporting_html/test_rccl_perf_*.html'))), 1)
            self.assertEqual(_cases(perf), PERF_CASES)
