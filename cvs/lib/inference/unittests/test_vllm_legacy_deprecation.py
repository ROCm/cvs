'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Guards the deprecated-but-restored legacy vLLM suites (OSS 3-month notice
window, 2026-08-11 -> 2026-11-11).

The legacy suites were hard-deleted in #223 without serving a notice period.
They are restored under ``cvs/tests/inference/vllm_legacy/`` -- deliberately
NOT under ``cvs/tests/inference/vllm/``, whose conftest sorts collected items
by a rank table keyed on the unified suite's test names. Legacy names are
absent from that table, so they would all tie at the default rank and get
reordered into an unrunnable sequence (inference before container launch).

These tests exist to keep the quarantine intact until the window closes.
'''

import ast
import datetime
import os
import unittest

import cvs.lib.inference.base_legacy as base_legacy
import cvs.lib.inference.vllm as legacy_vllm
from cvs.lib.inference.base_legacy import InferenceBaseJob as LegacyInferenceBaseJob
from cvs.lib.inference.vllm import DEPRECATION_REMOVAL_DATE, VllmJob as LegacyVllmJob

_CVS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_TESTS_ROOT = os.path.join(_CVS_ROOT, 'tests')
_LEGACY_SUITE_DIR = os.path.join(_TESTS_ROOT, 'inference', 'vllm_legacy')
_UNIFIED_SUITE_DIR = os.path.join(_TESTS_ROOT, 'inference', 'vllm')

LEGACY_SUITE_STEMS = (
    'vllm_gpt_oss_120b_single',
    'vllm_qwen3_80b_single',
    'vllm_qwen3_235b_single',
    'vllm_deepseek31_685b_single',
)


class TestLegacyVllmJobRestored(unittest.TestCase):
    """The deleted lib module is importable again and still its own type."""

    def test_vllm_job_subclasses_frozen_legacy_base(self):
        # Not the live base: dtni's base.py changed the bench driver, readiness
        # polling, and result-key names. Inheriting it would mean the
        # "deprecated" path behaves differently from what was deleted.
        self.assertTrue(issubclass(LegacyVllmJob, LegacyInferenceBaseJob))

    def test_vllm_job_does_not_subclass_live_base(self):
        from cvs.lib.inference.base import InferenceBaseJob as LiveInferenceBaseJob

        self.assertFalse(issubclass(LegacyVllmJob, LiveInferenceBaseJob))

    def test_legacy_hooks_preserved(self):
        for name, expected in (
            ('get_result_filename', 'vllm_test_result.json'),
            ('get_log_subdir', 'vllm'),
        ):
            with self.subTest(hook=name):
                self.assertEqual(getattr(LegacyVllmJob, name)(LegacyVllmJob), expected)

    def test_completion_pattern_matches_legacy_marker(self):
        pattern = LegacyVllmJob.get_completion_pattern(LegacyVllmJob)
        self.assertTrue(pattern.search('End-to-end Latency (ms): 12.5'))


class TestFrozenLegacyBase(unittest.TestCase):
    """base_legacy is a frozen copy, not a re-export of the evolving base."""

    def test_is_a_distinct_module_from_live_base(self):
        import cvs.lib.inference.base as live_base

        self.assertIsNot(base_legacy, live_base)
        self.assertIsNot(base_legacy.InferenceBaseJob, live_base.InferenceBaseJob)

    def test_retains_pre_dtni_bench_clone_behaviour(self):
        # dtni turned clone_bench_serving_repo into a no-op and dropped the
        # benchmark_script_repo default. The frozen copy must still do neither,
        # or restored configs silently stop working the way they used to.
        self.assertIn('benchmark_script_repo', base_legacy.LEGACY_BENCH_DEFAULTS)

    def test_retains_legacy_random_range_ratio_key_spelling(self):
        # The deleted code spelled it 'random_range_ration'; dtni fixed the
        # typo. Legacy configs still carry the misspelling, so the frozen base
        # must keep honouring it.
        self.assertIn('random_range_ration', base_legacy.LEGACY_BENCH_DEFAULTS)


class TestLegacySuiteQuarantine(unittest.TestCase):
    """Restored suites live outside the unified suite's conftest scope."""

    def test_legacy_suites_present_in_quarantine_dir(self):
        for stem in LEGACY_SUITE_STEMS:
            with self.subTest(suite=stem):
                self.assertTrue(os.path.isfile(os.path.join(_LEGACY_SUITE_DIR, stem + '.py')))

    def test_quarantine_dir_has_no_conftest(self):
        # A conftest here would reintroduce exactly the reordering hazard the
        # quarantine exists to avoid.
        self.assertFalse(os.path.isfile(os.path.join(_LEGACY_SUITE_DIR, 'conftest.py')))

    def test_legacy_suites_not_restored_into_unified_suite_dir(self):
        for stem in LEGACY_SUITE_STEMS:
            with self.subTest(suite=stem):
                self.assertFalse(os.path.isfile(os.path.join(_UNIFIED_SUITE_DIR, stem + '.py')))

    def test_legacy_suites_do_not_import_the_unified_job(self):
        # Each restored suite must bind to the legacy VllmJob, not vllm_job.
        for stem in LEGACY_SUITE_STEMS:
            with self.subTest(suite=stem):
                path = os.path.join(_LEGACY_SUITE_DIR, stem + '.py')
                with open(path) as fp:
                    tree = ast.parse(fp.read())
                modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
                self.assertIn('cvs.lib.inference.vllm', modules)
                self.assertNotIn('cvs.lib.inference.vllm_job', modules)

    def test_unified_suite_still_present(self):
        # Restoration must not disturb the replacement suite.
        self.assertTrue(os.path.isfile(os.path.join(_UNIFIED_SUITE_DIR, 'vllm.py')))


class TestDeprecationNotice(unittest.TestCase):
    """The notice window is explicit and machine-checkable."""

    def test_removal_date_is_three_months_after_notice_start(self):
        self.assertEqual(DEPRECATION_REMOVAL_DATE, datetime.date(2026, 11, 11))

    def test_importing_legacy_module_warns(self):
        import importlib
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            importlib.reload(legacy_vllm)
        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning) for w in caught),
            'importing cvs.lib.inference.vllm must raise a DeprecationWarning',
        )

    def test_warning_names_the_replacement_and_the_date(self):
        import importlib
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            importlib.reload(legacy_vllm)
        messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertTrue(any('2026-11-11' in m for m in messages))
        self.assertTrue(any('cvs.tests.inference.vllm' in m for m in messages))


if __name__ == '__main__':
    unittest.main()
