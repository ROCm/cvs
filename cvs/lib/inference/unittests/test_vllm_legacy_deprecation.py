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
import hashlib
import os
import re
import unittest

import cvs.lib.inference.base as live_base
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


class _FakePhdl:
    """Records commands. InferenceBaseJob.__init__ only needs host_list and exec()."""

    host_list = ['node0']

    def __init__(self):
        self.commands = []

    def exec(self, cmd, *args, **kwargs):
        self.commands.append(cmd)
        return {'node0': ''}

    def exec_cmd_list(self, cmd_list, *args, **kwargs):
        self.commands.extend(cmd_list)
        return {'node0': ''}


def _build_job(base_cls):
    """Build base_cls against a config that omits random_range_ratio.

    Both bases take the same __init__ signature, so one builder serves the
    frozen and the live class and the assertions can contrast them directly.
    """
    return base_cls(
        c_phdl=_FakePhdl(),
        s_phdl=_FakePhdl(),
        model_name='m',
        inference_config_dict={'container_image': 'img:tag', 'benchmark_server_script_path': '/scripts'},
        benchmark_params_dict={'m': {'server_script': 'srv.sh', 'bench_serv_script': 'bench.py'}},
        hf_token='tok',
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

    def test_frozen_base_clones_bench_serving_and_live_base_does_not(self):
        # dtni turned clone_bench_serving_repo into a no-op and dropped the
        # benchmark_script_repo default; the frozen copy must still clone.
        legacy_job = _build_job(base_legacy.InferenceBaseJob)
        legacy_job.clone_bench_serving_repo('/app')
        self.assertIn(
            'git clone https://github.com/kimbochen/bench_serving.git',
            legacy_job.c_phdl.commands[-1],
        )

        live_job = _build_job(live_base.InferenceBaseJob)
        live_job.clone_bench_serving_repo('/app')
        self.assertEqual(live_job.c_phdl.commands, [])

    def test_frozen_base_requires_random_range_ratio_in_config(self):
        # base_legacy setdefaults the misspelled 'random_range_ration', so the
        # reads of 'random_range_ratio' have no fallback and a config omitting
        # the key raises. dtni fixed the spelling. Legacy configs must set it.
        legacy_job = _build_job(base_legacy.InferenceBaseJob)
        with self.assertRaises(KeyError):
            legacy_job.bp_dict['random_range_ratio']

        live_job = _build_job(live_base.InferenceBaseJob)
        self.assertEqual(live_job.bp_dict['random_range_ratio'], '1.0')


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


class TestFrozenContentPins(unittest.TestCase):
    """The restored files still hash to their pre-#223 content on main.

    Pinned by hash rather than compared against ``git show origin/main:...``:
    CI checks out at depth 1 with no ``origin/main`` ref, these tests also run
    from the sdist copy which has no ``.git`` at all, and ``origin/main`` is not
    an ancestor of ``dev/dtni``. Hashes are of the file with the added
    deprecation header stripped, taken against main at commit d8da9a43.
    """

    # sha256 of each file minus its inserted deprecation block.
    FROZEN_SHA256 = {
        'lib/inference/base_legacy.py': '7744e396a547ba22820a2d55be8f8ef742b3ab65b64fd9fa62211f1361fc0a47',
        'tests/inference/vllm_legacy/vllm_gpt_oss_120b_single.py': (
            'f3ad2f1cf58691401745fa5ef08ab89cac52a29579596a7486a08dffd97c0b99'
        ),
        'tests/inference/vllm_legacy/vllm_qwen3_80b_single.py': (
            '25edc96e9b7fff58b69414f99bb07d5551de5388b0d2b64c1c051885a08a9135'
        ),
        'tests/inference/vllm_legacy/vllm_qwen3_235b_single.py': (
            '84ce576cbcc477f967d01ef63cb687d92330ac66cb7d0d0c7415b2822adae52b'
        ),
        'tests/inference/vllm_legacy/vllm_deepseek31_685b_single.py': (
            'efa1593fb0e4fb0e604b4d0f9fc190af71e3332549e1e0241f363dd0ddcde1ce'
        ),
    }

    # The header is inserted inside the module docstring, not prepended, so
    # neither a byte- nor a suffix-compare identifies the frozen content.
    _SUITE_HEADER = re.compile(r'\n\nDEPRECATED -- scheduled.*?There is deliberately no conftest\.py here\.\n', re.S)
    _BASE_HEADER = re.compile(r'\n\nDEPRECATED -- frozen copy.*?once the window closes\.\n', re.S)

    def test_restored_files_match_their_pre_deletion_content(self):
        for relpath, expected in self.FROZEN_SHA256.items():
            with self.subTest(path=relpath):
                with open(os.path.join(_CVS_ROOT, relpath)) as fp:
                    body = fp.read()
                stripped = self._BASE_HEADER.sub('\n', self._SUITE_HEADER.sub('\n', body))
                self.assertNotEqual(stripped, body, 'deprecation header missing from ' + relpath)
                self.assertEqual(hashlib.sha256(stripped.encode()).hexdigest(), expected)


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
