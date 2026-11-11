"""
Unit tests for BaseRunner.execute()'s setup -> run -> teardown lifecycle.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cvs.runners._base_runner import BaseRunner, RunConfig, RunResult, RunStatus, _runner_cache_dir


class _FakeRunner(BaseRunner):
    """Minimal concrete BaseRunner for exercising execute()."""

    def __init__(self, config, setup_return=True, setup_raises=None, run_return=None, run_raises=None):
        super().__init__(config)
        self._setup_return = setup_return
        self._setup_raises = setup_raises
        self._run_return = run_return
        self._run_raises = run_raises
        self.teardown_calls = 0

    def setup(self) -> bool:
        if self._setup_raises is not None:
            raise self._setup_raises
        return self._setup_return

    def run(self, **kwargs) -> RunResult:
        if self._run_raises is not None:
            raise self._run_raises
        return self._run_return

    def teardown(self) -> bool:
        self.teardown_calls += 1
        return True


def _config() -> RunConfig:
    return RunConfig(nodes=["10.0.0.1"], username="testuser")


class TestRunConfigDefaultDirs(unittest.TestCase):
    def test_defaults_use_home_cache_when_xdg_unset(self):
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch("cvs.runners._base_runner.Path.home", return_value=Path("/home/alice")),
        ):
            os.environ.pop("XDG_CACHE_HOME", None)
            config = RunConfig(nodes=["n1"], username="u")

            self.assertEqual(_runner_cache_dir(), Path("/home/alice/.cache/cvs/runners"))
            self.assertEqual(config.work_dir, Path("/home/alice/.cache/cvs/runners/work"))
            self.assertEqual(config.output_dir, Path("/home/alice/.cache/cvs/runners/benchmark_output"))

    def test_defaults_honour_absolute_xdg_cache_home(self):
        with (
            mock.patch.dict(os.environ, {"XDG_CACHE_HOME": "/srv/cache/alice"}, clear=False),
            mock.patch("cvs.runners._base_runner.Path.home", return_value=Path("/home/alice")),
        ):
            config = RunConfig(nodes=["n1"], username="u")

            self.assertEqual(_runner_cache_dir(), Path("/srv/cache/alice/cvs/runners"))
            self.assertEqual(config.work_dir, Path("/srv/cache/alice/cvs/runners/work"))
            self.assertEqual(config.output_dir, Path("/srv/cache/alice/cvs/runners/benchmark_output"))

    def test_relative_xdg_cache_home_is_ignored(self):
        with (
            mock.patch.dict(os.environ, {"XDG_CACHE_HOME": "relative/cache"}, clear=False),
            mock.patch("cvs.runners._base_runner.Path.home", return_value=Path("/home/alice")),
        ):
            config = RunConfig(nodes=["n1"], username="u")

            self.assertEqual(_runner_cache_dir(), Path("/home/alice/.cache/cvs/runners"))
            self.assertEqual(config.work_dir, Path("/home/alice/.cache/cvs/runners/work"))
            self.assertEqual(config.output_dir, Path("/home/alice/.cache/cvs/runners/benchmark_output"))

    def test_defaults_are_not_under_tmp(self):
        with mock.patch("cvs.runners._base_runner.Path.home", return_value=Path("/home/alice")):
            for xdg in (None, "/srv/cache/alice"):
                with self.subTest(xdg=xdg), mock.patch.dict(os.environ, {}, clear=False):
                    if xdg is None:
                        os.environ.pop("XDG_CACHE_HOME", None)
                    else:
                        os.environ["XDG_CACHE_HOME"] = xdg
                    config = RunConfig(nodes=["n1"], username="u")

                    for path in (config.work_dir, config.output_dir):
                        self.assertNotEqual(path, Path("/tmp"))
                        self.assertNotIn(Path("/tmp"), path.parents)

    def test_construction_does_not_create_directories(self):
        with (
            tempfile.TemporaryDirectory() as cache_home,
            mock.patch.dict(os.environ, {"XDG_CACHE_HOME": cache_home}, clear=False),
            mock.patch("cvs.runners._base_runner.Path.home", return_value=Path("/home/alice")),
        ):
            config = RunConfig(nodes=["n1"], username="u")

            self.assertFalse(config.work_dir.exists())
            self.assertFalse(config.output_dir.exists())

    def test_explicit_dirs_override_defaults(self):
        with (
            mock.patch.dict(os.environ, {"XDG_CACHE_HOME": "/srv/cache/alice"}, clear=False),
            mock.patch("cvs.runners._base_runner.Path.home", return_value=Path("/home/alice")),
        ):
            config = RunConfig(nodes=["n1"], username="u", work_dir=Path("/data/w"), output_dir=Path("/data/o"))

            self.assertEqual(config.work_dir, Path("/data/w"))
            self.assertEqual(config.output_dir, Path("/data/o"))


class TestExecuteTeardownLifecycle(unittest.TestCase):
    def test_teardown_runs_after_successful_setup_and_run(self):
        run_result = RunResult(status=RunStatus.COMPLETED, start_time=0, end_time=1)
        runner = _FakeRunner(_config(), setup_return=True, run_return=run_result)

        result = runner.execute()

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(runner.teardown_calls, 1)

    def test_teardown_runs_when_setup_fails(self):
        runner = _FakeRunner(_config(), setup_return=False)

        result = runner.execute()

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.error_message, "Setup failed")
        self.assertEqual(runner.teardown_calls, 1)

    def test_teardown_runs_when_setup_raises(self):
        runner = _FakeRunner(_config(), setup_raises=RuntimeError("setup exploded"))

        result = runner.execute()

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertIn("setup exploded", result.error_message)
        self.assertEqual(runner.teardown_calls, 1)

    def test_teardown_runs_when_run_raises(self):
        runner = _FakeRunner(_config(), setup_return=True, run_raises=RuntimeError("boom"))

        result = runner.execute()

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertIn("boom", result.error_message)
        self.assertEqual(runner.teardown_calls, 1)

    def test_teardown_not_run_before_setup_attempted(self):
        runner = _FakeRunner(
            _config(), setup_return=True, run_return=RunResult(status=RunStatus.COMPLETED, start_time=0, end_time=1)
        )

        self.assertFalse(runner._setup_complete)
        runner.execute()
        self.assertTrue(runner._setup_complete)


if __name__ == "__main__":
    unittest.main()
