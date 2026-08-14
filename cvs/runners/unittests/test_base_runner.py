"""
Unit tests for BaseRunner.execute()'s setup -> run -> teardown lifecycle.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import unittest

from cvs.runners._base_runner import BaseRunner, RunConfig, RunResult, RunStatus


class _FakeRunner(BaseRunner):
    """Minimal concrete BaseRunner for exercising execute()."""

    def __init__(self, config, setup_return=True, run_return=None, run_raises=None):
        super().__init__(config)
        self._setup_return = setup_return
        self._run_return = run_return
        self._run_raises = run_raises
        self.teardown_calls = 0

    def setup(self) -> bool:
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
