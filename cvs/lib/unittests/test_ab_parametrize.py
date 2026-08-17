"""
Unit tests for the A/B test parametrization hook (pytest_generate_tests).

These verify the decoupling of *perf-regression* axes (collective x dtype x size)
from the optional NCCL *knob* matrix (`regression`):

  * With no `regression` block, every collective/dtype is still parametrized and
    each runs once under the production env (a single empty knob override).
  * With a `regression` block, the NCCL knob matrix is expanded as a Cartesian
    product (extra coverage) - the legacy behaviour.
"""

import json
import os
import tempfile
import unittest

import cvs.tests.rccl.rccl_ab_regression as ab


class _FakeConfig:
    def __init__(self, config_file):
        self._config_file = config_file

    def getoption(self, name):
        if name == "config_file":
            return self._config_file
        return None


class _FakeMetafunc:
    """Minimal stand-in for pytest's Metafunc, capturing parametrize() calls."""

    def __init__(self, config_file, fixturenames):
        self.config = _FakeConfig(config_file)
        self.fixturenames = fixturenames
        self.calls = {}  # argname -> {"argvalues": [...], "ids": [...]}

    def parametrize(self, argname, argvalues, ids=None):
        self.calls[argname] = {"argvalues": list(argvalues), "ids": list(ids) if ids else None}


def _write_cfg(tmp, rccl_block):
    path = os.path.join(tmp, "cfg.json")
    with open(path, "w") as fp:
        json.dump({"rccl": rccl_block}, fp)
    return path


FIXTURES = ["rccl_collective", "regression_params", "data_type"]


class TestAbParametrize(unittest.TestCase):
    def test_no_regression_block_runs_collectives_under_default_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_cfg(tmp, {
                "rccl_collective": ["all_reduce_perf", "all_gather_perf"],
                "data_types": ["float", "bfloat16"],
                # no "regression" key at all
            })
            mf = _FakeMetafunc(cfg, FIXTURES)
            ab.pytest_generate_tests(mf)

            self.assertEqual(mf.calls["rccl_collective"]["argvalues"],
                             ["all_reduce_perf", "all_gather_perf"])
            # Exactly one knob combo: the production-env default.
            self.assertEqual(mf.calls["regression_params"]["argvalues"], [{}])
            self.assertEqual(mf.calls["regression_params"]["ids"], ["default"])
            self.assertEqual(mf.calls["data_type"]["argvalues"], ["float", "bfloat16"])

    def test_empty_regression_block_is_treated_as_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_cfg(tmp, {
                "rccl_collective": ["all_reduce_perf"],
                "regression": {},
            })
            mf = _FakeMetafunc(cfg, FIXTURES)
            ab.pytest_generate_tests(mf)

            self.assertEqual(mf.calls["regression_params"]["argvalues"], [{}])
            self.assertEqual(mf.calls["regression_params"]["ids"], ["default"])

    def test_regression_block_expands_knob_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_cfg(tmp, {
                "rccl_collective": ["all_reduce_perf"],
                "regression": {"NCCL_PXN_DISABLE": ["0", "1"]},
            })
            mf = _FakeMetafunc(cfg, FIXTURES)
            ab.pytest_generate_tests(mf)

            self.assertEqual(
                mf.calls["regression_params"]["argvalues"],
                [{"NCCL_PXN_DISABLE": "0"}, {"NCCL_PXN_DISABLE": "1"}],
            )
            self.assertEqual(
                mf.calls["regression_params"]["ids"],
                ["NCCL_PXN_DISABLE=0", "NCCL_PXN_DISABLE=1"],
            )

    def test_cartesian_product_of_two_knobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_cfg(tmp, {
                "rccl_collective": ["all_reduce_perf"],
                "regression": {"NCCL_ALGO": ["Ring"], "NCCL_PXN_DISABLE": ["0", "1"]},
            })
            mf = _FakeMetafunc(cfg, FIXTURES)
            ab.pytest_generate_tests(mf)

            self.assertEqual(len(mf.calls["regression_params"]["argvalues"]), 2)
            for combo in mf.calls["regression_params"]["argvalues"]:
                self.assertEqual(combo["NCCL_ALGO"], "Ring")


class TestResolveDetectThresholds(unittest.TestCase):
    CONFIG_THR = {"small": 0.20, "mid": 0.15, "large": 0.075}
    DERIVED_THR = {"small": 0.10, "mid": 0.05, "large": 0.03}

    def _write_derived(self, tmp, payload):
        with open(os.path.join(tmp, "ab_derived_thresholds.json"), "w") as fp:
            json.dump(payload, fp)

    def test_derived_file_overrides_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_derived(tmp, {"thresholds": self.DERIVED_THR})
            out = ab._resolve_detect_thresholds(
                {"thresholds": self.CONFIG_THR}, {}, tmp)
            self.assertEqual(out["thresholds"], self.DERIVED_THR)

    def test_missing_file_keeps_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = ab._resolve_detect_thresholds(
                {"thresholds": self.CONFIG_THR}, {}, tmp)
            self.assertEqual(out["thresholds"], self.CONFIG_THR)

    def test_opt_out_keeps_config_even_if_file_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_derived(tmp, {"thresholds": self.DERIVED_THR})
            out = ab._resolve_detect_thresholds(
                {"thresholds": self.CONFIG_THR},
                {"use_derived_thresholds": False}, tmp)
            self.assertEqual(out["thresholds"], self.CONFIG_THR)

    def test_corrupt_file_keeps_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "ab_derived_thresholds.json"), "w") as fp:
                fp.write("{ not json")
            out = ab._resolve_detect_thresholds(
                {"thresholds": self.CONFIG_THR}, {}, tmp)
            self.assertEqual(out["thresholds"], self.CONFIG_THR)

    def test_file_without_thresholds_key_keeps_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_derived(tmp, {"noise": {}})
            out = ab._resolve_detect_thresholds(
                {"thresholds": self.CONFIG_THR}, {}, tmp)
            self.assertEqual(out["thresholds"], self.CONFIG_THR)

    def test_does_not_mutate_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_derived(tmp, {"thresholds": self.DERIVED_THR})
            original = {"thresholds": self.CONFIG_THR}
            ab._resolve_detect_thresholds(original, {}, tmp)
            self.assertEqual(original["thresholds"], self.CONFIG_THR)


if __name__ == "__main__":
    unittest.main()
