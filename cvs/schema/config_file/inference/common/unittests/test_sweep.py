"""Unit tests for shared inference sweep schema (inference/common/sweep.py)."""

import unittest
import warnings

from pydantic import ValidationError

from cvs.schema.config_file.inference.common.sweep import (
    GoodputSlo,
    Run,
    SeqCombo,
    Sweep,
    validate_sweep_selector,
    validate_thresholds_cover_sweep,
)


def _combo(name, isl="128", osl="2048"):
    return SeqCombo(name=name, isl=isl, osl=osl)


class TestSweepValidator(unittest.TestCase):
    def test_valid_runs_selector_constructs(self):
        sw = Sweep(
            sequence_combinations=[_combo("a"), _combo("b", osl="4096")],
            runs=[Run(combo="a", concurrency=16), Run(combo="b", concurrency=32)],
        )
        self.assertEqual([r.combo for r in sw.runs], ["a", "b"])

    def test_unknown_run_combo_raises(self):
        with self.assertRaises(ValidationError) as ctx:
            Sweep(
                sequence_combinations=[_combo("a")],
                runs=[Run(combo="typo", concurrency=16)],
            )
        self.assertIn("names no sequence_combination", str(ctx.exception))

    def test_duplicate_combo_names_raise(self):
        with self.assertRaises(ValidationError) as ctx:
            Sweep(
                sequence_combinations=[_combo("a"), _combo("a", osl="4096")],
                runs=[Run(combo="a", concurrency=16)],
            )
        self.assertIn("duplicate sequence_combination names", str(ctx.exception))

    def test_concurrency_levels_is_rejected(self):
        with self.assertRaises(ValidationError):
            Sweep(
                sequence_combinations=[_combo("a")],
                runs=[Run(combo="a", concurrency=16)],
                concurrency_levels=[16],
            )


class TestValidateSweepSelector(unittest.TestCase):
    def test_unknown_combo_raises(self):
        with self.assertRaisesRegex(ValueError, "names no sequence_combination"):
            validate_sweep_selector(["a"], ["b"])

    def test_duplicate_names_raises(self):
        with self.assertRaisesRegex(ValueError, "duplicate sequence_combination names"):
            validate_sweep_selector(["a", "a"], ["a"])


class TestValidateThresholdsCoverSweep(unittest.TestCase):
    _CELL = "ISL=128,OSL=2048,TP=8,CONC=16"
    _GATED = {"p99_ttft_ms", "failed"}

    def test_missing_cell_raises_when_enforced(self):
        with self.assertRaises(ValueError):
            validate_thresholds_cover_sweep(
                expected_cells=[self._CELL],
                thresholds={},
                enforce_thresholds=True,
                gated_metrics=self._GATED,
            )

    def test_missing_gated_metric_raises_when_enforced(self):
        with self.assertRaises(ValueError):
            validate_thresholds_cover_sweep(
                expected_cells=[self._CELL],
                thresholds={self._CELL: {}},
                enforce_thresholds=True,
                gated_metrics=self._GATED,
            )

    def test_missing_gated_metric_warns_when_record_only(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_thresholds_cover_sweep(
                expected_cells=[self._CELL],
                thresholds={self._CELL: {}},
                enforce_thresholds=False,
                gated_metrics=self._GATED,
            )
        self.assertTrue(any("missing gated-metric specs" in str(x.message) for x in caught))


class TestGoodputSlo(unittest.TestCase):
    def test_valid_goodput_slo_constructs(self):
        slo = GoodputSlo(ttft_ms=100.0, tpot_ms=50.0, e2el_ms=5000.0)
        self.assertEqual(slo.e2el_ms, 5000.0)

    def test_extra_key_raises(self):
        with self.assertRaises(ValidationError):
            GoodputSlo(ttft_ms=1.0, tpot_ms=1.0, e2el_ms=1.0, ttft_msec=1.0)


class TestSeqComboForbid(unittest.TestCase):
    def test_extra_key_raises(self):
        with self.assertRaises(ValidationError):
            SeqCombo(name="a", isl="128", osl="2048", unknown_field="x")


if __name__ == "__main__":
    unittest.main()
