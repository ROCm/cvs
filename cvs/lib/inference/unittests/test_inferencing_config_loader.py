'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.utils.config_loader (ModelSpec, BaseVariantConfig,
substitute_config) and cvs.lib.inference.utils.inferencing_config_loader
(Sweep, SeqCombo, GoodputSlo, Run, VariantConfig.expected_cells,
_check_thresholds_cover_sweep). No hardware.
'''

import unittest
import warnings

from pydantic import ValidationError

from cvs.lib.inference.utils.inferencing_config_loader import (
    GoodputSlo,
    Run,
    SeqCombo,
    Sweep,
    VariantConfig,
)
from cvs.lib.utils.config_loader import ModelSpec
from cvs.lib.inference.utils.vllm_parsing import GATED_METRICS


def _combo(name, isl="128", osl="2048"):
    return SeqCombo(name=name, isl=isl, osl=osl)


def _full_gated_specs():
    """A spec for every gated metric -- the minimum that satisfies coverage.

    Values are inert (a 0 floor / huge ceiling) so the set passes without
    asserting anything; these tests pin the coverage gate, not the numbers.
    """
    out = {}
    for m in GATED_METRICS:
        kind = "max_ms" if m.endswith("_ms") else "max" if m == "failed" else "min"
        out[f"client.{m}"] = {"kind": kind, "value": 0 if kind == "min" else 1e12}
    return out


def _variant(sweep, tp="8", thresholds=None, enforce_thresholds=False):
    """A minimal VariantConfig carrying just enough to exercise expected_cells.

    remote=0 (the remote guard would otherwise reject it) and
    enforce_thresholds=False so the empty threshold dict does not trip the
    coverage check -- this test pins the selector expansion, not the gate.
    """
    return VariantConfig(
        schema_version=1,
        framework="vllm_single",
        gpu_arch="mi300x",
        enforce_thresholds=enforce_thresholds,
        threshold_json="<changeme>",
        paths={
            "shared_fs": "/home/x",
            "models_dir": "/home/x/models",
            "log_dir": "/home/x/LOGS",
            "hf_token_file": "/home/x/.hf",
        },
        model={"id": "amd/Llama-3.1-70B-Instruct-FP8-KV", "remote": 0},
        container={
            "name": "c",
            "image": "rocm/vllm-dev:nightly-sshd",
            "runtime": {"name": "docker"},
        },
        params={"tensor_parallelism": tp},
        sweep=sweep,
        thresholds=thresholds or {},
    )


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
        # The old cartesian key must be gone (extra=forbid): a config still
        # carrying concurrency_levels should fail loudly, not silently ignore it.
        with self.assertRaises(ValidationError):
            Sweep(
                sequence_combinations=[_combo("a")],
                runs=[Run(combo="a", concurrency=16)],
                concurrency_levels=[16],
            )


class TestExpectedCells(unittest.TestCase):
    def test_runs_expand_to_exactly_their_cells(self):
        sw = Sweep(
            sequence_combinations=[_combo("a", isl="128", osl="2048"), _combo("b", isl="256", osl="4096")],
            runs=[
                Run(combo="a", concurrency=16),
                Run(combo="b", concurrency=32),
                Run(combo="a", concurrency=64),
            ],
        )
        vc = _variant(sw)
        self.assertEqual(
            vc.expected_cells(),
            [
                "ISL=128,OSL=2048,TP=8,CONC=16",
                "ISL=256,OSL=4096,TP=8,CONC=32",
                "ISL=128,OSL=2048,TP=8,CONC=64",
            ],
        )

    def test_no_cartesian_blowup(self):
        # Two combos + two runs must yield TWO cells, not 2x2=4 (the old bug).
        sw = Sweep(
            sequence_combinations=[_combo("a"), _combo("b", osl="4096")],
            runs=[Run(combo="a", concurrency=16), Run(combo="b", concurrency=16)],
        )
        self.assertEqual(len(_variant(sw).expected_cells()), 2)


class TestGatedMetricCoverage(unittest.TestCase):
    """The gated-metric axis of _check_thresholds_cover_sweep."""

    _CELL = "ISL=128,OSL=2048,TP=8,CONC=16"

    def _variant_with(self, thresholds, enforce):
        sw = Sweep(
            sequence_combinations=[_combo("a")],
            runs=[Run(combo="a", concurrency=16)],
        )
        return VariantConfig(
            schema_version=1,
            framework="vllm_single",
            gpu_arch="mi300x",
            enforce_thresholds=enforce,
            threshold_json="<changeme>",
            paths={
                "shared_fs": "/home/x",
                "models_dir": "/home/x/models",
                "log_dir": "/home/x/LOGS",
                "hf_token_file": "/home/x/.hf",
            },
            model={"id": "amd/Llama-3.1-70B-Instruct-FP8-KV", "remote": 0},
            container={"name": "c", "image": "rocm/vllm-dev:nightly-sshd", "runtime": {"name": "docker"}},
            params={"tensor_parallelism": "8"},
            sweep=sw,
            thresholds=thresholds,
        )

    def test_full_gated_set_constructs(self):
        vc = self._variant_with({self._CELL: _full_gated_specs()}, enforce=True)
        self.assertEqual(vc.enforce_thresholds, True)

    def test_missing_gated_metric_raises_when_enforced(self):
        specs = _full_gated_specs()
        del specs["client.p99_ttft_ms"]  # drop one gated metric
        with self.assertRaises(ValidationError) as ctx:
            self._variant_with({self._CELL: specs}, enforce=True)
        self.assertIn("missing gated-metric specs", str(ctx.exception))
        self.assertIn("client.p99_ttft_ms", str(ctx.exception))

    def test_missing_gated_metric_warns_when_record_only(self):
        specs = _full_gated_specs()
        del specs["client.failed"]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._variant_with({self._CELL: specs}, enforce=False)
        self.assertTrue(any("missing gated-metric specs" in str(x.message) for x in caught))

    def test_extra_non_gated_spec_is_allowed(self):
        # A spec for a non-gated metric (record-only display extra) must not
        # trip coverage -- gating is a floor, not an allow-list.
        specs = _full_gated_specs()
        specs["client.num_prompts"] = {"kind": "min", "value": 0}
        vc = self._variant_with({self._CELL: specs}, enforce=True)
        self.assertIn("client.num_prompts", vc.thresholds[self._CELL])


class TestModelSpecNoPrecision(unittest.TestCase):
    """precision must not be accepted by ModelSpec (removed field)."""

    def test_precision_field_is_rejected(self):
        with self.assertRaises(ValidationError):
            ModelSpec(id="amd/Llama-3.1-70B", remote=0, precision="fp8")

    def test_valid_model_spec_without_precision(self):
        ms = ModelSpec(id="amd/Llama-3.1-70B", remote=0)
        self.assertEqual(ms.id, "amd/Llama-3.1-70B")
        self.assertEqual(ms.remote, 0)


class TestThresholdJsonField(unittest.TestCase):
    """threshold_json is a required field on BaseVariantConfig / VariantConfig."""

    def _base_kwargs(self):
        sw = Sweep(
            sequence_combinations=[_combo("a")],
            runs=[Run(combo="a", concurrency=16)],
        )
        return dict(
            schema_version=1,
            framework="vllm_single",
            gpu_arch="mi300x",
            enforce_thresholds=False,
            paths={
                "shared_fs": "/home/x",
                "models_dir": "/home/x/models",
                "log_dir": "/home/x/LOGS",
                "hf_token_file": "/home/x/.hf",
            },
            model={"id": "amd/Llama-3.1-70B", "remote": 0},
            container={"name": "c", "image": "img", "runtime": {"name": "docker"}},
            params={"tensor_parallelism": "8"},
            sweep=sw,
            thresholds={},
        )

    def test_missing_threshold_json_raises(self):
        kwargs = self._base_kwargs()
        # threshold_json deliberately absent
        with self.assertRaises(ValidationError):
            VariantConfig(**kwargs)

    def test_threshold_json_present_constructs(self):
        kwargs = self._base_kwargs()
        kwargs["threshold_json"] = "/some/absolute/path/threshold.json"
        vc = VariantConfig(**kwargs)
        self.assertEqual(vc.threshold_json, "/some/absolute/path/threshold.json")


class TestCellCoverageAxis(unittest.TestCase):
    """Axis-1 of _check_thresholds_cover_sweep: cell vs threshold key mismatch."""

    _CELL = "ISL=128,OSL=2048,TP=8,CONC=16"

    def _variant_with(self, thresholds, enforce=True):
        sw = Sweep(
            sequence_combinations=[_combo("a")],
            runs=[Run(combo="a", concurrency=16)],
        )
        return VariantConfig(
            schema_version=1,
            framework="vllm_single",
            gpu_arch="mi300x",
            enforce_thresholds=enforce,
            threshold_json="<changeme>",
            paths={
                "shared_fs": "/home/x",
                "models_dir": "/home/x/models",
                "log_dir": "/home/x/LOGS",
                "hf_token_file": "/home/x/.hf",
            },
            model={"id": "amd/Llama-3.1-70B", "remote": 0},
            container={"name": "c", "image": "img", "runtime": {"name": "docker"}},
            params={"tensor_parallelism": "8"},
            sweep=sw,
            thresholds=thresholds,
        )

    def test_sweep_cell_with_no_threshold_entry_raises(self):
        # thresholds is empty -> sweep cell has no entry -> axis-1 fires
        with self.assertRaises(ValidationError) as ctx:
            self._variant_with(thresholds={}, enforce=True)
        self.assertIn("sweep cells with no threshold entry", str(ctx.exception))
        self.assertIn(self._CELL, str(ctx.exception))

    def test_threshold_key_matching_no_sweep_cell_raises(self):
        # thresholds has the real cell PLUS a bogus key -> extra set is non-empty
        specs = _full_gated_specs()
        with self.assertRaises(ValidationError) as ctx:
            self._variant_with(
                thresholds={self._CELL: specs, "ISL=999,OSL=999,TP=8,CONC=99": specs},
                enforce=True,
            )
        self.assertIn("threshold keys matching no sweep cell", str(ctx.exception))
        self.assertIn("ISL=999,OSL=999,TP=8,CONC=99", str(ctx.exception))

    def test_cell_mismatch_warns_when_record_only(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._variant_with(thresholds={}, enforce=False)
        self.assertTrue(any("sweep cells with no threshold entry" in str(w.message) for w in caught))


class TestExpectedCellsBoundaries(unittest.TestCase):
    """Boundary cases for VariantConfig.expected_cells."""

    def test_empty_runs_yields_empty_cells(self):
        sw = Sweep(sequence_combinations=[_combo("a")], runs=[])
        self.assertEqual(_variant(sw).expected_cells(), [])

    def test_unreferenced_combo_not_in_expected_cells(self):
        # combo 'unused' is declared but never referenced by any run
        sw = Sweep(
            sequence_combinations=[_combo("a"), _combo("unused", isl="999", osl="999")],
            runs=[Run(combo="a", concurrency=16)],
        )
        cells = _variant(sw).expected_cells()
        self.assertEqual(len(cells), 1)
        self.assertNotIn("ISL=999", cells[0])


class TestGoodputSlo(unittest.TestCase):
    """GoodputSlo is a _Forbid model with three required float fields."""

    def test_valid_goodput_slo_constructs(self):
        slo = GoodputSlo(ttft_ms=100.0, tpot_ms=50.0, e2el_ms=5000.0)
        self.assertEqual(slo.ttft_ms, 100.0)
        self.assertEqual(slo.tpot_ms, 50.0)
        self.assertEqual(slo.e2el_ms, 5000.0)

    def test_missing_required_field_raises(self):
        for missing in ("ttft_ms", "tpot_ms", "e2el_ms"):
            with self.subTest(missing=missing):
                kwargs = {"ttft_ms": 1.0, "tpot_ms": 1.0, "e2el_ms": 1.0}
                del kwargs[missing]
                with self.assertRaises(ValidationError):
                    GoodputSlo(**kwargs)

    def test_extra_key_raises(self):
        with self.assertRaises(ValidationError):
            GoodputSlo(ttft_ms=1.0, tpot_ms=1.0, e2el_ms=1.0, ttft_msec=1.0)

    def test_seq_combo_with_goodput_slo(self):
        slo = GoodputSlo(ttft_ms=1000.0, tpot_ms=50.0, e2el_ms=10000.0)
        combo = SeqCombo(name="a", isl="128", osl="2048", goodput_slo=slo)
        self.assertIsNotNone(combo.goodput_slo)
        self.assertEqual(combo.goodput_slo.e2el_ms, 10000.0)

    def test_seq_combo_without_goodput_slo(self):
        combo = SeqCombo(name="a", isl="128", osl="2048")
        self.assertIsNone(combo.goodput_slo)


class TestSeqComboForbid(unittest.TestCase):
    """SeqCombo is _Forbid: missing required fields and extra keys must raise."""

    def test_missing_required_field_raises(self):
        for missing in ("name", "isl", "osl"):
            with self.subTest(missing=missing):
                kwargs = {"name": "a", "isl": "128", "osl": "2048"}
                del kwargs[missing]
                with self.assertRaises(ValidationError):
                    SeqCombo(**kwargs)

    def test_extra_key_raises(self):
        with self.assertRaises(ValidationError):
            SeqCombo(name="a", isl="128", osl="2048", unknown_field="x")


if __name__ == "__main__":
    unittest.main()
