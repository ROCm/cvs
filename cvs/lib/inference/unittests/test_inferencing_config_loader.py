'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for the sweep selector in
cvs.lib.inference.utils.inferencing_config_loader: named combos + an explicit
`runs` array (replacing the old sequence_combinations x concurrency_levels
cartesian). No hardware. Covers the Sweep validator (unique names, known
run.combo) and VariantConfig.expected_cells (the runs->cell_key expansion).
'''

import unittest

from pydantic import ValidationError

from cvs.lib.inference.utils.inferencing_config_loader import (
    Run,
    SeqCombo,
    Sweep,
    VariantConfig,
)
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


def _variant(sweep, tp="8"):
    """A minimal VariantConfig carrying just enough to exercise expected_cells.

    remote=0 (the remote guard would otherwise reject it) and
    enforce_thresholds=False so the empty threshold dict does not trip the
    coverage check -- this test pins the selector expansion, not the gate.
    """
    return VariantConfig(
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
        model={"id": "amd/Llama-3.1-70B-Instruct-FP8-KV", "remote": 0, "precision": "fp8"},
        image={"tag": "rocm/vllm-dev:nightly-sshd", "remote": 1},
        container={
            "name": "c",
            "image": "rocm/vllm-dev:nightly-sshd",
            "runtime": {"name": "docker"},
        },
        params={"tensor_parallelism": tp},
        sweep=sweep,
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
        vc = _variant(sw)  # remote guard ok; enforce/thresholds set below
        # _variant builds enforce_thresholds=False with empty thresholds; rebuild
        # via construction so the validator runs against the real values.
        return VariantConfig(
            schema_version=1,
            framework="vllm_single",
            gpu_arch="mi300x",
            enforce_thresholds=enforce,
            paths={
                "shared_fs": "/home/x",
                "models_dir": "/home/x/models",
                "log_dir": "/home/x/LOGS",
                "hf_token_file": "/home/x/.hf",
            },
            model={"id": "amd/Llama-3.1-70B-Instruct-FP8-KV", "remote": 0, "precision": "fp8"},
            image={"tag": "rocm/vllm-dev:nightly-sshd", "remote": 1},
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
        import warnings as _w
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            self._variant_with({self._CELL: specs}, enforce=False)
        self.assertTrue(any("missing gated-metric specs" in str(x.message) for x in caught))

    def test_extra_non_gated_spec_is_allowed(self):
        # A spec for a non-gated metric (record-only display extra) must not
        # trip coverage -- gating is a floor, not an allow-list.
        specs = _full_gated_specs()
        specs["client.num_prompts"] = {"kind": "min", "value": 0}
        vc = self._variant_with({self._CELL: specs}, enforce=True)
        self.assertIn("client.num_prompts", vc.thresholds[self._CELL])


if __name__ == "__main__":
    unittest.main()
