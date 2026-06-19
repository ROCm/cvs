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


def _combo(name, isl="128", osl="2048"):
    return SeqCombo(name=name, isl=isl, osl=osl)


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


if __name__ == "__main__":
    unittest.main()
