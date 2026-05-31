"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

import os
import unittest
from pathlib import Path

from pydantic import ValidationError

from cvs.lib.config import ContainerSpec, expand_sweep, parse_config
from cvs.lib.config.loader import ConfigError
from cvs.lib.config.thresholds import (
    ConvergenceThreshold,
    GoodputThreshold,
    MonotonicityThreshold,
    PercentileThreshold,
    RateThreshold,
    ResultView,
    StabilityThreshold,
)
from cvs.lib.runtime.container_handle import ContainerHandle

BASE = {
    "schema_version": "2",
    "framework": "vllm",
    "target_gpu": "mi300",
    "model": "meta-llama/Llama-3.1-70B",
    "topology": {"roles": {"server": {"count": 1, "gpus_per_node": 8}}},
    "container": {"env": {"HF_TOKEN": "hf_secret_abc"}},
    "params": {"server_script": "s.sh"},
    "sweep": {"concurrency": [16, 64], "sequence_combinations": [{"isl": 1024, "osl": 1024, "name": "balanced"}]},
}


class TestConfigValidation(unittest.TestCase):
    def test_accepts_valid(self):
        cfg = parse_config(BASE)
        self.assertEqual(cfg.framework, "vllm")
        self.assertEqual(cfg.workload_kind, "inference")

    def test_rejects_unknown_framework(self):
        with self.assertRaises(ConfigError):
            parse_config({**BASE, "framework": "vlm"})

    def test_rejects_extra_key(self):
        with self.assertRaises(ConfigError):
            parse_config({**BASE, "bogus": 1})

    def test_workload_hash_excludes_container_env(self):
        # Security removed: the token lives in container.env (plaintext). It
        # must NOT be workload-defining, so workload_hash ignores `container`...
        a = parse_config(BASE)
        b = parse_config({**BASE, "container": {"env": {"HF_TOKEN": "different"}}})
        self.assertEqual(a.workload_hash(), b.workload_hash())
        # ...but it IS retained (config_hash distinguishes it), proving the env
        # is not simply dropped everywhere; verification identity is unaffected.
        self.assertNotEqual(a.config_hash(), b.config_hash())
        self.assertEqual(a.verification_hash(), b.verification_hash())

    def test_seed_is_workload_defining(self):
        # A different seed yields different numerical results, so it must change
        # workload identity (otherwise two runs collide in any hash-keyed cache).
        a = parse_config(BASE)
        b = parse_config({**BASE, "seed": 7})
        self.assertNotEqual(a.workload_hash(), b.workload_hash())
        self.assertNotEqual(a.config_hash(), b.config_hash())


class TestSweepExpansion(unittest.TestCase):
    def test_cartesian_times_paired(self):
        cfg = parse_config(BASE)
        cells = expand_sweep(cfg.sweep)
        self.assertEqual(len(cells), 2)
        ids = {c.id for c in cells}
        self.assertEqual(ids, {"concurrency16-balanced", "concurrency64-balanced"})

    def test_paired_bundle_fields_covary(self):
        cfg = parse_config(
            {
                **BASE,
                "sweep": {
                    "concurrency": [16],
                    "sequence_combinations": [
                        {"isl": 1024, "osl": 1024, "name": "balanced"},
                        {"isl": 8192, "osl": 1024, "name": "long_context"},
                    ],
                },
            }
        )
        cells = {c.id: c.params for c in expand_sweep(cfg.sweep)}
        self.assertEqual(cells["concurrency16-long_context"]["isl"], 8192)

    def test_unnamed_combos_get_unique_positional_tokens(self):
        cfg = parse_config(
            {
                **BASE,
                "sweep": {
                    "concurrency": [16],
                    "sequence_combinations": [{"isl": 1, "osl": 1}, {"isl": 2, "osl": 2}],
                },
            }
        )
        ids = [c.id for c in expand_sweep(cfg.sweep)]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn("concurrency16-None", ids)

    def test_duplicate_cell_ids_raise(self):
        with self.assertRaises(ValueError):
            expand_sweep({"concurrency": [16, 16], "sequence_combinations": [{"isl": 1, "osl": 1, "name": "a"}]})


class TestThresholds(unittest.TestCase):
    def test_percentile_explicit_op(self):
        view = ResultView(samples=[{"ttft_ms": x} for x in [10, 20, 30, 40, 200]])
        ok = PercentileThreshold(metric="ttft_ms", percentile=50, op="<=", value=100).evaluate(view)
        self.assertTrue(ok.passed)
        bad = PercentileThreshold(metric="ttft_ms", percentile=99, op="<=", value=50).evaluate(view)
        self.assertFalse(bad.passed)

    def test_rate(self):
        view = ResultView(scalars={"total_throughput": 1500})
        self.assertTrue(RateThreshold(metric="total_throughput", op=">=", value=1200).evaluate(view).passed)

    def test_goodput_filtered(self):
        samples = [{"ttft_ms": 100, "tpot_ms": 10} for _ in range(8)] + [{"ttft_ms": 999, "tpot_ms": 99}]
        view = ResultView(scalars={"elapsed_s": 1.0}, samples=samples)
        verdict = GoodputThreshold(op=">=", value=8, where={"ttft_ms": "<=300", "tpot_ms": "<=40"}).evaluate(view)
        self.assertEqual(verdict.actual, 8.0)
        self.assertTrue(verdict.passed)

    def test_convergence_full_vs_by_step(self):
        traj = [{"step": i, "metric": "loss", "value": v} for i, v in enumerate([5.0, 1.0, 0.05])]
        view = ResultView(trajectory=traj)
        self.assertTrue(ConvergenceThreshold(metric="loss", target=0.0, epsilon=0.1).evaluate(view).passed)
        # by_step=2 only inspects [5.0, 1.0] -> never within epsilon -> not converged.
        early = ConvergenceThreshold(metric="loss", target=0.0, epsilon=0.1, by_step=2).evaluate(view)
        self.assertFalse(early.passed)

    def test_stability_variance_and_thin_data(self):
        steady = ResultView(samples=[{"ttft_ms": 10.0} for _ in range(5)])
        self.assertTrue(StabilityThreshold(metric="ttft_ms", max_variance=1.0).evaluate(steady).passed)
        noisy = ResultView(samples=[{"ttft_ms": v} for v in [1.0, 100.0, 1.0, 100.0]])
        self.assertFalse(StabilityThreshold(metric="ttft_ms", max_variance=1.0).evaluate(noisy).passed)
        thin = StabilityThreshold(metric="ttft_ms", max_variance=1.0).evaluate(ResultView(samples=[{"ttft_ms": 10.0}]))
        self.assertFalse(thin.passed)
        self.assertIn("not enough", thin.detail)

    def test_missing_data_verdicts(self):
        empty = ResultView()
        for verdict in (
            RateThreshold(metric="total_throughput", op=">=", value=1).evaluate(empty),
            PercentileThreshold(metric="ttft_ms", op="<=", value=1).evaluate(empty),
            GoodputThreshold(op=">=", value=1, where={"ttft_ms": "<=1"}).evaluate(empty),
        ):
            self.assertFalse(verdict.passed)
            self.assertIsNone(verdict.actual)


THRESHOLD_TYPES = [
    {"type": "rate", "metric": "total_throughput", "op": ">=", "value": 1000},
    {"type": "percentile", "metric": "ttft_ms", "percentile": 99, "op": "<=", "value": 50},
    {"type": "goodput", "op": ">=", "value": 8, "where": {"ttft_ms": "<=300"}},
    {"type": "monotonicity", "metric": "loss", "direction": "non_increasing"},
    {"type": "convergence", "metric": "loss", "target": 0.0, "epsilon": 0.1},
    {"type": "stability", "metric": "ttft_ms", "max_variance": 5.0},
]


class TestThresholdUnionParsing(unittest.TestCase):
    """The discriminated union is the core G2 parsing contract; exercise it via
    parse_config (not just direct construction) so the discriminator and each
    member's extra="forbid" are actually covered."""

    def test_all_threshold_types_round_trip(self):
        cfg = parse_config({**BASE, "thresholds": THRESHOLD_TYPES})
        self.assertEqual(
            [t.type for t in cfg.thresholds],
            ["rate", "percentile", "goodput", "monotonicity", "convergence", "stability"],
        )

    def test_unknown_threshold_type_rejected(self):
        with self.assertRaises(ConfigError):
            parse_config({**BASE, "thresholds": [{"type": "bogus", "metric": "x"}]})

    def test_misspelled_threshold_field_rejected(self):
        bad = {"type": "rate", "metric": "x", "op": ">=", "value": 1, "bogus": 2}
        with self.assertRaises(ConfigError):
            parse_config({**BASE, "thresholds": [bad]})

    def test_b2_malformed_where_rejected_via_config(self):
        bad = {"type": "goodput", "op": ">=", "value": 1, "where": {"ttft_ms": "garbage"}}
        with self.assertRaises(ConfigError):
            parse_config({**BASE, "thresholds": [bad]})


class TestThresholdBakeIns(unittest.TestCase):
    def test_b1_monotonicity_guards_windowed_tail(self):
        # series len 3, window 0.25 -> tail = last 1 point. Pre-fix this would
        # vacuously pass (empty pairwise loop); post-fix it is not-enough-data.
        traj = [{"step": i, "metric": "loss", "value": v} for i, v in enumerate([3.0, 2.0, 1.0])]
        verdict = MonotonicityThreshold(metric="loss", window=0.25).evaluate(ResultView(trajectory=traj))
        self.assertFalse(verdict.passed)
        self.assertIn("not enough", verdict.detail)

    def test_b1_monotonicity_still_evaluates_full_window(self):
        traj = [{"step": i, "metric": "loss", "value": v} for i, v in enumerate([5.0, 4.0, 3.0, 2.0, 1.0])]
        verdict = MonotonicityThreshold(metric="loss", window=1.0).evaluate(ResultView(trajectory=traj))
        self.assertTrue(verdict.passed)

    def test_b2_goodput_rejects_malformed_where_at_load(self):
        with self.assertRaises(ValidationError):
            GoodputThreshold(op=">=", value=8, where={"ttft_ms": "garbage"})

    def test_b2_goodput_accepts_wellformed_where(self):
        GoodputThreshold(op=">=", value=8, where={"ttft_ms": "<=300", "tpot_ms": ">= 1.5"})


class TestLoaderEnv(unittest.TestCase):
    def test_b3_unset_env_raises(self):
        os.environ.pop("DTNI_TEST_MISSING", None)
        self.addCleanup(os.environ.pop, "DTNI_TEST_MISSING", None)
        cfg = {**BASE, "container": {"env": {"HF_TOKEN": "${env:DTNI_TEST_MISSING}"}}}
        with self.assertRaises(ConfigError):
            parse_config(cfg)

    def test_b3_empty_env_allowed(self):
        os.environ["DTNI_TEST_EMPTY"] = ""
        self.addCleanup(os.environ.pop, "DTNI_TEST_EMPTY", None)
        cfg = parse_config({**BASE, "container": {"env": {"HF_TOKEN": "${env:DTNI_TEST_EMPTY}"}}})
        self.assertEqual(cfg.container.env["HF_TOKEN"], "")

    def test_b3_embedded_env_ref_rejected(self):
        os.environ["DTNI_TEST_SET"] = "x"
        self.addCleanup(os.environ.pop, "DTNI_TEST_SET", None)
        cfg = {**BASE, "params": {"server_script": "run ${env:DTNI_TEST_SET}.sh"}}
        with self.assertRaises(ConfigError):
            parse_config(cfg)


class TestContainerSpecA3(unittest.TestCase):
    def test_stringifies_non_str_values(self):
        spec = ContainerSpec(ports={8888: 8888}, volumes={Path("/weights"): "/models"}, devices=[0, 1])
        self.assertEqual(spec.ports, {"8888": "8888"})
        self.assertEqual(spec.volumes, {"/weights": "/models"})
        self.assertEqual(spec.devices, ["0", "1"])

    def test_feeds_container_handle_without_typeerror(self):
        # A3 contract: the spec owns str conversion, so ContainerHandle's
        # shlex.quote (which raises TypeError on non-str) emits a clean command.
        spec = ContainerSpec(ports={8888: 8888}, env={"HF_TOKEN": "tok"})
        handle = ContainerHandle(image="img:tag", run_id="r1", runner=_FakeRunner(), **spec.to_handle_kwargs())
        cmd = handle.build_run_command()
        self.assertIn("8888:8888", cmd)
        self.assertIn("HF_TOKEN=tok", cmd)

    def test_rejects_non_token_values(self):
        # Fail-closed: None/bool/list must error at load, never become
        # "None"/"True"/"['x']" and emit a malformed docker arg.
        for bad in ({"X": None}, {"X": True}, {"X": ["a"]}):
            with self.assertRaises(ValidationError):
                ContainerSpec(env=bad)
        with self.assertRaises(ValidationError):
            ContainerSpec(devices=[None])

    def test_handle_fails_closed_on_raw_non_str(self):
        # The other half of the contract: bypass the spec and feed the handle a
        # non-str directly -> shlex.quote raises rather than emitting a bad arg.
        handle = ContainerHandle(image="i", run_id="r", runner=_FakeRunner(), ports={8888: 8888})
        with self.assertRaises(TypeError):
            handle.build_run_command()


class TestNumericFieldBounds(unittest.TestCase):
    """Operator-authored numeric fields must be range-checked at load, not
    indexed/sliced blindly at evaluate time (the percentile IndexError class)."""

    def test_percentile_out_of_range_rejected_directly(self):
        for bad in (150, -5, 100.1):
            with self.assertRaises(ValidationError):
                PercentileThreshold(metric="x", percentile=bad, op="<=", value=1)

    def test_percentile_out_of_range_rejected_via_config(self):
        for bad in (150, -5):
            thr = {"type": "percentile", "metric": "x", "op": "<=", "value": 1, "percentile": bad}
            with self.assertRaises(ConfigError):
                parse_config({**BASE, "thresholds": [thr]})

    def test_percentile_boundaries_accepted(self):
        for ok in (0, 99, 100):
            PercentileThreshold(metric="x", percentile=ok, op="<=", value=1)

    def test_convergence_by_step_must_be_positive(self):
        for bad in (0, -1):
            with self.assertRaises(ValidationError):
                ConvergenceThreshold(metric="loss", target=0.0, epsilon=0.1, by_step=bad)
        ConvergenceThreshold(metric="loss", target=0.0, epsilon=0.1, by_step=1)


class TestTopologyAndTpConsistency(unittest.TestCase):
    def test_empty_roles_rejected(self):
        with self.assertRaises(ConfigError):
            parse_config({**BASE, "topology": {"roles": {}}})

    def test_tp_matching_gpus_per_node_accepted(self):
        # BASE server role has gpus_per_node=8.
        cfg = parse_config({**BASE, "sweep": {**BASE["sweep"], "tensor_parallelism": [8]}})
        self.assertEqual(cfg.sweep.tensor_parallelism, [8])

    def test_tp_diverging_from_topology_rejected(self):
        with self.assertRaises(ConfigError):
            parse_config({**BASE, "sweep": {**BASE["sweep"], "tensor_parallelism": [4]}})


class TestContainerSpecRejectsFloat(unittest.TestCase):
    def test_float_token_rejected(self):
        # A float port (8888.0) would stringify to a malformed "-p 8888.0:..".
        with self.assertRaises(ValidationError):
            ContainerSpec(ports={8888.0: 8888})
        with self.assertRaises(ValidationError):
            ContainerSpec(env={"RATIO": 0.8})
        with self.assertRaises(ValidationError):
            ContainerSpec(devices=[0.5])


class _FakeRunner:
    def exec(self, cmd, timeout=None):
        return ""


if __name__ == "__main__":
    unittest.main()
