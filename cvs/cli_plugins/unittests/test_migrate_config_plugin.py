"""Tests for the `cvs migrate-config` CLI plugin (G2.2 / C5 CLI half).

The plugin is a thin shell over `cvs.lib.config.migrate.migrate_vllm_megaconfig`.
These tests pin the bake-in invariants from the G2.2 brief:

  * Mean-named metric predicates (C5) survive round-trip -- the plugin does NOT
    silently relabel `mean_*` as P99.
  * Exit code 2 on `ConfigError` from the resulting v2 file (the migrated YAML
    must itself satisfy G2's `parse_config` -- fail-fast at migration time,
    never ship a v2 file the loader rejects).

Plus adversarial cases: empty input, malformed YAML, exit-code checks, schema
equality (not subset) on derived threshold types, op/value direction pinning,
multi-model mega-config handling, container.env survival, target_gpu plumb.

These tests are self-contained: they construct v1 YAML inputs inline (no spike
fixtures, no spike-oracle imports). The plugin module under test is
`cvs.cli_plugins.migrate_config_plugin` -- it does not yet exist, so this
file is expected to fail collection with ImportError until the implementer
lands G2.2 in a subsequent stage.
"""

from __future__ import annotations

import argparse
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import yaml

from cvs.lib.config.loader import parse_config


# ---------------------------------------------------------------------------
# Fixtures (built inline -- no oracle imports)
# ---------------------------------------------------------------------------

def _v1_single_model_mega() -> dict:
    """A minimal v1 mega-config carrying one vLLM model with mean_* metrics.

    The `result_dict` mean-named metrics are the C5 trip-wire: a buggy
    migration could relabel them as P99 percentiles. The bake-in test below
    asserts they survive verbatim as `mean_ttft_ms` / `mean_tpot_ms`.
    """
    return {
        "config": {"nnodes": "1", "container_image": "img:tag"},
        "benchmark_params": {
            "gpt-oss-120b": {
                "backend": "vllm",
                "model": "openai/gpt-oss-120b",
                "concurrency_levels": [16, 32],
                "sequence_combinations": [
                    {"isl": "1024", "osl": "1024", "name": "balanced"}
                ],
                "tensor_parallelism": "1",
                "num_prompts": "3200",
                "max_model_length": "9216",
                "percentile_metrics": "ttft,tpot,itl,e2el",
                "server_script": "gpt.sh",
                "result_dict": {
                    "ISL=1024,OSL=1024,TP=1,CONC=16": {
                        "total_throughput_per_sec": "4651",
                        "mean_ttft_ms": "70",
                        "mean_tpot_ms": "8",
                    },
                    "ISL=1024,OSL=1024,TP=1,CONC=32": {
                        "total_throughput_per_sec": "4200",
                        "mean_ttft_ms": "95",
                        "mean_tpot_ms": "11",
                    },
                },
            },
        },
    }


def _v1_multi_model_mega() -> dict:
    """Two-model mega-config -- the engine returns a dict keyed by model slug.

    The plugin must NOT silently drop or merge models when the brief shape is
    `-o <single-v2-yaml>`: it either writes a multi-doc YAML containing both,
    or exits non-zero with a clear message. Either is acceptable; silent loss
    is not.
    """
    return {
        "config": {"nnodes": "1", "container_image": "img:tag"},
        "benchmark_params": {
            "gpt-oss-120b": {
                "backend": "vllm",
                "model": "openai/gpt-oss-120b",
                "concurrency_levels": [16],
                "sequence_combinations": [{"isl": "1024", "osl": "1024", "name": "balanced"}],
                "tensor_parallelism": "1",
                "server_script": "gpt.sh",
                "result_dict": {},
            },
            "qwen3-80b": {
                "backend": "vllm",
                "model": "Qwen/Qwen3-Next-80B",
                "concurrency_levels": [16],
                "sequence_combinations": [{"isl": "8192", "osl": "1024", "name": "long_context"}],
                "tensor_parallelism": "8",
                "server_script": "qwen.sh",
                "result_dict": {},
            },
        },
    }


def _v1_tp8_single_model() -> dict:
    return {
        "config": {"nnodes": "1", "container_image": "img:tag"},
        "benchmark_params": {
            "qwen3-80b": {
                "backend": "vllm",
                "model": "Qwen/Qwen3-Next-80B",
                "concurrency_levels": [16],
                "sequence_combinations": [{"isl": "8192", "osl": "1024", "name": "long_context"}],
                "tensor_parallelism": "8",
                "server_script": "qwen.sh",
                "result_dict": {},
            },
        },
    }


def _v1_with_changeme_sentinel() -> dict:
    return {
        "config": {"hf_token_file": "<changeme>"},
        "benchmark_params": {},
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(dir_path: Path, name: str, payload: dict) -> Path:
    out = dir_path / name
    out.write_text(yaml.safe_dump(payload, sort_keys=False))
    return out


def _write_raw(dir_path: Path, name: str, text: str) -> Path:
    out = dir_path / name
    out.write_text(text)
    return out


def _load_v2_docs(v2_path: Path) -> list:
    """Load a v2 output file as a list of one-or-more configs.

    The migrate engine returns a `{slug: cfg}` dict, so the plugin may write
    one config per file (single-model mega) or a multi-doc YAML / mapping for
    multi-model. Normalize all shapes to a list of v2 config dicts so the
    threshold assertions work either way.
    """
    text = v2_path.read_text()
    # Try multi-doc first; falls back to single doc cleanly.
    docs = list(yaml.safe_load_all(text))
    docs = [d for d in docs if d is not None]
    if len(docs) == 1 and isinstance(docs[0], dict) and "schema_version" not in docs[0]:
        # Plugin emitted a {slug: cfg} mapping. Unwrap.
        inner = list(docs[0].values())
        if inner and all(isinstance(v, dict) and "schema_version" in v for v in inner):
            return inner
    return docs


def _run_plugin(input_path: Path, output_path: Path, target_gpu: str = "mi355x"):
    """Invoke the plugin's `run(args)` directly and capture (stdout, stderr).

    The plugin is expected to surface CLI errors via `SystemExit(<code>)`.
    Callers wrap this in `assertRaises(SystemExit)` for the failure paths.
    """
    from cvs.cli_plugins.migrate_config_plugin import MigrateConfigPlugin

    plugin = MigrateConfigPlugin()
    args = argparse.Namespace(
        input_file=str(input_path),
        output=str(output_path),
        target_gpu=target_gpu,
    )
    out_buf, err_buf = io.StringIO(), io.StringIO()
    with redirect_stdout(out_buf), redirect_stderr(err_buf):
        plugin.run(args)
    return out_buf.getvalue(), err_buf.getvalue()


# ---------------------------------------------------------------------------
# Surface tests -- the plugin exposes the right SubcommandPlugin contract.
# ---------------------------------------------------------------------------

class TestMigrateConfigPluginSurface(unittest.TestCase):
    def test_plugin_class_importable(self):
        from cvs.cli_plugins.migrate_config_plugin import MigrateConfigPlugin

        self.assertTrue(callable(MigrateConfigPlugin))

    def test_subcommand_name_is_migrate_config(self):
        from cvs.cli_plugins.migrate_config_plugin import MigrateConfigPlugin

        self.assertEqual(MigrateConfigPlugin().get_name(), "migrate-config")

    def test_plugin_subclasses_subcommandplugin(self):
        """The CLI loader dispatches by SubcommandPlugin -- a stand-alone class
        that merely duck-types the interface would not be discovered."""
        from cvs.cli_plugins.base import SubcommandPlugin
        from cvs.cli_plugins.migrate_config_plugin import MigrateConfigPlugin

        self.assertTrue(issubclass(MigrateConfigPlugin, SubcommandPlugin))

    def test_parser_registers_input_and_output_args(self):
        from cvs.cli_plugins.migrate_config_plugin import MigrateConfigPlugin

        root = argparse.ArgumentParser()
        subparsers = root.add_subparsers(dest="cmd")
        MigrateConfigPlugin().get_parser(subparsers)

        # The brief: `cvs migrate-config <v1-yaml> -o <v2-yaml>`.
        # The positional input must be present; `-o` must be a parsable flag.
        with tempfile.TemporaryDirectory() as td:
            v1 = _write_yaml(Path(td), "in.yaml", _v1_single_model_mega())
            v2 = Path(td) / "out.yaml"
            parsed = root.parse_args(["migrate-config", str(v1), "-o", str(v2)])
            self.assertEqual(parsed.cmd, "migrate-config")
            values = [getattr(parsed, k) for k in vars(parsed) if isinstance(getattr(parsed, k), str)]
            self.assertIn(str(v1), values)
            self.assertIn(str(v2), values)

    def test_parser_input_attribute_is_named_input_file(self):
        """`_run_plugin` (and any caller that hand-builds the Namespace) keys
        on `input_file`. Pin the argparse `dest=` so the plugin's `run()`
        and these tests agree on the attribute name."""
        from cvs.cli_plugins.migrate_config_plugin import MigrateConfigPlugin

        root = argparse.ArgumentParser()
        subparsers = root.add_subparsers(dest="cmd")
        MigrateConfigPlugin().get_parser(subparsers)
        parsed = root.parse_args(["migrate-config", "in.yaml", "-o", "out.yaml"])
        self.assertEqual(getattr(parsed, "input_file", None), "in.yaml",
                         "positional input must bind to `input_file` (matches `run(args)` contract)")
        self.assertEqual(getattr(parsed, "output", None), "out.yaml",
                         "`-o` must bind to `output`")


# ---------------------------------------------------------------------------
# Bake-in #1 -- mean-named metric predicates (C5) survive round-trip.
# ---------------------------------------------------------------------------

class TestMeanNamedMetricsSurvive(unittest.TestCase):
    """C5 trip-wire: the plugin must not silently relabel mean_* as P99."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tdir = Path(self._tmp.name)
        # parse_config resolves `${env:HF_TOKEN}` -- the engine emits that
        # placeholder unconditionally, so the self-check would otherwise fail
        # with ConfigError("required env var 'HF_TOKEN' is not set") on a
        # bare shell. Set it for the happy-path tests; the failure-path tests
        # below unset it deliberately as one ConfigError trigger.
        self._env = mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def _migrate_and_load(self) -> dict:
        v1 = _write_yaml(self.tdir, "in.yaml", _v1_single_model_mega())
        v2_path = self.tdir / "out.yaml"
        _run_plugin(v1, v2_path)
        self.assertTrue(v2_path.exists(), "plugin must write the v2 YAML on success")
        docs = _load_v2_docs(v2_path)
        self.assertEqual(len(docs), 1, "single-model mega must produce exactly one v2 config")
        return docs[0]

    def test_mean_ttft_metric_name_preserved(self):
        v2 = self._migrate_and_load()
        metric_names = {t["metric"] for t in v2["thresholds"]}
        self.assertIn("mean_ttft_ms", metric_names,
                      "mean_ttft_ms must survive verbatim -- never relabeled as p99")
        self.assertIn("mean_tpot_ms", metric_names,
                      "mean_tpot_ms must survive verbatim -- never relabeled as p99")

    def test_no_p99_relabel_anywhere_in_thresholds(self):
        v2 = self._migrate_and_load()
        for t in v2["thresholds"]:
            self.assertNotIn("p99", str(t.get("metric", "")).lower(),
                             f"threshold {t!r} silently relabels a mean metric as p99")
            self.assertNotEqual(t.get("type"), "percentile",
                                f"derived threshold must not be typed 'percentile' (source was a mean scalar): {t!r}")
            # The 'percentile' KEY itself must not be set on any derived threshold,
            # even if 'type' is rate -- a stray percentile field is a relabeling smell.
            self.assertNotIn("percentile", t,
                             f"threshold {t!r} has a stray 'percentile' field")

    def test_threshold_types_schema_equality_not_subset(self):
        """Pin the exact derived-threshold shape from a mean-only input.

        Subset checks ("contains rate") let a buggy migration sneak in extra
        percentile predicates unnoticed; equality on the (type, metric) set
        catches that.
        """
        v2 = self._migrate_and_load()
        derived = {(t["type"], t["metric"]) for t in v2["thresholds"]}
        expected = {
            ("rate", "total_throughput"),
            ("rate", "mean_ttft_ms"),
            ("rate", "mean_tpot_ms"),
        }
        self.assertSetEqual(derived, expected,
                            "derived threshold set must equal expected exactly (no quiet additions)")
        # Defensive: assertSetEqual with an empty derived set against an empty
        # expected would pass; pin non-empty explicitly so a mutant that emits
        # [] cannot align by both sides going to {}.
        self.assertEqual(len(v2["thresholds"]), 3,
                         "mean-only input must yield exactly 3 thresholds (throughput floor + 2 latency ceils)")

    def test_threshold_ops_pin_direction(self):
        """Throughput is a floor (>=), latencies are ceilings (<=).

        A mutant that flips the op would reverse the safety semantics: a
        throughput ceiling would let regressions pass, a latency floor would
        let slowdowns pass. Pinning ops catches this independent of the
        numeric value or threshold count.
        """
        v2 = self._migrate_and_load()
        by_metric = {t["metric"]: t for t in v2["thresholds"]}
        self.assertEqual(by_metric["total_throughput"]["op"], ">=",
                         "throughput is a floor -- op must be '>='")
        self.assertEqual(by_metric["mean_ttft_ms"]["op"], "<=",
                         "mean latency is a ceiling -- op must be '<='")
        self.assertEqual(by_metric["mean_tpot_ms"]["op"], "<=",
                         "mean latency is a ceiling -- op must be '<='")

    def test_threshold_values_pin_envelope_direction(self):
        """Throughput floor is the OBSERVED MIN across cells; latency ceilings
        are the OBSERVED MAX. A mutant that swaps min/max would silently widen
        or tighten the envelope past the data.

        Input has two cells: throughput {4651, 4200}; ttft {70, 95}; tpot {8, 11}.
        """
        v2 = self._migrate_and_load()
        by_metric = {t["metric"]: t for t in v2["thresholds"]}
        self.assertEqual(by_metric["total_throughput"]["value"], 4200.0,
                         "throughput floor must be MIN observed (4200), not MAX (4651)")
        self.assertEqual(by_metric["mean_ttft_ms"]["value"], 95.0,
                         "ttft ceiling must be MAX observed (95), not MIN (70)")
        self.assertEqual(by_metric["mean_tpot_ms"]["value"], 11.0,
                         "tpot ceiling must be MAX observed (11), not MIN (8)")


# ---------------------------------------------------------------------------
# Bake-in #2 -- the plugin must self-check via parse_config and exit 2 on
# ConfigError.
# ---------------------------------------------------------------------------

class TestExitCodeOnConfigError(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tdir = Path(self._tmp.name)

    def test_exit_code_2_when_self_check_raises_config_error(self):
        """With HF_TOKEN unset, the migrated YAML's `${env:HF_TOKEN}` cannot
        resolve, so the plugin's `parse_config` self-check must raise
        ConfigError, and the plugin must exit with code 2 (not 0, not 1) --
        never write a v2 file the loader would reject."""
        v1 = _write_yaml(self.tdir, "in.yaml", _v1_single_model_mega())
        v2_path = self.tdir / "out.yaml"

        env_no_token = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        with mock.patch.dict(os.environ, env_no_token, clear=True):
            with self.assertRaises(SystemExit) as ctx:
                _run_plugin(v1, v2_path)
        self.assertEqual(ctx.exception.code, 2,
                         "ConfigError from self-check must surface as exit code 2")

    def test_no_v2_file_written_when_self_check_fails(self):
        """Fail-fast: never ship a v2 file the loader rejects."""
        v1 = _write_yaml(self.tdir, "in.yaml", _v1_single_model_mega())
        v2_path = self.tdir / "out.yaml"

        env_no_token = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        with mock.patch.dict(os.environ, env_no_token, clear=True):
            with self.assertRaises(SystemExit):
                _run_plugin(v1, v2_path)
        self.assertFalse(v2_path.exists(),
                         "plugin must not write a v2 file when parse_config rejects it")

    def test_self_check_runs_via_parse_config_not_validate_only(self):
        """`validate_config` is schema-only (no env resolution); only
        `parse_config` triggers ConfigError on unresolved `${env:HF_TOKEN}`.
        If the plugin self-checks with `validate_config` instead, the
        HF_TOKEN-unset case above would not raise. This test pins that
        `parse_config` (imported into the plugin module) is actually called --
        proven by spying on it."""
        v1 = _write_yaml(self.tdir, "in.yaml", _v1_single_model_mega())
        v2_path = self.tdir / "out.yaml"
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"}):
            import cvs.cli_plugins.migrate_config_plugin as mod
            with mock.patch.object(mod, "parse_config", wraps=mod.parse_config) as spy:
                _run_plugin(v1, v2_path)
                self.assertGreater(spy.call_count, 0,
                                   "plugin must invoke parse_config (not validate_config) as self-check")

    def test_happy_path_v2_parses_clean(self):
        """With HF_TOKEN set, the written v2 YAML must round-trip parse_config."""
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"}):
            v1 = _write_yaml(self.tdir, "in.yaml", _v1_single_model_mega())
            v2_path = self.tdir / "out.yaml"
            _run_plugin(v1, v2_path)
            self.assertTrue(v2_path.exists())
            docs = _load_v2_docs(v2_path)
            self.assertEqual(len(docs), 1)
            cfg = parse_config(docs[0])  # would raise ConfigError on bad shape
            self.assertEqual(cfg.framework, "vllm")
            self.assertEqual(cfg.schema_version, "2")


# ---------------------------------------------------------------------------
# Engine-coupling -- the plugin must delegate to migrate_vllm_megaconfig and
# faithfully preserve its outputs (HF_TOKEN env ref, topology TP, target_gpu).
# ---------------------------------------------------------------------------

class TestEngineCoupling(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tdir = Path(self._tmp.name)
        self._env = mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_hf_token_env_reference_preserved_verbatim(self):
        """The engine emits `container.env.HF_TOKEN: "${env:HF_TOKEN}"` --
        a deferred placeholder, never the resolved secret. A buggy plugin that
        re-serialized after `parse_config` (which RESOLVES placeholders) would
        leak "hf_test_token" into the output file. Pin that the literal
        deferred form survives."""
        v1 = _write_yaml(self.tdir, "in.yaml", _v1_single_model_mega())
        v2_path = self.tdir / "out.yaml"
        _run_plugin(v1, v2_path)
        docs = _load_v2_docs(v2_path)
        self.assertEqual(len(docs), 1)
        self.assertEqual(
            docs[0]["container"]["env"]["HF_TOKEN"], "${env:HF_TOKEN}",
            "plugin must preserve the deferred ${env:HF_TOKEN} placeholder; never the resolved secret"
        )
        self.assertNotIn("hf_test_token", v2_path.read_text(),
                         "resolved HF_TOKEN value must not leak into the written file")

    def test_topology_gpus_per_node_reflects_tensor_parallelism(self):
        """TP lives once, in topology.roles.server.gpus_per_node. A plugin that
        loses the engine's topology block (e.g. by rebuilding from parse_config
        output) would fail this."""
        v1 = _write_yaml(self.tdir, "in.yaml", _v1_tp8_single_model())
        v2_path = self.tdir / "out.yaml"
        _run_plugin(v1, v2_path)
        docs = _load_v2_docs(v2_path)
        self.assertEqual(len(docs), 1)
        self.assertEqual(
            docs[0]["topology"]["roles"]["server"]["gpus_per_node"], 8,
            "TP=8 must surface as topology.roles.server.gpus_per_node=8"
        )

    def test_target_gpu_is_plumbed_into_output(self):
        """The `-t/--target-gpu` flag must reach the engine and land on every
        emitted config; a plugin that hard-codes the GPU silently mis-stamps
        runs."""
        v1 = _write_yaml(self.tdir, "in.yaml", _v1_single_model_mega())
        v2_path = self.tdir / "out.yaml"
        _run_plugin(v1, v2_path, target_gpu="mi300x")
        docs = _load_v2_docs(v2_path)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["target_gpu"], "mi300x",
                         "target_gpu from args must land in the v2 output")
        self.assertEqual(docs[0]["topology"]["roles"]["server"]["selector"], "mi300x",
                         "target_gpu must also plumb into topology selector")

    def test_multi_model_mega_does_not_silently_drop_models(self):
        """The engine returns `{slug: cfg}` for N models; with a single `-o`
        the plugin's options are: (a) write a multi-doc / mapping containing
        BOTH models, or (b) exit non-zero with a clear message. Silently
        keeping only the last model is the bug this test catches."""
        v1 = _write_yaml(self.tdir, "in.yaml", _v1_multi_model_mega())
        v2_path = self.tdir / "out.yaml"
        try:
            _run_plugin(v1, v2_path)
        except SystemExit as exc:
            # Option (b): refused multi-model with single -o.
            self.assertNotEqual(exc.code, 0,
                                "refusing multi-model must be a non-zero exit")
            if v2_path.exists() and v2_path.stat().st_size > 0:
                docs = _load_v2_docs(v2_path)
                self.assertGreaterEqual(len(docs), 2,
                                        "if refusing, must not also write a single-model file")
            return
        # Option (a): wrote both. Both model identities must be present.
        docs = _load_v2_docs(v2_path)
        models_seen = {d.get("model") for d in docs}
        self.assertIn("openai/gpt-oss-120b", models_seen,
                      "gpt-oss-120b model must not be silently dropped from multi-model output")
        self.assertIn("Qwen/Qwen3-Next-80B", models_seen,
                      "qwen3-80b model must not be silently dropped from multi-model output")


# ---------------------------------------------------------------------------
# Adversarial cases -- empty / malformed / sentinel inputs.
# ---------------------------------------------------------------------------

class TestAdversarialInputs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tdir = Path(self._tmp.name)
        self._env = mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_empty_file_exits_nonzero(self):
        """An empty input file has no `benchmark_params` -- nothing to migrate.

        The plugin must fail (non-zero exit), not silently write an empty v2.
        """
        v1 = _write_raw(self.tdir, "empty.yaml", "")
        v2_path = self.tdir / "out.yaml"
        with self.assertRaises(SystemExit) as ctx:
            _run_plugin(v1, v2_path)
        self.assertNotEqual(ctx.exception.code, 0,
                            "empty input must exit non-zero, not pretend success")
        self.assertFalse(v2_path.exists(),
                         "empty input must not produce a v2 file")

    def test_malformed_yaml_exits_via_systemexit(self):
        """Unparseable YAML must surface as a non-zero SystemExit, not a raw
        traceback. (Tightened from earlier lenient `(SystemExit, YAMLError)`:
        letting YAMLError escape means the user sees a bare stack trace, which
        is a CLI bug -- the plugin must catch and translate it.)"""
        v1 = _write_raw(self.tdir, "bad.yaml", "key: : : [unclosed\n  - oops")
        v2_path = self.tdir / "out.yaml"
        with self.assertRaises(SystemExit) as ctx:
            _run_plugin(v1, v2_path)
        self.assertNotEqual(ctx.exception.code, 0,
                            "malformed YAML must exit non-zero")
        self.assertFalse(v2_path.exists(),
                         "malformed YAML must not produce a v2 file")

    def test_changeme_sentinel_rejected(self):
        """The engine refuses `<changeme>` sentinels; the plugin must surface
        that as a non-zero SystemExit, not let the ValueError escape."""
        v1 = _write_yaml(self.tdir, "sentinel.yaml", _v1_with_changeme_sentinel())
        v2_path = self.tdir / "out.yaml"
        with self.assertRaises(SystemExit) as ctx:
            _run_plugin(v1, v2_path)
        self.assertNotEqual(ctx.exception.code, 0)
        self.assertFalse(v2_path.exists())

    def test_missing_input_file_exits_via_systemexit(self):
        """A nonexistent input must exit non-zero with a clear error, not a
        bare `FileNotFoundError` traceback. (Tightened from earlier lenient
        `(SystemExit, FileNotFoundError)`: a CLI must translate IO errors.)"""
        v1 = self.tdir / "does_not_exist.yaml"
        v2_path = self.tdir / "out.yaml"
        with self.assertRaises(SystemExit) as ctx:
            _run_plugin(v1, v2_path)
        self.assertNotEqual(ctx.exception.code, 0,
                            "missing input must exit non-zero, not raise FileNotFoundError")
        self.assertFalse(v2_path.exists())


if __name__ == "__main__":
    unittest.main()
