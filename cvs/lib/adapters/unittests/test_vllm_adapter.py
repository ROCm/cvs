"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Offline gate for the vLLM adapter (PR-Y Integration Milestone).

Tests by state-space card + adversarial-review-strengthened guards:

- ``test_registered`` -- the @register_adapter("vllm") decorator runs at
  import time and the registry returns this class.
- ``test_read_bench_result_happy_path`` -- the pure, static parser maps
  oracle vLLM ``benchmark_serving --save-result`` JSON keys into
  CVS-internal scalar names and zips per-request lists into G3.2's
  long-format samples (ttft_ms / tpot_ms / itl_ms / e2el_ms /
  output_tokens).
- ``test_read_bench_result_missing_file`` -- parse may be called before
  the bench produced a file; missing/None must return ``({}, [])``.
- ``test_bench_command_consumes_sweep_axes`` -- adversarial-review
  regression guard: the bench command must honor
  ``sweep.concurrency[0]`` / ``sweep.sequence_combinations[0]`` /
  ``sweep.tensor_parallelism[0]`` from the typed config, not silently
  fall through to hardcoded magic-number defaults (the v1 inference
  layer's exact failure mode the refactor kills).

``launch`` / ``progress_predicate`` are NOT unit-tested here: they need a
real ``RunContext`` + Pssh + ``ContainerHandle`` and would only verify
mocks. ``--collect-only`` proves the dispatch wiring; the real-HW gate
(separate, human-driven) covers end-to-end behavior.
"""

from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

from cvs.lib.adapters.vllm_adapter import VllmAdapter
from cvs.lib.registry import get_adapter


class TestVllmAdapterRegistration(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_adapter("vllm"), VllmAdapter)
        self.assertEqual(VllmAdapter.framework, "vllm")


class TestVllmAdapterReadBenchResult(unittest.TestCase):
    def test_read_bench_result_happy_path(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        rp = tmp / "bench_result.json"
        rp.write_text(
            json.dumps(
                {
                    "duration": 60.0,
                    "request_throughput": 53.3,
                    "output_throughput": 4321.0,
                    "total_token_throughput": 4651.0,
                    "mean_ttft_ms": 70.0,
                    "p99_ttft_ms": 120.0,
                    "mean_tpot_ms": 9.5,
                    "p99_tpot_ms": 18.0,
                    "ttfts": [10.0, 20.0, 30.0],
                    "tpots": [2.0, 3.0, 4.0],
                    "itls": [1.5, 1.6, 1.7],
                    "e2els": [100.0, 110.0, 120.0],
                    "output_lens": [128, 128, 128],
                }
            )
        )

        scalars, samples = VllmAdapter.read_bench_result(str(rp))

        self.assertEqual(scalars["elapsed_s"], 60.0)
        self.assertEqual(scalars["request_throughput"], 53.3)
        self.assertEqual(scalars["output_throughput"], 4321.0)
        self.assertEqual(scalars["total_throughput"], 4651.0)
        self.assertEqual(scalars["mean_ttft_ms"], 70.0)
        self.assertEqual(scalars["p99_ttft_ms"], 120.0)
        self.assertEqual(scalars["mean_tpot_ms"], 9.5)
        self.assertEqual(scalars["p99_tpot_ms"], 18.0)

        self.assertEqual(len(samples), 3)
        first = samples[0]
        self.assertEqual(first["request_id"], 0)
        self.assertEqual(first["ttft_ms"], 10.0)
        self.assertEqual(first["tpot_ms"], 2.0)
        self.assertEqual(first["itl_ms"], 1.5)
        self.assertEqual(first["e2el_ms"], 100.0)
        self.assertEqual(first["output_tokens"], 128.0)
        self.assertEqual(first["role"], "server")

    def test_read_bench_result_missing_file(self) -> None:
        scalars, samples = VllmAdapter.read_bench_result("/nonexistent/bench.json")
        self.assertEqual(scalars, {})
        self.assertEqual(samples, [])

        scalars, samples = VllmAdapter.read_bench_result(None)
        self.assertEqual(scalars, {})
        self.assertEqual(samples, [])


class TestVllmAdapterBenchCommandConsumesSweepAxes(unittest.TestCase):
    """Adversarial-review regression: the bench command must reflect the
    typed sweep values (concurrency / isl / osl / TP), not silently fall
    through to hardcoded magic-number defaults. Single-cell barebones."""

    def test_bench_command_consumes_sweep_axes(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        # A duck-typed ctx -- just enough surface for _bench_command +
        # _single_axis. Avoids spinning up a full RunContext / Pssh.
        cfg = types.SimpleNamespace(
            params=types.SimpleNamespace(
                # Legacy escape-hatch mode: ``bench_serv_script`` is set so
                # _bench_command takes the pass-through path. The composed
                # path is exercised in
                # ``TestVllmAdapterComposesBenchArgv`` below.
                server_script=None,
                bench_serv_script="benchmark_serving.py",
                bench_extra_args=[],
                backend="vllm",
                base_url="http://0.0.0.0",
                port_no=8888,
                dataset_name="random",
                num_prompts=3200,
                request_rate="inf",
                burstiness=1.0,
                tokenizer_mode="auto",
                extra_percentile_metrics=[],
                tensor_parallelism=8,
                concurrency=42,
                isl=2048,
                osl=4096,
            ),
            thresholds=[],
            fabric=None,
            model="meta-llama/Llama-3.1-70B-Instruct",
            seed=7,
        )
        ctx = types.SimpleNamespace(
            config=cfg,
            cell=None,
            layout=types.SimpleNamespace(logs_dir=str(tmp)),
        )

        cmd = VllmAdapter()._bench_command(ctx, Path(tmp) / "bench.json", "vllm_test")

        # Every swept axis must appear in the command verbatim. If the
        # adapter regressed to the hardcoded 16/1024/1024 defaults these
        # asserts would fail (the YAML's 42/2048/4096 would never reach
        # the bench).
        self.assertIn("--max-concurrency 42", cmd)
        self.assertIn("--random-input-len 2048", cmd)
        self.assertIn("--random-output-len 4096", cmd)
        # Seed and detached-exec shape also asserted as regression guards.
        self.assertIn("--seed 7", cmd)
        self.assertIn("docker exec -d vllm_test", cmd)


class TestVllmAdapterDerivesPercentileMetricsFromThresholds(unittest.TestCase):
    """B1 regression: ``--percentile-metrics`` / ``--metric-percentiles`` must
    come from ``thresholds[]`` (single source of truth), not from removed
    ``params.percentile_metrics`` / ``params.metric_percentiles`` fields.
    ``extra_percentile_metrics`` is union'd in for measure-only debug metrics.
    A config with no PercentileThreshold falls back to the legacy default
    set."""

    def _make_ctx(self, thresholds, extra_metrics=None):
        from cvs.lib.adapters.vllm_adapter import VllmAdapter  # noqa: F401

        tmp = Path(tempfile.mkdtemp())
        cfg = types.SimpleNamespace(
            params=types.SimpleNamespace(
                server_script=None,
                bench_serv_script="benchmark_serving.py",
                bench_extra_args=[],
                backend="vllm",
                base_url="http://0.0.0.0",
                port_no=8888,
                dataset_name="random",
                num_prompts=3200,
                request_rate="inf",
                burstiness=1.0,
                tokenizer_mode="auto",
                extra_percentile_metrics=extra_metrics or [],
                tensor_parallelism=8,
                concurrency=16,
                isl=1024,
                osl=1024,
            ),
            thresholds=thresholds,
            fabric=None,
            model="meta-llama/Llama-3.1-70B-Instruct",
            seed=0,
        )
        ctx = types.SimpleNamespace(
            config=cfg,
            cell=None,
            layout=types.SimpleNamespace(logs_dir=str(tmp)),
        )
        return ctx, tmp

    def test_derives_from_percentile_thresholds(self):
        from cvs.lib.adapters.vllm_adapter import VllmAdapter
        from cvs.lib.config.thresholds import PercentileThreshold, RateThreshold

        thresholds = [
            RateThreshold(metric="total_throughput", op=">=", value=1200),
            PercentileThreshold(metric="ttft_ms", percentile=99, op="<=", value=200),
            PercentileThreshold(metric="tpot_ms", percentile=99, op="<=", value=30),
        ]
        ctx, tmp = self._make_ctx(thresholds)
        cmd = VllmAdapter()._bench_command(ctx, Path(tmp) / "b.json", "c")
        # Derived from threshold metrics with _ms stripped, in declaration order.
        self.assertIn("--percentile-metrics ttft,tpot", cmd)
        self.assertIn("--metric-percentiles 99", cmd)
        # Rate threshold metric must NOT leak into the percentile-metrics flag.
        self.assertNotIn("total_throughput", cmd.split("--percentile-metrics ")[1].split(" ")[0])

    def test_extra_percentile_metrics_union(self):
        from cvs.lib.adapters.vllm_adapter import VllmAdapter
        from cvs.lib.config.thresholds import PercentileThreshold

        thresholds = [
            PercentileThreshold(metric="ttft_ms", percentile=99, op="<=", value=200),
        ]
        ctx, tmp = self._make_ctx(thresholds, extra_metrics=["itl", "e2el"])
        cmd = VllmAdapter()._bench_command(ctx, Path(tmp) / "b.json", "c")
        # ttft (from threshold) + itl, e2el (from extra), preserving order.
        self.assertIn("--percentile-metrics ttft,itl,e2el", cmd)

    def test_no_thresholds_falls_back_to_default_set(self):
        from cvs.lib.adapters.vllm_adapter import VllmAdapter

        ctx, tmp = self._make_ctx([])
        cmd = VllmAdapter()._bench_command(ctx, Path(tmp) / "b.json", "c")
        self.assertIn("--percentile-metrics ttft,tpot,itl,e2el", cmd)
        self.assertIn("--metric-percentiles 99", cmd)


class TestVllmAdapterComposesServerCommand(unittest.TestCase):
    """Adapter composes ``vllm serve`` argv from typed params when
    ``params.server_script`` is None. The legacy escape hatch (set the
    string) is the WHEN-NOT path -- here we exercise the default."""

    def test_server_command_baseline(self):
        from cvs.lib.adapters.vllm_adapter import VllmAdapter

        cfg = types.SimpleNamespace(
            params=types.SimpleNamespace(
                server_script=None,
                max_model_len=9216,
                gpu_memory_utilization=0.9,
                quantization=None,
                dtype=None,
                trust_remote_code=False,
                download_dir=None,
                port_no=8888,
                server_extra_args=[],
            ),
            model="meta-llama/Llama-3.3-70B-Instruct",
        )
        ctx = types.SimpleNamespace(config=cfg)
        cmd = VllmAdapter()._server_command(ctx, tp=8)
        # Mandatory shape: ``vllm serve <model> --tensor-parallel-size N ...``
        self.assertIn("vllm serve meta-llama/Llama-3.3-70B-Instruct", cmd)
        self.assertIn("--tensor-parallel-size 8", cmd)
        self.assertIn("--host 0.0.0.0", cmd)
        self.assertIn("--port 8888", cmd)
        self.assertIn("--max-model-len 9216", cmd)
        self.assertIn("--gpu-memory-utilization 0.9", cmd)
        # Optional knobs default-absent.
        self.assertNotIn("--quantization", cmd)
        self.assertNotIn("--dtype", cmd)
        self.assertNotIn("--trust-remote-code", cmd)
        self.assertNotIn("--download-dir", cmd)

    def test_server_command_optionals_emitted_when_set(self):
        from cvs.lib.adapters.vllm_adapter import VllmAdapter

        cfg = types.SimpleNamespace(
            params=types.SimpleNamespace(
                server_script=None,
                max_model_len=4096,
                gpu_memory_utilization=0.85,
                quantization="fp8",
                dtype="bfloat16",
                trust_remote_code=True,
                download_dir="/models",
                port_no=8888,
                server_extra_args=["--enable-prefix-caching", "--enforce-eager"],
            ),
            model="meta-llama/Llama-3.3-70B-Instruct",
        )
        ctx = types.SimpleNamespace(config=cfg)
        cmd = VllmAdapter()._server_command(ctx, tp=4)
        self.assertIn("--quantization fp8", cmd)
        self.assertIn("--dtype bfloat16", cmd)
        self.assertIn("--trust-remote-code", cmd)
        self.assertIn("--download-dir /models", cmd)
        self.assertIn("--enable-prefix-caching", cmd)
        self.assertIn("--enforce-eager", cmd)
        # Order guard: extra-args land AFTER the composed flags so an
        # operator override actually wins (last-write-wins on argparse).
        self.assertGreater(cmd.index("--enforce-eager"), cmd.index("--gpu-memory-utilization"))


class TestVllmAdapterComposesBenchArgv(unittest.TestCase):
    """Composed-mode ``_bench_command`` builds the ``vllm bench serve`` argv
    via shlex-joined arguments routed through
    ``python -m vllm.entrypoints.cli.main bench serve``. The base-url
    carries the port (new vLLM CLI shape -- ``--port`` is silently
    ignored by the new CLI, so we MUST inline)."""

    def _ctx(self, **overrides):
        from pathlib import Path as _Path

        tmp = _Path(tempfile.mkdtemp())
        cfg = types.SimpleNamespace(
            params=types.SimpleNamespace(
                # Composed mode: bench_serv_script intentionally None.
                server_script=None,
                bench_serv_script=None,
                bench_extra_args=[],
                backend="vllm",
                base_url="http://0.0.0.0",
                port_no=8888,
                dataset_name="random",
                num_prompts=64,
                request_rate="inf",
                burstiness=1.0,
                tokenizer_mode="auto",
                extra_percentile_metrics=[],
                tensor_parallelism=8,
                concurrency=16,
                isl=1024,
                osl=1024,
            ),
            thresholds=[],
            fabric=None,
            model="meta-llama/Llama-3.3-70B-Instruct",
            seed=0,
        )
        for k, v in overrides.items():
            setattr(cfg.params, k, v)
        ctx = types.SimpleNamespace(config=cfg, cell=None, layout=types.SimpleNamespace(logs_dir=str(tmp)))
        return ctx, tmp

    def test_bench_command_composed_uses_module_invocation(self):
        from cvs.lib.adapters.vllm_adapter import VllmAdapter

        ctx, tmp = self._ctx()
        cmd = VllmAdapter()._bench_command(ctx, Path(tmp) / "bench.json", "vllm_test")
        # New-CLI module invocation, not bare benchmark_serving.py.
        self.assertIn("python -m vllm.entrypoints.cli.main bench serve", cmd)
        # Base-url carries port (the new-CLI shape the old --port could not).
        self.assertIn("--base-url http://0.0.0.0:8888", cmd)
        # Critical: bare ``--port 8888`` flag must NOT appear -- the new CLI
        # silently ignores it and tries 0.0.0.0:80, the bug that bit us.
        # (Use a word-boundary guard so the port inside --base-url passes.)
        import re

        self.assertIsNone(
            re.search(r"(?<!:)\b--port\s+8888\b", cmd),
            f"bare --port flag must not appear in composed bench cmd: {cmd}",
        )
        self.assertIn("--save-result", cmd)
        self.assertIn("--result-filename ", cmd)

    def test_bench_extra_args_appended(self):
        from cvs.lib.adapters.vllm_adapter import VllmAdapter

        ctx, tmp = self._ctx(bench_extra_args=["--ignore-eos", "--debug"])
        cmd = VllmAdapter()._bench_command(ctx, Path(tmp) / "bench.json", "c")
        self.assertIn("--ignore-eos", cmd)
        self.assertIn("--debug", cmd)
        self.assertGreater(cmd.index("--debug"), cmd.index("--save-result"))


class TestPercentileThresholdSamplesFallback(unittest.TestCase):
    """When per-request samples are absent (e.g. the new ``vllm bench serve``
    CLI dropped the arrays), fall back to the matching scalar
    ``p{int(percentile)}_{metric}``. The verdict ``detail`` annotates
    which path produced the value."""

    def test_uses_samples_when_present(self):
        from cvs.lib.config.thresholds import PercentileThreshold, ResultView

        view = ResultView(
            samples=[{"ttft_ms": v} for v in [100.0, 110.0, 120.0, 5000.0]],
            scalars={"p99_ttft_ms": 999.0},
        )
        verdict = PercentileThreshold(metric="ttft_ms", percentile=99, op="<=", value=10000).evaluate(view)
        # Computed from samples (not the 999 scalar): not the framework fallback.
        self.assertTrue(verdict.passed)
        self.assertIsNone(verdict.detail)
        self.assertNotEqual(verdict.actual, 999.0)

    def test_falls_back_to_scalar_when_samples_empty(self):
        from cvs.lib.config.thresholds import PercentileThreshold, ResultView

        view = ResultView(
            samples=[],
            scalars={"p99_ttft_ms": 150.0},
        )
        verdict = PercentileThreshold(metric="ttft_ms", percentile=99, op="<=", value=200).evaluate(view)
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.actual, 150.0)
        # Detail must name the scalar so a manifest reader can tell which
        # path produced the verdict.
        self.assertIn("p99_ttft_ms", verdict.detail)
        self.assertIn("from framework scalar", verdict.detail)

    def test_no_samples_no_scalar_returns_null_actual(self):
        from cvs.lib.config.thresholds import PercentileThreshold, ResultView

        view = ResultView(samples=[], scalars={})
        verdict = PercentileThreshold(metric="ttft_ms", percentile=99, op="<=", value=200).evaluate(view)
        self.assertFalse(verdict.passed)
        self.assertIsNone(verdict.actual)
        self.assertIn("no fallback scalar", verdict.detail)


if __name__ == "__main__":
    unittest.main()
