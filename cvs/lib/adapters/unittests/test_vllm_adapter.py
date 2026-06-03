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
                bench_serv_script="benchmark_serving.py",
                backend="vllm",
                base_url="http://0.0.0.0",
                port_no=8888,
                dataset_name="random",
                num_prompts=3200,
                request_rate="inf",
                burstiness=1.0,
                tokenizer_mode="auto",
                percentile_metrics=["ttft", "tpot", "itl", "e2el"],
                metric_percentiles=99,
            ),
            sweep=types.SimpleNamespace(
                tensor_parallelism=[8],
                concurrency=[42],
                sequence_combinations=[types.SimpleNamespace(isl=2048, osl=4096, name="ynot")],
            ),
            model="meta-llama/Llama-3.1-70B-Instruct",
            seed=7,
        )
        ctx = types.SimpleNamespace(
            config=cfg,
            cell=None,
            layout=types.SimpleNamespace(logs_dir=str(tmp)),
        )

        # _bench_command consults ctx.param; emulate the param() fallback
        # rule from RunContext (cell.params first, then config.params).
        def _param(name, default=None):
            cell_params = getattr(ctx.cell, "params", None)
            if isinstance(cell_params, dict) and name in cell_params:
                return cell_params[name]
            params = getattr(cfg, "params", None)
            if params is not None and hasattr(params, name):
                return getattr(params, name)
            return default

        ctx.param = _param

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


if __name__ == "__main__":
    unittest.main()
