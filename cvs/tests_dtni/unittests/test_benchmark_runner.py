"""Unit tests for cvs.lib.benchmarks.runner orchestration.

Projector behavior is covered separately in test_projectors.py — this
file only exercises the runner's docker-exec / file-discovery loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from cvs.lib.benchmarks.runner import run_benchmarks
from cvs.lib.errors import WorkloadError


@dataclass
class _FakeRunner:
    """Drops per-benchmark result JSONs into <out_dir>/<bid>/."""
    out_dir: Path
    payloads: dict
    file_name: dict  # bid -> filename
    last_cmd: str = ""

    def exec(self, cmd: str, timeout: int = 0) -> str:
        self.last_cmd = cmd
        for bid, payload in self.payloads.items():
            # Match either lm-eval's "/<bid>" (--output_path is the dir) or
            # vllm bench serve's "--result-dir .../<bid>" prefix path.
            if f"/{bid}" in cmd:
                sub = self.out_dir / bid
                sub.mkdir(parents=True, exist_ok=True)
                (sub / self.file_name[bid]).write_text(json.dumps(payload))
                return ""
        return ""


@dataclass
class _FakeHandle:
    name: str
    runner: _FakeRunner


def test_run_benchmarks_lm_eval_then_vllm_serve_in_one_pass(tmp_path: Path):
    # Mix accuracy + perf in one call — exercises both file globs.
    payloads = {
        "mmlu":  {"results": {"mmlu":  {"acc,none": 0.71, "acc_stderr,none": 0.01}}},
        "serve_synth_short": {
            "completed": 200,
            "request_throughput": 12.5,
            "output_throughput": 4321.0,
            "percentiles_ttft_ms": [[95, 84.2]],
        },
    }
    file_name = {"mmlu": "results_20260606.json", "serve_synth_short": "result.json"}
    runner = _FakeRunner(out_dir=tmp_path, payloads=payloads, file_name=file_name)
    handle = _FakeHandle(name="cvs_smoke", runner=runner)

    scalars = run_benchmarks(
        benchmarks=["mmlu", "serve_synth_short"],
        server_handle=handle,
        base_url="http://localhost:8000",
        model_id="llama-3.1-8b",
        model_path="/models/foo",
        output_dir_in_container=str(tmp_path),
        output_dir_on_host=tmp_path,
    )

    # lm-eval projects to bare id (back-compat with Step-4 thresholds).
    assert scalars["mmlu"] == 0.71
    assert scalars["mmlu_stderr"] == 0.01
    # vllm serve projects to dotted namespace under the bench id.
    assert scalars["serve_synth_short.completed"] == 200.0
    assert scalars["serve_synth_short.output_throughput"] == 4321.0
    assert scalars["serve_synth_short.ttft_p95_ms"] == 84.2


def test_run_benchmarks_unknown_id_raises_before_exec(tmp_path: Path):
    runner = _FakeRunner(out_dir=tmp_path, payloads={}, file_name={})
    handle = _FakeHandle(name="cvs_smoke", runner=runner)
    with pytest.raises(KeyError):
        run_benchmarks(
            benchmarks=["does-not-exist"],
            server_handle=handle,
            base_url="http://localhost:8000",
            model_id="x",
            model_path="/models/foo",
            output_dir_in_container=str(tmp_path),
            output_dir_on_host=tmp_path,
        )
    assert runner.last_cmd == ""


def test_run_benchmarks_missing_output_raises(tmp_path: Path):
    runner = _FakeRunner(out_dir=tmp_path, payloads={}, file_name={})
    handle = _FakeHandle(name="cvs_smoke", runner=runner)
    with pytest.raises(WorkloadError) as exc_info:
        run_benchmarks(
            benchmarks=["mmlu"],
            server_handle=handle,
            base_url="http://localhost:8000",
            model_id="x",
            model_path="/models/foo",
            output_dir_in_container=str(tmp_path),
            output_dir_on_host=tmp_path,
        )
    assert "no result JSON" in str(exc_info.value)


def test_run_benchmarks_invalid_json_raises(tmp_path: Path):
    # Pre-create the per-benchmark subdir with a malformed result file
    # named to match the vllm-serve glob.
    sub = tmp_path / "serve_synth_short"
    sub.mkdir()
    (sub / "result.json").write_text("{not json")
    runner = _FakeRunner(out_dir=tmp_path, payloads={}, file_name={})
    handle = _FakeHandle(name="cvs_smoke", runner=runner)
    with pytest.raises(WorkloadError) as exc_info:
        run_benchmarks(
            benchmarks=["serve_synth_short"],
            server_handle=handle,
            base_url="http://localhost:8000",
            model_id="x",
            model_path="/models/foo",
            output_dir_in_container=str(tmp_path),
            output_dir_on_host=tmp_path,
        )
    assert "not valid JSON" in str(exc_info.value)
