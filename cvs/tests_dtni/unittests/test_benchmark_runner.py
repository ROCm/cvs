"""Unit tests for cvs.lib.benchmarks.runner — projector + orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from cvs.lib.benchmarks.registry import lookup
from cvs.lib.benchmarks.runner import _project_scalars, run_benchmarks
from cvs.lib.errors import WorkloadError


# ---- projector ------------------------------------------------------------


def test_project_scalars_reads_bare_metric_key():
    payload = {"results": {"mmlu": {"acc": 0.732, "acc_stderr": 0.011}}}
    out = _project_scalars(lookup("mmlu"), payload)
    assert out == {"mmlu": 0.732, "mmlu_stderr": 0.011}


def test_project_scalars_prefers_filter_suffixed_key_for_gsm8k():
    # gsm8k spec uses score_filter="strict-match" — must beat ",none".
    payload = {"results": {"gsm8k": {
        "exact_match,strict-match": 0.812,
        "exact_match_stderr,strict-match": 0.009,
        "exact_match,none": 0.99,  # red herring — should NOT be picked
    }}}
    out = _project_scalars(lookup("gsm8k"), payload)
    assert out == {"gsm8k": 0.812, "gsm8k_stderr": 0.009}


def test_project_scalars_falls_back_to_comma_none():
    # mmlu spec uses score_filter="none"; verify that lookup hits.
    payload = {"results": {"mmlu": {"acc,none": 0.71}}}
    out = _project_scalars(lookup("mmlu"), payload)
    assert out == {"mmlu": 0.71}


def test_project_scalars_missing_task_returns_empty():
    payload = {"results": {}}
    out = _project_scalars(lookup("mmlu"), payload)
    assert out == {}


def test_project_scalars_skips_non_numeric_metric():
    payload = {"results": {"mmlu": {"acc": "n/a"}}}
    out = _project_scalars(lookup("mmlu"), payload)
    assert out == {}


# ---- runner orchestration -------------------------------------------------


@dataclass
class _FakeRunner:
    """Drops a per-benchmark results JSON into <out_dir>/<bid>/results.json."""
    out_dir: Path
    payloads: dict
    last_cmd: str = ""

    def exec(self, cmd: str, timeout: int = 0) -> str:
        self.last_cmd = cmd
        for bid, payload in self.payloads.items():
            if f"/{bid} " in cmd or cmd.endswith(f"/{bid}"):
                sub = self.out_dir / bid
                sub.mkdir(parents=True, exist_ok=True)
                (sub / "results_20260606.json").write_text(json.dumps(payload))
                return ""
        return ""


@dataclass
class _FakeHandle:
    name: str
    runner: _FakeRunner


def test_run_benchmarks_emits_scalars_per_id(tmp_path: Path):
    payloads = {
        "mmlu":  {"results": {"mmlu":  {"acc,none": 0.71, "acc_stderr,none": 0.01}}},
        "gsm8k": {"results": {"gsm8k": {"exact_match,strict-match": 0.82}}},
    }
    runner = _FakeRunner(out_dir=tmp_path, payloads=payloads)
    handle = _FakeHandle(name="cvs_smoke", runner=runner)

    scalars = run_benchmarks(
        benchmarks=["mmlu", "gsm8k"],
        server_handle=handle,
        base_url="http://localhost:8000",
        model_id="llama-3.1-8b",
        output_dir_in_container=str(tmp_path),
        output_dir_on_host=tmp_path,
    )

    assert scalars["mmlu"] == 0.71
    assert scalars["mmlu_stderr"] == 0.01
    assert scalars["gsm8k"] == 0.82


def test_run_benchmarks_unknown_id_raises_before_exec(tmp_path: Path):
    runner = _FakeRunner(out_dir=tmp_path, payloads={})
    handle = _FakeHandle(name="cvs_smoke", runner=runner)
    with pytest.raises(KeyError):
        run_benchmarks(
            benchmarks=["does-not-exist"],
            server_handle=handle,
            base_url="http://localhost:8000",
            model_id="x",
            output_dir_in_container=str(tmp_path),
            output_dir_on_host=tmp_path,
        )
    assert runner.last_cmd == ""


def test_run_benchmarks_missing_output_raises(tmp_path: Path):
    # FakeRunner won't write anything because payloads is empty.
    runner = _FakeRunner(out_dir=tmp_path, payloads={})
    handle = _FakeHandle(name="cvs_smoke", runner=runner)
    with pytest.raises(WorkloadError) as exc_info:
        run_benchmarks(
            benchmarks=["mmlu"],
            server_handle=handle,
            base_url="http://localhost:8000",
            model_id="x",
            output_dir_in_container=str(tmp_path),
            output_dir_on_host=tmp_path,
        )
    assert "no results JSON" in str(exc_info.value)


def test_run_benchmarks_invalid_json_raises(tmp_path: Path):
    # Pre-create the per-benchmark subdir with a malformed results file.
    sub = tmp_path / "mmlu"
    sub.mkdir()
    (sub / "results_bad.json").write_text("{not json")

    # Fake runner that does NOT overwrite the file.
    runner = _FakeRunner(out_dir=tmp_path, payloads={})
    handle = _FakeHandle(name="cvs_smoke", runner=runner)
    with pytest.raises(WorkloadError) as exc_info:
        run_benchmarks(
            benchmarks=["mmlu"],
            server_handle=handle,
            base_url="http://localhost:8000",
            model_id="x",
            output_dir_in_container=str(tmp_path),
            output_dir_on_host=tmp_path,
        )
    assert "not valid JSON" in str(exc_info.value)
