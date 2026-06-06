"""Filename sanitization + write_artifacts roundtrip."""

import json

import pandas as pd

from cvs.lib.artifact_writer import artifact_basename, sanitize_for_filename, write_artifacts


def test_sanitize_strips_slashes_and_lowercases():
    assert sanitize_for_filename("Meta-LLAMA/Llama-3.1") == "meta-llama-llama-3.1"


def test_sanitize_strips_leading_dashes():
    assert sanitize_for_filename("//foo") == "foo"


def test_basename_format():
    name = artifact_basename(arch="mi300x", model_id="qwen3-next-80b",
                             framework="vllm", workload_hash="abc123def456", ts="20260605T143000Z")
    assert name == "mi300x_qwen3-next-80b_vllm_abc123def456_20260605T143000Z"


def test_write_artifacts_writes_all_three(tmp_path):
    df = pd.DataFrame([{"a": 1.0, "b": 2.0}])
    out = write_artifacts(
        out_dir=tmp_path, basename="x",
        samples_df=df, log_text="hello",
        verdict={"passed": True, "verdicts": []},
    )
    assert (tmp_path / "x.parquet").exists()
    assert (tmp_path / "x.log").read_text() == "hello"
    assert json.loads((tmp_path / "x.verdict.json").read_text())["passed"] is True
    assert set(out) == {"parquet", "log", "verdict"}


def test_write_skips_parquet_when_no_samples(tmp_path):
    out = write_artifacts(
        out_dir=tmp_path, basename="x",
        samples_df=None, log_text="",
        verdict={"failed_phase": "launch"},
    )
    assert "parquet" not in out
    assert (tmp_path / "x.verdict.json").exists()
