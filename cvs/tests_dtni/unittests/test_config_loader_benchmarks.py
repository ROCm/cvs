"""Unit tests for cvs.lib.config_loader benchmark validation.

Scope is narrow on purpose: this file exists to lock in the v1 behavior that
load_workload rejects unknown benchmark ids at load time with a did-you-mean
hint, against BENCHMARK_REGISTRY.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cvs.lib.catalog import Catalog, ModelEntry
from cvs.lib.config_loader import load_workload
from cvs.lib.errors import CatalogError


def _minimal_workload(benchmarks: list[str]) -> dict:
    return {
        "schema_version": 1,
        "framework": "vllm_single",
        "gpu_arch": "mi300x",
        "paths": {
            "shared_fs": "/tmp/x",
            "models_dir": "/tmp/x/models",
            "datasets_dir": "/tmp/x/datasets",
            "artifacts_dir": "/tmp/x/artifacts",
            "images_dir": "/tmp/x/images",
        },
        "model": {"id": "llama-3.1-8b", "remote": 0, "precision": "bf16"},
        "benchmarks": benchmarks,
        "image": {"tag": "rocm/vllm:latest", "remote": 1},
        "topology": {"roles": {"server": {"count": 1, "gpus_per_node": 1}}},
        "roles": {
            "server": {
                "port": 8000,
                "health_path": "/health",
                "command": "vllm serve x",
            }
        },
    }


def _catalog() -> Catalog:
    return Catalog(
        models={"llama-3.1-8b": ModelEntry(id="llama-3.1-8b", hf_repo="x/y")},
        datasets={},
        benchmarks=(),
    )


def _write(tmp_path: Path, wl: dict) -> Path:
    p = tmp_path / "config.json"
    p.write_text(json.dumps(wl))
    return p


def test_known_benchmarks_load_clean(tmp_path: Path):
    p = _write(tmp_path, _minimal_workload(["mmlu", "gsm8k"]))
    wl = load_workload(p, catalog=_catalog())
    assert wl.benchmarks == ["mmlu", "gsm8k"]


def test_empty_benchmarks_load_clean(tmp_path: Path):
    p = _write(tmp_path, _minimal_workload([]))
    wl = load_workload(p, catalog=_catalog())
    assert wl.benchmarks == []


def test_unknown_benchmark_raises_with_did_you_mean(tmp_path: Path):
    p = _write(tmp_path, _minimal_workload(["mmmlu"]))
    with pytest.raises(CatalogError) as exc_info:
        load_workload(p, catalog=_catalog())
    msg = str(exc_info.value)
    assert "mmmlu" in msg
    assert "mmlu" in msg  # did-you-mean suggestion


def test_unknown_benchmark_no_close_match_lists_known(tmp_path: Path):
    p = _write(tmp_path, _minimal_workload(["totally-unrelated"]))
    with pytest.raises(CatalogError) as exc_info:
        load_workload(p, catalog=_catalog())
    msg = str(exc_info.value)
    assert "totally-unrelated" in msg
    assert "mmlu" in msg and "gsm8k" in msg
