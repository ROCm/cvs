"""Catalog: load + suggest 'did you mean'."""

import json

from cvs.lib.catalog import Catalog, ModelEntry, load_catalog


def test_suggest_close_match():
    cat = Catalog(
        models={
            "llama-3.1-8b": ModelEntry("llama-3.1-8b", "meta-llama/Llama-3.1-8B-Instruct"),
            "llama-3.1-70b-fp8": ModelEntry("llama-3.1-70b-fp8", "x"),
        },
    )
    assert cat.suggest("model", "llama-3.1-70-fp8") == "llama-3.1-70b-fp8"


def test_suggest_no_match_returns_none():
    cat = Catalog(models={"a": ModelEntry("a", "x")})
    assert cat.suggest("model", "totally-different") is None


def test_load_catalog_roundtrip(tmp_path):
    cat_dir = tmp_path / "configs" / "catalog"
    cat_dir.mkdir(parents=True)
    (cat_dir / "models.json").write_text(json.dumps({
        "llama-3.1-8b": {"hf_repo": "meta-llama/Llama-3.1-8B-Instruct"},
    }))
    (cat_dir / "datasets.json").write_text(json.dumps({
        "mmlu": {"hf_repo": "cais/mmlu"},
    }))
    cat = load_catalog(tmp_path)
    assert cat.models["llama-3.1-8b"].hf_repo == "meta-llama/Llama-3.1-8B-Instruct"
    assert cat.datasets["mmlu"].hf_repo == "cais/mmlu"
