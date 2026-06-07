"""Catalog loader. Loads models.json + datasets.json from cvs/input/configs/catalog/.

Exposes:
- `load_catalog(input_dir)` -> Catalog with `.models`, `.datasets`, `.benchmarks`
- `Catalog.model_literal()` / `.dataset_literal()` -> typing.Literal[...] of known ids
- `Catalog.suggest(kind, bad_id)` -> str | None ("did you mean ...")

Benchmarks come from cvs.lib.dtni.benchmarks.registry (code, not JSON).
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ModelEntry:
    id: str
    hf_repo: str


@dataclass(frozen=True)
class DatasetEntry:
    id: str
    hf_repo: str


@dataclass(frozen=True)
class Catalog:
    models: dict[str, ModelEntry] = field(default_factory=dict)
    datasets: dict[str, DatasetEntry] = field(default_factory=dict)
    benchmarks: tuple[str, ...] = ()

    def suggest(self, kind: str, bad_id: str) -> str | None:
        pool = {
            "model": self.models.keys(),
            "dataset": self.datasets.keys(),
            "benchmark": self.benchmarks,
        }.get(kind, ())
        matches = difflib.get_close_matches(bad_id, pool, n=1, cutoff=0.6)
        return matches[0] if matches else None


def load_catalog(input_dir: Path) -> Catalog:
    """Load catalog/{models,datasets}.json. Benchmarks injected separately."""
    cat_dir = Path(input_dir) / "configs" / "catalog"
    models = _load_json_dict(cat_dir / "models.json", "models")
    datasets = _load_json_dict(cat_dir / "datasets.json", "datasets")
    return Catalog(
        models={k: ModelEntry(id=k, hf_repo=v["hf_repo"]) for k, v in models.items()},
        datasets={k: DatasetEntry(id=k, hf_repo=v["hf_repo"]) for k, v in datasets.items()},
        benchmarks=(),  # filled by registry import-time injection
    )


def _load_json_dict(path: Path, what: str) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"catalog file missing: {path} (expected {what})")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected top-level object, got {type(data).__name__}")
    for k, v in data.items():
        if not isinstance(v, dict) or "hf_repo" not in v:
            raise ValueError(f"{path}: entry {k!r} missing required field 'hf_repo'")
    return data
