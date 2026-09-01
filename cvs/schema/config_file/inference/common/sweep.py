"""
Shared inference sweep selector types and threshold coverage helpers.
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any, Dict, List, Optional

from pydantic import model_validator

from cvs.schema.base import _Forbid

NON_SWEEP_THRESHOLD_KEYS = {"accuracy", "mtp_quality", "long_context_accuracy", "quant_parity"}


class RoleServer(_Forbid):
    serve_args: Dict[str, Any] = {}
    env: Dict[str, str] = {}


class Roles(_Forbid):
    server: RoleServer = RoleServer()


class GoodputSlo(_Forbid):
    ttft_ms: float
    tpot_ms: float
    e2el_ms: float


class SeqCombo(_Forbid):
    name: str
    isl: str
    osl: str
    goodput_slo: Optional[GoodputSlo] = None


class Run(_Forbid):
    combo: str
    concurrency: int


def validate_thresholds_cover_sweep(
    *,
    expected_cells,
    thresholds,
    enforce_thresholds: bool,
    gated_metrics=None,
    gated_gpu_metrics=None,
) -> None:
    """Shared sweep/threshold coverage check for inference variant configs."""
    expected = set(expected_cells)
    present = set(thresholds.keys()) - NON_SWEEP_THRESHOLD_KEYS
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    problems = []
    if missing:
        problems.append(f"sweep cells with no threshold entry: {missing}")
    if extra:
        problems.append(f"threshold keys matching no sweep cell (typo?): {extra}")
    gated = gated_metrics if gated_metrics is not None else set()
    gated_keys = [f"client.{m}" for m in sorted(gated)]
    if gated_gpu_metrics:
        gated_keys += [f"gpu.{m}" for m in sorted(gated_gpu_metrics)]
    gated_gaps = {}
    for cell in sorted(expected & present):
        specs = thresholds.get(cell) or {}
        absent = [k for k in gated_keys if k not in specs]
        if absent:
            gated_gaps[cell] = absent
    if gated_gaps:
        problems.append(f"cells missing gated-metric specs: {gated_gaps}")
    if problems:
        msg = "threshold.json does not match the sweep matrix; " + "; ".join(problems)
        if enforce_thresholds:
            raise ValueError(msg)
        warnings.warn(f"{msg} (enforce_thresholds=false -> record-only)", stacklevel=3)


def validate_sweep_selector(combo_names, run_combo_refs):
    """Sweep-selector rule: combo names unique, every run.combo names one."""
    counts = Counter(combo_names)
    dupes = sorted(name for name, count in counts.items() if count > 1)
    if dupes:
        raise ValueError(f"duplicate sequence_combination names: {dupes}")
    known = set(counts)
    unknown = sorted({r for r in run_combo_refs if r not in known})
    if unknown:
        raise ValueError(f"run.combo names no sequence_combination: {unknown} (known: {sorted(known)})")


class Sweep(_Forbid):
    sequence_combinations: List[SeqCombo]
    runs: List[Run]

    @model_validator(mode="after")
    def _check_runs_reference_known_combos(self):
        validate_sweep_selector(
            [c.name for c in self.sequence_combinations],
            [r.combo for r in self.runs],
        )
        return self
