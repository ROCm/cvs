'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Inference-specific config schema for the vllm_single suite.

The framework-agnostic machinery (paths/model/image/container schema, the
3-pass placeholder substitution, the `enforce_thresholds` gate, and the
`substitute_config` file-read helper) lives in `cvs.lib.utils.config_loader`.
This module holds the inference half: the sweep selector
(`SeqCombo`/`Run`/`Sweep`), the goodput SLO, the framework `Params`, the
`server` role, and `VariantConfig(BaseVariantConfig)` with the
ISL/OSL/TP/CONC `cell_key` and its sweep-coverage check.

`Sweep`/`SeqCombo`/`GoodputSlo`/`Roles`/`cell_key` are inference-generic (any
serving framework sweeps sequence shapes at concurrencies); only `Params` (the
`vllm bench serve` flags) is framework-flavored and is the seam to subclass
when a second serving framework lands.
'''

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any, Dict, List, Optional

from pydantic import model_validator
from typing_extensions import Literal

from cvs.lib.utils.config_loader import BaseVariantConfig, _Forbid, substitute_config
from cvs.lib.inference.utils.vllm_parsing import GATED_METRICS


# ---------- pydantic models (inference) ----------


class RoleServer(_Forbid):
    # The `vllm serve` command is built in Python (cvs.lib.inference.vllm_single),
    # not cloned from a `.sh` script, so a run is self-contained. Per-model
    # server quirks live here: extra `vllm serve` flags (serve_args) and env vars
    # merged over the defaults the orchestrator sets. Both default empty; the
    # fp8-kv cell sets its --kv-cache-dtype via serve_args, kept out of the
    # generic driver so it stays model-agnostic. A {flag: value} map (flag
    # without the leading --): a scalar renders `--flag value`, True renders a
    # bare `--flag`, and a list renders the flag once per element. Cleaner than
    # a flat [flag, value, flag, value] list and still covers vllm's bare and
    # repeatable flags.
    serve_args: Dict[str, Any] = {}
    env: Dict[str, str] = {}


class Roles(_Forbid):
    server: RoleServer = RoleServer()


class GoodputSlo(_Forbid):
    # Per-cell SLOs for the goodput gate, in milliseconds. An INPUT to the run
    # (passed to `vllm bench serve --goodput`), NOT a threshold to assert -- so
    # it lives in the sweep, not threshold.json. Attached per seq_combo because
    # e2el scales ~linearly with osl. _Forbid: a typo'd key fails load, not a
    # silently-dropped SLO.
    ttft_ms: float
    tpot_ms: float
    e2el_ms: float


class SeqCombo(_Forbid):
    # `name` is the join key the `runs` selector references; required.
    name: str
    isl: str
    osl: str
    goodput_slo: Optional[GoodputSlo] = None


class Run(_Forbid):
    # One sweep cell: a named combo run at a single concurrency. The explicit
    # list of Runs replaces the old `sequence_combinations x concurrency_levels`
    # cartesian -- you enumerate exactly the cells you want, no NxM explosion.
    combo: str
    concurrency: int


def validate_thresholds_cover_sweep(
    *,
    expected_cells,
    thresholds,
    enforce_thresholds: bool,
    gated_metrics=None,
) -> None:
    """Shared sweep/threshold coverage check for inference variant configs."""
    expected = set(expected_cells)
    present = set(thresholds.keys())
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    problems = []
    if missing:
        problems.append(f"sweep cells with no threshold entry: {missing}")
    if extra:
        problems.append(f"threshold keys matching no sweep cell (typo?): {extra}")
    gated = gated_metrics if gated_metrics is not None else GATED_METRICS
    gated_keys = [f"client.{m}" for m in sorted(gated)]
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
    """The sweep-selector rule: combo names unique, every run.combo names one.

    The single home for this check, shared by the typed `Sweep` validator (load
    time) and `pytest_generate_tests` (collection time, which reads raw JSON
    before the loader runs) so the two can never drift. Operates on plain lists
    of strings so both call sites can feed it.

    Without it a duplicate name silently shadows a combo and a typo'd
    `run.combo` is a silently-dropped cell -- either way the sweep runs a
    different matrix than the config reads.
    """
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


class Params(_Forbid):
    backend: str = "vllm"
    base_url: str = "http://0.0.0.0"
    port_no: str = "8888"
    dataset_name: str = "random"
    burstiness: str = "1.0"
    seed: str = "0"
    request_rate: str = "inf"
    random_range_ratio: str = "0.8"
    random_prefix_len: str = "0"
    tensor_parallelism: str = "1"
    tokenizer_mode: str = "auto"
    percentile_metrics: str = "ttft,tpot,itl,e2el"
    metric_percentiles: str = "50,90,95,99"
    num_prompts: str = "3200"
    # Completion-poll budget for the bench client = client_poll_count * 60s
    # (plus a 120s initial wait). Large-output cells (high osl) need a bigger
    # budget; the poll loop exits as soon as the client finishes, so raising
    # this never slows down fast cells. See regressions REG-20260609-001.
    client_poll_count: str = "20"


class VariantConfig(BaseVariantConfig):
    framework: Literal["vllm_single"]
    gpu_arch: str
    roles: Roles = Roles()
    params: Params
    sweep: Sweep

    def cell_key(self, isl, osl, concurrency):
        """The canonical threshold key for one sweep cell.

        Single source of truth shared by the loader's coverage check and the
        test's verdict lookup -- so the two can never drift on whitespace,
        ordering, or field names.
        """
        return f"ISL={isl},OSL={osl},TP={self.params.tensor_parallelism},CONC={concurrency}"

    def expected_cells(self):
        """Every (isl, osl, conc) cell the sweep's `runs` selector picks."""
        by_name = {c.name: c for c in self.sweep.sequence_combinations}
        return [self.cell_key(by_name[r.combo].isl, by_name[r.combo].osl, r.concurrency) for r in self.sweep.runs]

    @model_validator(mode="after")
    def _check_thresholds_cover_sweep(self):
        """Fail at load time if any sweep cell lacks a threshold entry."""
        validate_thresholds_cover_sweep(
            expected_cells=self.expected_cells(),
            thresholds=self.thresholds,
            enforce_thresholds=self.enforce_thresholds,
        )
        return self


# ---------- public API (inference) ----------


def load_variant(config_path, cluster_dict):
    """Load and validate a vllm_single variant config + its sibling threshold file.

    Delegates the file read + placeholder substitution + threshold discovery to
    the generic `substitute_config`, then attaches the thresholds and builds the
    typed `VariantConfig`.
    """
    raw, thresholds = substitute_config(config_path, cluster_dict)
    raw["thresholds"] = thresholds
    return VariantConfig(**raw)
