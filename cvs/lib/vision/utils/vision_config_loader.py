'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Config schema for the pytorch_vision suite.

The framework-agnostic machinery (paths/model/image/container schema, the 3-pass
placeholder substitution, the `enforce_thresholds` gate, and the
`substitute_config` file-read helper) lives in `cvs.lib.utils.config_loader`.
This module holds the vision half: the model sweep selector
(`ModelCombo`/`Run`/`Sweep`), the benchmark `Params`, and
`VariantConfig(BaseVariantConfig)` with a MODEL/PREC/RES/BS `cell_key` and its
sweep-coverage check.

A vision benchmark does NOT sweep sequence lengths (isl/osl) the way a serving
suite does, so this defines its own sweep dimensions -- named
`(architecture, precision, input-resolution)` combos selected at explicit batch
sizes -- rather than reusing the inference sweep schema. Everything else
(container/paths/image handling, the coverage-check shape) mirrors the
`vllm_single` reference so the container image is pulled and launched by the
shared ContainerOrchestrator with no suite-specific plumbing.
'''

from __future__ import annotations

import warnings
from collections import Counter
from typing import Dict, List

from pydantic import model_validator
from typing_extensions import Literal

from cvs.lib.utils.config_loader import BaseVariantConfig, _Forbid, substitute_config
from cvs.lib.vision.utils.vision_parsing import GATED_METRICS


# ---------- pydantic models (vision) ----------


class ModelCombo(_Forbid):
    # `name` is the join key the `runs` selector references; required.
    # `arch` is a torchvision model name (e.g. "resnet50", "vit_b_16") the driver
    # instantiates via torchvision.models.get_model(arch, weights=None) -- random
    # weights, so the run is offline and measures compute, not accuracy.
    name: str
    arch: str
    precision: Literal["fp32", "fp16", "bf16"] = "fp16"
    # Square input resolution fed to the model (H == W). A model property, so it
    # lives on the combo rather than global Params.
    input_size: str = "224"


class Run(_Forbid):
    # One sweep cell: a named model combo run at a single batch size. The explicit
    # list of runs replaces an implicit combos x batch_sizes cartesian -- you
    # enumerate exactly the cells you want, no NxM explosion.
    combo: str
    batch_size: int


def validate_model_sweep(combo_names, run_combo_refs):
    """The sweep-selector rule: combo names unique, every run.combo names one.

    The single home for this check, shared by the typed `Sweep` validator (load
    time) and `pytest_generate_tests` (collection time, which reads raw JSON
    before the loader runs) so the two can never drift. Operates on plain lists
    of strings so both call sites can feed it.
    """
    counts = Counter(combo_names)
    dupes = sorted(name for name, count in counts.items() if count > 1)
    if dupes:
        raise ValueError(f"duplicate model_combination names: {dupes}")
    known = set(counts)
    unknown = sorted({r for r in run_combo_refs if r not in known})
    if unknown:
        raise ValueError(f"run.combo names no model_combination: {unknown} (known: {sorted(known)})")


class Sweep(_Forbid):
    model_combinations: List[ModelCombo]
    runs: List[Run]

    @model_validator(mode="after")
    def _check_runs_reference_known_combos(self):
        validate_model_sweep(
            [c.name for c in self.model_combinations],
            [r.combo for r in self.runs],
        )
        return self


class Params(_Forbid):
    # Benchmark knobs shared by every cell. All fields str (rendered as CLI args
    # to the in-container benchmark script), mirroring the vllm Params convention.
    warmup_iters: str = "10"
    bench_iters: str = "50"
    # "1"/"0": run the model with channels_last memory format (a common inference
    # win for conv nets on ROCm) and/or torch.compile. Parsed to a bool in the job.
    channels_last: str = "1"
    use_torch_compile: str = "0"
    # Per-cell benchmark wall-clock budget. torch.compile / large models on the
    # first cell can take minutes; the exec returns as soon as the script exits,
    # so a bigger budget only lengthens the failure path.
    bench_timeout_s: str = "1800"


class VariantConfig(BaseVariantConfig):
    framework: Literal["pytorch_vision"]
    gpu_arch: str
    # Extra env vars exported inside the container before the benchmark runs
    # (e.g. HIP_VISIBLE_DEVICES, MIOPEN flags). Merged over the orchestrator's
    # defaults; empty by default.
    env: Dict[str, str] = {}
    params: Params
    sweep: Sweep

    def cell_key(self, arch, precision, input_size, batch_size):
        """The canonical threshold key for one sweep cell.

        Single source of truth shared by the loader's coverage check and the
        test's verdict lookup -- so the two can never drift on whitespace,
        ordering, or field names.
        """
        return f"MODEL={arch},PREC={precision},RES={input_size},BS={batch_size}"

    def expected_cells(self):
        """Every (arch, precision, input_size, batch_size) cell the sweep picks."""
        by_name = {c.name: c for c in self.sweep.model_combinations}
        cells = []
        for run in self.sweep.runs:
            c = by_name[run.combo]
            cells.append(self.cell_key(c.arch, c.precision, c.input_size, run.batch_size))
        return cells

    @model_validator(mode="after")
    def _check_thresholds_cover_sweep(self):
        """Fail at load time if the threshold file does not match the sweep.

        Two axes are checked. (1) Cell coverage: every sweep cell has a threshold
        entry, and no threshold key names a cell the sweep does not run. (2)
        Gated-metric coverage: every present cell carries a spec for every
        GATED_METRICS member. Without axis 2 a gated metric with no spec falls
        through test_metric's `spec is None` record-only branch and reports a
        green PASS with zero assertions, even under enforce_thresholds=true.

        When `enforce_thresholds` is false the same mismatch is a warning rather
        than an error -- the config loads as a record-only scaffold (metrics
        captured, no assertions) instead of failing. This is the expected mode
        for a fresh, un-calibrated suite.
        """
        expected = set(self.expected_cells())
        present = set(self.thresholds.keys())
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        problems = []
        if missing:
            problems.append(f"sweep cells with no threshold entry: {missing}")
        if extra:
            problems.append(f"threshold keys matching no sweep cell (typo?): {extra}")
        gated_keys = [f"vision.{m}" for m in sorted(GATED_METRICS)]
        gated_gaps = {}
        for cell in sorted(expected & present):
            specs = self.thresholds.get(cell) or {}
            absent = [k for k in gated_keys if k not in specs]
            if absent:
                gated_gaps[cell] = absent
        if gated_gaps:
            problems.append(f"cells missing gated-metric specs: {gated_gaps}")
        if problems:
            msg = "threshold.json does not match the sweep matrix; " + "; ".join(problems)
            if self.enforce_thresholds:
                raise ValueError(msg)
            warnings.warn(f"{msg} (enforce_thresholds=false -> record-only)", stacklevel=2)
        return self


# ---------- public API (vision) ----------


def load_variant(config_path, cluster_dict):
    """Load and validate a pytorch_vision variant config + its sibling threshold file.

    Delegates the file read + placeholder substitution + threshold discovery to
    the generic `substitute_config`, then attaches the thresholds and builds the
    typed `VariantConfig`.
    """
    raw, thresholds = substitute_config(config_path, cluster_dict)
    raw["thresholds"] = thresholds
    return VariantConfig(**raw)
