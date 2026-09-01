"""
TorchTitan training variant config schema.

Mirrors ``cvs/input/config_file/training/torchtitan/``.
"""

import warnings
from collections import Counter
from typing import Any, Dict, List

from pydantic import Field, model_validator
from typing_extensions import Literal

from cvs.schema.common.base import ContainerSpec
from cvs.schema.base import _Forbid


class TorchTitanSweepCombo(_Forbid):
    name: str
    micro_batch_size: str
    global_batch_size: str
    precision: str = ""


def validate_sweep_selector(combo_keys, run_refs):
    """The sweep-selector rule: combination keys unique, every run references one."""
    counts = Counter(combo_keys)
    dupes = sorted(k for k, count in counts.items() if count > 1)
    if dupes:
        raise ValueError(f"duplicate sweep.combinations keys: {dupes}")
    known = set(counts)
    unknown = sorted(r for r in run_refs if r not in known)
    if unknown:
        raise ValueError(f"sweep.runs references unknown combinations: {unknown} (known: {sorted(known)})")


def validate_thresholds_cover_sweep(
    *,
    expected_cells,
    thresholds,
    enforce_thresholds: bool,
    gated_metrics=None,
) -> None:
    """Shared sweep/threshold coverage check for training variant configs."""
    expected = set(expected_cells)
    present = set(thresholds.keys())
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    problems = []
    if missing:
        problems.append(f"sweep cells with no threshold entry: {missing}")
    if extra:
        problems.append(f"threshold keys matching no sweep cell (typo?): {extra}")
    gated = gated_metrics if gated_metrics is not None else set()
    gated_keys = [f"training.{m}" for m in sorted(gated)]
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


class TorchTitanSweep(_Forbid):
    combinations: Dict[str, TorchTitanSweepCombo]
    runs: List[str]

    @model_validator(mode="after")
    def _check_runs_reference_known_combos(self):
        validate_sweep_selector(
            list(self.combinations.keys()),
            self.runs,
        )
        return self


class ScalingBaseline(_Forbid):
    tokens_per_sec_total: float = 0.0
    num_nodes: int = 1


class LossCurveConfig(_Forbid):
    sample_every: int = 10
    milestone_steps: List[int] = Field(default_factory=lambda: [100, 500, 1000, 5000])
    max_slope: float = 0.0
    enforce: bool = True


class ConvergenceConfig(_Forbid):
    target_metric: Literal["auto", "train_loss", "eval_loss"] = "auto"
    target_value: float = 0.0


class CheckpointConfig(_Forbid):
    enforce: bool = False
    save_interval: int = 20
    save_iters: int = 21
    resume_iters: int = 25
    loss_rtol: float = 0.05
    checkpoint_dir: str = ""


class TorchTitanVariantConfig(_Forbid):
    schema_version: Literal[1]
    framework: Literal["torchtitan_single", "torchtitan_distributed"]
    gpu_arch: str
    enforce_thresholds: bool = True
    threshold_json: str = ""
    scaling_baseline: ScalingBaseline = Field(default_factory=ScalingBaseline)
    loss_curve: LossCurveConfig = Field(default_factory=LossCurveConfig)
    convergence: ConvergenceConfig = Field(default_factory=ConvergenceConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    config: Dict[str, Any]
    model_params: Dict[str, Any]
    container: ContainerSpec
    sweep: TorchTitanSweep
    thresholds: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    def cell_key(self, combo_key: str) -> str:
        combo = self.sweep.combinations[combo_key]
        return f"MBS={combo.micro_batch_size},GBS={combo.global_batch_size},PRECISION={combo.precision}"

    def expected_cells(self) -> List[str]:
        return [self.cell_key(k) for k in self.sweep.runs]

    @model_validator(mode="after")
    def _check_thresholds_cover_sweep(self):
        validate_thresholds_cover_sweep(
            expected_cells=self.expected_cells(),
            thresholds=self.thresholds,
            enforce_thresholds=self.enforce_thresholds,
            gated_metrics=set(),
        )
        return self
