'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Training-specific config schema for Megatron suites (single-node and distributed).

The framework-agnostic machinery (ContainerSpec, RuntimeSpec, placeholder
substitution, threshold file discovery) lives in `cvs.lib.utils.config_loader`.
This module holds the training half: MegatronSweepCombo, MegatronSweep,
MegatronVariantConfig, and load_training_variant.

Thresholds live in a sibling *threshold.json file (not inline in result_dict).
The threshold file is discovered via the `threshold_json` field in the config or
auto-discovered as the sole *threshold.json sibling. Cell keys in the threshold
file must match the combination keys in sweep.combinations exactly.

enforce_thresholds gates whether threshold specs are asserted in test_metric.

Both megatron_single and megatron_distributed are covered by MegatronVariantConfig
via the framework field, which is a validated schema tag / config discriminator.
'''

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any, Dict, List

from pydantic import Field, model_validator
from typing_extensions import Literal

from cvs.lib.utils.config_loader import (
    ContainerSpec,
    _Forbid,
    substitute_config,
)


# ---------- pydantic models (training) ----------


class MegatronSweepCombo(_Forbid):
    name: str
    micro_batch_size: str
    global_batch_size: str
    precision: str = ""


def validate_sweep_selector(combo_keys, run_refs):
    """The sweep-selector rule: combination keys unique, every run references one.

    Single home for this check, shared by the typed MegatronSweep validator
    (load time) and pytest_generate_tests (collection time, which reads raw
    JSON before the loader runs) so the two can never drift.

    Without it a typo'd run key is a silently-dropped cell — the sweep runs
    a different matrix than the config reads.
    """
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
    """Shared sweep/threshold coverage check for training variant configs.

    Checks every sweep cell has a threshold entry and no threshold key is
    orphaned. Individual metrics within a cell are optional — absent specs
    are skipped in test_metric (record-only for that metric).
    """
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


class MegatronSweep(_Forbid):
    combinations: Dict[str, MegatronSweepCombo]
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
    enforce: bool = False  # if False, test_checkpoint is skipped entirely
    save_interval: int = 20  # checkpoint written every N steps
    save_iters: int = 21  # save phase total; last checkpoint = floor(save/interval)*interval
    resume_iters: int = 25  # load phase total (must be > last_ckpt_step)
    loss_rtol: float = 0.05  # max allowed fractional loss increase across boundary
    checkpoint_dir: str = ""  # shared path for distributed; empty = derive from log_dir (single-node)


class MegatronVariantConfig(_Forbid):
    schema_version: Literal[1]
    framework: Literal["megatron_single", "megatron_distributed"]
    gpu_arch: str
    enforce_thresholds: bool = True
    threshold_json: str = ""
    scaling_baseline: ScalingBaseline = Field(default_factory=ScalingBaseline)
    loss_curve: LossCurveConfig = Field(default_factory=LossCurveConfig)
    convergence: ConvergenceConfig = Field(default_factory=ConvergenceConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    config: Dict[str, Any]  # training knobs: megatron_root, nccl_*, nic_type, ...
    model_params: Dict[str, Any]  # model knobs: model_name, precision, tp, pp, ...
    container: ContainerSpec
    sweep: MegatronSweep
    thresholds: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    def cell_key(self, combo_key: str) -> str:
        """Canonical threshold lookup key for a sweep combo.

        Constructs a key from the combo's micro_batch_size, global_batch_size,
        and precision — must match the top-level keys in the threshold file exactly.
        """
        combo = self.sweep.combinations[combo_key]
        return f"MBS={combo.micro_batch_size},GBS={combo.global_batch_size},PRECISION={combo.precision}"

    def expected_cells(self) -> List[str]:
        """Return the threshold cell key for every run in sweep.runs."""
        return [self.cell_key(k) for k in self.sweep.runs]

    @model_validator(mode="after")
    def _check_thresholds_cover_sweep(self):
        """Every sweep cell must have a threshold entry; no metric within it is
        mandatory. test_metric treats an absent ``training.*`` spec as
        "don't gate this metric" (skips the assertion), so a threshold.json
        is free to gate only the metrics an operator cares about.
        """
        validate_thresholds_cover_sweep(
            expected_cells=self.expected_cells(),
            thresholds=self.thresholds,
            enforce_thresholds=self.enforce_thresholds,
            gated_metrics=set(),
        )
        return self


# ---------- public API (training) ----------


def _check_no_changeme(node, path="", _offenders=None):
    """Recursively collect config fields whose value still contains '<changeme>'.

    Collects all offending dotted paths so the caller can report them all at once.
    """
    if _offenders is None:
        _offenders = []
    if isinstance(node, dict):
        for k, v in node.items():
            _check_no_changeme(v, f"{path}.{k}" if path else k, _offenders)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _check_no_changeme(v, f"{path}[{i}]", _offenders)
    elif isinstance(node, str) and "<changeme>" in node:
        _offenders.append(path)
    if not path:
        if _offenders:
            raise ValueError(f"config has unfilled placeholder '<changeme>' in: {', '.join(_offenders)}")


def load_training_variant(config_path, cluster_dict) -> MegatronVariantConfig:
    """Load and validate a Megatron training variant config + its threshold file.

    Delegates file read, placeholder substitution, and threshold file discovery
    to the generic substitute_config. The threshold file is located via the
    threshold_json field in the config (relative to the config file's directory)
    or auto-discovered as the sole *threshold.json sibling.

    Cell keys in the threshold file must match MegatronVariantConfig.cell_key()
    output exactly — MBS=<mbs>,GBS=<gbs>,PRECISION=<precision>. A load-time
    validator checks that every sweep cell has a threshold entry and no key is
    orphaned.
    """
    raw, thresholds = substitute_config(config_path, cluster_dict)

    # When checkpoint testing is disabled, checkpoint_dir and its shared-FS
    # volume mount are unused — exempt both from the <changeme> check so
    # operators can use the template as-is without filling in checkpoint paths.
    if not raw.get("checkpoint", {}).get("enforce", False):
        raw.get("checkpoint", {}).pop("checkpoint_dir", None)
        try:
            vols = raw["container"]["runtime"]["args"]["volumes"]
            raw["container"]["runtime"]["args"]["volumes"] = [
                v for v in vols if "<changeme>" not in v
            ]
        except (KeyError, TypeError):
            pass

    _check_no_changeme(raw)

    known = {k: v for k, v in raw.items() if k in MegatronVariantConfig.model_fields}
    known["thresholds"] = thresholds
    return MegatronVariantConfig(**known)
