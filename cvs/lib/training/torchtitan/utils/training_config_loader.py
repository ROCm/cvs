'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Training-specific config schema for TorchTitan suites (single-node and distributed).

The framework-agnostic machinery (ContainerSpec, RuntimeSpec, placeholder
substitution, threshold file discovery) lives in `cvs.lib.utils.config_loader`.
This module holds the training half: TorchTitanSweepCombo, TorchTitanSweep,
TorchTitanVariantConfig, and load_training_variant.

Thresholds live in a sibling *threshold.json file (not inline in result_dict).
The threshold file is discovered via the `threshold_json` field in the config or
auto-discovered as the sole *threshold.json sibling. Cell keys in the threshold
file must match the combination keys in sweep.combinations exactly.

enforce_thresholds gates whether threshold specs are asserted in test_metric.

Both torchtitan_single and torchtitan_distributed are covered by TorchTitanVariantConfig
via the framework field, which is a validated schema tag / config discriminator.
'''

from __future__ import annotations

import re
import warnings
from collections import Counter
from typing import Any, Dict, List

from pydantic import Field, field_validator, model_validator
from typing_extensions import Literal

from cvs.lib.utils.config_loader import (
    ContainerSpec,
    _Allow,
    _Forbid,
    substitute_config,
)


# ---------- constants ----------

DEFAULT_SWEEP_NAME = "default"
_DEFAULT_MBS = "1"
_DEFAULT_GBS = "16"
_DEFAULT_PRECISION = "BF16"
_SWEEP_KEY_PATTERN = re.compile(
    r"^MBS=(?P<micro_batch_size>[^,]+),"
    r"GBS=(?P<global_batch_size>[^,]+),"
    r"PRECISION=(?P<precision>[^,]+)$"
)

# ---------- pydantic models (training) ----------


class TorchTitanSweepCombo(_Allow):
    name: str
    micro_batch_size: str
    global_batch_size: str
    precision: str = ""


def parse_sweep_cell_key(key):
    """Parse MBS, GBS, and precision from a canonical sweep cell key."""
    match = _SWEEP_KEY_PATTERN.fullmatch(key)
    if not match:
        raise ValueError(
            f"invalid sweep combination key {key!r}; expected "
            "MBS=<micro_batch_size>,GBS=<global_batch_size>,PRECISION=<precision>"
        )
    return match.groupdict()


def sweep_cell_key(combo) -> str:
    """Threshold / combination key: MBS=<mbs>,GBS=<gbs>,PRECISION=<precision>."""
    if isinstance(combo, dict):
        mbs = combo["micro_batch_size"]
        gbs = combo["global_batch_size"]
        precision = combo.get("precision", "")
    else:
        mbs = combo.micro_batch_size
        gbs = combo.global_batch_size
        precision = combo.precision
    return f"MBS={mbs},GBS={gbs},PRECISION={precision}"


def validate_combo_keys_match_params(combinations) -> None:
    """Combination dict keys must equal sweep_cell_key() for that cell."""
    mismatches = [
        f"{key!r} (expected {sweep_cell_key(combo)!r})"
        for key, combo in combinations.items()
        if key != DEFAULT_SWEEP_NAME and key != sweep_cell_key(combo)
    ]
    if mismatches:
        raise ValueError(
            "sweep.combinations keys must equal "
            "MBS=<micro_batch_size>,GBS=<global_batch_size>,PRECISION=<precision>: " + "; ".join(mismatches)
        )


def validate_sweep_selector(combo_keys, run_refs):
    """The sweep-selector rule: combination keys unique, every run references one.

    Single home for this check, shared by the typed TorchTitanSweep validator
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


class TorchTitanSweep(_Forbid):
    combinations: Dict[str, TorchTitanSweepCombo]
    runs: List[str]

    @model_validator(mode="before")
    @classmethod
    def _assign_params_from_keys(cls, data):
        """Parse MBS/GBS/PRECISION from combination keys and inject into combo bodies.

        Allows minimal combo bodies (just name + overlays) while validating key format.
        """
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        normalized["combinations"] = {}
        for key, raw_combo in data.get("combinations", {}).items():
            combo = dict(raw_combo)
            if key == DEFAULT_SWEEP_NAME:
                combo.setdefault("micro_batch_size", _DEFAULT_MBS)
                combo.setdefault("global_batch_size", _DEFAULT_GBS)
                combo.setdefault("precision", _DEFAULT_PRECISION)
                normalized["combinations"][key] = combo
                continue
            parsed = parse_sweep_cell_key(key)
            for param, value in parsed.items():
                configured = combo.get(param)
                if configured is not None and str(configured) != value:
                    raise ValueError(
                        f"sweep combination {key!r} sets {param}={configured!r}; the key defines {param}={value!r}"
                    )
                combo[param] = value
            normalized["combinations"][key] = combo
        return normalized

    @model_validator(mode="after")
    def _check_runs_reference_known_combos(self):
        validate_sweep_selector(
            list(self.combinations.keys()),
            self.runs,
        )
        validate_combo_keys_match_params(self.combinations)
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


class SmokeConfig(_Forbid):
    """Fixed cell for test_smoke (opt-OUT; on by default).

    Empty global_batch_size lets the suite use its topology default
    (single-node 8, distributed 16).
    """

    enabled: bool = True
    iters: int = 10
    micro_batch_size: str = "1"
    global_batch_size: str = ""
    precision: str = "BF16"


class CheckpointConfig(_Forbid):
    enforce: bool = False  # if False, test_checkpoint is skipped entirely
    save_interval: int = 20  # checkpoint written every N steps
    save_iters: int = 21  # save phase total; last checkpoint = floor(save/interval)*interval
    resume_iters: int = 25  # load phase total (must be > last_ckpt_step)
    loss_rtol: float = 0.05  # max allowed fractional loss increase across boundary
    checkpoint_dir: str = ""  # shared path for distributed; empty = derive from log_dir (single-node)


class TorchTitanPaths(_Forbid):
    hf_token_file: str
    log_dir: str
    scripts_dir: str
    data_cache_dir: str
    rocm_dir: str = ""


class TorchTitanContainerSpec(ContainerSpec):
    env: Dict[str, str] = Field(default_factory=dict)


# container.env uses real process env names; jobs still read the lowercase aliases.
_CONTAINER_ENV_TO_JOB = {
    "NNODES": "nnodes",
    "MASTER_ADDR": "master_address",
    "NCCL_IB_HCA": "nccl_ib_hca",
    "NCCL_SOCKET_IFNAME": "nccl_socket_ifname",
    "GLOO_SOCKET_IFNAME": "gloo_socket_ifname",
    "NCCL_DEBUG": "nccl_debug",
    "NCCL_IB_GID_INDEX": "nccl_ib_gid_index",
}


class TorchTitanVariantConfig(_Forbid):
    gpu_name: str
    enforce_thresholds: bool = True
    threshold_json: str = ""
    paths: TorchTitanPaths
    verify_network_errors: str = "False"
    scaling_baseline: ScalingBaseline = Field(default_factory=ScalingBaseline)
    smoke: SmokeConfig = Field(default_factory=SmokeConfig)
    loss_curve: LossCurveConfig = Field(default_factory=LossCurveConfig)
    convergence: ConvergenceConfig = Field(default_factory=ConvergenceConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    train_params: Dict[str, Any]
    container: TorchTitanContainerSpec
    sweep: TorchTitanSweep
    thresholds: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("gpu_name")
    @classmethod
    def _uppercase_gpu_name(cls, value: str) -> str:
        return value.strip().upper()

    @property
    def gpu_arch(self) -> str:
        """Backward compatibility alias for gpu_name."""
        return self.gpu_name

    @property
    def model_params(self) -> Dict[str, Any]:
        """Backward compatibility alias for train_params."""
        return self.train_params

    def job_config_dict(self) -> Dict[str, Any]:
        """Flatten paths + container.env + train_params.training_iterations into the job dict."""
        merged: Dict[str, Any] = {}
        merged["verify_network_errors"] = self.verify_network_errors
        merged["hf_token_file"] = self.paths.hf_token_file
        merged["log_dir"] = self.paths.log_dir
        merged["scripts_dir"] = self.paths.scripts_dir
        merged["data_cache_dir"] = self.paths.data_cache_dir
        merged["rocm_dir"] = self.paths.rocm_dir
        merged.update(self.container.env)
        for env_key, job_key in _CONTAINER_ENV_TO_JOB.items():
            if env_key in self.container.env:
                merged[job_key] = self.container.env[env_key]
        iters = self.train_params.get("training_iterations")
        if iters is not None:
            merged["training_iterations"] = iters
        # TorchTitan-specific: add torchtitan_root and nic_type if present
        if "torchtitan_root" in self.train_params:
            merged["torchtitan_root"] = self.train_params["torchtitan_root"]
        if "nic_type" in self.train_params:
            merged["nic_type"] = self.train_params["nic_type"]
        return merged

    def cell_key(self, combo_key: str) -> str:
        """Canonical threshold lookup key for a sweep combo.

        Constructs a key from the combo's micro_batch_size, global_batch_size,
        and precision — must match the top-level keys in the threshold file exactly.
        """
        return sweep_cell_key(self.sweep.combinations[combo_key])

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


def load_training_variant(config_path, cluster_dict) -> TorchTitanVariantConfig:
    """Load and validate a TorchTitan training variant config + its threshold file.

    Delegates file read, placeholder substitution, and threshold file discovery
    to the generic substitute_config. The threshold file is located via the
    threshold_json field in the config (relative to the config file's directory)
    or auto-discovered as the sole *threshold.json sibling.

    Cell keys in the threshold file must match TorchTitanVariantConfig.cell_key()
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
            raw["container"]["runtime"]["args"]["volumes"] = [v for v in vols if "<changeme>" not in v]
        except (KeyError, TypeError):
            pass

    _check_no_changeme(raw)

    known = {k: v for k, v in raw.items() if k in TorchTitanVariantConfig.model_fields}
    known["thresholds"] = thresholds
    return TorchTitanVariantConfig(**known)
